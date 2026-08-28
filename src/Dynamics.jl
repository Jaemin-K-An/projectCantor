# ============================================================================
# Dynamics.jl — the gated 1-D state equation and its integrators.
#
#     dh/dt = -α (h - 1/2) + g(h) · δ(t),        h(0) = h₀
#
# NAMING: at this stage the vector field contains NO trained parameters, so
# this is a *Cantor-gated dynamical system*, not a Neural ODE. The term
# "Neural ODE" is used only in src/NeuralODEModel.jl, where f_θ is learned.
#
# ---------------------------------------------------------------------------
# TWO INTEGRATORS, ON PURPOSE
#
# `simulate_rk4`      fixed-step classical RK4. The workhorse for the sweeps.
# `simulate_adaptive` Tsit5 / Vern9 from the SciML stack. The cross-check.
#
# The hard gate makes the right-hand side DISCONTINUOUS in the state, so the
# solution is a Carathéodory/Filippov solution and adaptive error control is
# not meaningful across a switching surface (an adaptive solver can also stall
# by rejecting steps forever near a sliding mode). A fixed step of Δt is a
# well-defined regularisation at scale Δt, is deterministic, and never stalls,
# which is why it is the primary method. `scripts/run_numerical_analysis.jl`
# quantifies the disagreement between the two on a representative subset;
# results are reported, not hidden.
#
# BOUNDARY EXTENSION CONVENTION: the gate is evaluated at `clamp(h, 0, 1)`.
# Since 0 and 1 belong to K_n, and to the pass set of G4, the perturbation is
# NOT blocked when the state leaves the unit interval. This reproduces the
# original study's `clamp(h, 0.001, 0.999)` behaviour. G3 (periodic) is the one
# family whose pass set excludes 0 and 1; the sensitivity of the ablation to
# this convention is measured in scripts/run_ablation.jl.
# ============================================================================

"""
    make_rhs(gate, δ, α) -> f(h, t)

Scalar right-hand side of the gated system. `gate` is any [`Gate`](@ref),
`δ` any [`Perturb`](@ref), `α > 0` the restoring-force constant.
"""
function make_rhs(gate, δ, α::Float64)
    return (h, t) -> -α * (h - H_SAFE) + gate_value(gate, clamp(h, 0.0, 1.0)) * δ(t)
end

"""
    simulate_rk4(gate, δ; α, h0, T, dt, save_every) -> (ts, hs, gs)

Fixed-step RK4 on `[0,T]`. Returns the saved time grid, the state and the gate
value `g(h(t))` on that grid. The saved grid is uniform with spacing
`dt*save_every`, as required by [`compute_metrics`](@ref).
"""
function simulate_rk4(gate, δ; α::Float64 = 0.8, h0::Float64 = 0.15,
                      T::Float64 = 30.0, dt::Float64 = 0.001,
                      save_every::Int = 2)
    f = make_rhs(gate, δ, α)
    nsteps = round(Int, T / dt)
    nsave  = nsteps ÷ save_every
    ts = Vector{Float64}(undef, nsave + 1)
    hs = Vector{Float64}(undef, nsave + 1)
    gs = Vector{Float64}(undef, nsave + 1)
    h = h0; t = 0.0
    ts[1] = 0.0; hs[1] = h; gs[1] = gate_value(gate, clamp(h, 0.0, 1.0))
    j = 1
    @inbounds for k in 1:nsteps
        k1 = f(h, t)
        k2 = f(h + dt/2 * k1, t + dt/2)
        k3 = f(h + dt/2 * k2, t + dt/2)
        k4 = f(h + dt * k3,   t + dt)
        h += dt/6 * (k1 + 2k2 + 2k3 + k4)
        t = k * dt
        if k % save_every == 0
            j += 1
            j > nsave + 1 && break
            ts[j] = t; hs[j] = h; gs[j] = gate_value(gate, clamp(h, 0.0, 1.0))
        end
    end
    return ts, hs, gs
end

"""
    run_case(gate, δ; α, h0, T, dt, save_every, hold) -> TrajectoryMetrics

Integrate once and reduce to metrics. The trajectory itself is discarded, which
is what makes the parameter sweep memory-feasible.
"""
function run_case(gate, δ; α = 0.8, h0 = 0.15, T = 30.0, dt = 0.001,
                  save_every = 2, hold = 5.0)
    ts, hs, gs = simulate_rk4(gate, δ; α = α, h0 = h0, T = T, dt = dt,
                              save_every = save_every)
    return compute_metrics(ts, hs, gs; hold = hold)
end

# ---------------------------------------------------------------------------
# SciML cross-check
# ---------------------------------------------------------------------------

using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqTsit5: Tsit5
using OrdinaryDiffEqVerner: Vern9

"""
    simulate_adaptive(gate, δ; α, h0, T, dt_save, alg, abstol, reltol, dtmax)

Adaptive integration via the SciML stack, sampled on the same uniform
`dt_save` grid. `alg` is `Tsit5()` or `Vern9()`. Used for the tolerance and
solver sanity checks, never for the bulk sweeps.
"""
function simulate_adaptive(gate, δ; α::Float64 = 0.8, h0::Float64 = 0.15,
                           T::Float64 = 30.0, dt_save::Float64 = 0.002,
                           alg = Tsit5(), abstol = 1e-8, reltol = 1e-6,
                           dtmax = 0.005)
    f = make_rhs(gate, δ, α)
    rhs!(du, u, p, t) = (du[1] = f(u[1], t))
    prob = ODEProblem(rhs!, [h0], (0.0, T))
    sol = solve(prob, alg; saveat = dt_save, dtmax = dtmax,
                abstol = abstol, reltol = reltol, maxiters = 10^8)
    ts = collect(sol.t)
    hs = [sol.u[i][1] for i in eachindex(sol.u)]
    gs = [gate_value(gate, clamp(h, 0.0, 1.0)) for h in hs]
    return ts, hs, gs
end

"""
    analytic_no_filter(t; α, A, ω, h0)

Closed-form solution of the ungated system `dh/dt = -α(h-1/2) + A sin(ωt)`:

    h(t) = 1/2 + (h₀ - 1/2 + Aω/(α²+ω²)) e^{-αt}
           + A (α sin ωt - ω cos ωt)/(α²+ω²)

Used to validate the integrators against exact truth (test/runtests.jl) and to
give the analytic steady-state occupancy quoted in docs/BASELINE_AUDIT.md.
"""
function analytic_no_filter(t; α = 0.8, A = 1.5, ω = 4.0, h0 = 0.15)
    d = α^2 + ω^2
    c = h0 - H_SAFE + A * ω / d
    return H_SAFE + c * exp(-α * t) + A * (α * sin(ω * t) - ω * cos(ω * t)) / d
end

"""
    analytic_no_filter_occupancy(; α, A, ω)

Asymptotic (transient-free) safe occupancy of the ungated system. In steady
state `h - 1/2 = R sin(ωt - ψ)` with `R = A/√(α²+ω²)`, so the fraction of time
with `|h-1/2| ≤ 1/6` is `(2/π) asin(1/(6R))` when `R > 1/6`, else `1`.
"""
function analytic_no_filter_occupancy(; α = 0.8, A = 1.5, ω = 4.0)
    R = A / sqrt(α^2 + ω^2)
    R ≤ 1/6 && return 1.0
    return (2 / π) * asin(1 / (6R))
end
