# ============================================================================
# BarrierDynamics.jl — the V2 synthetic threat-coordinate dynamics.
#
#     ṙ = f(r) + δ(t) − η V_L'(r),      r ∈ [0,1] (threat coordinate)
#
# r ≈ 0 deep safe, r = ½ decision boundary, r ≈ 1 unsafe.
#
# ---------------------------------------------------------------------------
# WHY THE STEP SIZE IS ADAPTIVE
#
# The level-k barriers have width 3^{-k}. A fixed step Δt lets the state move
# |ṙ|Δt per step, and if that exceeds a barrier's width the integrator TUNNELS
# straight through it — silently deleting exactly the fine structure this study
# is about, and doing so more to Cantor's deepest levels than to anything else.
# So the step is capped by a displacement budget ε tied to the narrowest gap:
#
#     Δt = clamp( ε / (|ṙ| + tiny), Δt_min, Δt_max ),   ε = 3^{-n} / step_safety
#
# Cost is proportional to path length, not to T, so slow phases are cheap.
# `scripts/v2/run_synthetic_barrier.jl` re-runs a subset at halved ε as a
# convergence check and reports the disagreement.
# ============================================================================

module BarrierDynamics

using Random, Statistics
using ..CantorBarrier

export ThreatField, linear_field, cubic_field, bistable_field,
       simulate_barrier, BarrierMetrics, barrier_metrics,
       containment_point, containment_curve

# ---------------------------------------------------------------------------
# nominal (uncontrolled) dynamics
# ---------------------------------------------------------------------------

"""A named nominal vector field `f(r)` with a recorded `sup_r f` on [0,1]."""
struct ThreatField
    f::Function
    name::String
    supf::Float64
end
(F::ThreatField)(r) = F.f(r)

"""A — linear relaxation toward a nominal safe level `r₀`."""
function linear_field(; α = 1.0, r0 = 0.2)
    f = r -> -α * (r - r0)
    ThreatField(f, "linear", maximum(f, range(0, 1; length = 4001)))
end

"""B — cubic-stiffened relaxation; still a single global attractor at `r₀`."""
function cubic_field(; α = 1.0, γ = 4.0, r0 = 0.2)
    f = r -> -α * (r - r0) - γ * (r - r0)^3
    ThreatField(f, "cubic", maximum(f, range(0, 1; length = 4001)))
end

"""
C — bistable: stable safe attractor at `rs`, stable unsafe attractor at `ru`,
UNSTABLE separatrix at `rc = ½`. This is the model that matches the safety
picture: a jailbreak is a disturbance that pushes the state over the
separatrix into the unsafe basin, where it stays without further help.
"""
function bistable_field(; k = 12.0, rs = 0.15, rc = 0.5, ru = 0.85)
    f = r -> -k * (r - rs) * (r - rc) * (r - ru)
    ThreatField(f, "bistable", maximum(f, range(0, 1; length = 4001)))
end

# ---------------------------------------------------------------------------
# integrator
# ---------------------------------------------------------------------------

"""
    simulate_barrier(L, F, δ; η, r0, T, n_for_scale, step_safety, dt_max, dt_min,
                     n_save) -> (ts, rs)

Adaptive-step RK4 of `ṙ = f(r) + δ(t) − η V_L'(r)`, sampled onto a uniform grid
of `n_save+1` points by linear interpolation (so metrics see a uniform grid).
The controller is evaluated at `clamp(r,0,1)`; the state itself is NOT clamped,
because every nominal field above already points inward outside `[0,1]`.
"""
function simulate_barrier(L::BarrierLayout, F::ThreatField, δ;
                          η::Float64 = 1.0, r0::Float64 = 0.2, T::Float64 = 8.0,
                          n_for_scale::Int = L.n, step_safety::Float64 = 8.0,
                          dt_max::Float64 = 2e-3, dt_min::Float64 = 1e-7,
                          n_save::Int = 800, max_steps::Int = 4_000_000)
    ε = 3.0^(-n_for_scale) / step_safety
    g(r, t) = F.f(r) + δ(t) - η * barrier_field(L, clamp(r, 0.0, 1.0))

    ts = collect(range(0.0, T; length = n_save + 1))
    rs = Vector{Float64}(undef, n_save + 1)
    rs[1] = r0
    t = 0.0; r = r0; j = 2; nst = 0
    while t < T && nst < max_steps
        v = abs(g(r, t))
        dt = clamp(ε / (v + 1e-12), dt_min, dt_max)
        dt = min(dt, T - t)
        k1 = g(r, t)
        k2 = g(r + dt/2 * k1, t + dt/2)
        k3 = g(r + dt/2 * k2, t + dt/2)
        k4 = g(r + dt * k3,   t + dt)
        rn = r + dt/6 * (k1 + 2k2 + 2k3 + k4)
        tn = t + dt
        while j ≤ n_save + 1 && ts[j] ≤ tn + 1e-15      # linear interpolation onto the grid
            θ = dt < 1e-300 ? 1.0 : (ts[j] - t) / dt
            rs[j] = r + θ * (rn - r); j += 1
        end
        r = rn; t = tn; nst += 1
    end
    while j ≤ n_save + 1; rs[j] = r; j += 1; end
    return ts, rs
end

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

"""
    BarrierMetrics

* `safe_frac`  — fraction of time with `r < ½` (on the safe side of the boundary)
* `r_max`      — worst excursion `max_t r(t)`  (PRIMARY worst-case quantity)
* `r_end`      — terminal threat coordinate
* `captured`   — `r_end > ½`, i.e. the run ended on the unsafe side
* `crossed`    — `r` ever exceeded ½
* `t_cross`    — first crossing time (`Inf` if never)
* `mean_r`     — time-averaged threat coordinate
* `ctrl_action`— `(1/T)∫ η|V'(r(t))| dt`, the control effort actually spent
"""
struct BarrierMetrics
    safe_frac::Float64
    r_max::Float64
    r_end::Float64
    captured::Bool
    crossed::Bool
    t_cross::Float64
    mean_r::Float64
    ctrl_action::Float64
end

function barrier_metrics(ts, rs, L::BarrierLayout; η::Float64 = 1.0)
    dt = ts[2] - ts[1]; T = ts[end] - ts[1]
    trap(v) = (sum(v) - 0.5 * (v[1] + v[end])) * dt / T
    ind = [r < 0.5 ? 1.0 : 0.0 for r in rs]
    act = [η * barrier_field(L, clamp(r, 0.0, 1.0)) for r in rs]
    ic = findfirst(r -> r ≥ 0.5, rs)
    tc = ic === nothing ? Inf : ts[ic]
    BarrierMetrics(trap(ind), maximum(rs), rs[end], rs[end] > 0.5,
                   ic !== nothing, tc, trap(rs), trap(act))
end

# ---------------------------------------------------------------------------
# exact analysis for a CONSTANT attack (no integration needed)
# ---------------------------------------------------------------------------

"""
    containment_point(L, F, A; η, r0, N) -> Float64

For a constant attack `δ ≡ A`, Theorem C says the state started at `r0` cannot
pass the first `r* > r0` with `η V_L'(r*) > f(r*) + A`. This returns that `r*`
(or `1.0` if no such point exists), computed by a fine scan — **exactly, with
no ODE integration**, which is what makes a dense continuous sweep over
`log A` affordable.
"""
function containment_point(L::BarrierLayout, F::ThreatField, A::Real;
                           η::Float64 = 1.0, r0::Float64 = 0.2, N::Int = 400_000)
    i0 = max(1, ceil(Int, r0 * N))
    @inbounds for i in i0:N
        r = i / N
        η * barrier_field(L, r) > F.f(r) + A && return r
    end
    return 1.0
end

"""
    containment_curve(L, F, As; η, r0, N) -> Vector{Float64}

`containment_point` over a vector of amplitudes — the geometric robustness
curve `r*(A)`. Lower is better. This is the pre-registered primary quantity of
the theory-validation phase.
"""
containment_curve(L::BarrierLayout, F::ThreatField, As::AbstractVector;
                  η::Float64 = 1.0, r0::Float64 = 0.2, N::Int = 400_000) =
    [containment_point(L, F, A; η = η, r0 = r0, N = N) for A in As]

end # module
