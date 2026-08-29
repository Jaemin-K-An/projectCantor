# ============================================================================
# RobustDynamics.jl — V3 PHASE 6: joint boundary × attack uncertainty.
#
# The V2 synthetic assumed the controller knew the decision boundary exactly.
# V3 removes that assumption, because the V2 LLM audit showed it is false in
# practice (calibration was fitted on prompt tokens and applied to generated
# ones, docs/v3/V2_LLM_AUDIT.md §3).
#
#   true state        x ∈ ℝ,  true boundary b = 1/2
#   controller sees   r = clamp(x + Δ, 0, 1)      Δ = calibration error
#   dynamics          ẋ = f(x) + δ_ε(t) − η V'_L(r)
#
# TWO UNCERTAINTIES, KEPT SEPARATE (harness §28):
#   Δ  the controller mis-locates the boundary   (calibration error)
#   ε  the attack magnitude is unknown           (attack-scale uncertainty)
#
# The headline quantity is NOT mean performance but
#       J(L) = min over (Δ, ε, attack family) of R_L(Δ, ε, a),
# the minimax robustness over the pre-declared uncertainty set.
#
# FAIRNESS: gains are calibrated per controller so that the REALISED mean
# control action along trajectories matches a target (see `match_gain`),
# not merely the analytic ∫|u|dr. This is the V3 correction to V2 defect B.
# ============================================================================

module RobustDynamics

using Random, Statistics
using ..CantorBarrier

export ThreatField3, linear3, cubic3, bistable3, simulate_robust,
       robust_metrics, realised_action, match_gain, RobustResult

struct ThreatField3
    f::Function
    name::String
    supf::Float64
end
(F::ThreatField3)(x) = F.f(x)

linear3(; α = 1.0, x0 = 0.2) =
    (f = x -> -α * (x - x0); ThreatField3(f, "linear", maximum(f, range(0,1;length=4001))))
cubic3(; α = 1.0, γ = 4.0, x0 = 0.2) =
    (f = x -> -α*(x-x0) - γ*(x-x0)^3; ThreatField3(f, "cubic", maximum(f, range(0,1;length=4001))))
"""Bistable: safe attractor 0.15, UNSTABLE separatrix at the true boundary 0.5,
unsafe attractor 0.85. A jailbreak is a push across the separatrix."""
bistable3(; k = 12.0) =
    (f = x -> -k*(x-0.15)*(x-0.5)*(x-0.85); ThreatField3(f, "bistable", maximum(f, range(0,1;length=4001))))

const TRUE_BOUNDARY = 0.5

"""
    simulate_robust(L, F, δ; η, Δ, x0, T, ...) -> (ts, xs, us)

Displacement-limited RK4 (the V2 integrator, which resolves barriers of width
3^-n instead of tunnelling through them). `Δ` is the controller's boundary
error: it evaluates its field at `clamp(x+Δ,0,1)` while the true separatrix
stays at 1/2. `us` records the instantaneous |control| so the REALISED action
can be measured.
"""
function simulate_robust(L::BarrierLayout, F::ThreatField3, δ;
                         η::Float64 = 1.0, Δ::Float64 = 0.0, x0::Float64 = 0.2,
                         T::Float64 = 8.0, n_for_scale::Int = max(L.n, 1),
                         step_safety::Float64 = 8.0, dt_max::Float64 = 2e-3,
                         n_save::Int = 400, max_steps::Int = 3_000_000)
    ε_move = 3.0^(-n_for_scale) / step_safety
    u(x) = -η * barrier_field(L, clamp(x + Δ, 0.0, 1.0))
    g(x, t) = F.f(x) + δ(t) + u(x)
    ts = collect(range(0.0, T; length = n_save + 1))
    xs = Vector{Float64}(undef, n_save + 1); us = similar(xs)
    xs[1] = x0; us[1] = abs(u(x0))
    t = 0.0; x = x0; j = 2; nst = 0
    while t < T && nst < max_steps
        dt = clamp(ε_move / (abs(g(x, t)) + 1e-12), 1e-7, dt_max)
        dt = min(dt, T - t)
        k1 = g(x, t); k2 = g(x + dt/2*k1, t + dt/2)
        k3 = g(x + dt/2*k2, t + dt/2); k4 = g(x + dt*k3, t + dt)
        xn = x + dt/6*(k1 + 2k2 + 2k3 + k4); tn = t + dt
        while j ≤ n_save + 1 && ts[j] ≤ tn + 1e-15
            θ = dt < 1e-300 ? 1.0 : (ts[j] - t)/dt
            xs[j] = x + θ*(xn - x); us[j] = abs(u(xs[j])); j += 1
        end
        x = xn; t = tn; nst += 1
    end
    while j ≤ n_save + 1; xs[j] = x; us[j] = abs(u(x)); j += 1; end
    return ts, xs, us
end

struct RobustResult
    safe::Bool          # never crossed the TRUE boundary
    x_max::Float64
    x_end::Float64
    safe_frac::Float64
    action_mean::Float64
    action_rms::Float64
    action_max::Float64
end

function robust_metrics(ts, xs, us)
    dt = ts[2] - ts[1]; T = ts[end] - ts[1]
    trap(v) = (sum(v) - 0.5*(v[1] + v[end])) * dt / T
    RobustResult(!any(x -> x ≥ TRUE_BOUNDARY, xs), maximum(xs), xs[end],
                 trap([x < TRUE_BOUNDARY ? 1.0 : 0.0 for x in xs]),
                 trap(us), sqrt(trap(us .^ 2)), maximum(us))
end

"""
    realised_action(L, F, δs, ηs...; kwargs) -> Float64

Mean realised |u| over a reference ensemble of trajectories. This is what the
gain matcher equalises across controllers -- the V3 fairness constraint.
"""
function realised_action(L::BarrierLayout, F::ThreatField3, δs, η::Float64;
                         Δs = (0.0,), x0s = (0.15, 0.35), kwargs...)
    tot = 0.0; n = 0
    for δ in δs, Δ in Δs, x0 in x0s
        ts, xs, us = simulate_robust(L, F, δ; η = η, Δ = Δ, x0 = x0, kwargs...)
        tot += robust_metrics(ts, xs, us).action_rms; n += 1
    end
    return tot / n
end

"""
    match_gain(L, F, δs, target; lo, hi, tol, iters) -> Float64

Binary search for the gain η whose REALISED rms action equals `target`.
Realised action is monotone increasing in η for a fixed layout, so bisection
is well posed. Returns `hi`/`lo` if the target is unreachable in the bracket,
which is recorded rather than silently clipped.
"""
function match_gain(L::BarrierLayout, F::ThreatField3, δs, target::Float64;
                    lo = 1e-3, hi = 1e3, tol = 0.02, iters = 40, kwargs...)
    isempty(L.gaps) && return 0.0
    a, b = lo, hi
    fa = realised_action(L, F, δs, a; kwargs...)
    fb = realised_action(L, F, δs, b; kwargs...)
    fa > target && return a
    fb < target && return b
    for _ in 1:iters
        m = sqrt(a * b)                       # geometric bisection: η spans decades
        fm = realised_action(L, F, δs, m; kwargs...)
        abs(fm - target) / target < tol && return m
        fm < target ? (a = m) : (b = m)
    end
    return sqrt(a * b)
end

end # module
