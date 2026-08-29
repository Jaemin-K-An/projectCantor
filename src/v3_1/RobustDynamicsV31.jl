# ============================================================================
# RobustDynamicsV31.jl — V3.1 dynamics, accepting TrueConstant as a controller.
# Identical to src/v3/RobustDynamics.jl except that the control term goes
# through `ctrl_field`, so a genuinely state-independent controller is possible.
# ============================================================================
module RobustDynamicsV31

using Random, Statistics
using ..CantorBarrier, ..V31Controllers

export ThreatField31, linear31, cubic31, bistable31, simulate31, metrics31,
       realised_action31, match_gain31, TRUE_BOUNDARY31

const TRUE_BOUNDARY31 = 0.5

struct ThreatField31; f::Function; name::String; supf::Float64; end
(F::ThreatField31)(x) = F.f(x)
linear31(; α=1.0, x0=0.2) = (f = x -> -α*(x-x0);
    ThreatField31(f, "linear", maximum(f, range(0,1;length=4001))))
cubic31(; α=1.0, γ=4.0, x0=0.2) = (f = x -> -α*(x-x0)-γ*(x-x0)^3;
    ThreatField31(f, "cubic", maximum(f, range(0,1;length=4001))))
bistable31(; k=12.0) = (f = x -> -k*(x-0.15)*(x-0.5)*(x-0.85);
    ThreatField31(f, "bistable", maximum(f, range(0,1;length=4001))))

"""
    simulate31(C, F, δ; η, Δ, x0, T, ...) -> (ts, xs, us)

`ẋ = f(x) + δ(t) − η·ctrl_field(C, clamp(x+Δ,0,1))`. `Δ` is the controller's
boundary error; the true separatrix stays at 1/2. A `TrueConstant` ignores
both `x` and `Δ` by construction, which is the point.
"""
function simulate31(C, F::ThreatField31, δ; η::Float64=1.0, Δ::Float64=0.0,
                    x0::Float64=0.2, T::Float64=8.0, n_for_scale::Int=5,
                    step_safety::Float64=8.0, dt_max::Float64=2e-3,
                    n_save::Int=400, max_steps::Int=3_000_000)
    ε = 3.0^(-n_for_scale)/step_safety
    u(x) = -η*ctrl_field(C, clamp(x+Δ, 0.0, 1.0))
    g(x,t) = F.f(x) + δ(t) + u(x)
    ts = collect(range(0.0, T; length=n_save+1))
    xs = Vector{Float64}(undef, n_save+1); us = similar(xs)
    xs[1] = x0; us[1] = abs(u(x0)); t = 0.0; x = x0; j = 2; nst = 0
    while t < T && nst < max_steps
        dt = clamp(ε/(abs(g(x,t))+1e-12), 1e-7, dt_max); dt = min(dt, T-t)
        k1=g(x,t); k2=g(x+dt/2*k1,t+dt/2); k3=g(x+dt/2*k2,t+dt/2); k4=g(x+dt*k3,t+dt)
        xn = x + dt/6*(k1+2k2+2k3+k4); tn = t+dt
        while j ≤ n_save+1 && ts[j] ≤ tn+1e-15
            θ = dt < 1e-300 ? 1.0 : (ts[j]-t)/dt
            xs[j] = x + θ*(xn-x); us[j] = abs(u(xs[j])); j += 1
        end
        x = xn; t = tn; nst += 1
    end
    while j ≤ n_save+1; xs[j] = x; us[j] = abs(u(x)); j += 1; end
    return ts, xs, us
end

function metrics31(ts, xs, us)
    dt = ts[2]-ts[1]; T = ts[end]-ts[1]
    trap(v) = (sum(v) - 0.5*(v[1]+v[end]))*dt/T
    (safe = !any(x -> x ≥ TRUE_BOUNDARY31, xs), x_max = maximum(xs),
     safe_frac = trap([x < TRUE_BOUNDARY31 ? 1.0 : 0.0 for x in xs]),
     action_rms = sqrt(trap(us.^2)), action_max = maximum(us))
end

"""Mean realised rms action over a reference ensemble — the matched quantity."""
function realised_action31(C, F, δs, η; Δs=(0.0,), x0s=(0.15,0.35), kw...)
    tot = 0.0; n = 0
    for δ in δs, Δ in Δs, x0 in x0s
        ts, xs, us = simulate31(C, F, δ; η=η, Δ=Δ, x0=x0, kw...)
        tot += metrics31(ts, xs, us).action_rms; n += 1
    end
    tot/n
end

"""Geometric bisection on η so the REALISED rms action hits `target`."""
function match_gain31(C, F, δs, target::Float64; lo=1e-3, hi=1e3, tol=0.02,
                      iters=40, kw...)
    (C isa BarrierLayout && isempty(C.gaps)) && return 0.0
    a, b = lo, hi
    realised_action31(C,F,δs,a; kw...) > target && return a
    realised_action31(C,F,δs,b; kw...) < target && return b
    for _ in 1:iters
        m = sqrt(a*b); fm = realised_action31(C,F,δs,m; kw...)
        abs(fm-target)/target < tol && return m
        fm < target ? (a = m) : (b = m)
    end
    sqrt(a*b)
end

end # module
