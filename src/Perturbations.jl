# ============================================================================
# Perturbations.jl — the external disturbance library δ(t).
#
# FAIRNESS INVARIANT: every perturbation is normalised so that sup_t |δ(t)| ≤ A
# with equality (or near equality) for the deterministic waveforms. Without
# this the OOD comparison would confound "different waveform" with
# "different energy", and the robust-invariance bound B|g| < α/6 of
# docs/MATHEMATICAL_ANALYSIS.md would not be comparable across families.
# ============================================================================

using Random

"""
    Perturb(f, name, params, bound)

A callable disturbance. `f(t)` returns δ(t); `bound` is the analytic (or
measured) `sup_t |δ(t)|`; `params` is logged verbatim into the result tables.
"""
struct Perturb{F}
    f::F
    name::String
    params::NamedTuple
    bound::Float64
end
@inline (p::Perturb)(t) = p.f(t)

# ---------------------------------------------------------------------------
# P1–P6: deterministic, closed-form
# ---------------------------------------------------------------------------

"""P1 — `δ(t) = A sin(ωt + φ)`. The disturbance used by the original study."""
sinusoid(A, ω; φ = 0.0) =
    Perturb(t -> A * sin(ω * t + φ), "sinusoid", (A=A, ω=ω, φ=φ), abs(A))

"""P4 — square wave `δ(t) = A·sign(sin(ωt+φ))`. Same amplitude and
fundamental frequency as P1 but with all odd harmonics present."""
square_wave(A, ω; φ = 0.0) =
    Perturb(t -> A * sign(sin(ω * t + φ)), "square", (A=A, ω=ω, φ=φ), abs(A))

"""P5 — linear chirp sweeping `ω₀ → ω₁` over `[0,T]` (instantaneous frequency
is linear in `t`); amplitude `A` throughout."""
chirp(A, ω0, ω1, T) = Perturb(
    t -> A * sin(ω0 * t + (ω1 - ω0) * t^2 / (2T)),
    "chirp", (A=A, ω0=ω0, ω1=ω1, T=T), abs(A))

"""
    impulse_train(A, times, width)

P3 — a train of Gaussian impulses of height `A` and standard deviation
`width` centred at `times`. Peaks are well separated so `sup|δ| ≈ A`.
"""
function impulse_train(A, times::AbstractVector, width)
    ts = collect(float.(times))
    Perturb(t -> A * sum(exp(-((t - tk) / width)^2) for tk in ts),
            "impulse", (A=A, n_impulse=length(ts), width=width), abs(A))
end

"""
    multifreq(A, ωs; φs)

P6 — `δ(t) = A · Σ aₖ sin(ωₖ t + φₖ) / Σ|aₖ|` with equal weights `aₖ = 1`,
so `sup|δ| ≤ A`. Energy is spread across incommensurate frequencies.
"""
function multifreq(A, ωs::AbstractVector; φs = zeros(length(ωs)))
    ws = collect(float.(ωs)); ps = collect(float.(φs)); k = length(ws)
    Perturb(t -> A * sum(sin(ws[i] * t + ps[i]) for i in 1:k) / k,
            "multifreq", (A=A, ωs=Tuple(ws), k=k), abs(A))
end

# ---------------------------------------------------------------------------
# P7–P2: sampled-then-interpolated (deterministic given a seed)
# ---------------------------------------------------------------------------

"""
    _interp_path(ts, vs)

Linear interpolation of a pre-sampled path, clamped outside `[ts[1], ts[end]]`.
Keeping the path pre-sampled (rather than drawing noise inside the RHS) is what
lets the "stochastic" perturbations stay ordinary ODEs: the realisation is
fixed before integration, so the solver sees a deterministic, Lipschitz δ(t)
and no SDE machinery is needed. This is a *randomised ODE*, not an SDE, and is
labelled as such throughout.
"""
function _interp_path(ts::Vector{Float64}, vs::Vector{Float64})
    dt = ts[2] - ts[1]; t0 = ts[1]; N = length(ts)
    return function (t)
        u = (t - t0) / dt
        i = clamp(floor(Int, u) + 1, 1, N - 1)
        θ = clamp(u - (i - 1), 0.0, 1.0)
        return (1 - θ) * vs[i] + θ * vs[i+1]
    end
end

"""
    piecewise_random(A, T, dt_hold, seed)

P7 — piecewise-random disturbance: i.i.d. `U(-1,1)` values held for `dt_hold`
and linearly interpolated, rescaled so `sup|δ| = A`.
"""
function piecewise_random(A, T, dt_hold, seed::Int)
    rng = Xoshiro(seed)
    ts = collect(0.0:dt_hold:(T + dt_hold))
    vs = 2 .* rand(rng, length(ts)) .- 1
    vs .*= A / maximum(abs, vs)
    Perturb(_interp_path(ts, vs), "piecewise_random",
            (A=A, dt_hold=dt_hold, seed=seed), abs(A))
end

"""
    ou_noise(A, T, θ, σ, dt, seed)

P2 — an Ornstein–Uhlenbeck path (`dX = -θX dt + σ dW`) pre-sampled on a grid of
step `dt` by Euler–Maruyama, linearly interpolated, and rescaled so
`sup|δ| = A`. Gives temporally-correlated "Gaussian-like" disturbance while the
state equation stays a deterministic ODE (see [`_interp_path`](@ref)).
"""
function ou_noise(A, T, θ, σ, dt, seed::Int)
    rng = Xoshiro(seed)
    ts = collect(0.0:dt:(T + dt)); N = length(ts)
    vs = zeros(N)
    for i in 2:N
        vs[i] = vs[i-1] - θ * vs[i-1] * dt + σ * sqrt(dt) * randn(rng)
    end
    mx = maximum(abs, vs); mx > 0 && (vs .*= A / mx)
    Perturb(_interp_path(ts, vs), "ou_noise",
            (A=A, θ=θ, σ=σ, dt=dt, seed=seed), abs(A))
end

"""
    zero_perturbation()

The clean (unperturbed) case, used for the Neural ODE fitting loss.
"""
zero_perturbation() = Perturb(t -> 0.0, "none", NamedTuple(), 0.0)
