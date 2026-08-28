# ============================================================================
# Metrics.jl — trajectory summary statistics.
#
# All time integrals use the composite trapezoid rule on the UNIFORM saveat
# grid. The original study used `mean(indicator)`, i.e. a rectangle rule that
# double-weights the two endpoints; the difference is O(Δt/T) and is quantified
# in docs/BASELINE_AUDIT.md rather than silently corrected.
# ============================================================================

using Statistics

const SAFE_LO = 1/3
const SAFE_HI = 2/3
const H_SAFE  = 0.5

"""`true` iff `h` is in the safe set `S = [1/3, 2/3]`."""
@inline in_safe(h) = SAFE_LO ≤ h ≤ SAFE_HI

"""
    trapz_mean(vals, dt, T) -> Float64

`(1/T)∫₀ᵀ v(t) dt` by the composite trapezoid rule on a uniform grid of
spacing `dt`.
"""
function trapz_mean(vals::AbstractVector, dt::Real, T::Real)
    n = length(vals)
    n < 2 && return Float64(vals[1])
    s = sum(vals) - 0.5 * (vals[1] + vals[end])
    return s * dt / T
end

"""
    TrajectoryMetrics

The six metrics of the experiment protocol plus two diagnostics.

* `R_safe`   — safe-set occupancy `(1/T)∫ 1_S(h) dt`
* `R_safe_rect` — the original study's rectangle-rule occupancy (audit only)
* `tau_S`    — first hitting time `inf{t : h(t) ∈ S}`, `Inf` if never entered
* `D_max`    — `max_t |h(t) - 1/2|`
* `D_mean`   — `(1/T)∫ |h(t) - 1/2| dt`
* `T_rec`    — recovery time: first `t` after which `h` stays in `S` for at
               least `hold` time units (`Inf` if no such `t` exists in `[0,T]`)
* `R_gate`   — gate pass ratio `(1/T)∫ g(h(t)) dt`
* `h_end`    — terminal state
"""
struct TrajectoryMetrics
    R_safe::Float64
    R_safe_rect::Float64
    tau_S::Float64
    D_max::Float64
    D_mean::Float64
    T_rec::Float64
    R_gate::Float64
    h_end::Float64
end

"""
    first_hitting_time(ts, hs) -> Float64

`inf{t : h(t) ∈ S}` estimated from a uniform sample. If the first in-set
sample is `hs[k]` and `hs[k-1]` was outside, the crossing time is refined by
linear interpolation of `h` to the boundary it crossed, giving `O(Δt²)`
accuracy on a monotone crossing. Returns `0.0` if `h(0) ∈ S` and `Inf` if the
set is never reached.
"""
function first_hitting_time(ts::AbstractVector, hs::AbstractVector)
    in_safe(hs[1]) && return 0.0
    @inbounds for k in 2:length(hs)
        if in_safe(hs[k])
            hprev, hcur = hs[k-1], hs[k]
            target = hprev < SAFE_LO ? SAFE_LO : SAFE_HI
            denom = hcur - hprev
            θ = abs(denom) < 1e-14 ? 1.0 : clamp((target - hprev) / denom, 0.0, 1.0)
            return ts[k-1] + θ * (ts[k] - ts[k-1])
        end
    end
    return Inf
end

"""
    recovery_time(ts, hs; hold) -> Float64

The smallest sample time `t` such that `h(s) ∈ S` for every sampled
`s ∈ [t, t+hold]`. Requires a full `hold` window inside `[0,T]`, so it is
`Inf` whenever the trajectory only settles in the last `hold` time units —
a deliberately conservative convention, stated so that `Inf` counts are
reported rather than dropped.
"""
function recovery_time(ts::AbstractVector, hs::AbstractVector; hold::Real = 5.0)
    n = length(hs)
    dt = ts[2] - ts[1]
    w = max(1, round(Int, hold / dt))
    n ≤ w && return Inf
    # run[k] = length of the in-S run starting at k (computed backwards)
    run = 0
    lastgood = -1
    @inbounds for k in n:-1:1
        run = in_safe(hs[k]) ? run + 1 : 0
        run > w && (lastgood = k)
    end
    return lastgood == -1 ? Inf : ts[lastgood]
end

"""
    compute_metrics(ts, hs, gvals; hold) -> TrajectoryMetrics

`ts` must be uniformly spaced; `gvals[i] = g(hs[i])`.
"""
function compute_metrics(ts::AbstractVector, hs::AbstractVector,
                         gvals::AbstractVector; hold::Real = 5.0)
    dt = ts[2] - ts[1]
    T  = ts[end] - ts[1]
    ind = Float64.(in_safe.(hs))
    dev = abs.(hs .- H_SAFE)
    TrajectoryMetrics(
        trapz_mean(ind, dt, T),
        mean(ind),
        first_hitting_time(ts, hs),
        maximum(dev),
        trapz_mean(dev, dt, T),
        recovery_time(ts, hs; hold = hold),
        trapz_mean(gvals, dt, T),
        hs[end],
    )
end

"""Column names used by every raw-result CSV, in order."""
const METRIC_COLS = [:R_safe, :R_safe_rect, :tau_S, :D_max, :D_mean,
                     :T_rec, :R_gate, :h_end]

metrics_tuple(m::TrajectoryMetrics) =
    (R_safe=m.R_safe, R_safe_rect=m.R_safe_rect, tau_S=m.tau_S, D_max=m.D_max,
     D_mean=m.D_mean, T_rec=m.T_rec, R_gate=m.R_gate, h_end=m.h_end)
