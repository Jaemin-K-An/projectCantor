# ============================================================================
# Controls.jl — the gate families compared in the ablation study (G0–G6).
#
# Every gate is a map g : [0,1] → [0,1] applied to the state h. The ablation's
# fairness principle is that G1–G5 have IDENTICAL pass measure ∫₀¹ g = (2/3)^n,
# so any performance difference is attributable to spatial arrangement, not to
# "how much" of the state space is shielded. G0 (no filter) and G6 (smooth)
# deliberately break that match and are reported separately.
# ============================================================================

using Random

abstract type Gate end

"""G0 — no filter: `g ≡ 1`, the perturbation passes everywhere."""
struct NoGate <: Gate end

"""
    IntervalGate(los, his, label, family, n)

A hard {0,1} gate that is `1` on the union of the closed intervals
`[los[i], his[i]]` (sorted, disjoint) and `0` elsewhere. Evaluation is
`O(log m)` via binary search.
"""
struct IntervalGate <: Gate
    los::Vector{Float64}
    his::Vector{Float64}
    label::String
    family::String
    n::Int
end

"""
    SmoothGate(base, β)

G6 — a differentiable relaxation of an [`IntervalGate`](@ref):

    g_β(x) = clamp( Σ_i σ(β(x-aᵢ))·σ(β(bᵢ-x)), 0, 1 ),   σ = logistic

`β → ∞` recovers the hard gate; small `β` leaks perturbation into the flat
regions (which is exactly what the robust-invariance bound of
`docs/MATHEMATICAL_ANALYSIS.md` predicts should destroy invariance).
Only intervals within `cutoff/β` of `x` are summed, so cost stays `O(log m)`
in practice.
"""
struct SmoothGate <: Gate
    base::IntervalGate
    β::Float64
    label::String
end
SmoothGate(base::IntervalGate, β::Real) =
    SmoothGate(base, Float64(β), "$(base.label)-smooth(β=$(β))")

@inline logistic(z) = 1 / (1 + exp(-z))

gate_value(::NoGate, x) = 1.0

@inline function gate_value(g::IntervalGate, x)
    (x < g.los[1] || x > g.his[end]) && return 0.0
    i = searchsortedlast(g.los, x)
    i == 0 && return 0.0
    return x ≤ g.his[i] ? 1.0 : 0.0
end

function gate_value(g::SmoothGate, x)
    b = g.base
    reach = 40.0 / g.β                      # σ(-40) < 1e-17
    ilo = max(1, searchsortedfirst(b.his, x - reach))
    ihi = min(length(b.los), searchsortedlast(b.los, x + reach))
    s = zero(typeof(float(x)))
    @inbounds for i in ilo:ihi
        s += logistic(g.β * (x - b.los[i])) * logistic(g.β * (b.his[i] - x))
    end
    return clamp(s, zero(s), one(s))
end

(g::Gate)(x) = gate_value(g, x)

gate_label(::NoGate) = "no_filter"
gate_label(g::IntervalGate) = g.label
gate_label(g::SmoothGate) = g.label

gate_family(::NoGate) = "G0_none"
gate_family(g::IntervalGate) = g.family
gate_family(g::SmoothGate) = "G6_smooth_" * g.base.family

"""
    pass_measure_of(g) -> Float64

`∫₀¹ g(x) dx`. Exact for hard interval gates; a fine Riemann sum
(2 000 001 nodes) for [`SmoothGate`](@ref), where no closed form is used.
"""
pass_measure_of(::NoGate) = 1.0
pass_measure_of(g::IntervalGate) = sum(g.his .- g.los)
function pass_measure_of(g::SmoothGate; N::Int = 2_000_000)
    dx = 1 / N
    s = 0.0
    for i in 0:N
        w = (i == 0 || i == N) ? 0.5 : 1.0
        s += w * gate_value(g, i * dx)
    end
    return s * dx
end

# ---------------------------------------------------------------------------
# Constructors — G1 … G5, all with pass measure exactly (2/3)^n
# ---------------------------------------------------------------------------

"""G1 — the Cantor gate: `g = 1_{K_n}`, the 2^n level-`n` Cantor intervals."""
function cantor_interval_gate(n::Int)
    iv = cantor_intervals(n)
    IntervalGate(Float64.(first.(iv)), Float64.(last.(iv)),
                 "cantor_n$(n)", "G1_cantor", n)
end

"""
    random_matched_gate(n, rng)

G2 — measure-matched random layout: `2^n` intervals of width `3^{-n}` placed
by a uniformly random non-overlapping packing of `[0,1]`. Same interval count
and same individual widths as G1, but no self-similar structure and gap widths
that are exchangeable (Dirichlet) rather than hierarchical.
"""
function random_matched_gate(n::Int, rng::AbstractRNG)
    m = 2^n
    w = 3.0^(-n)
    free = 1 - m * w
    s = sort!(rand(rng, m) .* free)
    los = [s[i] + (i - 1) * w for i in 1:m]
    IntervalGate(los, los .+ w, "random_n$(n)", "G2_random", n)
end

"""
    periodic_gate(n)

G3 — same `2^n` intervals of width `3^{-n}`, laid out on a perfectly regular
lattice of period `2^{-n}` (each interval centred in its cell). Single length
scale, no hierarchy.
"""
function periodic_gate(n::Int)
    m = 2^n
    w = 3.0^(-n)
    p = 1 / m
    los = [(i - 1) * p + (p - w) / 2 for i in 1:m]
    IntervalGate(los, los .+ w, "periodic_n$(n)", "G3_periodic", n)
end

"""
    central_gate(n)

G4 — the blocked measure `1-(2/3)^n` packed into a single interval centred on
the safe state `h = 1/2`; the pass set is the two outer intervals. For `n = 1`
this coincides exactly with the Cantor gate. Represents the "just shield the
safe state" strategy with no fractal structure at all.
"""
function central_gate(n::Int)
    m = 1 - (2/3)^n              # blocked measure
    lo, hi = 0.5 - m/2, 0.5 + m/2
    IntervalGate([0.0, hi], [lo, 1.0], "central_n$(n)", "G4_central", n)
end

"""
    shuffled_multiscale_gate(n, rng)

G5 — topology-matched control. Keeps the exact multiset of Cantor gap widths
`{3^{-k} × 2^{k-1} : k=1..n}` and the `2^n` pass intervals of width `3^{-n}`,
but randomly permutes the ORDER in which the gaps appear along `[0,1]`.

This is a strictly sharper control than G2: it matches interval count, all
interval widths, all gap widths and the total measure, differing from G1 only
in the *arrangement* of the multiscale gaps. If G1 beats G5 the effect is
genuinely about self-similar spatial organisation.
"""
function shuffled_multiscale_gate(n::Int, rng::AbstractRNG)
    w = 3.0^(-n)
    gaps = Float64.(cantor_gap_widths(n))
    shuffle!(rng, gaps)
    m = 2^n
    los = Vector{Float64}(undef, m)
    x = 0.0
    for i in 1:m
        los[i] = x
        x += w
        i ≤ length(gaps) && (x += gaps[i])
    end
    IntervalGate(los, los .+ w, "shuffled_n$(n)", "G5_shuffled", n)
end

"""
    build_gate(family, n; rng, β) -> Gate

Dispatch table used by the config-driven scripts. `family` ∈
`"G0_none"`, `"G1_cantor"`, `"G2_random"`, `"G3_periodic"`, `"G4_central"`,
`"G5_shuffled"`, `"G6_smooth_cantor"`.
"""
function build_gate(family::AbstractString, n::Int;
                    rng::AbstractRNG = Random.default_rng(), β::Real = 50.0)
    family == "G0_none"        && return NoGate()
    family == "G1_cantor"      && return cantor_interval_gate(n)
    family == "G2_random"      && return random_matched_gate(n, rng)
    family == "G3_periodic"    && return periodic_gate(n)
    family == "G4_central"     && return central_gate(n)
    family == "G5_shuffled"    && return shuffled_multiscale_gate(n, rng)
    family == "G6_smooth_cantor" && return SmoothGate(cantor_interval_gate(n), β)
    throw(ArgumentError("unknown gate family: $family"))
end

const GATE_FAMILIES = ["G0_none", "G1_cantor", "G2_random", "G3_periodic",
                       "G4_central", "G5_shuffled", "G6_smooth_cantor"]
