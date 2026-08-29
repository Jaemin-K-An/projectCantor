# ============================================================================
# CantorSelfSimilarity.jl — V3 PHASE 5.
#
# THE ONE STRUCTURAL PROPERTY THAT IS GENUINELY CANTOR-SPECIFIC.
#
# V2 proved Theorems A and B but also proved they are ORDERING-INVARIANT:
# every width-matched control satisfies them identically, so neither could
# support a Cantor-specific claim. This module derives and tests a property
# that shuffled layouts provably do NOT have.
#
# ---------------------------------------------------------------------------
# THEOREM S (exact scale-equivariance of the Cantor barrier field)
#
# Let T_0(r) = r/3 and T_2(r) = 2/3 + r/3 be the two contractions generating
# the Cantor set, and let V_n be the scale-compensated barrier of order n with
# per-level budget E0 (src/v2/CantorBarrier.jl). Then for all r in [0,1] and
# i in {0,2}:
#
#       V'_n(T_i(r)) = (3/2) · V'_{n-1}(r)
#       V_n(T_i(r)) − V_n(T_i(0)) = (1/2) · V_{n-1}(r)
#
# PROOF. The gaps of the level-n construction lying inside [0,1/3] are exactly
# the T_0-images of the gaps of the level-(n−1) construction on [0,1]: a gap of
# level k at (a, a+w_k) maps to a gap of level k+1 at (a/3, a/3 + w_k/3), and
# indeed w_{k+1} = 3^{-(k+1)} = w_k/3. Its energy is e_{k+1} = E0/2^k = e_k/2.
# Inside a gap the field is (e/w)·Φ'((r−a)/w), and Φ' is unchanged because the
# normalised coordinate (T_0(r) − T_0(a))/(w_k/3) = (r−a)/w_k. Hence the field
# is scaled by (e_k/2)/(w_k/3) ÷ (e_k/w_k) = 3/2. Outside the gaps both sides
# vanish. The same argument applies verbatim to T_2, whose image is [2/3,1].
# Integrating, V_n(T_i(r)) − V_n(T_i(0)) = ∫_0^r (3/2)V'_{n-1}(u)·(1/3)du
# = (1/2)V_{n-1}(r). ∎
#
# COROLLARY S.1 (self-similar coverage). Define the guaranteed force within a
# window of half-width Δ,
#       M_n(Δ) = min_r max_{|u−r| ≤ Δ} V'_n(u).
# Theorem S gives M_n(Δ/3) = (3/2)·M_{n-1}(Δ) on the sub-copies, i.e. the
# coverage curve of the Cantor layout is an exact power law in Δ. A shuffled
# layout has the same gap multiset but no such relation.
#
# WHY THIS MATTERS FOR V3. A boundary-calibration error Δ means the controller
# evaluates its field at r+Δ instead of r. Theorem S says the Cantor field
# "looks the same at every scale", so the response to a shift is
# scale-covariant rather than scale-arbitrary. That is the only mechanism by
# which recursive ORDERING (as opposed to width multiset) could buy robustness
# to an unknown Δ. Whether it actually does is an empirical question that
# PHASE 6 and PHASE 12 answer; this module only establishes that the property
# exists and is exclusive to Cantor.
# ============================================================================

module CantorSelfSimilarity

using Statistics, Random
using ..CantorBarrier

export T0, T2, selfsim_residual, selfsim_score, coverage_curve,
       coverage_powerlaw_residual, layout_geometry

"""Left contraction of the Cantor IFS."""
T0(r) = r / 3
"""Right contraction of the Cantor IFS."""
T2(r) = 2/3 + r / 3

"""
    selfsim_residual(Ln, Lnm1, rs) -> (rel_L2, max_abs)

Relative L2 and sup residual of Theorem S applied to a layout pair:

    resid(r) = V'_n(T_i(r)) − (3/2)·V'_{n-1}(r),  i ∈ {0,2}

For the Cantor layout this is 0 to machine precision. For any other layout with
the same widths it is not, and the size of the residual is exactly the
"how far from self-similar" score used in PHASE 6.
"""
function selfsim_residual(Ln::BarrierLayout, Lnm1::BarrierLayout,
                          rs::AbstractVector = range(0, 1; length = 60_001))
    num = 0.0; den = 0.0; mx = 0.0
    @inbounds for r in rs
        ref = 1.5 * barrier_field(Lnm1, r)
        for T in (T0, T2)
            d = barrier_field(Ln, T(r)) - ref
            num += d^2; den += ref^2; mx = max(mx, abs(d))
        end
    end
    return (sqrt(num / max(den, 1e-300)), mx)
end

"""
    selfsim_score(L; nref) -> Float64 ∈ [0,1]

`1 − min(1, rel_L2)` of Theorem S, comparing layout `L` (order n) against the
Cantor layout of order n−1 as the reference sub-copy. 1.0 means exactly
self-similar. Used in PHASE 6 to regress robustness on self-similarity across
many shuffled layouts, so the mechanism claim is tested on a CONTINUUM rather
than on Cantor-versus-one-shuffle.
"""
function selfsim_score(L::BarrierLayout)
    L.n ≥ 2 || return NaN
    ref = cantor_layout(L.n - 1, L.E0)
    rel, _ = selfsim_residual(L, ref)
    return 1 - min(1.0, rel)
end

"""
    coverage_curve(L, Δs; N) -> Vector{Float64}

`M_L(Δ) = min_r max_{|u−r| ≤ Δ} V'_L(u)`: the force a controller is GUARANTEED
to apply somewhere within Δ of any state. This is the quantity a boundary
error of size Δ actually degrades, and it is the bridge from Theorem S to the
robustness experiments.
"""
function coverage_curve(L::BarrierLayout, Δs::AbstractVector; N::Int = 120_000)
    y = [barrier_field(L, i / N) for i in 0:N]
    out = Float64[]
    for Δ in Δs
        m = max(1, 2 * round(Int, Δ * N) + 1)
        if m ≥ length(y)
            push!(out, maximum(y)); continue
        end
        best = Inf; dq = Int[]
        @inbounds for i in eachindex(y)
            while !isempty(dq) && y[dq[end]] ≤ y[i]; pop!(dq); end
            push!(dq, i)
            while dq[1] ≤ i - m; popfirst!(dq); end
            i ≥ m && (best = min(best, y[dq[1]]))
        end
        push!(out, best)
    end
    return out
end

"""
    coverage_powerlaw_residual(L, Δs) -> Float64

How well `log M_L(Δ)` is linear in `log Δ` (Corollary S.1 predicts an exact
power law for Cantor). Returns 1 − R² of that fit; 0 = perfect power law.
"""
function coverage_powerlaw_residual(L::BarrierLayout, Δs::AbstractVector)
    M = coverage_curve(L, Δs)
    ok = M .> 0
    sum(ok) < 3 && return NaN
    x = log.(collect(Δs)[ok]); y = log.(M[ok])
    b = cov(x, y) / var(x); a = mean(y) - b * mean(x)
    ŷ = a .+ b .* x
    return 1 - 1 - (-1) * (sum((y .- ŷ).^2) / sum((y .- mean(y)).^2))
end

"""
    layout_geometry(L) -> NamedTuple

Geometric statistics used in PHASE 6 to separate "because it is Cantor" from
"because of a specific measurable property" (harness §46):

* `selfsim`      — Theorem-S score (Cantor-exclusive)
* `max_weak_run` — longest stretch containing no gap of level ≤ `strong_level`
* `mean_nn`      — mean nearest-barrier distance over a fine grid
* `discrepancy`  — star discrepancy of the gap centres (uniformity of spread)
* `entropy`      — Shannon entropy of the level sequence read left to right
* `max_gap_span` — longest run of [0,1] with no barrier at all
"""
function layout_geometry(L::BarrierLayout; strong_level::Int = 0, N::Int = 20_000)
    isempty(L.gaps) && return (selfsim = NaN, max_weak_run = 1.0, mean_nn = 0.5,
                               discrepancy = 1.0, entropy = 0.0, max_gap_span = 1.0)
    sl = strong_level == 0 ? max(1, L.n - 1) : strong_level
    cs = sort([(g.a + g.b) / 2 for g in L.gaps if g.level ≥ sl])
    mwr = isempty(cs) ? 1.0 : maximum(vcat(cs[1], diff(cs), 1 - cs[end]))
    allc = sort([(g.a + g.b) / 2 for g in L.gaps])
    nn = 0.0
    for i in 0:N
        r = i / N
        k = searchsortedfirst(allc, r)
        d = min(k ≤ length(allc) ? allc[k] - r : Inf,
                k > 1 ? r - allc[k-1] : Inf)
        nn += d
    end
    # star discrepancy of the gap centres
    m = length(allc); disc = 0.0
    for (i, c) in enumerate(allc)
        disc = max(disc, abs(i / m - c), abs((i - 1) / m - c))
    end
    # entropy of the left-to-right level sequence
    lv = [g.level for g in L.gaps]
    p = [count(==(k), lv) / length(lv) for k in 1:L.n]
    ent = -sum(q > 0 ? q * log(q) : 0.0 for q in p)
    spans = vcat(L.gaps[1].a, [L.gaps[i+1].a - L.gaps[i].b for i in 1:length(L.gaps)-1],
                 1 - L.gaps[end].b)
    return (selfsim = selfsim_score(L), max_weak_run = mwr, mean_nn = nn / (N + 1),
            discrepancy = disc, entropy = ent, max_gap_span = maximum(spans))
end

end # module
