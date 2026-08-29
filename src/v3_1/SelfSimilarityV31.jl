# ============================================================================
# SelfSimilarityV31.jl — V3.1 PHASE 4: corrected Theorem S corollary.
#
# V3 DEFECT D5: V3 claimed the GLOBAL coverage relation
#       M_n(Δ/3) = (3/2)·M_{n-1}(Δ),   M_n(Δ) = min_r max_{|u-r|≤Δ} V'_n(u)
# follows from Theorem S. It does not: the global minimising window may
# straddle the two Cantor copies, the central gap, or the interval ends, where
# no self-similar relation applies. Direct computation gives ratios
# 0.225–0.282, not 1.0.
#
# The LOCAL statement below is what actually follows, and it is proved.
# ============================================================================

module SelfSimilarityV31

using Statistics
using ..CantorBarrier

export local_coverage, local_corollary_ratio, sup_field_derivative_layout

"""
    local_coverage(L, Δ, i; N) -> Float64

`M^loc_{n,i}(Δ)` — the guaranteed force within a window of half-width `Δ`,
minimised ONLY over windows entirely contained in the `i`-th Cantor copy
(`i = 0` → `[0,1/3]`, `i = 2` → `[2/3,1]`).

**Corollary S.1′ (local, proved).** For `i ∈ {0,2}` and any `Δ ∈ (0, 1/3)`,

        M^loc_{n,i}(Δ/3) = (3/2)·M_{n-1}(Δ)

*Proof.* Windows contained in `T_i([0,1])` are exactly the `T_i`-images of
windows contained in `[0,1]`, and `T_i` scales lengths by `1/3`, so a window of
half-width `Δ/3` in the copy is the image of one of half-width `Δ`. By
Theorem S, `V'_n(T_i(r)) = (3/2)V'_{n-1}(r)` pointwise, so the max over the
image window is `(3/2)` times the max over the pre-image window, and the min
over admissible windows is preserved by the bijection. ∎

The global version fails precisely because the global minimiser need not be an
admissible window of either copy.
"""
function local_coverage(L::BarrierLayout, Δ::Real, i::Int; N::Int = 200_000)
    lo, hi = i == 0 ? (0.0, 1/3) : (2/3, 1.0)
    a, b = lo + Δ, hi - Δ
    a ≥ b && return NaN
    best = Inf
    M = max(2, round(Int, (b - a) * N))
    for j in 0:M
        r = a + (b - a) * j / M
        m = 0.0
        K = 400
        for q in 0:K
            m = max(m, barrier_field(L, r - Δ + 2Δ * q / K))
        end
        best = min(best, m)
    end
    return best
end

"""
    local_corollary_ratio(n, E0, Δs) -> Vector

`M^loc_{n,i}(Δ/3) / [(3/2)·M_{n-1}(Δ)]`, which Corollary S.1′ says is 1.
Returned for both copies so the test can check them independently.
"""
function local_corollary_ratio(n::Int, E0::Float64, Δs::AbstractVector; N::Int = 60_000)
    Ln = cantor_layout(n, E0); Lm = cantor_layout(n - 1, E0)
    out = Float64[]
    for Δ in Δs, i in (0, 2)
        lhs = local_coverage(Ln, Δ / 3, i; N = N)
        # reference: same construction on the full interval at order n-1
        a, b = Δ, 1 - Δ
        best = Inf; M = max(2, round(Int, (b - a) * N))
        for j in 0:M
            r = a + (b - a) * j / M
            m = 0.0; K = 400
            for q in 0:K
                m = max(m, barrier_field(Lm, r - Δ + 2Δ * q / K))
            end
            best = min(best, m)
        end
        push!(out, lhs / (1.5 * best))
    end
    return out
end

"""    sup_field_derivative_layout(L) = ‖u′‖_∞ = 6·max_k e_k/w_k²  (Theorem T)."""
sup_field_derivative_layout(L::BarrierLayout) =
    isempty(L.gaps) ? 0.0 : 6 * maximum(L.est[i] / L.wid[i]^2 for i in eachindex(L.gaps))

end # module
