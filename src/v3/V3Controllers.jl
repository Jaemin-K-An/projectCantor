# ============================================================================
# V3Controllers.jl — the strong baseline set (harness §14).
#
# V2's line-up was too weak in two places, both fixed here:
#   * only ONE central barrier width was tried (L2). A wide central barrier is
#     a genuinely different and possibly better strategy under boundary error,
#     so L3_wide_central is added.
#   * no minimax-optimised controller existed. L9 is fitted on the DEV
#     uncertainty set to maximise WORST-CASE robustness, which is exactly the
#     objective Cantor is being judged on. If Cantor cannot beat a controller
#     explicitly optimised for that objective, that is reported.
# ============================================================================

module V3Controllers

using Random, Statistics
using ..CantorBarrier

export v3_layout, V3_FAMILIES, V3_RANDOMISED, piecewise_layout

const V3_FAMILIES = ["L0_none", "L1_constant", "L2_narrow_central",
                     "L3_wide_central", "L4_periodic", "L5_random",
                     "L6_shuffled", "L7_center_anchored", "L8_cantor",
                     "L9_minimax", "L10_spline"]
const V3_RANDOMISED = Set(["L5_random", "L6_shuffled", "L7_center_anchored"])

"""
    piecewise_layout(weights, n, E0, label, family)

A controller made of `m` adjacent bins covering `[0,1]`, with per-bin energy
shares `weights` renormalised to the same total action `n·E0`. Because it uses
the same gap machinery as every other layout, its action bookkeeping and
evaluation path are identical -- only the energy PLACEMENT differs. L9 and L10
are both instances of this.
"""
function piecewise_layout(weights::Vector{Float64}, n::Int, E0::Float64,
                          label::String, family::String)
    w = clamp.(weights, 1e-6, Inf); w ./= sum(w)
    m = length(w); edges = range(0, 1; length = m + 1)
    gaps = [Gap(1, Float64(edges[i]), Float64(edges[i+1])) for i in 1:m]
    L = BarrierLayout(gaps, n, 1.0, label, family)
    est = w .* (n * E0)
    return BarrierLayout(L.gaps, n, E0, label, family, L.las, L.lbs, L.wid,
                         est, est ./ L.wid,
                         vcat(0.0, cumsum(est)))
end

"""
    v3_layout(family, n, E0; rng, weights) -> BarrierLayout

L0-L8 reuse the V2 constructors (so the ablation is still ordering-only for the
width-matched families). L2/L3 differ only in the width of the single central
barrier: 1/3 (matching the Cantor level-1 gap) versus 2/3.
"""
function v3_layout(family::AbstractString, n::Int, E0::Float64;
                   rng::AbstractRNG = Random.default_rng(),
                   weights::Union{Nothing,Vector{Float64}} = nothing)
    family == "L0_none"  && return no_layout(n, E0)
    family == "L1_constant" && return piecewise_layout(ones(1), n, E0, "constant", family)
    family == "L2_narrow_central" &&
        return BarrierLayout([Gap(1, 1/3, 2/3)], n, n * E0, "narrow_central", family)
    family == "L3_wide_central" &&
        return BarrierLayout([Gap(1, 1/6, 5/6)], n, n * E0, "wide_central", family)
    family == "L4_periodic"  && return _re(periodic_layout(n, E0), family)
    family == "L5_random"    && return _re(random_matched_layout(n, E0, rng), family)
    family == "L6_shuffled"  && return _re(shuffled_layout(n, E0, rng), family)
    family == "L7_center_anchored" &&
        return _re(center_anchored_shuffled_layout(n, E0, rng), family)
    family == "L8_cantor"    && return _re(cantor_layout(n, E0), family)
    if family in ("L9_minimax", "L10_spline")
        w = weights === nothing ? ones(8) : weights
        return piecewise_layout(w, n, E0, family, family)
    end
    throw(ArgumentError("unknown V3 family: $family"))
end

_re(L::BarrierLayout, fam) = BarrierLayout(L.gaps, L.n, L.E0, L.label, fam,
                                           L.las, L.lbs, L.wid, L.est, L.coef, L.cum)

end # module
