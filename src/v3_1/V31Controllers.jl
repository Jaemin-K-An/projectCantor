# ============================================================================
# V31Controllers.jl — V3.1 PHASE 1: the TRUE constant controller, and the
# corrected baseline set.
#
# V3 DEFECT D1 (docs/v3_1/V3_AUDIT.md): V3's "L1_constant" was
# `piecewise_layout(ones(1))`, i.e. ONE smoothstep barrier spanning [0,1].
# Its field is u(r) ∝ Φ'(r) = 6r(1−r): zero at both ends, maximal at r = ½,
# range 1.500. That is a wide STATE-DEPENDENT barrier, so V3's headline
# reading -- "constant steering is insensitive to Δ and therefore wins" --
# had no basis.
#
# Here the two are separated and both are kept:
#   S1_true_constant  u(r) ≡ −c            ∂u/∂r = 0 exactly
#   S2_global_smooth  the old ones(1) barrier, preserved under an honest name
#
# The phrase "state-independence" is reserved for S1 alone.
# ============================================================================

module V31Controllers

using Random, Statistics
using ..CantorBarrier

export TrueConstant, ControllerV31, ctrl_field, ctrl_label, ctrl_family,
       build_v31, V31_FAMILIES, V31_RANDOMISED, is_true_constant,
       analytic_action, sup_field_derivative

"""
    TrueConstant(c)

`u(r) = −c` for every `r`. Not a barrier: it has no gaps, no smoothstep and
no state dependence at all. Its analytic L1 action over `[0,1]` is `c`, so it
is directly comparable to a barrier of total action `c`.
"""
struct TrueConstant
    c::Float64
end

"""A V3.1 controller is either a barrier layout or a true constant."""
const ControllerV31 = Union{BarrierLayout, TrueConstant}

"""    ctrl_field(C, r) -> |u(r)| ≥ 0. The applied control is `−η·ctrl_field`."""
@inline ctrl_field(C::BarrierLayout, r::Real) = barrier_field(C, r)
@inline ctrl_field(C::TrueConstant, ::Real) = C.c

ctrl_label(C::BarrierLayout) = C.label
ctrl_label(C::TrueConstant) = "true_constant"
ctrl_family(C::BarrierLayout) = C.family
ctrl_family(C::TrueConstant) = "S1_true_constant"

is_true_constant(C) = C isa TrueConstant

"""    analytic_action(C) = ∫₀¹|u| dr. Equals `c` for TrueConstant, `n·E0` for a layout."""
analytic_action(C::TrueConstant) = C.c
analytic_action(C::BarrierLayout) = isempty(C.gaps) ? 0.0 : sum(C.est)

"""
    sup_field_derivative(C) -> ‖u′‖_∞

The quantity Theorem T bounds. Exactly 0 for `TrueConstant`; for a barrier it
is `6·max_k e_k/w_k²` (see docs/v3_1/MATHEMATICAL_THEORY.md).
"""
sup_field_derivative(C::TrueConstant) = 0.0
function sup_field_derivative(C::BarrierLayout)
    isempty(C.gaps) && return 0.0
    return 6 * maximum(C.est[i] / C.wid[i]^2 for i in eachindex(C.gaps))
end

const V31_FAMILIES = ["S0_none", "S1_true_constant", "S2_global_smooth",
                      "S3_narrow_central", "S4_wide_central", "S5_periodic",
                      "S6_random", "S7_shuffled", "S8_center_anchored",
                      "S9_cantor", "S10_minimax", "S11_spline"]
const V31_RANDOMISED = Set(["S6_random", "S7_shuffled", "S8_center_anchored"])

"""Piecewise-bin layout with free per-bin energies (used by S2, S10, S11)."""
function bin_layout(weights::Vector{Float64}, n::Int, E0::Float64,
                    label::String, family::String)
    w = clamp.(weights, 1e-9, Inf); w = w ./ sum(w)
    m = length(w); edges = range(0, 1; length = m + 1)
    gaps = [Gap(1, Float64(edges[i]), Float64(edges[i+1])) for i in 1:m]
    L = BarrierLayout(gaps, n, 1.0, label, family)
    est = w .* (n * E0)
    return BarrierLayout(L.gaps, n, E0, label, family, L.las, L.lbs, L.wid,
                         est, est ./ L.wid, vcat(0.0, cumsum(est)))
end

_rename(L::BarrierLayout, fam) = BarrierLayout(L.gaps, L.n, L.E0, L.label, fam,
    L.las, L.lbs, L.wid, L.est, L.coef, L.cum)

"""
    build_v31(family, n, E0; rng, weights) -> ControllerV31

All controllers carry the same ANALYTIC action `n·E0`; the realised action is
matched separately by gain bisection (V3.1 fairness constraint).
"""
function build_v31(family::AbstractString, n::Int, E0::Float64;
                   rng::AbstractRNG = Random.default_rng(),
                   weights::Union{Nothing,Vector{Float64}} = nothing)
    family == "S0_none"           && return no_layout(n, E0)
    family == "S1_true_constant"  && return TrueConstant(n * E0)
    family == "S2_global_smooth"  && return bin_layout(ones(1), n, E0, "global_smooth", family)
    family == "S3_narrow_central" && return BarrierLayout([Gap(1, 1/3, 2/3)], n, n*E0,
                                                          "narrow_central", family)
    family == "S4_wide_central"   && return BarrierLayout([Gap(1, 1/6, 5/6)], n, n*E0,
                                                          "wide_central", family)
    family == "S5_periodic"       && return _rename(periodic_layout(n, E0), family)
    family == "S6_random"         && return _rename(random_matched_layout(n, E0, rng), family)
    family == "S7_shuffled"       && return _rename(shuffled_layout(n, E0, rng), family)
    family == "S8_center_anchored"&& return _rename(center_anchored_shuffled_layout(n, E0, rng), family)
    family == "S9_cantor"         && return _rename(cantor_layout(n, E0), family)
    if family in ("S10_minimax", "S11_spline")
        return bin_layout(weights === nothing ? ones(8) : weights, n, E0, family, family)
    end
    throw(ArgumentError("unknown V3.1 family: $family"))
end

end # module
