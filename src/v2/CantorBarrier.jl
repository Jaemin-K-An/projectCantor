# ============================================================================
# CantorBarrier.jl — V2 scale-compensated barrier controllers.
#
# V1 (src/Controls.jl) normalised C'_n by (3/2)^n and kept only set membership,
# so the controller saw a {0,1} mask. V2 keeps the level-dependent magnitude:
# the controller acts INSIDE the removed gaps, with strength set by the level.
#
#   level k :  N_k = 2^{k-1} gaps of width w_k = 3^{-k}
#   energy  :  e_k = E0 / N_k                      (equal budget per LEVEL)
#   barrier :  V_{k,j}(r) = e_k Φ((r-a)/w_k),  Φ(u) = 3u² - 2u³
#   field   :  u_C(r) = -V_C'(r) ≤ 0              (always pushes r toward 0)
#
# Theorem A : Σ_j ∫|V'_{k,j}| = E0 for every k            (level action is flat)
# Theorem B : ‖V'_k‖_∞ = 3E0 (3/2)^k                      ((3/2)^k restored)
#
# BOTH THEOREMS ARE ORDERING-INVARIANT: every width-matched control below
# satisfies them exactly. They are properties of the scale-compensation scheme,
# NOT of the Cantor arrangement. See docs/v2/MATHEMATICAL_THEORY.md §3.
# ============================================================================

module CantorBarrier

using Random, Statistics

export Gap, BarrierLayout, smoothstep, dsmoothstep,
       cantor_gap_list, cantor_gap_list_exact, kstar_for, layout_from_order, barrier_potential, barrier_field,
       peak_of_level, level_action, total_action, level_energy,
       cantor_layout, periodic_layout, random_matched_layout,
       shuffled_layout, center_anchored_shuffled_layout, single_central_layout,
       constant_layout, no_layout, build_layout, LAYOUT_FAMILIES,
       guaranteed_peak, worst_displacement, min_window_energy

# ---------------------------------------------------------------------------
# smoothstep
# ---------------------------------------------------------------------------

"""
    smoothstep(u) = 3u² − 2u³, clamped to [0,1].

`Φ(0)=0`, `Φ(1)=1`, `Φ'(0)=Φ'(1)=0`, `Φ' ≥ 0`. The vanishing end-derivatives
are what make the assembled potential C¹, so V2 has no switching surfaces —
unlike V1's hard gate, which produced the sliding modes documented in
`docs/BASELINE_AUDIT.md`.
"""
@inline function smoothstep(u::Real)
    u ≤ 0 && return 0.0
    u ≥ 1 && return 1.0
    return u * u * (3 - 2u)
end

"""    dsmoothstep(u) = Φ'(u) = 6u(1−u) on [0,1], 0 outside. max = 3/2 at u = ½."""
@inline function dsmoothstep(u::Real)
    (u ≤ 0 || u ≥ 1) && return 0.0
    return 6u * (1 - u)
end

# ---------------------------------------------------------------------------
# gaps and layouts
# ---------------------------------------------------------------------------

"""A removed middle third: `level` k, open interval `(a, b)` with `b-a = 3^{-k}`."""
struct Gap
    level::Int
    a::Float64
    b::Float64
end
width(g::Gap) = g.b - g.a
centre(g::Gap) = (g.a + g.b) / 2

"""
    BarrierLayout(gaps, n, E0, label, family)

A controller: a set of disjoint barrier gaps, sorted by position, together with
the per-level energy budget `E0`. `E0` is the energy given to EACH level, so the
total L1 control action is `n·E0` (Corollary A.1); pass `E0 = B_total/n` to
compare different `n` at a fixed budget.
"""
struct BarrierLayout
    gaps::Vector{Gap}
    n::Int
    E0::Float64
    label::String
    family::String
    # --- precomputed, sorted by position, for O(log m) evaluation -----------
    # The sweep integrates ~10^5 trajectories × 1.2·10^5 RHS calls; a linear
    # scan over 2^n gaps would dominate the runtime, so position lookup is a
    # binary search and the potential uses a prefix sum of the level energies.
    las::Vector{Float64}      # gap left endpoints
    lbs::Vector{Float64}      # gap right endpoints
    wid::Vector{Float64}      # widths
    est::Vector{Float64}      # e_k for each gap
    coef::Vector{Float64}     # e_k / w_k
    cum::Vector{Float64}      # cum[i] = Σ_{j<i} e_j  (energy fully passed before gap i)
end

function BarrierLayout(gaps::Vector{Gap}, n::Int, E0::Float64,
                       label::String, family::String)
    g = sort(gaps; by = x -> x.a)
    las = [x.a for x in g]; lbs = [x.b for x in g]
    wid = lbs .- las
    est = [E0 / 2.0^(x.level - 1) for x in g]
    coef = est ./ wid
    cum = Vector{Float64}(undef, length(g) + 1)
    cum[1] = 0.0
    for i in eachindex(g); cum[i+1] = cum[i] + est[i]; end
    BarrierLayout(g, n, E0, label, family, las, lbs, wid, est, coef, cum)
end

"""    level_energy(L, k) = e_k = E0 / 2^{k-1}."""
level_energy(L::BarrierLayout, k::Int) = L.E0 / 2.0^(k - 1)

"""    peak_of_level(L, k) = ‖V'_k‖_∞ = 3·E0·(3/2)^k   (Theorem B)."""
peak_of_level(L::BarrierLayout, k::Int) = 3 * L.E0 * (1.5)^k

"""
    cantor_gap_list(n) -> Vector{Gap}

Every removed middle third down to level `n`, sorted left to right.
`2^n − 1` gaps; level `k` contributes `2^{k-1}` gaps of width `3^{-k}`.
"""
function cantor_gap_list(n::Int)
    n ≥ 1 || throw(ArgumentError("n ≥ 1"))
    n ≤ 16 || throw(ArgumentError("2^n gaps; keep n ≤ 16"))
    # Endpoints are built in EXACT rational arithmetic and rounded once at the
    # end. The naive float recursion (a+w, a+2w with w=(b-a)/3) amplifies
    # representation error by 3 per level — the same conditioning defect V1
    # documented for its recursive digit map (docs/BASELINE_AUDIT.md). The
    # right endpoint is then set as a + w_k so that every stored width is the
    # correctly rounded 3^-k rather than a difference of two large numbers.
    out = Gap[]
    function rec(a::Rational{BigInt}, b::Rational{BigInt}, k::Int)
        k > n && return
        w = (b - a) // 3
        lo = a + w
        af = Float64(lo)
        push!(out, Gap(k, af, af + 3.0^(-k)))
        rec(a, a + w, k + 1)
        rec(a + 2w, b, k + 1)
    end
    rec(Rational{BigInt}(0), Rational{BigInt}(1), 1)
    sort!(out; by = g -> g.a)
    return out
end

"""
    cantor_gap_list_exact(n) -> Vector{Tuple{Int,Rational{BigInt},Rational{BigInt}}}

The same gaps with exact rational endpoints. Used only by the test suite, to
check the Float64 construction against ground truth rather than against itself.
"""
function cantor_gap_list_exact(n::Int)
    out = Tuple{Int,Rational{BigInt},Rational{BigInt}}[]
    function rec(a::Rational{BigInt}, b::Rational{BigInt}, k::Int)
        k > n && return
        w = (b - a) // 3
        push!(out, (k, a + w, a + 2w))
        rec(a, a + w, k + 1); rec(a + 2w, b, k + 1)
    end
    rec(Rational{BigInt}(0), Rational{BigInt}(1), 1)
    sort!(out; by = t -> t[2])
    return out
end

"""
    layout_from_order(gaps, order, n) -> Vector{Gap}

Re-place the gaps left to right in the given permutation, separating them by
survivor intervals of width `3^{-n}`. This preserves EXACTLY: the number of
gaps, every gap width, every gap level (hence every energy `e_k`), the number
of survivors and their common width `3^{-n}`, and the total length 1.
**Only the ordering changes** — this is what makes the ablation a test of
arrangement rather than of measure.
"""
function layout_from_order(gaps::Vector{Gap}, order::AbstractVector{<:Integer}, n::Int)
    surv = 3.0^(-n)
    out = Gap[]
    x = 0.0
    for i in order
        g = gaps[i]
        x += surv
        push!(out, Gap(g.level, x, x + width(g)))
        x += width(g)
    end
    return out
end

# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------

"""    barrier_potential(L, r) = V_C(r), monotone non-decreasing."""
function barrier_potential(L::BarrierLayout, r::Real)
    isempty(L.las) && return 0.0
    i = searchsortedlast(L.las, r)
    i == 0 && return 0.0
    @inbounds if r < L.lbs[i]
        return L.cum[i] + L.est[i] * smoothstep((r - L.las[i]) / L.wid[i])
    end
    @inbounds return L.cum[i+1]
end

"""
    barrier_field(L, r) = V_C'(r) ≥ 0.

The control term applied to the dynamics is `−η·barrier_field(L, r)`, i.e.
`u_C = −V_C'`, which always decreases the threat coordinate. Gaps are disjoint,
so at most one term is non-zero; the loop is short-circuited accordingly.
"""
@inline function barrier_field(L::BarrierLayout, r::Real)
    isempty(L.las) && return 0.0
    i = searchsortedlast(L.las, r)
    i == 0 && return 0.0
    @inbounds (r ≥ L.lbs[i]) && return 0.0
    @inbounds return L.coef[i] * dsmoothstep((r - L.las[i]) / L.wid[i])
end

"""    level_action(L, k) = Σ_j ∫|V'_{k,j}| computed by quadrature (Theorem A ⇒ E0)."""
function level_action(L::BarrierLayout, k::Int; N::Int = 20_000)
    tot = 0.0
    for g in L.gaps
        g.level == k || continue
        w = g.b - g.a
        e = L.E0 / 2.0^(g.level - 1)
        h = w / N
        s = 0.0
        for i in 0:N
            wt = (i == 0 || i == N) ? 0.5 : 1.0
            s += wt * (e / w) * dsmoothstep(i / N)
        end
        tot += s * h
    end
    return tot
end

"""    total_action(L) = Σ_k level_action(L,k) — must equal n·E0."""
total_action(L::BarrierLayout; N::Int = 20_000) =
    sum(level_action(L, k; N = N) for k in 1:L.n)

# ---------------------------------------------------------------------------
# the seven layout families
# ---------------------------------------------------------------------------

_mk(gs, n, E0, lab, fam) = BarrierLayout(gs, n, E0, lab, fam)

"""B0 — no controller."""
no_layout(n::Int, E0::Float64) = _mk(Gap[], n, E0, "none", "B0_none")

"""
    constant_layout(n, E0)

B1 — constant restoring controller with the SAME total L1 action `n·E0`,
realised as a single barrier spanning all of `[0,1]`. Its force is spread
maximally thin: peak `3·n·E0` at `r = ½`, no multi-scale structure at all.
"""
constant_layout(n::Int, E0::Float64) =
    BarrierLayout([Gap(1, 0.0, 1.0)], n, n * E0, "constant", "B1_constant")

"""
    single_central_layout(n, E0)

B2 — all `n·E0` of action concentrated in ONE barrier of width `1/3` centred on
the decision boundary `r = ½`. The strongest "just defend the boundary"
strategy; V1 found its analogue (`G4_central`) to be the best control.
"""
single_central_layout(n::Int, E0::Float64) =
    BarrierLayout([Gap(1, 1/3, 2/3)], n, n * E0, "central", "B2_central")

"""B7 — the Cantor hierarchy itself."""
cantor_layout(n::Int, E0::Float64) =
    _mk(cantor_gap_list(n), n, E0, "cantor_n$(n)", "B7_cantor")

"""
    periodic_layout(n, E0)

B3 — identical gap multiset laid out coarse→fine on a regular lattice. Matches
every width and energy but destroys the interleaving completely; Proposition E
predicts this is by far the worst arrangement.
"""
function periodic_layout(n::Int, E0::Float64)
    gaps = cantor_gap_list(n)
    order = sortperm(gaps; by = g -> (g.level, g.a))
    _mk(layout_from_order(gaps, order, n), n, E0, "periodic_n$(n)", "B3_periodic")
end

"""
    random_matched_layout(n, E0, rng)

B4 — gap WIDTHS and energies preserved, positions drawn as a uniformly random
non-overlapping packing (survivor widths therefore vary, unlike B5/B6).
"""
function random_matched_layout(n::Int, E0::Float64, rng::AbstractRNG)
    gaps = shuffle(rng, cantor_gap_list(n))
    tot = sum(width, gaps)
    free = 1 - tot
    m = length(gaps)
    cuts = sort!(rand(rng, m) .* free)
    out = Gap[]
    x = 0.0
    for (i, g) in enumerate(gaps)
        lo = cuts[i] + x
        push!(out, Gap(g.level, lo, lo + width(g)))
        x += width(g)
    end
    _mk(out, n, E0, "random_n$(n)", "B4_random")
end

"""B5 — fully shuffled ordering, survivors all exactly `3^{-n}`."""
function shuffled_layout(n::Int, E0::Float64, rng::AbstractRNG)
    gaps = cantor_gap_list(n)
    _mk(layout_from_order(gaps, randperm(rng, length(gaps)), n), n, E0,
        "shuffled_n$(n)", "B5_shuffled")
end

"""
    center_anchored_shuffled_layout(n, E0, rng)

B6 — THE decisive control. The level-1 gap stays centred on the decision
boundary `r = ½`; everything else is randomised.

Construction (exact, no slack): for each level `k ≥ 2` the Cantor layout puts
`2^{k-2}` of its `2^{k-1}` gaps left of centre and `2^{k-2}` right of centre.
We keep those per-level counts but choose WHICH gaps go to which side at
random, and shuffle the order within each side. Because

    Σ_{k=2}^n 2^{k-2}·3^{-k} = 1/3 − ½(2/3)^n = (1/3)(1 − (2/3)^{n-1}),

each side has exactly the length the Cantor layout gives it, so the level-1
gap's centre lands on `r = ½` to machine precision and the layout still spans
exactly `[0,1]`.

Preserved vs Cantor: every gap width, every gap level, every energy `e_k`, the
total action, the survivor width `3^{-n}`, the per-side level census, and the
central barrier's position. Randomised: which gap sits where, and the entire
ordering within each half — i.e. the recursive nesting is destroyed while the
"a big barrier guards the boundary" explanation is held fixed. This is what
makes B6, not B5, the control that H4 must beat.
"""
function center_anchored_shuffled_layout(n::Int, E0::Float64, rng::AbstractRNG)
    gaps = cantor_gap_list(n)
    i1 = findfirst(g -> g.level == 1, gaps)
    left = Int[]; right = Int[]
    for k in 2:n
        idx = shuffle(rng, [i for i in eachindex(gaps) if gaps[i].level == k])
        half = length(idx) ÷ 2                     # = 2^{k-2}
        append!(left, idx[1:half]); append!(right, idx[half+1:end])
    end
    shuffle!(rng, left); shuffle!(rng, right)
    order = vcat(left, [i1], right)
    _mk(layout_from_order(gaps, order, n), n, E0, "canchor_n$(n)", "B6_center_anchored")
end

const LAYOUT_FAMILIES = ["B0_none", "B1_constant", "B2_central", "B3_periodic",
                         "B4_random", "B5_shuffled", "B6_center_anchored",
                         "B7_cantor"]

"""    build_layout(family, n, E0; rng) -> BarrierLayout — config-driven dispatch."""
function build_layout(family::AbstractString, n::Int, E0::Float64;
                      rng::AbstractRNG = Random.default_rng())
    family == "B0_none"            && return no_layout(n, E0)
    family == "B1_constant"        && return constant_layout(n, E0)
    family == "B2_central"         && return single_central_layout(n, E0)
    family == "B3_periodic"        && return periodic_layout(n, E0)
    family == "B4_random"          && return random_matched_layout(n, E0, rng)
    family == "B5_shuffled"        && return shuffled_layout(n, E0, rng)
    family == "B6_center_anchored" && return center_anchored_shuffled_layout(n, E0, rng)
    family == "B7_cantor"          && return cantor_layout(n, E0)
    throw(ArgumentError("unknown layout family: $family"))
end

# ---------------------------------------------------------------------------
# geometry statistics used by the pre-registered predictions
# ---------------------------------------------------------------------------

"""
    guaranteed_peak(L, ℓ; N) -> Float64

`P_L(ℓ) = min_r max_{s∈[r,r+ℓ]} |V'(s)|` (Corollary C.1) on a uniform grid of
`N` samples. Proposition D predicts this is nearly ordering-invariant.
"""
function guaranteed_peak(L::BarrierLayout, ℓ::Real; N::Int = 200_000)
    y = [barrier_field(L, i / N) for i in 0:N]
    m = max(1, round(Int, ℓ * N))
    m ≥ length(y) && return maximum(y)
    best = Inf
    # sliding-window maximum, O(N) with a monotone deque
    dq = Int[]
    for i in eachindex(y)
        while !isempty(dq) && y[dq[end]] ≤ y[i]; pop!(dq); end
        push!(dq, i)
        while dq[1] ≤ i - m; popfirst!(dq); end
        i ≥ m && (best = min(best, y[dq[1]]))
    end
    return best
end

"""
    min_window_energy(L, ℓ; N) -> Float64

`W_L(ℓ) = min_r [V(r+ℓ) − V(r)]`, the least barrier ENERGY in any window.
"""
function min_window_energy(L::BarrierLayout, ℓ::Real; N::Int = 200_000)
    V = [barrier_potential(L, i / N) for i in 0:N]
    m = max(1, round(Int, ℓ * N))
    m ≥ length(V) && return V[end] - V[1]
    return minimum(V[1+m:end] .- V[1:end-m])
end

"""
    worst_displacement(L, kstar) -> Float64

`D_L(k*)`, the worst-case rightward distance to the first gap of level ≥ `k*`
(Proposition E). Smaller is more robust. An attack of amplitude `A` can only
be stopped by gaps of level `k ≥ k*(A) = ⌈log_{3/2}(A/(3ηE0))⌉`.
"""
function worst_displacement(L::BarrierLayout, kstar::Int)
    pk = sort!([centre(g) for g in L.gaps if g.level ≥ kstar])
    isempty(pk) && return 1.0
    d = pk[1]
    for i in 2:length(pk)
        d = max(d, pk[i] - pk[i-1])
    end
    return max(d, 1.0 - pk[end])
end

"""    kstar_for(A, η, E0) — the finest level an attack of amplitude `A` can pass."""
kstar_for(A::Real, η::Real, E0::Real) =
    max(1, ceil(Int, log(A / (3 * η * E0)) / log(1.5)))

end # module
