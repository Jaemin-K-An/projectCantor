# ============================================================================
# CantorCore.jl — finite-order Cantor staircase, its derivative, and the
# induced state-dependent gate.
#
# CONVENTIONS FIXED HERE (do not change silently; tests depend on them):
#
#   K_n  := the level-n Cantor set, the union of 2^n CLOSED intervals of
#           width 3^{-n} that remain after n middle-third removals.
#   F_n  := [0,1] \ K_n, the OPEN "flat" set, |F_n| = 1 - (2/3)^n.
#
#   C_n(x)  : the n-th staircase approximation (piecewise linear, slope
#             (3/2)^n on K_n, slope 0 on F_n).
#   C'_n(x) := (3/2)^n  for x ∈ K_n
#              0         for x ∈ F_n
#   g_n(x)  := C'_n(x) / (3/2)^n = 1_{K_n}(x)   (the normalised gate)
#
# The closed/open split means g_n(1/3) = g_n(2/3) = 1: the endpoints of the
# removed middle third belong to K_n. This matches the ORIGINAL study's
# implementation (verified in test/runtests.jl) and only matters on a
# Lebesgue-null set, but it is the convention used in the forward-invariance
# proof of docs/MATHEMATICAL_ANALYSIS.md, so it is stated explicitly.
# ============================================================================

"""
    cantor_staircase(x, n) -> Float64

`C_n(x)`, the n-th finite approximation of the Cantor staircase, defined by
`C_0(x) = x` and the self-similar recursion

    C_n(x) = ½ C_{n-1}(3x)         0 ≤ x < 1/3
             ½                     1/3 ≤ x ≤ 2/3
             ½ + ½ C_{n-1}(3x-2)   2/3 < x ≤ 1

Values outside `[0,1]` are clamped to `C_n(0)=0`, `C_n(1)=1`.
This is the ORIGINAL study's implementation, kept verbatim so that the
reproduction is faithful.
"""
function cantor_staircase(x::Real, n::Int)::Float64
    n == 0 && return Float64(clamp(x, 0, 1))
    x ≤ 0 && return 0.0
    x ≥ 1 && return 1.0
    if x < 1/3
        return 0.5 * cantor_staircase(3x, n - 1)
    elseif x ≤ 2/3
        return 0.5
    else
        return 0.5 + 0.5 * cantor_staircase(3x - 2, n - 1)
    end
end

"""
    in_cantor_set(x, n) -> Bool

`true` iff `x ∈ K_n`, i.e. the first `n` ternary digits of `x` can be chosen
in `{0,2}`. Implemented by the iterated affine map used by the original code
(`x ↦ 3x` on the left third, `x ↦ 3x-2` on the right third); a point is
rejected as soon as it lands strictly inside a middle third.

This is implementation #1 of two mutually-checked implementations; see
[`in_cantor_set_intervals`](@ref).
"""
function in_cantor_set(x::Real, n::Int)::Bool
    (x < 0 || x > 1) && return false
    xf = Float64(x)
    for _ in 1:n
        if 1/3 < xf < 2/3
            return false
        elseif xf ≤ 1/3
            xf = 3xf
        else
            xf = 3xf - 2
        end
    end
    return true
end

"""
    cantor_derivative(x, n) -> Float64

`C'_n(x)`: `(3/2)^n` on `K_n`, `0` on the flat set. Undefined in the classical
sense at the 2^{n+1} interval endpoints; the closed-interval convention above
assigns them the value `(3/2)^n`.
"""
cantor_derivative(x::Real, n::Int)::Float64 =
    in_cantor_set(x, n) ? (3/2)^n : 0.0

"""
    cantor_gate(x, n) -> Float64

Normalised gate `g_n(x) = C'_n(x)/(3/2)^n ∈ {0,1}`. `1` lets an external
perturbation through, `0` blocks it completely.
"""
cantor_gate(x::Real, n::Int)::Float64 = in_cantor_set(x, n) ? 1.0 : 0.0

# ---------------------------------------------------------------------------
# Exact interval representation (independent implementation #2)
# ---------------------------------------------------------------------------

"""
    cantor_intervals(n) -> Vector{Tuple{Rational{BigInt},Rational{BigInt}}}

The 2^n closed intervals of `K_n`, in increasing order, as exact rationals.
Interval `k` (0-based) has left endpoint `Σ_{i=1}^{n} 2·b_i·3^{-i}` where
`b_1…b_n` are the binary digits of `k` (most significant first), and width
`3^{-n}`. Only call for `n ≤ 20` (memory grows as `2^n`).
"""
function cantor_intervals(n::Int)
    n ≥ 0 || throw(ArgumentError("n must be ≥ 0"))
    n ≤ 20 || throw(ArgumentError("cantor_intervals is exponential; n ≤ 20"))
    w = Rational{BigInt}(1, BigInt(3)^n)
    out = Vector{Tuple{Rational{BigInt},Rational{BigInt}}}(undef, 2^n)
    for k in 0:(2^n - 1)
        a = Rational{BigInt}(0)
        for i in 1:n
            bit = (k >> (n - i)) & 1
            a += 2 * bit * Rational{BigInt}(1, BigInt(3)^i)
        end
        out[k+1] = (a, a + w)
    end
    return out
end

"""
    in_cantor_set_intervals(x, n) -> Bool

Membership test by explicit interval enumeration — an implementation that
shares no code path with [`in_cantor_set`](@ref). Used only in tests and for
building gate objects; `O(2^n)` per call.
"""
function in_cantor_set_intervals(x::Real, n::Int)::Bool
    (x < 0 || x > 1) && return false
    for (a, b) in cantor_intervals(n)
        (Float64(a) ≤ x ≤ Float64(b)) && return true
    end
    return false
end

"""
    cantor_flat_interval(x, n) -> Union{Nothing,NamedTuple}

If `x` lies in the flat set `F_n`, return
`(level=k, lo=a, hi=b)` where `k ≤ n` is the removal level at which `x` was
first discarded and `(a,b)` is the exact open middle third removed at that
level (as `Rational{BigInt}`). Returns `nothing` if `x ∈ K_n`.

This is the tool used in `docs/BASELINE_AUDIT.md` to locate `h₀ = 0.15`.
"""
function cantor_flat_interval(x::Real, n::Int)
    (x < 0 || x > 1) && return nothing
    # Track the exact affine map from the current rescaled coordinate back to [0,1].
    lo = Rational{BigInt}(0)
    scale = Rational{BigInt}(1)          # current window width
    xf = Float64(x)
    for k in 1:n
        if 1/3 < xf < 2/3
            return (level = k,
                    lo = lo + scale * Rational{BigInt}(1, 3),
                    hi = lo + scale * Rational{BigInt}(2, 3))
        elseif xf ≤ 1/3
            xf = 3xf
            scale = scale // 3
        else
            xf = 3xf - 2
            lo = lo + 2 * scale // 3
            scale = scale // 3
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Analytic quantities (exact; never to be confused with numerical estimates)
# ---------------------------------------------------------------------------

"""`|K_n| = (2/3)^n`: the measure of the non-flat (pass) set."""
pass_measure(n::Int) = (2/3)^n

"""`|F_n| = 1 - (2/3)^n`: the measure of the flat (blocked) set."""
flat_measure(n::Int) = 1 - (2/3)^n

"""`(3/2)^n`: the value of `C'_n` on the non-flat set."""
nonflat_derivative(n::Int) = (3/2)^n

"""
    analytic_integral(n) -> Float64

`∫₀¹ C'_n(x) dx = |K_n| · (3/2)^n = (2/3)^n (3/2)^n = 1`, exactly, for every
finite `n`. Returned as the *analytic truth*, never as a numerical estimate.
"""
analytic_integral(::Int) = 1.0

"""
    cantor_gap_widths(n) -> Vector{Rational{BigInt}}

The widths of the `2^n - 1` flat gaps of `F_n` that lie strictly between
consecutive intervals of `K_n`, in left-to-right order. At level `k` there are
`2^{k-1}` gaps of width `3^{-k}`, so the multiset is
`{3^{-k} with multiplicity 2^{k-1} : k = 1..n}` and the widths sum to
`1 - (2/3)^n`. Used to build the topology-matched control `G5`.
"""
function cantor_gap_widths(n::Int)
    iv = cantor_intervals(n)
    return [iv[i+1][1] - iv[i][2] for i in 1:(length(iv)-1)]
end

# ---------------------------------------------------------------------------
# Float64 stability of the two membership tests
# ---------------------------------------------------------------------------

"""
    in_cantor_set_stable(x, n, los, his) -> Bool

Membership by binary search over PRE-ROUNDED `Float64` interval endpoints.
Unlike [`in_cantor_set`](@ref) it performs no arithmetic on `x`, so it inherits
only the ≤1 ulp error of rounding the endpoints, instead of amplifying the
representation error of `x` by a factor 3 per level.

This is the test actually used by [`IntervalGate`](@ref) — and therefore by
every experiment. `in_cantor_set` is retained because it is the original
study's algorithm and the baseline reproduction must be faithful.
"""
@inline function in_cantor_set_stable(x::Real, los::Vector{Float64}, his::Vector{Float64})
    (x < los[1] || x > his[end]) && return false
    k = searchsortedlast(los, x)
    return k ≥ 1 && x ≤ his[k]
end

"""
    endpoint_instability(n) -> NamedTuple

Quantifies the conditioning defect of the recursive digit map. Every one of the
`2^{n+1}` exact endpoints of `K_n` is a rational `p/3^k` that in general has no
exact `Float64` representation; the maps `x ↦ 3x` and `x ↦ 3x-2` multiply that
representation error by 3 at every level, so after `n` levels the accumulated
error is `O(3^n · eps)` and a point can be pushed across a switching surface.

Returns `(n_endpoints, n_misclassified, frac)` where "misclassified" means
[`in_cantor_set`](@ref) returns `false` at a point that provably lies in `K_n`.
Reported in docs/MATHEMATICAL_ANALYSIS.md; it affects a Lebesgue-null set and
so has no effect on any integral or occupancy, but it is the same
scale-resolution failure that drives the quadrature study.
"""
function endpoint_instability(n::Int)
    iv = cantor_intervals(n)
    bad = 0; tot = 0
    for (a, b) in iv
        for r in (a, b)
            tot += 1
            in_cantor_set(Float64(r), n) || (bad += 1)
        end
    end
    return (n_endpoints = tot, n_misclassified = bad, frac = bad / tot)
end
