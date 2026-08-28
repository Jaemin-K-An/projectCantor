# ============================================================================
# NumericalAnalysis.jl — how quadrature of C'_n fails when the grid is coarser
# than the fractal scale.
#
# ANALYTIC TRUTH (never estimated): ∫₀¹ C'_n(x) dx = (2/3)^n · (3/2)^n = 1 for
# every finite n. Any deviation produced below is a property of the QUADRATURE
# RULE, not of the function.
#
# The relevant dimensionless number is
#
#     ρ = Δx / 3^{-n} = Δx · 3^n
#
# the grid spacing measured in units of the smallest Cantor interval. ρ ≪ 1
# resolves every interval; ρ ≳ 1 means most of the 2^n intervals contain no
# node at all and the sum degenerates into an aliasing problem.
# ============================================================================

"""
    riemann_integral_left(n, dx) -> Float64

The original study's rule: `sum(C'_n.(0:dx:1)) * dx`. Note this uses `N+1`
nodes for an interval of length `1`, and evaluates the guarded endpoints
`x = 0` and `x = 1`. Reproduced exactly so its bias can be decomposed.
"""
function riemann_integral_left(n::Int, dx::Float64)
    N = round(Int, 1 / dx)
    peak = (3/2)^n
    c = 0
    @inbounds for i in 0:N
        x = i * dx
        (x ≤ 0 || x ≥ 1) && continue          # the original's guard
        in_cantor_set(x, n) && (c += 1)
    end
    return c * peak * dx
end

"""
    riemann_integral_midpoint(n, dx) -> Float64

Midpoint rule with `N = 1/dx` cells: `Σ C'_n((i+½)dx) · dx`. Unbiased with
respect to the endpoint guard, so comparing it with
[`riemann_integral_left`](@ref) separates endpoint bias from aliasing.
"""
function riemann_integral_midpoint(n::Int, dx::Float64)
    N = round(Int, 1 / dx)
    peak = (3/2)^n
    c = 0
    @inbounds for i in 0:(N-1)
        in_cantor_set((i + 0.5) * dx, n) && (c += 1)
    end
    return c * peak * dx
end

"""
    exact_interval_integral(n) -> Float64

Integration by exact interval arithmetic: sum the `2^n` closed Cantor
intervals' exact rational widths and multiply by `(3/2)^n`. Structure-aware
quadrature — it returns `1` to machine precision for every `n ≤ 20`, which is
the point: the failure is in the *uniform grid*, not the integrand.
"""
function exact_interval_integral(n::Int)
    iv = cantor_intervals(n)
    w = sum(Float64(b - a) for (a, b) in iv)
    return w * (3/2)^n
end

"""
    adaptive_integral(n, x0, x1; rtol, maxdepth) -> Float64

Recursive adaptive Simpson on `C'_n`. Included to show that adaptivity does
**not** rescue the computation: the integrand is a piecewise-constant
comb whose local Simpson estimate is exact (and therefore "converged") on any
subinterval that happens to miss the comb, so the refinement criterion is
blind to the intervals it has skipped.
"""
function adaptive_integral(n::Int, x0::Float64 = 0.0, x1::Float64 = 1.0;
                           rtol::Float64 = 1e-8, maxdepth::Int = 24)
    f(x) = cantor_derivative(x, n)
    function simpson(a, b, fa, fm, fb)
        (b - a) / 6 * (fa + 4fm + fb)
    end
    function rec(a, b, fa, fm, fb, whole, depth)
        m  = (a + b) / 2
        lm = (a + m) / 2; rm = (m + b) / 2
        flm = f(lm); frm = f(rm)
        left  = simpson(a, m, fa, flm, fm)
        right = simpson(m, b, fm, frm, fb)
        if depth ≥ maxdepth || abs(left + right - whole) ≤ 15 * rtol * max(1, abs(whole))
            return left + right + (left + right - whole) / 15
        end
        return rec(a, m, fa, flm, fm, left, depth + 1) +
               rec(m, b, fm, frm, fb, right, depth + 1)
    end
    fa = f(x0); fb = f(x1); fm = f((x0 + x1) / 2)
    return rec(x0, x1, fa, fm, fb, simpson(x0, x1, fa, fm, fb), 0)
end

"""
    node_hit_statistics(n, dx) -> NamedTuple

Diagnostic decomposition of the uniform-grid failure. Returns

* `n_intervals`  — `2^n`
* `hit`          — how many of them contain at least one midpoint node
* `nodes_in_K`   — total nodes landing in `K_n`
* `expected`     — `(2/3)^n / dx`, the count a perfectly equidistributed grid
                   would give
* `rho`          — `dx · 3^n`

The estimate equals `nodes_in_K · (3/2)^n · dx`, so the relative error is
exactly `nodes_in_K/expected - 1`: a pure *counting* error.
"""
function node_hit_statistics(n::Int, dx::Float64)
    N = round(Int, 1 / dx)
    iv = cantor_intervals(n)
    los = Float64.(first.(iv)); his = Float64.(last.(iv))
    hit = falses(length(los))
    nodes = 0
    @inbounds for i in 0:(N-1)
        x = (i + 0.5) * dx
        k = searchsortedlast(los, x)
        if k ≥ 1 && x ≤ his[k]
            nodes += 1
            hit[k] = true
        end
    end
    return (n_intervals = length(los), hit = count(hit), nodes_in_K = nodes,
            expected = (2/3)^n / dx, rho = dx * 3.0^n)
end
