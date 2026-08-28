# ============================================================================
# PHASE C — numerical analysis of ∫₀¹ C'_n(x) dx.
#   julia --project=. scripts/run_numerical_analysis.jl
#
# ANALYTIC TRUTH: the integral equals 1 EXACTLY for every finite n. Everything
# computed here is a property of the quadrature rule, never of the integrand.
#
# Output: results/raw/numerical_integration.csv
#         results/raw/numerical_node_stats.csv
#         results/raw/solver_discontinuity_check.csv
#         results/tables/numerical_summary.md
# ============================================================================

using DataFrames, Statistics, Printf
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

cfg = load_config("numerical.toml")
ns  = Int.(cfg["grid"]["n"]); dxs = Float64.(cfg["grid"]["dx"])

println("="^78)
println("PHASE C.1 — I(n, Δx) for four quadrature rules   (truth = 1 exactly)")
println("="^78)

rows = DataFrame()
for n in ns, dx in dxs
    il = riemann_integral_left(n, dx)
    im = riemann_integral_midpoint(n, dx)
    st = node_hit_statistics(n, dx)
    push!(rows, (n = n, dx = dx, rho = dx * 3.0^n,
                 I_left = il, I_mid = im,
                 E_left = abs(il - 1), E_mid = abs(im - 1),
                 n_intervals = st.n_intervals, intervals_hit = st.hit,
                 hit_fraction = st.hit / st.n_intervals,
                 nodes_in_K = st.nodes_in_K, nodes_expected = st.expected,
                 interval_width = 3.0^(-n), peak = (3/2)^n))
end
# rules that do not depend on Δx
exact_rows = DataFrame(n = ns,
                       I_exact  = [exact_interval_integral(n) for n in ns],
                       I_adapt  = [adaptive_integral(n) for n in ns])
exact_rows.E_exact = abs.(exact_rows.I_exact .- 1)
exact_rows.E_adapt = abs.(exact_rows.I_adapt .- 1)

@printf("%4s", "n\\ρ")
for dx in dxs; @printf("%12.0e", dx); end
println("        exact     adaptive")
for n in ns
    @printf("%4d", n)
    for dx in dxs
        @printf("%12.4f", only(rows[(rows.n .== n) .& (rows.dx .== dx), :I_mid]))
    end
    e = exact_rows[exact_rows.n .== n, :]
    @printf("   %10.6f  %10.6f\n", e.I_exact[1], e.I_adapt[1])
end

println("\nintervals resolved (hit fraction), midpoint grid:")
@printf("%4s", "n\\Δx")
for dx in dxs; @printf("%12.0e", dx); end; println()
for n in ns
    @printf("%4d", n)
    for dx in dxs
        @printf("%12.3f", only(rows[(rows.n .== n) .& (rows.dx .== dx), :hit_fraction]))
    end
    println()
end

# ------------------------------------------------------------- the ρ collapse
println("\n" * "="^78)
println("PHASE C.2 — the error collapses onto ρ = Δx·3ⁿ (grid spacing in units")
println("            of the smallest Cantor interval 3^-n)")
println("="^78)
bins = [(0.0,1e-3),(1e-3,1e-2),(1e-2,1e-1),(1e-1,1.0),(1.0,10.0),
        (10.0,1e2),(1e2,1e3),(1e3,Inf)]
@printf("%18s %6s %12s %12s %12s\n", "ρ range", "cases", "median E_mid", "max E_mid", "median hit")
for (lo, hi) in bins
    s = rows[(rows.rho .≥ lo) .& (rows.rho .< hi), :]
    nrow(s) == 0 && continue
    @printf("%8.0e–%-8.0e %6d %12.2e %12.2e %12.3f\n", lo, hi, nrow(s),
            median(s.E_mid), maximum(s.E_mid), median(s.hit_fraction))
end

# ------------------- reproduce the ORIGINAL study's own quadrature (dx = 1e-4)
println("\n" * "="^78)
println("PHASE C.3 — the original study's own rule: sum(C'_n.(0:1e-4:1))*1e-4")
println("="^78)
@printf("%4s %14s %14s %10s %12s\n", "n", "I_left(orig)", "I_mid", "ρ", "hit frac")
for n in 1:12
    r = only(eachrow(rows[(rows.n .== n) .& (rows.dx .== 1e-4), :]))
    @printf("%4d %14.6f %14.6f %10.3f %12.4f\n", n, r.I_left, r.I_mid, r.rho, r.hit_fraction)
end

# ------------------------- endpoint conditioning of the recursive digit map
println("\n" * "="^78)
println("PHASE C.4 — conditioning of the recursive membership test at the")
println("            2^(n+1) exact interval endpoints")
println("="^78)
endp = DataFrame()
for n in 1:14
    e = endpoint_instability(n)
    push!(endp, (n = n, n_endpoints = e.n_endpoints,
                 n_misclassified = e.n_misclassified, frac = e.frac,
                 err_amplification = 3.0^n * eps(), interval_width = 3.0^(-n)))
end
for r in eachrow(endp)
    @printf("  n=%2d  %6d endpoints, %6d misclassified (%.4f)   3ⁿ·eps = %.2e vs width %.2e\n",
            r.n, r.n_endpoints, r.n_misclassified, r.frac,
            r.err_amplification, r.interval_width)
end

write_raw(rows, "numerical_integration.csv"; overwrite = true,
          meta = Dict("phase"=>"C", "analytic_truth"=>1.0, "config"=>"numerical.toml"))
write_raw(exact_rows, "numerical_structure_aware.csv"; overwrite = true,
          meta = Dict("phase"=>"C"))
write_raw(endp, "numerical_endpoint_conditioning.csv"; overwrite = true,
          meta = Dict("phase"=>"C"))

open(tabpath("numerical_summary.md"), "w") do io
    println(io, "# Quadrature of C'_n  (analytic truth = 1 for every finite n)\n")
    println(io, "## Midpoint-rule estimate I_mid(n, Δx)\n")
    print(io, "| n |"); for dx in dxs; print(io, " Δx=$(dx) |"); end
    println(io, " exact-interval | adaptive Simpson |")
    print(io, "|---|"); for _ in dxs; print(io, "---|"); end; println(io, "---|---|")
    for n in ns
        print(io, "| $n |")
        for dx in dxs
            print(io, " ", fmtf(only(rows[(rows.n.==n).&(rows.dx.==dx), :I_mid]), 4), " |")
        end
        e = exact_rows[exact_rows.n .== n, :]
        println(io, " ", fmtf(e.I_exact[1], 6), " | ", fmtf(e.I_adapt[1], 6), " |")
    end
end
println("\ndone.")
