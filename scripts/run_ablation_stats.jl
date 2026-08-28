# ============================================================================
# PHASE F/§13 — statistical analysis of the ablation.
#   julia --project=. scripts/run_ablation_stats.jl
#
# Effect sizes and bootstrap intervals are reported IN PREFERENCE TO p-values:
# with 30 layouts × 225 conditions almost any difference is "significant", so
# the scientifically meaningful question is how LARGE the Cantor advantage is
# relative to the spread of measure-matched controls.
#
# Output: results/tables/ablation_stats.csv, results/tables/ablation_stats.md
# ============================================================================

using DataFrames, CSV, Statistics, Random, Printf, StatsBase
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

df = CSV.read(rawpath("ablation_main.csv"), DataFrame)

"""Hedges-corrected Cohen's d of a single value against a sample."""
function effect_size(x::Real, sample::AbstractVector)
    s = std(sample)
    s < 1e-12 && return x ≈ mean(sample) ? 0.0 : sign(x - mean(sample)) * Inf
    return (x - mean(sample)) / s
end

"""Percentile-bootstrap CI of the mean of `v`."""
function boot_ci(v::AbstractVector, rng; B = 4000, level = 0.95)
    n = length(v)
    ms = [mean(v[rand(rng, 1:n, n)]) for _ in 1:B]
    a = (1 - level) / 2
    return quantile(ms, a), quantile(ms, 1 - a)
end

const METRICS = [:R_safe, :D_mean, :D_max]
rng = Xoshiro(20260828)

out = DataFrame()
conds = unique(df[:, [:n, :h0, :A, :omega, :alpha]])
for ctrl in ("G2_random", "G5_shuffled", "G3_periodic", "G4_central"), met in METRICS
    for n in sort(unique(df[df.n .> 0, :n]))
        dvals = Float64[]; pct = Float64[]; eff = Float64[]
        for c in eachrow(conds)
            c.n == n || continue
            sel(f) = df[(df.gate_family .== f) .& (df.n .== n) .& (df.h0 .== c.h0) .&
                        (df.A .== c.A) .& (df.omega .== c.omega) .& (df.alpha .== c.alpha), met]
            rc = sel("G1_cantor"); rr = sel(ctrl)
            (isempty(rc) || isempty(rr)) && continue
            x = rc[1]
            push!(dvals, x - mean(rr))
            push!(pct, 100 * (count(<(x), rr) + 0.5*count(==(x), rr)) / length(rr))
            length(rr) > 1 && push!(eff, effect_size(x, rr))
        end
        isempty(dvals) && continue
        lo, hi = boot_ci(dvals, rng)
        push!(out, (control = ctrl, metric = String(met), n = n,
                    n_conditions = length(dvals),
                    mean_delta = mean(dvals), median_delta = median(dvals),
                    ci_lo = lo, ci_hi = hi,
                    frac_delta_pos = count(>(0), dvals) / length(dvals),
                    mean_pct_rank = isempty(pct) ? NaN : mean(pct),
                    mean_effect_size = isempty(eff) ? NaN : mean(filter(isfinite, eff))))
    end
end
write_table(out, "ablation_stats.csv")

println("="^108)
println("Cantor (G1) minus measure-matched control — metric R_safe (higher = better)")
println("Δ>0 means the Cantor ARRANGEMENT helps beyond its measure.")
println("="^108)
@printf("%-14s %3s %8s %10s %10s %18s %10s %10s\n",
        "control", "n", "#cond", "meanΔ", "medianΔ", "95% bootstrap CI",
        "frac Δ>0", "pctile")
for r in eachrow(out[out.metric .== "R_safe", :])
    @printf("%-14s %3d %8d %+10.4f %+10.4f  [%+7.4f,%+7.4f] %10.3f %10.1f\n",
            r.control, r.n, r.n_conditions, r.mean_delta, r.median_delta,
            r.ci_lo, r.ci_hi, r.frac_delta_pos, r.mean_pct_rank)
end

# ---- where in parameter space is ΔR positive? (for FIG 9) ------------------
grid = DataFrame()
for ctrl in ("G2_random", "G5_shuffled")
    for c in eachrow(conds)
        c.n == 0 && continue
        sel(f) = df[(df.gate_family .== f) .& (df.n .== c.n) .& (df.h0 .== c.h0) .&
                    (df.A .== c.A) .& (df.omega .== c.omega) .& (df.alpha .== c.alpha), :R_safe]
        rc = sel("G1_cantor"); rr = sel(ctrl)
        (isempty(rc) || isempty(rr)) && continue
        push!(grid, (control = ctrl, n = c.n, h0 = c.h0, A = c.A, omega = c.omega,
                     alpha = c.alpha, R_cantor = rc[1], R_ctrl_mean = mean(rr),
                     R_ctrl_sd = std(rr), delta = rc[1] - mean(rr),
                     pct_rank = 100 * count(<(rc[1]), rr) / length(rr)))
    end
end
write_table(grid, "ablation_delta_grid.csv")

println("\n" * "="^108)
println("VERDICT — is the Cantor arrangement better than a measure-matched control?")
println("="^108)
for ctrl in ("G2_random", "G5_shuffled", "G3_periodic", "G4_central")
    s = out[(out.control .== ctrl) .& (out.metric .== "R_safe"), :]
    pooled = mean(s.mean_delta)
    consistent = all(s.ci_lo .> 0) ? "ALL n: CI excludes 0 and Δ>0" :
                 all(s.ci_hi .< 0) ? "ALL n: CI excludes 0 and Δ<0" :
                 "SIGN VARIES with n"
    @printf("  vs %-14s  pooled meanΔ = %+.4f   %s\n", ctrl, pooled, consistent)
end

open(tabpath("ablation_stats.md"), "w") do io
    println(io, "# Ablation statistics (auto-generated)\n")
    println(io, "Δ = metric(Cantor) − mean over measure-matched control layouts.\n")
    for met in METRICS
        println(io, "\n## metric: `$met`\n")
        println(io, "| control | n | #cond | mean Δ | median Δ | 95% CI | frac Δ>0 | mean pctile | effect size |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        for r in eachrow(out[out.metric .== String(met), :])
            println(io, "| $(r.control) | $(r.n) | $(r.n_conditions) | ",
                    fmtf(r.mean_delta,4), " | ", fmtf(r.median_delta,4), " | [",
                    fmtf(r.ci_lo,4), ", ", fmtf(r.ci_hi,4), "] | ",
                    fmtf(r.frac_delta_pos,3), " | ", fmtf(r.mean_pct_rank,1), " | ",
                    fmtf(r.mean_effect_size,3), " |")
        end
    end
end
println("\ndone.")
