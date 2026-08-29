# ============================================================================
# V3 PHASE 6b — mechanism analysis (harness §45, §46).
#
# Two questions the headline table cannot answer:
#   1. Is L9's collapse to uniform weights real, or a weak search? -> stronger
#      search with local refinement. A stronger L9 can only HURT Cantor, so
#      this check is conservative in Cantor's favour.
#   2. Is robustness explained by SELF-SIMILARITY (Theorem S), by
#      ANTI-CLUSTERING, or by neither? Regress worst-case robustness on
#      layout geometry across MANY shuffled layouts -- not Cantor-versus-one.
# ============================================================================

using DataFrames, CSV, Statistics, Printf, Random, TOML, Dates, Base.Threads
include(joinpath(@__DIR__, "..", "..", "src", "CantorGate.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v2", "CantorBarrier.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v3", "V3Controllers.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v3", "RobustDynamics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v3", "CantorSelfSimilarity.jl"))
using .CantorGate: sinusoid, square_wave
using .CantorBarrier, .V3Controllers, .RobustDynamics, .CantorSelfSimilarity

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
out(f) = joinpath(ROOT, "results", "v3", "raw", f)
cfg = TOML.parsefile(joinpath(ROOT, "configs", "v3", "synthetic.toml"))
n = 5; E0 = 1.0/n; T = 8.0
F = bistable3(); x0s = [0.15, 0.35]
Δs = Float64.(cfg["uncertainty"]["delta"])
εs = 10 .^ range(-0.52, 0.90; length = 6)
ref_atk = [sinusoid(2.0, 4.0), square_wave(2.0, 1.5)]
mk(a, A, rng) = a == "sinusoid" ? sinusoid(A, 10^(-0.3+1.8rand(rng)); φ=2π*rand(rng)) :
                                  square_wave(A, 10^(-0.3+1.8rand(rng)); φ=2π*rand(rng))
ATK = ["sinusoid", "square"]

"""Worst-case graded robustness on the REPORTING grid at a matched budget."""
function worstR(L::BarrierLayout, target::Float64)
    η = isempty(L.gaps) ? 0.0 :
        match_gain(L, F, ref_atk, target; tol=0.03, T=T, n_for_scale=n)
    w = Inf
    for Δ in Δs, ε in εs, a in ATK
        δ = mk(a, ε, Xoshiro(hash((a, ε)) % 10^9))
        acc = 0.0
        for x0 in x0s
            ts, xs, us = simulate_robust(L, F, δ; η=η, Δ=Δ, x0=x0, T=T, n_for_scale=n)
            acc += robust_metrics(ts, xs, us).safe_frac
        end
        w = min(w, acc/length(x0s))
    end
    return w, η
end

# ---------------------------------------------- 1. stronger L9 minimax search
println("="^96)
println("STRONGER L9 SEARCH — is uniform really minimax-optimal?")
println("="^96)
rng = Xoshiro(4242)
best = (-Inf, ones(8))
cands = [ones(8)]
for _ in 1:120; push!(cands, rand(rng, 8).^2 .+ 1e-3); end
# add structured candidates: centred, edge-heavy, two-scale, cantor-like
push!(cands, [0.02,0.05,0.15,0.28,0.28,0.15,0.05,0.02])       # centred
push!(cands, [0.28,0.15,0.05,0.02,0.02,0.05,0.15,0.28])       # edge heavy
push!(cands, [0.05,0.20,0.05,0.20,0.20,0.05,0.20,0.05])       # two-scale
function search_l9(cands)
    b = (-Inf, ones(8))
    for c in cands
        L = piecewise_layout(collect(c), n, E0, "L9", "L9_minimax")
        v, _ = worstR(L, 0.30)
        v > b[1] && (b = (v, collect(c)))
    end
    # coordinate refinement around the incumbent
    for _ in 1:3, i in 1:8, m in (0.6, 1.6)
        c = copy(b[2]); c[i] *= m
        L = piecewise_layout(c, n, E0, "L9", "L9_minimax")
        v, _ = worstR(L, 0.30)
        v > b[1] && (b = (v, c))
    end
    return b
end
best = search_l9(cands)
w = best[2] ./ sum(best[2])
@printf("  best worst-case graded R = %.4f\n  weights = %s\n", best[1],
        string(round.(w; digits=4)))
@printf("  uniform-weight R = %.4f   ->  %s\n",
        worstR(piecewise_layout(ones(8), n, E0, "u", "L9_minimax"), 0.30)[1],
        maximum(abs.(w .- 1/8)) < 0.02 ? "UNIFORM IS OPTIMAL" : "non-uniform optimum found")

# ------------------------------------- 2. self-similarity vs robustness (§45)
println("\n" * "="^96)
println("MECHANISM — does SELF-SIMILARITY or ANTI-CLUSTERING explain robustness?")
println("(many shuffled layouts, not Cantor-versus-one)")
println("="^96)
rows = NamedTuple[]
layouts = Tuple{String,Int,BarrierLayout}[("L8_cantor", 0, v3_layout("L8_cantor", n, E0))]
for s in 1:40
    push!(layouts, ("L6_shuffled", s, v3_layout("L6_shuffled", n, E0; rng=Xoshiro(1000+s))))
    push!(layouts, ("L7_center_anchored", s, v3_layout("L7_center_anchored", n, E0; rng=Xoshiro(2000+s))))
end
push!(layouts, ("L4_periodic", 0, v3_layout("L4_periodic", n, E0)))
res = Vector{NamedTuple}(undef, length(layouts))
@threads for i in eachindex(layouts)
    fam, sd, L = layouts[i]
    g = layout_geometry(L)
    v, η = worstR(L, 0.30)
    res[i] = (family=fam, seed=sd, R_worst=v, eta=η, selfsim=g.selfsim,
              max_weak_run=g.max_weak_run, mean_nn=g.mean_nn,
              discrepancy=g.discrepancy, entropy=g.entropy,
              max_gap_span=g.max_gap_span)
end
df = DataFrame(res)
CSV.write(out("v3_mechanism.csv"), df)
@info "wrote $(nrow(df)) rows → $(out("v3_mechanism.csv"))"

cantor = df[df.family .== "L8_cantor", :]
@printf("\n  Cantor: selfsim=%.4f  R_worst=%.4f\n", cantor.selfsim[1], cantor.R_worst[1])
sh = df[df.family .!= "L8_cantor", :]
@printf("  shuffled/c-anchored (n=%d): selfsim %.3f–%.3f, R_worst %.4f–%.4f (median %.4f)\n",
        nrow(sh), minimum(sh.selfsim), maximum(sh.selfsim),
        minimum(sh.R_worst), maximum(sh.R_worst), median(sh.R_worst))
@printf("  Cantor percentile among controls: %.1f\n",
        100*count(<(cantor.R_worst[1]), sh.R_worst)/nrow(sh))
cor2(a,b) = (v = cor(a,b); isnan(v) ? 0.0 : v)
println("\n  Pearson correlation of R_worst with each geometric statistic:")
for c in (:selfsim, :max_weak_run, :mean_nn, :discrepancy, :entropy, :max_gap_span)
    @printf("    %-16s r = %+.3f\n", c, cor2(df.R_worst, df[!, c]))
end
println("\n  ⇒ if r(selfsim) ≈ 0 the self-similarity mechanism is REJECTED as an")
println("    explanation of robustness under boundary uncertainty.")
