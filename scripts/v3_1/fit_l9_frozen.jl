# V3.1 PHASE 7 — fit the minimax controller ONCE on a disjoint DEV grid and
# FREEZE its weights. V3 defect D7: two scripts refitted L9 independently and
# obtained different controllers (0.4613 vs 0.5225).
using TOML, Random, Printf, Statistics, Dates
R = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(R,"src","CantorGate.jl")); include(joinpath(R,"src","v2","CantorBarrier.jl"))
include(joinpath(R,"src","v3_1","V31Controllers.jl")); include(joinpath(R,"src","v3_1","RobustDynamicsV31.jl"))
using .CantorGate: sinusoid, square_wave
using .CantorBarrier, .V31Controllers, .RobustDynamicsV31

n, E0, T = 5, 1/5, 8.0
F = bistable31(); x0s = [0.15, 0.35]
# DEV grid — disjoint from the reporting grid in configs/v3_1/synthetic.toml
dΔ = [-0.13, -0.07, 0.0, 0.07, 0.13]
dε = 10 .^ range(-0.4, 0.75; length = 4)
ref = [sinusoid(2.0, 4.0), square_wave(2.0, 1.5)]
mk(a,A,rng) = a == "sin" ? sinusoid(A, 10^(-0.3+1.8rand(rng)); φ=2π*rand(rng)) :
                           square_wave(A, 10^(-0.3+1.8rand(rng)); φ=2π*rand(rng))

function devworst(C, η)
    w = Inf
    for Δ in dΔ, ε in dε, a in ("sin","sq")
        δ = mk(a, ε, Xoshiro(hash((a,ε))%10^9)); acc = 0.0
        for x0 in x0s
            ts,xs,us = simulate31(C,F,δ; η=η,Δ=Δ,x0=x0,T=T,n_for_scale=n)
            acc += metrics31(ts,xs,us).safe_frac
        end
        w = min(w, acc/length(x0s))
    end
    w
end
function score(wv)
    C = build_v31("S10_minimax", n, E0; weights = collect(wv))
    η = match_gain31(C, F, ref, 0.30; tol=0.03, T=T, n_for_scale=n)
    devworst(C, η)
end

rng = Xoshiro(90210)
cands = Vector{Vector{Float64}}([ones(8)])
for _ in 1:160; push!(cands, rand(rng,8).^2 .+ 1e-3); end
push!(cands, [0.02,0.05,0.15,0.28,0.28,0.15,0.05,0.02])
push!(cands, [0.28,0.15,0.05,0.02,0.02,0.05,0.15,0.28])
push!(cands, [0.05,0.20,0.05,0.20,0.20,0.05,0.20,0.05])
function search(cands)
    b = (-Inf, ones(8))
    for c in cands
        v = score(c); v > b[1] && (b = (v, copy(c)))
    end
    for _ in 1:4, i in 1:8, m in (0.55, 0.8, 1.3, 1.8)
        c = copy(b[2]); c[i] *= m
        v = score(c); v > b[1] && (b = (v, c))
    end
    return b
end
best = search(cands)
w = best[2] ./ sum(best[2])
@printf("frozen L9: DEV worst-case = %.4f\n  weights = %s\n", best[1], string(round.(w; digits=6)))
open(joinpath(R,"configs","v3_1","l9_frozen_weights.toml"),"w") do io
    TOML.print(io, Dict("weights"=>collect(w), "dev_score"=>best[1],
        "search_seed"=>90210, "n_candidates"=>length(cands),
        "refinement"=>"4 rounds x 8 coords x 4 multipliers",
        "dev_delta"=>dΔ, "dev_eps"=>collect(dε), "budget_target"=>0.30,
        "timestamp"=>string(now()),
        "note"=>"FROZEN. The main sweep must READ these weights, never refit."))
end
println("wrote configs/v3_1/l9_frozen_weights.toml")
