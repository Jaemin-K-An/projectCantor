# ============================================================================
# V3.1 PHASE 8 — synthetic rerun with the CORRECTED controller set.
#   julia --project=. -t auto scripts/v3_1/run_synthetic_v31.jl
#
# Same uncertainty grid as V3 so the two are directly comparable. Changes:
#   * S1_true_constant is genuinely constant (V3 defect D1)
#   * the old ones(1) barrier is kept as S2_global_smooth
#   * S10_minimax READS frozen weights (V3 defect D7)
#   * gains matched on REALISED rms action, per controller per budget
# ============================================================================
using DataFrames, CSV, Statistics, Printf, Random, TOML, Dates, Base.Threads
R = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(R,"src","CantorGate.jl")); include(joinpath(R,"src","v2","CantorBarrier.jl"))
include(joinpath(R,"src","v3_1","V31Controllers.jl")); include(joinpath(R,"src","v3_1","RobustDynamicsV31.jl"))
using .CantorGate: sinusoid, square_wave, chirp, ou_noise
using .CantorBarrier, .V31Controllers, .RobustDynamicsV31
out(f) = joinpath(R, "results", "v3_1", "raw", f)

cfg = TOML.parsefile(joinpath(R,"configs","v3_1","synthetic.toml"))
U,S,B,C,SV = cfg["uncertainty"], cfg["system"], cfg["budget"], cfg["controllers"], cfg["solver"]
Δs = Float64.(U["delta"])
εs = 10 .^ range(Float64(U["log_eps_min"]), Float64(U["log_eps_max"]); length=Int(U["n_eps"]))
n = Int(S["n"]); E0 = 1.0/n; T = Float64(S["T"]); x0s = Float64.(S["x0"])
ATK = String.(S["attacks"]); TG = Float64.(B["targets"]); NS = Int(C["layout_seeds"])
F = bistable31()
L9W = Float64.(TOML.parsefile(joinpath(R,"configs","v3_1","l9_frozen_weights.toml"))["weights"])
println("L9 weights loaded from FROZEN file (never refitted here)")

function mkatk(fam, A, rng)
    ω = 10^(Float64(S["log_w_min"]) + (Float64(S["log_w_max"])-Float64(S["log_w_min"]))*rand(rng))
    fam=="sinusoid" && return sinusoid(A,ω;φ=2π*rand(rng))
    fam=="square"   && return square_wave(A,ω;φ=2π*rand(rng))
    fam=="chirp"    && return chirp(A,ω/3,6ω,T)
    fam=="ou_noise" && return ou_noise(A,T,1.0,2.0,2e-3,Int(abs(rand(rng,Int32))))
    error(fam)
end
ref = [mkatk("sinusoid",2.0,Xoshiro(1)), mkatk("square",2.0,Xoshiro(2))]

INST = Tuple{String,Int,Any}[]
for fam in V31_FAMILIES
    if fam in V31_RANDOMISED
        for s in 1:NS; push!(INST,(fam,s,build_v31(fam,n,E0;rng=Xoshiro(hash((fam,s))%10^9)))); end
    elseif fam in ("S10_minimax","S11_spline")
        push!(INST,(fam,0,build_v31(fam,n,E0; weights = fam=="S10_minimax" ? L9W : ones(8))))
    else
        push!(INST,(fam,0,build_v31(fam,n,E0)))
    end
end
@printf("%d instances × %d Δ × %d ε × %d attacks × %d x0 × %d budgets\n",
        length(INST), length(Δs), length(εs), length(ATK), length(x0s), length(TG))

println("\nmatching gains on REALISED rms action…")
GAIN = Dict{Tuple{String,Int,Float64},Float64}()
ACH  = Dict{Tuple{String,Int,Float64},Float64}()
lk = ReentrantLock()
@threads for i in eachindex(INST)
    fam,sd,Cc = INST[i]
    for tg in TG
        η = fam=="S0_none" ? 0.0 : match_gain31(Cc,F,ref,tg; tol=Float64(B["match_tol"]),
              T=T,n_save=Int(SV["n_save"]),step_safety=Float64(SV["step_safety"]),
              dt_max=Float64(SV["dt_max"]),n_for_scale=n)
        a = fam=="S0_none" ? 0.0 : realised_action31(Cc,F,ref,η; T=T,n_save=Int(SV["n_save"]),
              step_safety=Float64(SV["step_safety"]),dt_max=Float64(SV["dt_max"]),n_for_scale=n)
        lock(lk) do; GAIN[(fam,sd,tg)]=η; ACH[(fam,sd,tg)]=a; end
    end
end
for tg in TG
    println("  target $tg:")
    for fam in V31_FAMILIES
        fam=="S0_none" && continue
        v = [ACH[(fam,sd,tg)] for (f,sd,_) in INST if f==fam]
        @printf("    %-22s %.4f (%+.1f%%)\n", fam, mean(v), 100*(mean(v)-tg)/tg)
    end
end

cases = NamedTuple[]
for (fam,sd,_) in INST, tg in TG, Δ in Δs, ε in εs, a in ATK, x0 in x0s
    push!(cases,(fam=fam,sd=sd,tg=tg,Δ=Δ,ε=ε,a=a,x0=x0))
end
N = length(cases); @printf("\nsweep: %d simulations on %d threads\n", N, nthreads())
res = Vector{NamedTuple}(undef,N); t0 = time()
@threads for i in 1:N
    c = cases[i]
    Cc = INST[findfirst(x->x[1]==c.fam && x[2]==c.sd, INST)][3]
    δ = mkatk(c.a,c.ε,Xoshiro(hash((c.a,c.ε,c.x0))%10^9))
    ts,xs,us = simulate31(Cc,F,δ; η=GAIN[(c.fam,c.sd,c.tg)],Δ=c.Δ,x0=c.x0,T=T,
        n_for_scale=n,step_safety=Float64(SV["step_safety"]),dt_max=Float64(SV["dt_max"]),
        n_save=Int(SV["n_save"]))
    m = metrics31(ts,xs,us)
    res[i] = (family=c.fam,seed=c.sd,budget=c.tg,delta=c.Δ,eps=c.ε,log_eps=log10(c.ε),
              attack=c.a,x0=c.x0,eta=GAIN[(c.fam,c.sd,c.tg)],safe=Int(m.safe),
              x_max=m.x_max,safe_frac=m.safe_frac,action_rms=m.action_rms,
              sup_deriv=sup_field_derivative(Cc))
end
@printf("elapsed %.1f s (%.2f ms/sim)\n", time()-t0, 1000*(time()-t0)/N)
df = DataFrame(res); mkpath(dirname(out("x"))); CSV.write(out("v31_synthetic.csv"), df)
gc = try strip(read(`git -C $R rev-parse --short HEAD`,String)) catch; "nogit" end
open(out("v31_synthetic.csv.meta.toml"),"w") do io
    TOML.print(io, Dict("timestamp"=>string(now()),"git_commit"=>gc,"julia"=>string(VERSION),
        "n_cases"=>N,"phase"=>"V3.1-PHASE8","config"=>"configs/v3_1/synthetic.toml",
        "l9_frozen"=>"configs/v3_1/l9_frozen_weights.toml"))
end
@info "wrote $(nrow(df)) rows → $(out("v31_synthetic.csv"))"

println("\n" * "="^104)
println("PRIMARY — R_worst = min over (Δ,ε,attack) of graded safe_frac")
println("="^104)
for tg in TG
    println("\n--- realised budget $tg ---")
    @printf("%-22s %10s %10s %12s %10s\n","controller","R_worst","R_mean","‖u'‖_inf","worst Δ")
    rows = []
    for fam in V31_FAMILIES
        s = df[(df.family.==fam) .& (df.budget.==tg), :]
        isempty(s) && continue
        cell = combine(groupby(s,[:delta,:eps,:attack]), :safe_frac=>mean=>:R)
        push!(rows,(fam,minimum(cell.R),mean(s.safe_frac),s.sup_deriv[1],cell.delta[argmin(cell.R)]))
    end
    for r in sort(rows; by=x->-x[2])
        @printf("%-22s %10.4f %10.4f %12.1f %10.3f\n", r...)
    end
end
