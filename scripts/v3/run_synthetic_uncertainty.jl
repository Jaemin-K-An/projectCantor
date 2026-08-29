# ============================================================================
# V3 PHASE 6 — joint boundary × attack uncertainty sweep.
#   julia --project=. -t auto scripts/v3/run_synthetic_uncertainty.jl
#
# For every controller: match the gain so the REALISED rms action hits the
# target budget, then sweep Delta × eps × attack × x0 and record whether the
# TRUE boundary (1/2) was ever crossed.
#
# Writes incrementally so a session interruption cannot lose finished work.
# ============================================================================

using DataFrames, CSV, Statistics, Printf, Random, TOML, Dates, Base.Threads
include(joinpath(@__DIR__, "..", "..", "src", "CantorGate.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v2", "CantorBarrier.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v3", "V3Controllers.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v3", "RobustDynamics.jl"))
using .CantorGate: sinusoid, square_wave, chirp, ou_noise
using .CantorBarrier, .V3Controllers, .RobustDynamics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
out(f) = joinpath(ROOT, "results", "v3", "raw", f)

cfg = TOML.parsefile(joinpath(ROOT, "configs", "v3", "synthetic.toml"))
U, S, B, C, SV = cfg["uncertainty"], cfg["system"], cfg["budget"],
                 cfg["controllers"], cfg["solver"]
Δs  = Float64.(U["delta"])
εs  = 10 .^ range(Float64(U["log_eps_min"]), Float64(U["log_eps_max"]);
                  length = Int(U["n_eps"]))
n   = Int(S["n"]); E0 = 1.0 / n; T = Float64(S["T"])
x0s = Float64.(S["x0"]); ATK = String.(S["attacks"])
TARGETS = Float64.(B["targets"]); NSEED = Int(C["layout_seeds"])

FIELDS = Dict("bistable" => bistable3(), "cubic" => cubic3(), "linear" => linear3())
F = FIELDS[first(String.(S["fields"]))]

function mkattack(fam::String, A::Float64, rng::AbstractRNG)
    ω = 10 ^ (Float64(S["log_w_min"]) +
              (Float64(S["log_w_max"]) - Float64(S["log_w_min"])) * rand(rng))
    fam == "sinusoid" && return sinusoid(A, ω; φ = 2π*rand(rng))
    fam == "square"   && return square_wave(A, ω; φ = 2π*rand(rng))
    fam == "chirp"    && return chirp(A, ω/3, 6ω, T)
    fam == "ou_noise" && return ou_noise(A, T, 1.0, 2.0, 2e-3, Int(abs(rand(rng, Int32))))
    error("unknown attack $fam")
end

# ---------------------------------------------------------------------------
# FIT L9 (minimax) on the DISJOINT dev grid, before anything else.
# L9 is the strong baseline explicitly optimised for the SAME objective Cantor
# is judged on. Without this step L9 was identical to L10 (uniform weights) and
# was not a baseline at all.
# ---------------------------------------------------------------------------
const ref_atk0 = [mkattack("sinusoid", 2.0, Xoshiro(1)),
                  mkattack("square", 2.0, Xoshiro(2))]
DV = cfg["dev"]
dΔ = Float64.(DV["delta"])
dε = 10 .^ range(Float64(DV["log_eps_min"]), Float64(DV["log_eps_max"]);
                 length = Int(DV["n_eps"]))
dATK = String.(DV["attacks"])

"""Worst-case graded robustness of a layout on the DEV uncertainty grid."""
function dev_worst(L::BarrierLayout, η::Float64)
    w = Inf
    for Δ in dΔ, ε in dε, a in dATK
        δ = mkattack(a, ε, Xoshiro(hash((a, ε, 0.15)) % 10^9))
        acc = 0.0
        for x0 in x0s
            ts, xs, us = simulate_robust(L, F, δ; η=η, Δ=Δ, x0=x0, T=T,
                n_for_scale=n, step_safety=Float64(SV["step_safety"]),
                dt_max=Float64(SV["dt_max"]), n_save=Int(SV["n_save"]))
            acc += robust_metrics(ts, xs, us).safe_frac
        end
        w = min(w, acc / length(x0s))
    end
    return w
end

println("fitting L9_minimax on the disjoint DEV grid ($(Int(DV["n_candidates"])) candidates)...")
let rng = Xoshiro(31337), best = -Inf, bestw = ones(8)
    for c in 1:Int(DV["n_candidates"])
        w = c == 1 ? ones(8) : rand(rng, 8) .^ 2 .+ 1e-3
        Lc = piecewise_layout(collect(w), n, E0, "L9", "L9_minimax")
        η = match_gain(Lc, F, ref_atk0, 0.30; tol=0.03, T=T,
                       n_save=Int(SV["n_save"]), step_safety=Float64(SV["step_safety"]),
                       dt_max=Float64(SV["dt_max"]), n_for_scale=n)
        v = dev_worst(Lc, η)
        v > best && (best = v; bestw = collect(w))
    end
    global L9_WEIGHTS = bestw
    @printf("  best DEV worst-case graded R = %.4f   weights = %s
", best,
            string(round.(bestw ./ sum(bestw); digits=3)))
end

# controller instances: deterministic once, randomised × seeds
INST = Tuple{String,Int,BarrierLayout}[]
for fam in V3_FAMILIES
    if fam in V3_RANDOMISED
        for s in 1:NSEED
            push!(INST, (fam, s, v3_layout(fam, n, E0; rng = Xoshiro(hash((fam,s))%10^9))))
        end
    else
        push!(INST, (fam, 0, fam == "L9_minimax" ?
                     piecewise_layout(L9_WEIGHTS, n, E0, "L9_minimax", "L9_minimax") :
                     v3_layout(fam, n, E0)))
    end
end
@printf("%d controller instances, %d Δ × %d ε × %d attacks × %d x0 × %d budgets\n",
        length(INST), length(Δs), length(εs), length(ATK), length(x0s), length(TARGETS))

# ---- gain matching: realised rms action == target, per instance per budget ---
println("\nmatching gains to REALISED budget (V3 fairness constraint)...")
ref_atk = ref_atk0
GAIN = Dict{Tuple{String,Int,Float64},Float64}()
ACH  = Dict{Tuple{String,Int,Float64},Float64}()
lk = ReentrantLock()
@threads for i in eachindex(INST)
    fam, sd, L = INST[i]
    for tg in TARGETS
        η = fam == "L0_none" ? 0.0 :
            match_gain(L, F, ref_atk, tg; tol = Float64(B["match_tol"]),
                       T = T, n_save = Int(SV["n_save"]),
                       step_safety = Float64(SV["step_safety"]),
                       dt_max = Float64(SV["dt_max"]), n_for_scale = n)
        a = fam == "L0_none" ? 0.0 : realised_action(L, F, ref_atk, η; T = T,
                n_save = Int(SV["n_save"]), step_safety = Float64(SV["step_safety"]),
                dt_max = Float64(SV["dt_max"]), n_for_scale = n)
        lock(lk) do; GAIN[(fam,sd,tg)] = η; ACH[(fam,sd,tg)] = a; end
    end
end
println("  achieved realised rms action (target → mean achieved over seeds):")
for tg in TARGETS, fam in V3_FAMILIES
    fam == "L0_none" && continue
    v = [ACH[(fam,sd,tg)] for (f2,sd,_) in INST if f2 == fam]
    @printf("    target %.2f  %-20s %.4f  (dev %+.1f%%)\n", tg, fam, mean(v),
            100*(mean(v)-tg)/tg)
end

# ---------------------------------------------------------------- main sweep
cases = NamedTuple[]
for (fam, sd, _) in INST, tg in TARGETS, Δ in Δs, ε in εs, a in ATK, x0 in x0s
    push!(cases, (fam=fam, sd=sd, tg=tg, Δ=Δ, ε=ε, a=a, x0=x0))
end
N = length(cases); @printf("\nsweep: %d simulations on %d threads\n", N, nthreads())
res = Vector{NamedTuple}(undef, N)
t0 = time()
@threads for i in 1:N
    c = cases[i]
    L = INST[findfirst(x -> x[1] == c.fam && x[2] == c.sd, INST)][3]
    η = GAIN[(c.fam, c.sd, c.tg)]
    δ = mkattack(c.a, c.ε, Xoshiro(hash((c.a, c.ε, c.x0)) % 10^9))
    ts, xs, us = simulate_robust(L, F, δ; η=η, Δ=c.Δ, x0=c.x0, T=T,
                                 n_for_scale=n, step_safety=Float64(SV["step_safety"]),
                                 dt_max=Float64(SV["dt_max"]), n_save=Int(SV["n_save"]))
    m = robust_metrics(ts, xs, us)
    res[i] = (family=c.fam, seed=c.sd, budget=c.tg, delta=c.Δ, eps=c.ε,
              log_eps=log10(c.ε), attack=c.a, x0=c.x0, eta=η,
              safe=Int(m.safe), x_max=m.x_max, safe_frac=m.safe_frac,
              action_rms=m.action_rms, action_max=m.action_max)
end
@printf("elapsed %.1f s (%.2f ms/sim)\n", time()-t0, 1000*(time()-t0)/N)
df = DataFrame(res)
mkpath(dirname(out("x"))); CSV.write(out("v3_synthetic_joint.csv"), df)
gc = try strip(read(`git -C $ROOT rev-parse --short HEAD`, String)) catch; "nogit" end
open(out("v3_synthetic_joint.csv.meta.toml"), "w") do io
    TOML.print(io, Dict("timestamp"=>string(now()), "git_commit"=>gc,
                        "julia"=>string(VERSION), "n_cases"=>N,
                        "phase"=>"V3-PHASE6", "config"=>"configs/v3/synthetic.toml"))
end
@info "wrote $(nrow(df)) rows → $(out("v3_synthetic_joint.csv"))"

# ------------------------------------------------ PRIMARY: minimax robustness
println("\n" * "="^100)
println("PRIMARY ENDPOINT — R_worst = min over (Δ, ε, attack) of GRADED safe_frac")
println("(pre-registered in configs/v3/synthetic.toml)")
println("="^100)
for tg in TARGETS
    println("\n--- realised budget target $tg ---")
    @printf("%-22s %10s %10s %10s %10s %10s\n", "controller", "R_worst",
            "R_mean", "AUC_joint", "Rw_binary", "worst Δ")
    for fam in V3_FAMILIES
        s = df[(df.family .== fam) .& (df.budget .== tg), :]
        isempty(s) && continue
        cell = combine(groupby(s, [:delta, :eps, :attack]), :safe_frac => mean => :R)
        cellb = combine(groupby(s, [:delta, :eps, :attack]), :safe => mean => :Rb)
        wi = argmin(cell.R)
        # joint AUC: trapezoid over Δ and log ε, averaged over attacks
        auc = 0.0; nA = 0
        for a in unique(cell.attack)
            ca = cell[cell.attack .== a, :]
            M = [only(ca[(ca.delta .== d) .& (ca.eps .== e), :R]) for d in Δs, e in εs]
            iy = [sum((M[i, 1:end-1] .+ M[i, 2:end]) ./ 2 .*
                      diff(log10.(collect(εs)))) for i in 1:size(M,1)]
            auc += sum((iy[1:end-1] .+ iy[2:end]) ./ 2 .* diff(Δs)); nA += 1
        end
        auc /= (nA * (Δs[end]-Δs[1]) * (log10(εs[end])-log10(εs[1])))
        @printf("%-22s %10.4f %10.4f %10.4f %10.4f %10.3f\n", fam, minimum(cell.R),
                mean(s.safe_frac), auc, minimum(cellb.Rb), cell.delta[wi])
    end
end
println("\ndone.")
