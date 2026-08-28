# ============================================================================
# PHASE I — seen / unseen perturbation benchmark for the TRAINED Neural ODEs.
#   julia --project=. scripts/run_neural_benchmark.jl
#
# Loads results/processed/trained_params.jls (written by run_neural_ode.jl) so
# the benchmark can be re-run without retraining.
#
# TRAIN/TEST SPLIT WAS FIXED IN configs/neural_ode.toml BEFORE TRAINING.
#   TRAIN   : sinusoids, A∈[0.5,1.5], ω∈[2,6], h0∈[0.05,0.95]
#   TEST-ID : fresh draws from that same distribution
#   TEST-OOD: square / chirp / multifreq / impulse / OU / piecewise-random
#             + amplitude extrapolation A=3 + frequency extrapolation ω=16
#             + an adversarial random search inside the TRAINING amplitude
# No test condition influenced any training choice.
#
# INTEGRATOR: fixed-step RK4, the same choice as everywhere else in this study
# (`src/Dynamics.jl`): the trained closed loop f_θ + g·δ is still discontinuous
# in h, so an adaptive solver can chatter on a switching surface. An adaptive
# CROSS-CHECK with a capped `maxiters` is run on a subset and its failure rate
# is reported rather than hidden.
#
# Output: results/raw/neural_benchmark.csv
#         results/raw/neural_solver_crosscheck.csv
#         results/tables/neural_summary.{csv,md}
# ============================================================================

using DataFrames, Statistics, Printf, Random, Lux, ComponentArrays, Serialization
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

cfg = load_config("neural_ode.toml")
TR, MO, EV = cfg["train"], cfg["models"], cfg["eval"]
κ, γ  = cfg["reference"]["kappa"], cfg["reference"]["gamma"]
n_gate = Int(MO["n"])
T_eval, dt_save = Float64(EV["T"]), Float64(EV["dt_save"])

D = deserialize(procpath("trained_params.jls"))
nvf, _ = init_nvf(Xoshiro(Int(TR["seed"])); width = Int(D["width"]),
                  depth = Int(D["depth"]), T = Float64(D["T"]))
P = D["params"]

gc = cantor_interval_gate(n_gate)
variants = Tuple{String,Any}[("neural_none", NoGate()), ("neural_cantor_n$(n_gate)", gc)]
for s in 1:Int(MO["random_seeds"])
    push!(variants, ("neural_random_n$(n_gate)_s$(s)",
                     random_matched_gate(n_gate, Xoshiro(seed_for("node_random", n_gate, s)))))
end
for β in Float64.(MO["smooth_betas"])
    push!(variants, ("neural_smooth_b$(Int(β))", SmoothGate(gc, β)))
end
variants = filter(v -> haskey(P, v[1]), variants)
@printf("%d trained models loaded\n", length(variants))

# ------------------------------------------------------------- test batteries
rng_id = Xoshiro(987654)
id_cases = [(A = TR["A_lo"] + (TR["A_hi"]-TR["A_lo"])*rand(rng_id),
             ω = TR["w_lo"] + (TR["w_hi"]-TR["w_lo"])*rand(rng_id),
             φ = 2π*rand(rng_id)) for _ in 1:Int(EV["n_id"])]

Aood, ωood = Float64(EV["ood_A_extrap"]), Float64(EV["ood_w_extrap"])
Amid = (TR["A_lo"] + TR["A_hi"]) / 2      # 1.0 — the training amplitude midpoint
ωmid = (TR["w_lo"] + TR["w_hi"]) / 2      # 4.0

function ood_battery()
    out = Tuple{String,String,Perturb}[]
    for s in Int.(EV["ood_seeds"])
        push!(out, ("square",      "waveform",   square_wave(Amid, ωmid; φ = s)))
        push!(out, ("chirp",       "waveform",   chirp(Amid, 0.5, 20.0, T_eval)))
        push!(out, ("multifreq",   "waveform",   multifreq(Amid, [ωmid, 1.7ωmid, 3.3ωmid];
                                                           φs = [s, 2s, 3s])))
        push!(out, ("impulse",     "waveform",   impulse_train(Amid,
                                                 collect(1.5:2.5:T_eval) .+ 0.1s, 0.12)))
        push!(out, ("ou_noise",    "stochastic", ou_noise(Amid, T_eval, 1.0, 2.0, 0.005, s)))
        push!(out, ("pw_random",   "stochastic", piecewise_random(Amid, T_eval, 0.35, s)))
        push!(out, ("amp_extrap",  "amplitude",  sinusoid(Aood, ωmid; φ = s)))
        push!(out, ("freq_extrap", "frequency",  sinusoid(Amid, ωood; φ = s)))
    end
    return out
end

h0_test = Float64.(EV["h0_test"])
bench = DataFrame()
cross = DataFrame()
t0 = time()

for (label, sg) in variants
    ps = P[label]
    function ev(δ, h0)
        ts, hs, gs = evaluate_neural_ode(nvf, ps, sg, δ; h0 = h0, T = T_eval,
                                         dt_save = dt_save)
        (mse = tracking_mse(ts, hs; h0 = h0, κ = κ, γ = γ),
         m = compute_metrics(ts, hs, gs; hold = 3.0), ts = ts, hs = hs)
    end
    for c in id_cases, h0 in h0_test
        r = ev(sinusoid(c.A, c.ω; φ = c.φ), h0)
        push!(bench, merge((model = label, split = "ID", family = "sinusoid",
                            kind = "sinusoid", h0 = h0, A = c.A, omega = c.ω,
                            mse = r.mse), metrics_tuple(r.m)); cols = :union)
    end
    for (name, fam, δ) in ood_battery(), h0 in h0_test
        r = ev(δ, h0)
        push!(bench, merge((model = label, split = "OOD", family = fam, kind = name,
                            h0 = h0, A = δ.bound, omega = NaN, mse = r.mse),
                           metrics_tuple(r.m)); cols = :union)
    end
    # ADVERSARIAL: random search over 3-tone mixtures inside the TRAINING
    # amplitude budget, maximising mean deviation from the safe state.
    rng_a = Xoshiro(seed_for("adv", label))
    worst_val, worst_δ, worst_h0 = -Inf, nothing, h0_test[1]
    for _ in 1:Int(EV["adversarial_budget"])
        ωs = TR["w_lo"] .+ (ωood - TR["w_lo"]) .* rand(rng_a, 3)
        δ  = multifreq(Amid, ωs; φs = 2π .* rand(rng_a, 3))
        h0 = rand(rng_a, h0_test)
        r  = ev(δ, h0)
        r.m.D_mean > worst_val && ((worst_val, worst_δ, worst_h0) = (r.m.D_mean, δ, h0))
    end
    r = ev(worst_δ, worst_h0)
    push!(bench, merge((model = label, split = "OOD", family = "adversarial",
                        kind = "adversarial", h0 = worst_h0, A = Amid, omega = NaN,
                        mse = r.mse), metrics_tuple(r.m)); cols = :union)

    # adaptive cross-check on a fixed subset (one h0 per OOD kind + 4 ID cases)
    subset = vcat([(("sinusoid"), sinusoid(c.A, c.ω; φ = c.φ)) for c in id_cases[1:4]],
                  [(name, δ) for (name, _, δ) in ood_battery()[1:8]])
    for (name, δ) in subset, h0 in (0.20, 0.72)
        ts, hs, gs = evaluate_neural_ode(nvf, ps, sg, δ; h0 = h0, T = T_eval, dt_save = dt_save)
        m_rk4 = compute_metrics(ts, hs, gs; hold = 3.0)
        ta, ha, ga, ok = evaluate_neural_ode_adaptive(nvf, ps, sg, δ; h0 = h0,
                                                      T = T_eval, dt_save = dt_save)
        m_ad = ok ? compute_metrics(ta, ha, ga; hold = 3.0) : nothing
        push!(cross, (model = label, kind = name, h0 = h0,
                      R_safe_rk4 = m_rk4.R_safe,
                      R_safe_adaptive = ok ? m_ad.R_safe : NaN,
                      abs_diff = ok ? abs(m_rk4.R_safe - m_ad.R_safe) : NaN,
                      adaptive_ok = ok))
    end
    @printf("  %-28s done (%.0f s elapsed)\n", label, time() - t0)
end

write_raw(bench, "neural_benchmark.csv"; overwrite = true,
          meta = Dict("phase"=>"I", "config"=>"neural_ode.toml",
                      "n_rows"=>nrow(bench), "integrator"=>"rk4_fixed_step",
                      "dt"=>dt_save/2, "T"=>T_eval))
write_raw(cross, "neural_solver_crosscheck.csv"; overwrite = true,
          meta = Dict("phase"=>"I", "note"=>"RK4 vs Tsit5 (maxiters capped at 2e5)"))

# --------------------------------------------------------------- reporting
histf = rawpath("neural_training.csv")
using CSV
ht = CSV.read(histf, DataFrame)

summ = DataFrame()
for (label, _) in variants
    b   = bench[bench.model .== label, :]
    idb = b[b.split .== "ID", :]; oodb = b[b.split .== "OOD", :]
    h   = ht[ht.model .== label, :]
    push!(summ, (model = label,
                 mse_id = mean(idb.mse), mse_ood = mean(oodb.mse),
                 delta_ood_mse = mean(oodb.mse) - mean(idb.mse),
                 rsafe_id = mean(idb.R_safe), rsafe_ood = mean(oodb.R_safe),
                 delta_ood_rsafe = mean(idb.R_safe) - mean(oodb.R_safe),
                 dmax_id = mean(idb.D_max), dmax_ood = mean(oodb.D_max),
                 dmean_ood = mean(oodb.D_mean),
                 tau_inf_frac_ood = count(!isfinite, oodb.tau_S) / nrow(oodb),
                 trec_inf_frac_ood = count(!isfinite, oodb.T_rec) / nrow(oodb),
                 adversarial_R_safe = only(b[b.kind .== "adversarial", :R_safe]),
                 final_loss = h.loss[end], mean_gnorm = mean(h.gnorm)))
end
write_table(summ, "neural_summary.csv")

println("\n" * "="^116)
println("NEURAL ODE BENCHMARK — TEST-ID (seen) vs TEST-OOD (unseen)")
println("="^116)
@printf("%-24s %10s %10s %10s %10s %10s %10s %10s\n", "model", "MSE_ID", "MSE_OOD",
        "Rsafe_ID", "Rsafe_OOD", "ΔOOD", "D_max_OOD", "adv Rsafe")
for r in eachrow(summ)
    @printf("%-24s %10.5f %10.5f %10.4f %10.4f %+10.4f %10.4f %10.4f\n",
            replace(r.model, "neural_"=>""), r.mse_id, r.mse_ood, r.rsafe_id,
            r.rsafe_ood, r.delta_ood_rsafe, r.dmax_ood, r.adversarial_R_safe)
end

println("\nR_safe by OOD family:")
kinds = unique(bench[bench.split .== "OOD", :kind])
@printf("%-24s", "model"); for k in kinds; @printf("%12s", first(k, 11)); end; println()
for (label, _) in variants
    @printf("%-24s", replace(label, "neural_"=>""))
    for k in kinds
        s = bench[(bench.model .== label) .& (bench.kind .== k), :R_safe]
        @printf("%12.4f", isempty(s) ? NaN : mean(s))
    end
    println()
end

println("\nsolver cross-check (RK4 vs adaptive Tsit5, maxiters capped at 2e5):")
@printf("  adaptive solves that FAILED (chattering / maxiters): %d of %d (%.1f%%)\n",
        count(.!cross.adaptive_ok), nrow(cross),
        100*count(.!cross.adaptive_ok)/nrow(cross))
okc = cross[cross.adaptive_ok, :]
@printf("  where it succeeded: max |ΔR_safe| = %.2e, median = %.2e\n",
        maximum(okc.abs_diff), median(okc.abs_diff))
for (label, _) in variants
    s = cross[cross.model .== label, :]
    @printf("    %-24s failures %2d/%2d\n", replace(label,"neural_"=>""),
            count(.!s.adaptive_ok), nrow(s))
end

open(tabpath("neural_summary.md"), "w") do io
    println(io, "# Neural ODE benchmark (auto-generated)\n")
    println(io, "Integrator: fixed-step RK4, Δt = $(dt_save/2), T = $T_eval.\n")
    println(io, "| model | MSE-ID | MSE-OOD | R_safe ID | R_safe OOD | ΔOOD | D_max OOD | adversarial R_safe | final loss | mean ‖∇θL‖ |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for r in eachrow(summ)
        println(io, "| $(replace(r.model,"neural_"=>"")) | ", fmtf(r.mse_id,5), " | ",
                fmtf(r.mse_ood,5), " | ", fmtf(r.rsafe_id,4), " | ", fmtf(r.rsafe_ood,4),
                " | ", fmtf(r.delta_ood_rsafe,4), " | ", fmtf(r.dmax_ood,4), " | ",
                fmtf(r.adversarial_R_safe,4), " | ", fmtf(r.final_loss,6), " | ",
                fmtf(r.mean_gnorm,4), " |")
    end
    println(io, "\n## R_safe by OOD family\n")
    print(io, "| model |"); for k in kinds; print(io, " $k |"); end; println(io)
    print(io, "|---|"); for _ in kinds; print(io, "---|"); end; println(io)
    for (label, _) in variants
        print(io, "| $(replace(label,"neural_"=>"")) |")
        for k in kinds
            s = bench[(bench.model .== label) .& (bench.kind .== k), :R_safe]
            print(io, " ", isempty(s) ? "-" : fmtf(mean(s),4), " |")
        end
        println(io)
    end
    println(io, "\n## Solver cross-check\n")
    println(io, "- adaptive Tsit5 solves that failed (maxiters / chattering): ",
            "$(count(.!cross.adaptive_ok)) of $(nrow(cross))")
    println(io, "- where it succeeded: max |ΔR_safe| vs RK4 = ",
            fmtf(maximum(okc.abs_diff), 6))
end
println("\ndone.")
