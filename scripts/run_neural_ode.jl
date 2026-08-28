# ============================================================================
# PHASE H + PHASE I — trainable Neural ODE and the seen/unseen benchmark.
#   julia --project=. scripts/run_neural_ode.jl
#
# From here on the term "Neural ODE" is earned: dh/dt = f_θ(h,t) + g(h)δ(t)
# with θ learned by gradient descent.
#
# TRAIN/TEST SPLIT WAS FIXED IN configs/neural_ode.toml BEFORE THE FIRST RUN.
#   TRAIN   : sinusoids, A∈[0.5,1.5], ω∈[2,6], h0∈[0.05,0.95]
#   TEST-ID : fresh draws from that same distribution
#   TEST-OOD: square / chirp / impulse / multi-frequency / OU / piecewise-random
#             + amplitude extrapolation (A=3) + frequency extrapolation (ω=16)
#             + an adversarial search inside the training amplitude budget
# No test condition influenced any training choice.
#
# Output: results/raw/neural_training.csv, neural_benchmark.csv
#         results/tables/neural_summary.md
# ============================================================================

using DataFrames, Statistics, Printf, Random, Lux, ComponentArrays, Serialization
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

cfg  = load_config("neural_ode.toml")
TR, MO, EV = cfg["train"], cfg["models"], cfg["eval"]
κ, γ = cfg["reference"]["kappa"], cfg["reference"]["gamma"]
n_gate = Int(MO["n"])
T_eval = Float64(EV["T"]);  dt_save = Float64(EV["dt_save"])

# --------------------------------------------------------------- model zoo
# All variants share ONE network initialisation so that differences are due to
# the gate, not to a lucky init.
rng0 = Xoshiro(Int(TR["seed"]))
nvf, ps0 = init_nvf(rng0; width = Int(TR["width"]), depth = Int(TR["depth"]),
                    T = Float64(TR["T"]))
@printf("f_θ: MLP %d→%d(×%d, tanh)→1,  %d parameters\n",
        2, TR["width"], TR["depth"], length(ps0))

variants = Tuple{String,Any,Any}[]                 # (label, scalar gate, batch gate)
push!(variants, ("neural_none", NoGate(), BNone()))
gc = cantor_interval_gate(n_gate)
push!(variants, ("neural_cantor_n$(n_gate)", gc, to_batch_gate(gc)))
for s in 1:Int(MO["random_seeds"])
    gr = random_matched_gate(n_gate, Xoshiro(seed_for("node_random", n_gate, s)))
    push!(variants, ("neural_random_n$(n_gate)_s$(s)", gr, to_batch_gate(gr)))
end
for β in Float64.(MO["smooth_betas"])
    gs = SmoothGate(gc, β)
    push!(variants, ("neural_smooth_b$(Int(β))", gs, to_batch_gate(gs)))
end

# ------------------------------------------------------------------ training
println("\n" * "="^90); println("TRAINING"); println("="^90)
trained = Dict{String,Any}()
histdf  = DataFrame()
for (label, _, bg) in variants
    println("── $label")
    ps, hist = train_neural_ode(nvf, ps0, bg, TR; rng = Xoshiro(Int(TR["seed"]) + 1))
    trained[label] = ps
    for i in eachindex(hist.iter)
        push!(histdf, (model = label, iter = hist.iter[i],
                       loss = hist.loss[i], gnorm = hist.gnorm[i]))
    end
end
write_raw(histdf, "neural_training.csv"; overwrite = true,
          meta = Dict("phase"=>"H", "config"=>"neural_ode.toml",
                      "n_params"=>length(ps0), "iters"=>TR["iters"]))
serialize(joinpath(procpath("."), "trained_params.jls"),
          Dict("labels"=>collect(keys(trained)), "params"=>trained,
               "width"=>TR["width"], "depth"=>TR["depth"], "T"=>TR["T"]))

println("\nfinal training loss and mean gradient norm:")
@printf("  %-28s %14s %14s %14s\n", "model", "loss[1]", "loss[end]", "mean |∇|")
for (label, _, _) in variants
    h = histdf[histdf.model .== label, :]
    @printf("  %-28s %14.6e %14.6e %14.4e\n", label, h.loss[1], h.loss[end], mean(h.gnorm))
end

# --------------------------------------------------------------- benchmark
println("\n" * "="^90); println("BENCHMARK — TEST-ID vs TEST-OOD"); println("="^90)

rng_id = Xoshiro(987654)                      # ID draws, fixed seed
id_cases = [(A = TR["A_lo"] + (TR["A_hi"]-TR["A_lo"])*rand(rng_id),
             ω = TR["w_lo"] + (TR["w_hi"]-TR["w_lo"])*rand(rng_id),
             φ = 2π*rand(rng_id)) for _ in 1:Int(EV["n_id"])]

Aood, ωood = Float64(EV["ood_A_extrap"]), Float64(EV["ood_w_extrap"])
Amid = (TR["A_lo"] + TR["A_hi"]) / 2         # = 1.0, inside the training amplitude
ωmid = (TR["w_lo"] + TR["w_hi"]) / 2         # = 4.0

"""The a-priori OOD battery. Amplitudes stay at the TRAINING amplitude unless
the condition is explicitly an amplitude-extrapolation test."""
function ood_battery()
    out = Tuple{String,String,Perturb}[]
    for s in Int.(EV["ood_seeds"])
        push!(out, ("square",      "waveform",  square_wave(Amid, ωmid; φ = s)))
        push!(out, ("chirp",       "waveform",  chirp(Amid, 0.5, 20.0, T_eval)))
        push!(out, ("multifreq",   "waveform",  multifreq(Amid, [ωmid, 1.7ωmid, 3.3ωmid];
                                                          φs = [s, 2s, 3s])))
        push!(out, ("impulse",     "waveform",  impulse_train(Amid,
                                                collect(1.5:2.5:T_eval) .+ 0.1s, 0.12)))
        push!(out, ("ou_noise",    "stochastic",ou_noise(Amid, T_eval, 1.0, 2.0, 0.005, s)))
        push!(out, ("pw_random",   "stochastic",piecewise_random(Amid, T_eval, 0.35, s)))
        push!(out, ("amp_extrap",  "amplitude", sinusoid(Aood, ωmid; φ = s)))
        push!(out, ("freq_extrap", "frequency", sinusoid(Amid, ωood; φ = s)))
    end
    return unique(out)
end

h0_test = Float64.(EV["h0_test"])
bench = DataFrame()
for (label, sg, _) in variants
    ps = trained[label]
    function ev(δ, h0)
        ts, hs, gs = evaluate_neural_ode(nvf, ps, sg, δ; h0 = h0, T = T_eval,
                                         dt_save = dt_save)
        m = compute_metrics(ts, hs, gs; hold = 3.0)
        (mse = tracking_mse(ts, hs; h0 = h0, κ = κ, γ = γ), m = m)
    end
    # TEST-ID
    for c in id_cases, h0 in h0_test
        r = ev(sinusoid(c.A, c.ω; φ = c.φ), h0)
        push!(bench, merge((model = label, split = "ID", family = "sinusoid",
                            kind = "sinusoid", h0 = h0, A = c.A, omega = c.ω,
                            mse = r.mse), metrics_tuple(r.m)); cols = :union)
    end
    # TEST-OOD
    for (name, fam, δ) in ood_battery(), h0 in h0_test
        r = ev(δ, h0)
        push!(bench, merge((model = label, split = "OOD", family = fam, kind = name,
                            h0 = h0, A = δ.bound, omega = NaN, mse = r.mse),
                           metrics_tuple(r.m)); cols = :union)
    end
    # ADVERSARIAL: random search over multi-frequency phase/frequency mixtures
    # inside the TRAINING amplitude budget, maximising mean deviation.
    rng_a = Xoshiro(seed_for("adv", label))
    worst = (mse = -Inf, δ = nothing, h0 = h0_test[1])
    for _ in 1:Int(EV["adversarial_budget"])
        ωs = TR["w_lo"] .+ (ωood - TR["w_lo"]) .* rand(rng_a, 3)
        δ = multifreq(Amid, ωs; φs = 2π .* rand(rng_a, 3))
        h0 = rand(rng_a, h0_test)
        r = ev(δ, h0)
        r.m.D_mean > (worst.mse == -Inf ? -Inf : worst.mse) &&
            (worst = (mse = r.m.D_mean, δ = δ, h0 = h0))
    end
    r = ev(worst.δ, worst.h0)
    push!(bench, merge((model = label, split = "OOD", family = "adversarial",
                        kind = "adversarial", h0 = worst.h0, A = Amid, omega = NaN,
                        mse = r.mse), metrics_tuple(r.m)); cols = :union)
end
write_raw(bench, "neural_benchmark.csv"; overwrite = true,
          meta = Dict("phase"=>"H/I", "config"=>"neural_ode.toml",
                      "n_rows"=>nrow(bench), "solver"=>"Tsit5 adaptive"))

# ------------------------------------------------------------------ reporting
agg(df, col) = (mean(df[!, col]), median(df[!, col]))
println("\n%-28s %12s %12s %12s %12s %12s" |> x->@printf(x, "model", "MSE_ID",
        "MSE_OOD", "Rsafe_ID", "Rsafe_OOD", "ΔOOD_Rsafe"))
summ = DataFrame()
for (label, _, _) in variants
    b = bench[bench.model .== label, :]
    idb = b[b.split .== "ID", :]; oodb = b[b.split .== "OOD", :]
    row = (model = label,
           mse_id = mean(idb.mse), mse_ood = mean(oodb.mse),
           rsafe_id = mean(idb.R_safe), rsafe_ood = mean(oodb.R_safe),
           delta_ood_rsafe = mean(idb.R_safe) - mean(oodb.R_safe),
           dmax_id = mean(idb.D_max), dmax_ood = mean(oodb.D_max),
           trec_ood_inf_frac = count(!isfinite, oodb.T_rec) / nrow(oodb),
           tau_ood_inf_frac = count(!isfinite, oodb.tau_S) / nrow(oodb),
           final_loss = histdf[histdf.model .== label, :loss][end],
           mean_gnorm = mean(histdf[histdf.model .== label, :gnorm]))
    push!(summ, row)
    @printf("%-28s %12.5f %12.5f %12.4f %12.4f %+12.4f\n", label, row.mse_id,
            row.mse_ood, row.rsafe_id, row.rsafe_ood, row.delta_ood_rsafe)
end
write_table(summ, "neural_summary.csv")

println("\nby OOD family (mean R_safe):")
fams = unique(bench[bench.split .== "OOD", :kind])
@printf("%-28s", "model"); for f in fams; @printf("%12s", first(f, 11)); end; println()
for (label, _, _) in variants
    @printf("%-28s", label)
    for f in fams
        s = bench[(bench.model .== label) .& (bench.kind .== f), :R_safe]
        @printf("%12.4f", isempty(s) ? NaN : mean(s))
    end
    println()
end

open(tabpath("neural_summary.md"), "w") do io
    println(io, "# Neural ODE benchmark (auto-generated)\n")
    println(io, "| model | MSE-ID | MSE-OOD | R_safe ID | R_safe OOD | ΔOOD | final loss | mean ∇ |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for r in eachrow(summ)
        println(io, "| $(r.model) | ", fmtf(r.mse_id,5), " | ", fmtf(r.mse_ood,5), " | ",
                fmtf(r.rsafe_id,4), " | ", fmtf(r.rsafe_ood,4), " | ",
                fmtf(r.delta_ood_rsafe,4), " | ", fmtf(r.final_loss,6), " | ",
                fmtf(r.mean_gnorm,4), " |")
    end
end
println("\ndone.")
