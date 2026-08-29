# ============================================================================
# PHASE H — trainable Neural ODE: TRAINING ONLY.
#   julia --project=. scripts/run_neural_ode.jl
#
# The seen/unseen benchmark lives in scripts/run_neural_benchmark.jl, which is
# the single benchmark entry point. This script used to contain a second,
# obsolete copy of that benchmark (the one that stalled on a sliding mode with
# maxiters = 10^8); it was removed in the V2 hygiene pass. See
# docs/v2/V1_ERRATA.md item A. The trained parameters are serialised, so the
# benchmark can be re-run without retraining.
#
# From here on the term "Neural ODE" is earned: dh/dt = f_θ(h,t) + g(h)δ(t)
# with θ learned by gradient descent.
#
# TRAIN SPLIT WAS FIXED IN configs/neural_ode.toml BEFORE THE FIRST RUN:
#   sinusoids, A∈[0.5,1.5], ω∈[2,6], h0∈[0.05,0.95]. No test condition
#   influenced any training choice.
#
# Output: results/raw/neural_training.csv
#         results/processed/trained_params.jls
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

println("\ntraining complete. Run the benchmark with:")
println("  julia --project=. scripts/run_neural_benchmark.jl")
