# Run the entire study end to end.  julia --project=. scripts/run_all.jl
# Phases must run in this order: later scripts read earlier scripts' raw output.
const STEPS = [
    ("PHASE B  tests",              `julia --project=. test/runtests.jl`),
    ("PHASE A  baseline + audit",   `julia --project=. scripts/reproduce_baseline.jl`),
    ("PHASE C  numerical analysis", `julia --project=. scripts/run_numerical_analysis.jl`),
    ("PHASE D  parameter sweep",    `julia --project=. -t auto scripts/run_parameter_sweep.jl`),
    ("PHASE F  ablation",           `julia --project=. -t auto scripts/run_ablation.jl`),
    ("PHASE F' ablation stats",     `julia --project=. scripts/run_ablation_stats.jl`),
    ("PHASE G  math verification",  `julia --project=. scripts/run_math_analysis.jl`),
    ("PHASE H  neural ODE training", `julia --project=. scripts/run_neural_ode.jl`),
    ("PHASE I  ID/OOD benchmark",   `julia --project=. scripts/run_neural_benchmark.jl`),
    ("POST-HOC hitting time",       `julia --project=. scripts/run_posthoc_hitting.jl`),
    ("FIGURES",                     `julia --project=. scripts/generate_all_figures.jl`),
]
for (name, cmd) in STEPS
    println("\n" * "="^78); println("▶ ", name); println("="^78)
    t = @elapsed run(cmd)
    println("✓ $name  ($(round(t; digits=1)) s)")
end
println("\nall phases complete.")
