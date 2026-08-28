# ============================================================================
# PHASE D — full factorial parameter sweep of the Cantor-gated system.
#   julia --project=. -t auto scripts/run_parameter_sweep.jl
#
# 11 (n) × 99 (h0) × 6 (A) × 6 (ω) × 5 (α) = 196 020 trajectories.
# The single h₀ = 0.15 point of the original study is one cell of this grid;
# the point of the sweep is that a single cell cannot support a claim about
# robustness.
#
# Output: results/raw/sweep_full.csv  (+ .meta.toml)
# ============================================================================

using DataFrames, CSV, Statistics, Printf, Base.Threads
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

cfg = load_config("sweep.toml")
G, S = cfg["grid"], cfg["solver"]
ns     = Int.(G["n"])
h0s    = collect(eval(Meta.parse(G["h0"])))
As     = Float64.(G["A"]);      ωs = Float64.(G["omega"]);  αs = Float64.(G["alpha"])
T, dt, se, hold = S["T"], S["dt"], Int(S["save_every"]), S["hold"]

# Pre-build one gate object per order (thread-safe: read-only after construction)
const GATES = Dict{Int,Any}(0 => NoGate())
for n in ns; n > 0 && (GATES[n] = cantor_interval_gate(n)); end

cases = [(n, h0, A, ω, α) for n in ns, h0 in h0s, A in As, ω in ωs, α in αs] |> vec
N = length(cases)
@printf("sweep: %d cases on %d threads\n", N, nthreads())

res = Vector{NamedTuple}(undef, N)
t0 = time()
@threads for i in 1:N
    (n, h0, A, ω, α) = cases[i]
    m = run_case(GATES[n], sinusoid(A, ω); α = α, h0 = h0, T = T,
                 dt = dt, save_every = se, hold = hold)
    res[i] = merge((n = n, h0 = h0, A = A, omega = ω, alpha = α,
                    gate_family = n == 0 ? "G0_none" : "G1_cantor",
                    pass_measure = n == 0 ? 1.0 : (2/3)^n,
                    gate_at_h0 = gate_value(GATES[n], h0),
                    perturbation = "sinusoid", T = T, dt = dt, seed = 0),
                   metrics_tuple(m))
end
@printf("elapsed %.1f s  (%.2f ms/case)\n", time()-t0, 1000*(time()-t0)/N)

df = DataFrame(res)
write_raw(df, "sweep_full.csv"; overwrite = true,
          meta = Dict("phase" => "D", "config" => "sweep.toml",
                      "n_cases" => N, "integrator" => "rk4_fixed_step",
                      "dt" => dt, "save_dt" => dt*se, "T" => T, "hold" => hold))

# ------------------------------------------------------------ quick summaries
println("\nmean R_safe by n (over the WHOLE grid):")
for n in ns
    s = df[df.n .== n, :R_safe]
    @printf("  n=%2d   mean %.4f  median %.4f  IQR [%.4f, %.4f]  min %.4f\n",
            n, mean(s), median(s), quantile(s, .25), quantile(s, .75), minimum(s))
end
println("\nmean R_safe by n at the ORIGINAL operating point (A=1.5, ω=4, α=0.8):")
ref = df[(df.A .== 1.5) .& (df.omega .== 4.0) .& (df.alpha .== 0.8), :]
for n in ns
    s = ref[ref.n .== n, :R_safe]
    @printf("  n=%2d   mean over h0 %.4f   at h0=0.15 %.4f\n", n, mean(s),
            only(ref[(ref.n .== n) .& (isapprox.(ref.h0, 0.15; atol=1e-9)), :R_safe]))
end
