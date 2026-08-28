# ============================================================================
# PHASE F — ablation: is the CANTOR arrangement special, or only its measure?
#   julia --project=. -t auto scripts/run_ablation.jl
#
# All of G1..G5 have pass measure EXACTLY (2/3)^n, so a difference between them
# cannot be attributed to "the Cantor gate simply blocks more of the state
# space". G0 (no filter) is the unshielded reference.
#
#   G1 cantor    — self-similar hierarchy of gaps
#   G2 random    — same 2^n intervals of width 3^-n, uniformly random packing
#   G3 periodic  — same intervals on a regular lattice (single length scale)
#   G4 central   — all blocked measure in ONE interval centred on h = 1/2
#   G5 shuffled  — EXACT Cantor gap-width multiset, order randomly permuted
#                  (matches count, widths, gaps; differs only in arrangement)
#
# G2 and G5 are replicated over `random_seeds` layouts per condition, so the
# Cantor result is compared against a DISTRIBUTION, never a single draw.
#
# Output: results/raw/ablation_main.csv, results/raw/ablation_smooth.csv,
#         results/raw/ablation_outside_convention.csv
# ============================================================================

using DataFrames, Statistics, Printf, Random, Base.Threads
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

cfg = load_config("ablation.toml")
GA, CO, SM, S = cfg["gates"], cfg["conditions"], cfg["smooth"], cfg["solver"]
families = String.(GA["families"])
ns       = Int.(GA["n"])
nseeds   = Int(GA["random_seeds"])
h0s      = collect(eval(Meta.parse(CO["h0"])))
As       = Float64.(CO["A"]); ωs = Float64.(CO["omega"]); αs = Float64.(CO["alpha"])
T, dt, se, hold = S["T"], S["dt"], Int(S["save_every"]), S["hold"]

const RANDOMISED = Set(["G2_random", "G5_shuffled"])

# Cache the exact (BigInt-rational) Cantor gap-width multiset once per order:
# rebuilding it inside the threaded loop would dominate the runtime and the
# values are read-only afterwards.
const GAPS  = Dict(n => Float64.(cantor_gap_widths(n)) for n in ns)
const FIXED = Dict((f, n) => build_gate(f, n)
                   for f in ("G1_cantor", "G3_periodic", "G4_central"), n in ns)

"""One (family, n, seed) → a gate object. Deterministic in the seed."""
function gate_for(fam, n, seed)
    fam == "G0_none" && return NoGate()
    fam in RANDOMISED && return build_gate(fam, n; rng = Xoshiro(seed), gap_cache = GAPS[n])
    return FIXED[(fam, n)]
end

# ---- enumerate cases (fixed before execution) ------------------------------
cases = NamedTuple[]
for A in As, ω in ωs, α in αs, h0 in h0s
    push!(cases, (fam = "G0_none", n = 0, seed = 0, h0 = h0, A = A, ω = ω, α = α))
    for n in ns, fam in families
        fam == "G0_none" && continue
        if fam in RANDOMISED
            for s in 1:nseeds
                push!(cases, (fam = fam, n = n, seed = seed_for(fam, n, h0, A, ω, α, s),
                              h0 = h0, A = A, ω = ω, α = α))
            end
        else
            push!(cases, (fam = fam, n = n, seed = 0, h0 = h0, A = A, ω = ω, α = α))
        end
    end
end
N = length(cases)
@printf("ablation: %d cases on %d threads\n", N, nthreads())

res = Vector{NamedTuple}(undef, N)
t0 = time()
@threads for i in 1:N
    c = cases[i]
    g = gate_for(c.fam, c.n, c.seed)
    m = run_case(g, sinusoid(c.A, c.ω); α = c.α, h0 = c.h0, T = T,
                 dt = dt, save_every = se, hold = hold)
    res[i] = merge((gate_family = c.fam, n = c.n, seed = c.seed, h0 = c.h0,
                    A = c.A, omega = c.ω, alpha = c.α,
                    pass_measure = c.n == 0 ? 1.0 : (2/3)^c.n,
                    gate_at_h0 = gate_value(g, c.h0),
                    perturbation = "sinusoid", T = T),
                   metrics_tuple(m))
end
@printf("elapsed %.1f s (%.2f ms/case)\n", time()-t0, 1000*(time()-t0)/N)
df = DataFrame(res)
write_raw(df, "ablation_main.csv"; overwrite = true,
          meta = Dict("phase"=>"F", "config"=>"ablation.toml", "n_cases"=>N,
                      "random_seeds"=>nseeds, "integrator"=>"rk4_fixed_step"))

# ---- G6: hard vs smooth ----------------------------------------------------
sm_cases = NamedTuple[]
for n in Int.(SM["n"]), β in Float64.(SM["beta"]), A in As, ω in ωs, α in αs, h0 in h0s
    push!(sm_cases, (n = n, β = β, h0 = h0, A = A, ω = ω, α = α))
end
sres = Vector{NamedTuple}(undef, length(sm_cases))
@threads for i in eachindex(sm_cases)
    c = sm_cases[i]
    g = SmoothGate(cantor_interval_gate(c.n), c.β)
    m = run_case(g, sinusoid(c.A, c.ω); α = c.α, h0 = c.h0, T = T,
                 dt = dt, save_every = se, hold = hold)
    sres[i] = merge((gate_family = "G6_smooth_cantor", n = c.n, beta = c.β,
                     h0 = c.h0, A = c.A, omega = c.ω, alpha = c.α,
                     gate_at_h0 = gate_value(g, c.h0),
                     gate_at_safe_lo = gate_value(g, SAFE_LO),
                     gate_at_safe_hi = gate_value(g, SAFE_HI),
                     gate_at_half = gate_value(g, 0.5),
                     # robust-invariance test from docs/MATHEMATICAL_ANALYSIS.md
                     invariance_margin = c.α/6 - c.A*max(gate_value(g, SAFE_LO),
                                                         gate_value(g, SAFE_HI))),
                    metrics_tuple(m))
end
sdf = DataFrame(sres)
write_raw(sdf, "ablation_smooth.csv"; overwrite = true,
          meta = Dict("phase"=>"F", "config"=>"ablation.toml",
                      "note"=>"G6 smooth Cantor gate, β sweep"))

# ---- reporting -------------------------------------------------------------
println("\n" * "="^90)
println("Mean R_safe by family and n, pooled over 25 h0 × 3 A × 3 ω")
println("="^90)
@printf("%-14s", "family")
for n in ns; @printf("%9s", "n=$n"); end; println("   pass measure at n=3")
for fam in families
    fam == "G0_none" && continue
    @printf("%-14s", fam)
    for n in ns
        s = df[(df.gate_family .== fam) .& (df.n .== n), :R_safe]
        @printf("%9.4f", mean(s))
    end
    println()
end
g0 = mean(df[df.gate_family .== "G0_none", :R_safe])
@printf("%-14s%9.4f  (n-independent reference)\n", "G0_none", g0)

println("\n" * "="^90)
println("CANTOR vs the two randomised measure-matched controls")
println("(ΔR = R_cantor − mean over seeds;  pct = percentile rank of Cantor in")
println(" the seed distribution; >50 means Cantor beats the median layout)")
println("="^90)
@printf("%4s %14s %10s %10s %10s %10s %10s\n",
        "n", "control", "R_cantor", "mean_ctrl", "ΔR", "pct rank", "frac Δ>0")
for n in ns, ctrl in ("G2_random", "G5_shuffled")
    dR = Float64[]; pr = Float64[]
    for A in As, ω in ωs, α in αs, h0 in h0s
        rc = df[(df.gate_family .== "G1_cantor") .& (df.n .== n) .& (df.h0 .== h0) .&
                (df.A .== A) .& (df.omega .== ω) .& (df.alpha .== α), :R_safe]
        rr = df[(df.gate_family .== ctrl) .& (df.n .== n) .& (df.h0 .== h0) .&
                (df.A .== A) .& (df.omega .== ω) .& (df.alpha .== α), :R_safe]
        (isempty(rc) || isempty(rr)) && continue
        push!(dR, rc[1] - mean(rr))
        push!(pr, 100 * count(<(rc[1]), rr) / length(rr))
    end
    rcm = mean(df[(df.gate_family .== "G1_cantor") .& (df.n .== n), :R_safe])
    rrm = mean(df[(df.gate_family .== ctrl) .& (df.n .== n), :R_safe])
    @printf("%4d %14s %10.4f %10.4f %+10.4f %10.1f %10.3f\n",
            n, ctrl, rcm, rrm, mean(dR), mean(pr), count(>(0), dR)/length(dR))
end
println("\ndone.")
