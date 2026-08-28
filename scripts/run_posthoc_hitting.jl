# ============================================================================
# POST-HOC EXPERIMENT (§16) — the reaching problem.
#
#   *** THIS IS A POST-HOC ANALYSIS. ***
# It was designed AFTER seeing that R_safe is non-monotone in n (PHASE D) and
# that G4_central beats G1_cantor at large n (PHASE F). It is reported
# separately from the pre-registered analyses and no pre-registered conclusion
# depends on it. Its purpose is to EXPLAIN an observation, not to establish one.
#
# It reuses results/raw/sweep_full.csv; no new sweep grid is introduced.
#   julia --project=. scripts/run_posthoc_hitting.jl
#
# Output: results/tables/posthoc_hitting.csv, results/tables/posthoc_hitting.md
# ============================================================================

using DataFrames, CSV, Statistics, Printf
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

"""τ_relax — Proposition 3: hitting time when the gate is CLOSED all the way to ∂S."""
tau_relax(h0, α) = abs(h0 - 0.5) ≤ 1/6 ? 0.0 : log(6 * abs(h0 - 0.5)) / α

"""τ_min — Proposition 4: comparison-theorem lower bound when the gate is OPEN
and the disturbance is maximally helpful, |δ| ≤ B."""
function tau_min(h0, α, B)
    d0 = abs(h0 - 0.5)
    d0 ≤ 1/6 && return 0.0
    return log((α*d0 + B) / (α/6 + B)) / α
end

sw = CSV.read(rawpath("sweep_full.csv"), DataFrame)
ref = sw[(sw.A .== 1.5) .& (sw.omega .== 4.0) .& (sw.alpha .== 0.8), :]
α, B = 0.8, 1.5

out = DataFrame()
for r in eachrow(ref)
    abs(r.h0 - 0.5) ≤ 1/6 && continue
    push!(out, (n = r.n, h0 = r.h0, tau_S = r.tau_S,
                tau_relax = tau_relax(r.h0, α), tau_min = tau_min(r.h0, α, B),
                faster_than_relax = isfinite(r.tau_S) && r.tau_S < tau_relax(r.h0, α),
                bound_violated = isfinite(r.tau_S) && r.tau_S < tau_min(r.h0, α, B) - 1e-6,
                R_safe = r.R_safe))
end
write_table(out, "posthoc_hitting.csv")

println("="^92)
println("POST-HOC — does the gate help or hurt REACHING the safe set?")
println("Proposition 3: τ_relax = ln(6|h₀−½|)/α   (gate fully CLOSED outside S)")
println("Proposition 4: τ_min   = ln((α|h₀−½|+B)/(α/6+B))/α  (lower bound, gate OPEN)")
println("="^92)
@printf("  lower bound τ_min violated in %d of %d cases (must be 0)\n",
        count(out.bound_violated), nrow(out))
println()
@printf("%4s %10s %14s %14s %10s\n", "n", "median τ_S", "median τ_relax",
        "frac τ_S<τ_relax", "frac τ_S=∞")
for n in sort(unique(out.n))
    s = out[out.n .== n, :]
    fin = filter(isfinite, s.tau_S)
    @printf("%4d %10.4f %14.4f %14.3f %10.3f\n", n,
            isempty(fin) ? NaN : median(fin), median(s.tau_relax),
            mean(s.faster_than_relax), mean(.!isfinite.(s.tau_S)))
end
println("\n  ⇒ for n = 0 (no gate) the disturbance is always available to accelerate")
println("    entry; as n grows the flat set outside S grows and more trajectories")
println("    are locked onto the slow pure-relaxation rate τ_relax.")

println("\nrepresentative points at the ORIGINAL operating point (α=0.8, A=1.5, ω=4):")
@printf("%8s %10s %10s", "h0", "τ_relax", "τ_min")
for n in (0, 1, 3, 5, 8); @printf("%10s", "τ_S(n=$n)"); end; println()
for h0 in (0.05, 0.10, 0.15, 0.25, 0.75, 0.85, 0.90, 0.95)
    @printf("%8.2f %10.4f %10.4f", h0, tau_relax(h0, α), tau_min(h0, α, B))
    for n in (0, 1, 3, 5, 8)
        v = ref[(ref.n .== n) .& (isapprox.(ref.h0, h0; atol = 1e-9)), :tau_S]
        @printf("%10s", isempty(v) ? "-" : fmtf(v[1], 4))
    end
    println()
end

open(tabpath("posthoc_hitting.md"), "w") do io
    println(io, "# POST-HOC: the reaching problem (auto-generated)\n")
    println(io, "**This analysis is post-hoc** — see the header of ",
                "`scripts/run_posthoc_hitting.jl`.\n")
    println(io, "α = 0.8, A = 1.5, ω = 4.\n")
    println(io, "| h₀ | τ_relax (Prop 3) | τ_min (Prop 4) | τ_S n=0 | n=1 | n=3 | n=5 | n=8 |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for h0 in (0.05, 0.10, 0.15, 0.25, 0.75, 0.85, 0.90, 0.95)
        print(io, "| $h0 | ", fmtf(tau_relax(h0,α),4), " | ", fmtf(tau_min(h0,α,B),4), " |")
        for n in (0, 1, 3, 5, 8)
            v = ref[(ref.n .== n) .& (isapprox.(ref.h0, h0; atol=1e-9)), :tau_S]
            print(io, " ", isempty(v) ? "-" : fmtf(v[1],4), " |")
        end
        println(io)
    end
    println(io, "\nLower bound violations: $(count(out.bound_violated)) / $(nrow(out)).")
end
println("\ndone.")
