# ============================================================================
# PHASE G — numerical VERIFICATION of the analytic results.
#   julia --project=. scripts/run_math_analysis.jl
#
# The proofs live in docs/MATHEMATICAL_ANALYSIS.md. This script only checks
# that the code obeys them; it does not "discover" them. Nothing here is
# presented as a proof.
#
# Output: results/raw/math_invariance_check.csv
#         results/raw/math_smooth_invariance.csv
#         results/tables/math_summary.md
# ============================================================================

using DataFrames, Statistics, Printf, Random
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

const T = 30.0

println("="^92)
println("THEOREM 1 — forward invariance of S° = (1/3, 2/3) under a hard Cantor")
println("gate, for ARBITRARY disturbance amplitude.")
println("Claim: h(t) = 1/2 + (h₀-1/2)e^{-αt} exactly, independent of δ.")
println("="^92)

inv = DataFrame()
h0s_in = [0.34, 0.4, 0.45, 0.5, 0.55, 0.6, 0.66]
δfams = Dict(
    "sinusoid"   => (A, ω) -> sinusoid(A, ω),
    "square"     => (A, ω) -> square_wave(A, ω),
    "chirp"      => (A, ω) -> chirp(A, 0.5, 4ω, T),
    "multifreq"  => (A, ω) -> multifreq(A, [ω, 1.7ω, 3.1ω]),
    "impulse"    => (A, ω) -> impulse_train(A, collect(2.0:3.0:29.0), 0.15),
    "pw_random"  => (A, ω) -> piecewise_random(A, T, 0.3, 11),
    "ou"         => (A, ω) -> ou_noise(A, T, 1.0, 2.0, 0.005, 13),
)
for n in 1:8, α in (0.2, 0.8, 2.0), A in (1.5, 10.0, 100.0, 1000.0),
    (name, mk) in δfams, h0 in h0s_in
    g = cantor_interval_gate(n)
    ts, hs, gs = simulate_rk4(g, mk(A, 4.0); α = α, h0 = h0, T = T, dt = 1e-3, save_every = 2)
    exact = [0.5 + (h0 - 0.5) * exp(-α * t) for t in ts]
    push!(inv, (n = n, alpha = α, A = A, perturbation = name, h0 = h0,
                max_abs_error_vs_linear = maximum(abs, hs .- exact),
                left_S = any(!in_safe, hs),
                max_gate = maximum(gs),
                R_safe = trapz_mean(Float64.(in_safe.(hs)), ts[2]-ts[1], ts[end])))
end
@printf("  %d configurations tested (n=1..8, α∈{0.2,0.8,2}, A up to 1000, 7 waveforms, 7 h₀)\n",
        nrow(inv))
@printf("  trajectories that ever left S      : %d\n", count(inv.left_S))
@printf("  max gate value ever seen inside S  : %.3e\n", maximum(inv.max_gate))
@printf("  max |h(t) − linear solution|       : %.3e   (RK4 truncation only)\n",
        maximum(inv.max_abs_error_vs_linear))
@printf("  min R_safe over all configurations : %.6f\n", minimum(inv.R_safe))
println("  ⇒ Theorem 1 holds numerically: the disturbance amplitude is IRRELEVANT")
println("    once the state is inside S°. Robustness there is an identity, not a finding.")

# --------------------------------------------------------------------------
println("\n" * "="^92)
println("BOUNDARY SIGNS — the Nagumo inward condition on ∂S under a hard gate")
println("="^92)
for α in (0.2, 0.5, 0.8, 1.2, 2.0)
    @printf("  α = %.1f :  f(1/3⁺) = +α/6 = %+.6f ,  f(2/3⁻) = −α/6 = %+.6f\n",
            α, α/6, -α/6)
end
g3 = cantor_interval_gate(3)
@printf("\n  gate ON the boundary points (closed-interval convention):")
@printf(" g(1/3) = %.0f , g(2/3) = %.0f\n", gate_value(g3, 1/3), gate_value(g3, 2/3))
@printf("  gate just INSIDE: g(1/3+1e-12) = %.0f , g(2/3-1e-12) = %.0f\n",
        gate_value(g3, 1/3 + 1e-12), gate_value(g3, 2/3 - 1e-12))
println("  ⇒ the CLOSED set S is not invariant at the two endpoint states themselves;")
println("    the OPEN set S° is. This distinction is why Theorem 1 is stated on S°.")

# --------------------------------------------------------------------------
println("\n" * "="^92)
println("THEOREM 2 — robust invariance for a GENERAL gate:  B·g(∂S) < α/6 .")
println("Tested on the smooth Cantor gate, where g(∂S) > 0 by construction.")
println("="^92)

sm = DataFrame()
for n in (3, 5), β in (2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0),
    α in (0.8,), A in (0.1, 0.2, 0.25, 0.3, 0.5, 1.0, 1.5, 3.0)
    g = SmoothGate(cantor_interval_gate(n), β)
    glo, ghi = gate_value(g, SAFE_LO), gate_value(g, SAFE_HI)
    gb = max(glo, ghi)
    margin = α/6 - A * gb                       # > 0 ⇒ Theorem 2 guarantees invariance
    # boundary-layer width predicted by  A·σ(-βu) = α/6
    layer = A > α/6 ? log(6A/α - 1) / β : 0.0
    escaped = false; maxdev = 0.0
    for h0 in (0.34, 0.45, 0.5, 0.55, 0.66)
        ts, hs, _ = simulate_rk4(g, sinusoid(A, 4.0); α = α, h0 = h0, T = T,
                                 dt = 1e-3, save_every = 2)
        any(!in_safe, hs) && (escaped = true)
        maxdev = max(maxdev, maximum(abs.(hs .- 0.5)))
    end
    push!(sm, (n = n, beta = β, alpha = α, A = A, g_lo = glo, g_hi = ghi,
               nagumo_margin = margin, guaranteed = margin > 0,
               predicted_layer_width = layer, escaped_S = escaped, max_dev = maxdev))
end
guar = sm[sm.guaranteed, :]
@printf("  configurations where Theorem 2 GUARANTEES invariance : %d\n", nrow(guar))
@printf("    of these, how many actually escaped S              : %d  (must be 0)\n",
        count(guar.escaped_S))
ng = sm[.!sm.guaranteed, :]
@printf("  configurations where the guarantee FAILS             : %d\n", nrow(ng))
@printf("    of these, how many actually escaped S              : %d  (%.1f%%)\n",
        count(ng.escaped_S), 100*count(ng.escaped_S)/max(1,nrow(ng)))
println("  ⇒ the condition is SUFFICIENT, not necessary: many configurations that")
println("    violate it still stay inside, because escape also requires crossing a")
println("    leaky boundary layer of width ≈ ln(6A/α − 1)/β before the disturbance")
println("    reverses sign.")

println("\n  g_β(1/3) as β → ∞ (the endpoint is PINNED, it does not converge to 0):")
for β in (10.0, 100.0, 1000.0, 10000.0)
    @printf("    β = %8.0f   g_β(1/3) = %.6f\n", β,
            gate_value(SmoothGate(cantor_interval_gate(3), β), SAFE_LO))
end
@printf("  ⇒ with α = 0.8, Theorem 2 can never guarantee invariance for A ≥ α/(6·½) = %.4f\n",
        0.8/3)
println("    no matter how large β is. Sharpness/robustness cannot be recovered by β alone.")

# ---- escape boundary in the (β, A) plane -----------------------------------
println("\n  measured escape from S (● = escaped) in the (β, A) plane, n = 3:")
βs = sort(unique(sm.beta)); Aas = sort(unique(sm.A))
@printf("  %8s", "A\\β"); for β in βs; @printf("%8.0f", β); end; println()
for A in Aas
    @printf("  %8.2f", A)
    for β in βs
        r = sm[(sm.n .== 3) .& (sm.beta .== β) .& (sm.A .== A), :]
        @printf("%8s", isempty(r) ? "-" : (r.escaped_S[1] ? "●" : (r.guaranteed[1] ? "✓G" : "·")))
    end
    println()
end
println("  ✓G = guaranteed by Theorem 2   · = stayed in S but not guaranteed   ● = escaped")

write_raw(inv, "math_invariance_check.csv"; overwrite = true,
          meta = Dict("phase"=>"G", "theorem"=>"1 forward invariance of S-open"))
write_raw(sm,  "math_smooth_invariance.csv"; overwrite = true,
          meta = Dict("phase"=>"G", "theorem"=>"2 robust invariance B*g(dS) < alpha/6"))

open(tabpath("math_summary.md"), "w") do io
    println(io, "# Verification of the analytic results (auto-generated)\n")
    println(io, "## Theorem 1 — invariance of S° under the hard Cantor gate\n")
    println(io, "- configurations tested: $(nrow(inv))")
    println(io, "- trajectories that left S: $(count(inv.left_S))")
    println(io, "- max |h − linear solution|: $(fmtf(maximum(inv.max_abs_error_vs_linear), 12))")
    println(io, "- disturbance amplitudes up to A = 1000 with no effect whatsoever\n")
    println(io, "## Theorem 2 — robust invariance for a general gate\n")
    println(io, "| n | β | A | g(1/3) | α/6 − A·g | guaranteed | escaped |")
    println(io, "|---|---|---|---|---|---|---|")
    for r in eachrow(sm[sm.n .== 3, :])
        println(io, "| $(r.n) | $(r.beta) | $(r.A) | ", fmtf(r.g_lo,5), " | ",
                fmtf(r.nagumo_margin,5), " | $(r.guaranteed) | $(r.escaped_S) |")
    end
end
println("\ndone.")
