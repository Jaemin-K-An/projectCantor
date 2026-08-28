# ============================================================================
# PHASE A — reproduce the original study, then audit it.
#   julia --project=. scripts/reproduce_baseline.jl
#
# Produces:
#   results/raw/baseline_reproduction.csv   the three original models
#   results/raw/baseline_h0_audit.csv       the same models at other h₀
#   results/raw/baseline_solver_check.csv   RK4 vs Tsit5 vs Vern9, 2 tolerances
#   results/tables/baseline_summary.md
# ============================================================================

using DataFrames, Statistics, Printf, Random
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

cfg  = load_config("baseline.toml")
sys  = cfg["system"];  slv = cfg["solver"];  rep = cfg["reproduce"]
α, A, ω = sys["alpha"], sys["A"], sys["omega"]
T, h0₀  = sys["T"], sys["h0"]
dt, se, hold = slv["dt"], Int(slv["save_every"]), slv["hold"]
δ = sinusoid(A, ω)

gates = Dict("no_filter" => NoGate(),
             "C3" => cantor_interval_gate(3),
             "C5" => cantor_interval_gate(5))
ORDER = ["no_filter", "C3", "C5"]

println("="^78)
println("PHASE A.1–A.2 — reproduction of the original trajectories (h0 = $h0₀)")
println("="^78)

rows = DataFrame()
for k in ORDER
    m = run_case(gates[k], δ; α = α, h0 = h0₀, T = T, dt = dt,
                 save_every = se, hold = hold)
    push!(rows, merge((model = k, h0 = h0₀, alpha = α, A = A, omega = ω, T = T,
                       integrator = "rk4", dt = dt, save_dt = dt * se),
                      metrics_tuple(m)); cols = :union)
end
orig = Dict("no_filter" => rep["original_nofilter"], "C3" => rep["original_c3"],
            "C5" => rep["original_c5"])
rows.original_pct  = [orig[m] for m in rows.model]
rows.reproduced_pct = round.(100 .* rows.R_safe_rect; digits = 1)
rows.delta_pct      = rows.reproduced_pct .- rows.original_pct

@printf("%-10s %12s %12s %8s %10s %10s %10s\n",
        "model", "original %", "reproduced %", "Δ", "τ_S", "R_gate", "D_max")
for r in eachrow(rows)
    @printf("%-10s %12.1f %12.1f %8.1f %10s %10.5f %10.4f\n",
            r.model, r.original_pct, r.reproduced_pct, r.delta_pct,
            fmtf(r.tau_S, 3), r.R_gate, r.D_max)
end
println()
@printf("analytic steady-state occupancy of the ungated system = %.4f (%.1f%%)\n",
        analytic_no_filter_occupancy(; α = α, A = A, ω = ω),
        100 * analytic_no_filter_occupancy(; α = α, A = A, ω = ω))

# ---------------------------------------------------------------- A.3 / A.4
println("\n" * "="^78)
println("PHASE A.3–A.4 — the h0 = 0.15 audit")
println("="^78)
for n in 1:6
    fi = cantor_flat_interval(h0₀, n)
    @printf("  g_%d(0.15) = %.1f   %s\n", n, cantor_gate(h0₀, n),
            fi === nothing ? "0.15 ∈ K_$n (perturbation PASSES)" :
            "0.15 removed at level $(fi.level), flat interval ($(fi.lo), $(fi.hi)) " *
            "= ($(round(Float64(fi.lo);digits=6)), $(round(Float64(fi.hi);digits=6)))")
end
@printf("\n  1/9 = %.6f  <  h0 = %.2f  <  2/9 = %.6f\n", 1/9, h0₀, 2/9)
println("  ⇒ for every n ≥ 2 the ORIGINAL initial condition already sits in a")
println("    flat interval, so g_n(h(0)) = 0 and the perturbation is switched")
println("    OFF at t = 0 in both the C3 and the C5 run.")

# how long does the gate ever open?
for k in ORDER
    ts, hs, gs = simulate_rk4(gates[k], δ; α = α, h0 = h0₀, T = T, dt = dt, save_every = se)
    @printf("  %-10s gate open for %.3f%% of the horizon; first opening at t = %s\n",
            k, 100 * trapz_mean(gs, ts[2]-ts[1], ts[end]),
            (i = findfirst(>(0.5), gs); i === nothing ? "never" : fmtf(ts[i], 3)))
end

# h0 sweep over the audit points
audit = DataFrame()
for h0 in Float64.(rep["audit_h0"]), k in ORDER
    g = gates[k]
    n = k == "no_filter" ? 0 : parse(Int, k[2:end])
    m = run_case(g, δ; α = α, h0 = h0, T = T, dt = dt, save_every = se, hold = hold)
    push!(audit, merge((model = k, n = n, h0 = h0,
                        gate_at_h0 = gate_value(g, h0),
                        in_flat_at_h0 = gate_value(g, h0) == 0.0),
                       metrics_tuple(m)); cols = :union)
end
println("\n  occupancy R_safe as a function of h0 (same α, A, ω):")
@printf("  %-8s %10s %10s %10s %10s\n", "h0", "no_filter", "C3", "C5", "g_3(h0)")
for h0 in Float64.(rep["audit_h0"])
    sub = audit[audit.h0 .== h0, :]
    v(k) = only(sub[sub.model .== k, :R_safe])
    @printf("  %-8.2f %10.4f %10.4f %10.4f %10.0f\n",
            h0, v("no_filter"), v("C3"), v("C5"), cantor_gate(h0, 3))
end

# ------------------------------------------------------------- solver check
println("\n" * "="^78)
println("PHASE A / §23 — solver and tolerance sanity check")
println("="^78)
chk = DataFrame()
for k in ORDER, h0 in (0.15, 0.25, 0.75)
    g = gates[k]
    variants = [
        ("rk4_dt1e-3",  () -> simulate_rk4(g, δ; α=α, h0=h0, T=T, dt=1e-3, save_every=2)),
        ("rk4_dt1e-4",  () -> simulate_rk4(g, δ; α=α, h0=h0, T=T, dt=1e-4, save_every=20)),
        ("tsit5_normal",() -> simulate_adaptive(g, δ; α=α, h0=h0, T=T, dt_save=2e-3,
                              alg=Tsit5(), abstol=slv["adaptive_abstol_normal"],
                              reltol=slv["adaptive_reltol_normal"],
                              dtmax=slv["adaptive_dtmax_normal"])),
        ("vern9_strict",() -> simulate_adaptive(g, δ; α=α, h0=h0, T=T, dt_save=2e-3,
                              alg=Vern9(), abstol=slv["adaptive_abstol_strict"],
                              reltol=slv["adaptive_reltol_strict"],
                              dtmax=slv["adaptive_dtmax_strict"])),
    ]
    for (name, f) in variants
        el = @elapsed ((ts, hs, gs) = f())
        m = compute_metrics(ts, hs, gs; hold = hold)
        push!(chk, merge((model=k, h0=h0, integrator=name, seconds=el),
                         metrics_tuple(m)); cols = :union)
    end
end
println("  max |ΔR_safe| across integrators, per (model, h0):")
for k in ORDER, h0 in (0.15, 0.25, 0.75)
    s = chk[(chk.model .== k) .& (chk.h0 .== h0), :]
    @printf("    %-10s h0=%.2f   R_safe ∈ [%.5f, %.5f]   spread = %.2e\n",
            k, h0, minimum(s.R_safe), maximum(s.R_safe),
            maximum(s.R_safe) - minimum(s.R_safe))
end

meta = Dict("phase" => "A", "config" => "baseline.toml",
            "alpha" => α, "A" => A, "omega" => ω, "T" => T,
            "note" => "reproduction of the original study; no parameter was tuned")
write_raw(rows,  "baseline_reproduction.csv";  overwrite = true, meta = meta)
write_raw(audit, "baseline_h0_audit.csv";      overwrite = true, meta = meta)
write_raw(chk,   "baseline_solver_check.csv";  overwrite = true, meta = meta)

open(tabpath("baseline_summary.md"), "w") do io
    println(io, "# Baseline reproduction (auto-generated by scripts/reproduce_baseline.jl)\n")
    println(io, "| model | original % | reproduced % | Δ | τ_S | R_gate | D_max | D_mean |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for r in eachrow(rows)
        println(io, "| $(r.model) | $(r.original_pct) | $(r.reproduced_pct) | ",
                "$(round(r.delta_pct;digits=1)) | $(fmtf(r.tau_S,3)) | ",
                "$(fmtf(r.R_gate,5)) | $(fmtf(r.D_max,4)) | $(fmtf(r.D_mean,4)) |")
    end
    println(io, "\nAnalytic ungated steady-state occupancy: ",
            round(100*analytic_no_filter_occupancy(;α=α,A=A,ω=ω); digits=2), "%")
end
println("\ndone.")
