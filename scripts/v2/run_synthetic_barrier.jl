# ============================================================================
# V2 PHASE 2 — synthetic dynamic experiment.
#   julia --project=. -t auto scripts/v2/run_synthetic_barrier.jl
#
# ṙ = f(r) + δ(t) − η V_L'(r) integrated with the displacement-limited RK4 of
# src/v2/BarrierDynamics.jl, over a log-spaced amplitude grid and log-uniform
# frequencies (never a ternary-aligned scale).
#
# The perturbation library is V1's (src/Perturbations.jl): every waveform is
# normalised to sup|δ| = A, so "different waveform" is never confounded with
# "different energy".
#
# Output: results/v2/raw/synthetic_main.csv, synthetic_convergence.csv
# ============================================================================

using DataFrames, CSV, Statistics, Printf, Random, TOML, Dates, Base.Threads
include(joinpath(@__DIR__, "..", "..", "src", "CantorGate.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v2", "CantorBarrier.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v2", "BarrierDynamics.jl"))
using .CantorGate: sinusoid, square_wave, chirp, multifreq, ou_noise, piecewise_random
using .CantorBarrier, .BarrierDynamics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
v2raw(f) = joinpath(ROOT, "results", "v2", "raw", f)
function writev2(df, name; meta = Dict())
    p = v2raw(name); mkpath(dirname(p)); CSV.write(p, df)
    gc = try strip(read(`git -C $ROOT rev-parse --short HEAD`, String)) catch; "nogit" end
    open(p * ".meta.toml", "w") do io
        TOML.print(io, merge(Dict{String,Any}("timestamp" => string(now()),
            "julia" => string(VERSION), "git_commit" => gc,
            "nthreads" => Threads.nthreads()),
            Dict{String,Any}(string(k) => string(v) for (k, v) in meta)))
    end
    @info "wrote $(nrow(df)) rows → $p"
end

cfg = TOML.parsefile(joinpath(ROOT, "configs", "v2", "synthetic.toml"))
S, AT, CT, SV = cfg["system"], cfg["attack"], cfg["controllers"], cfg["solver"]
B, η, T = Float64(S["B_total"]), Float64(S["eta"]), Float64(S["T"])
ns  = Int.(S["n"]); r0s = Float64.(S["r0"])
As  = 10 .^ range(Float64(AT["logA_min"]), Float64(AT["logA_max"]); length = Int(AT["n_A"]))
nsd = Int(CT["layout_seeds"]); nph = Int(AT["phase_seeds"])

FIELDS = Dict("linear" => linear_field(), "cubic" => cubic_field(),
              "bistable" => bistable_field())

"""Build δ(t) with sup|δ| = A and a log-uniform frequency drawn from `rng`."""
function make_attack(fam::String, A::Float64, rng::AbstractRNG)
    ω = 10 ^ (Float64(AT["logw_min"]) +
              (Float64(AT["logw_max"]) - Float64(AT["logw_min"])) * rand(rng))
    fam == "sinusoid"  && return (sinusoid(A, ω; φ = 2π*rand(rng)), ω)
    fam == "square"    && return (square_wave(A, ω; φ = 2π*rand(rng)), ω)
    fam == "chirp"     && return (chirp(A, ω/3, 6ω, T), ω)
    fam == "multifreq" && return (multifreq(A, [ω, 1.7ω, 3.3ω]; φs = 2π .* rand(rng, 3)), ω)
    fam == "ou_noise"  && return (ou_noise(A, T, 1.0, 2.0, 2e-3, Int(abs(rand(rng, Int32)))), ω)
    error("unknown attack family $fam")
end

"""Controller instances: deterministic once, randomised × layout_seeds."""
function instances(n::Int, E0::Float64)
    out = Tuple{String,Int,BarrierLayout}[]
    for f in String.(CT["deterministic"]); push!(out, (f, 0, build_layout(f, n, E0))); end
    for f in String.(CT["randomised"]), s in 1:nsd
        push!(out, (f, s, build_layout(f, n, E0; rng = Xoshiro(hash((f, n, s)) % 10^9))))
    end
    out
end

# ------------------------------------------------------------ enumerate cases
cases = NamedTuple[]
for n in ns
    E0 = B / n
    for (fam, sd, L) in instances(n, E0), fld in String.(S["fields"]),
        A in As, af in String.(AT["families"]), ph in 1:nph, r0 in r0s
        push!(cases, (ctrl = fam, n = n, seed = sd, field = fld, A = A,
                      attack = af, phase = ph, r0 = r0))
    end
end
N = length(cases)
@printf("synthetic: %d simulations on %d threads\n", N, nthreads())

# pre-build layouts once (read-only afterwards)
LAY = Dict{Tuple{String,Int,Int},BarrierLayout}()
for n in ns, (fam, sd, L) in instances(n, B / n); LAY[(fam, n, sd)] = L; end

res = Vector{NamedTuple}(undef, N)
t0 = time()
@threads for i in 1:N
    c = cases[i]
    L = LAY[(c.ctrl, c.n, c.seed)]
    F = FIELDS[c.field]
    rng = Xoshiro(hash((c.attack, c.A, c.phase, c.field)) % 10^9)
    δ, ω = make_attack(c.attack, c.A, rng)
    ts, rs = simulate_barrier(L, F, δ; η = η, r0 = c.r0, T = T, n_for_scale = c.n,
                              step_safety = Float64(SV["step_safety"]),
                              dt_max = Float64(SV["dt_max"]), n_save = Int(SV["n_save"]))
    m = barrier_metrics(ts, rs, L; η = η)
    res[i] = (ctrl = c.ctrl, n = c.n, seed = c.seed, field = c.field, A = c.A,
              logA = log10(c.A), attack = c.attack, omega = ω, phase = c.phase,
              r0 = c.r0, r_max = m.r_max, safe_frac = m.safe_frac, r_end = m.r_end,
              captured = m.captured, crossed = m.crossed, t_cross = m.t_cross,
              mean_r = m.mean_r, ctrl_action = m.ctrl_action)
end
@printf("elapsed %.1f s (%.2f ms/sim)\n", time()-t0, 1000*(time()-t0)/N)
df = DataFrame(res)
writev2(df, "synthetic_main.csv"; meta = Dict("phase" => "V2-synthetic",
        "config" => "synthetic.toml", "n_cases" => N, "T" => T, "eta" => η))

# ------------------------------------------------------------ convergence check
println("\nstep-size convergence check (subset re-run at step_safety = $(SV["check_safety"]))")
sub = cases[1:max(1, N ÷ 400):N]
conv = DataFrame(ctrl=String[], n=Int[], A=Float64[], r_max_a=Float64[],
                 r_max_b=Float64[], abs_diff=Float64[])
for c in sub
    L = LAY[(c.ctrl, c.n, c.seed)]; F = FIELDS[c.field]
    rng = Xoshiro(hash((c.attack, c.A, c.phase, c.field)) % 10^9)
    δ, _ = make_attack(c.attack, c.A, rng)
    ta, ra = simulate_barrier(L, F, δ; η=η, r0=c.r0, T=T, n_for_scale=c.n,
                step_safety=Float64(SV["step_safety"]), dt_max=Float64(SV["dt_max"]),
                n_save=Int(SV["n_save"]))
    tb, rb = simulate_barrier(L, F, δ; η=η, r0=c.r0, T=T, n_for_scale=c.n,
                step_safety=Float64(SV["check_safety"]), dt_max=Float64(SV["dt_max"]),
                n_save=Int(SV["n_save"]))
    push!(conv, (c.ctrl, c.n, c.A, maximum(ra), maximum(rb), abs(maximum(ra)-maximum(rb))))
end
@printf("  %d re-runs: max |Δr_max| = %.3e, median = %.3e, 99th pct = %.3e\n",
        nrow(conv), maximum(conv.abs_diff), median(conv.abs_diff),
        quantile(conv.abs_diff, 0.99))
writev2(conv, "synthetic_convergence.csv"; meta = Dict("phase" => "V2-synthetic"))

# ------------------------------------------------------------------- reporting
println("\n" * "="^100)
println("PRIMARY: worst excursion r_max, pooled over attacks/phases/r0/fields")
println("(lower is better; 0.5 is the decision boundary)")
println("="^100)
@printf("%-22s %9s %9s %9s %9s %9s\n", "controller", "mean", "median",
        "WORST(A)", "P(cross)", "mean act")
for fam in vcat(String.(CT["deterministic"]), String.(CT["randomised"]))
    s = df[df.ctrl .== fam, :]
    isempty(s) && continue
    byA = combine(groupby(s, :A), :r_max => mean => :m)
    @printf("%-22s %9.4f %9.4f %9.4f %9.4f %9.4f\n", fam, mean(s.r_max),
            median(s.r_max), maximum(byA.m), mean(s.crossed), mean(s.ctrl_action))
end
println("\ndone.")
