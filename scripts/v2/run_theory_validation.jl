# ============================================================================
# V2 PHASE 1 — theory validation.
#   julia --project=. scripts/v2/run_theory_validation.jl
#
# Checks that the implementation obeys Theorems A and B, then computes the two
# geometry statistics the pre-registration rests on:
#   Proposition D : P(l), W(l)  — predicted to be ordering-INVARIANT
#   Proposition E : D(k*)       — predicted to favour Cantor, but only slightly
# and the exact constant-attack containment curve r*(A) (Theorem C), which
# needs no ODE integration and so can be swept densely over log A.
#
# Output: results/v2/raw/theory_{theorems,geometry,displacement,containment}.csv
# ============================================================================

using DataFrames, CSV, Statistics, Printf, Random, TOML, Dates
include(joinpath(@__DIR__, "..", "..", "src", "v2", "CantorBarrier.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "v2", "BarrierDynamics.jl"))
using .CantorBarrier, .BarrierDynamics

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
v2raw(f)  = joinpath(ROOT, "results", "v2", "raw", f)
v2tab(f)  = joinpath(ROOT, "results", "v2", "tables", f)

function provenance_v2(; extra...)
    d = Dict{String,Any}("timestamp" => string(now()), "julia" => string(VERSION),
                         "nthreads" => Threads.nthreads(), "host" => gethostname())
    try
        d["git_commit"] = strip(read(`git -C $ROOT rev-parse --short HEAD`, String))
        d["git_dirty"]  = !isempty(strip(read(`git -C $ROOT status --porcelain`, String)))
    catch; d["git_commit"] = "nogit" end
    for (k, v) in pairs(extra); d[string(k)] = string(v); end
    d
end
function writev2(df, name; meta = Dict())
    p = v2raw(name); mkpath(dirname(p)); CSV.write(p, df)
    open(p * ".meta.toml", "w") do io
        TOML.print(io, merge(Dict{String,Any}(provenance_v2()),
                             Dict{String,Any}(string(k) => v for (k, v) in meta)))
    end
    @info "wrote $(nrow(df)) rows → $p"
end

cfg = TOML.parsefile(joinpath(ROOT, "configs", "v2", "theory.toml"))
ns  = Int.(cfg["grid"]["n"]); B = Float64(cfg["grid"]["B_total"]); η = Float64(cfg["grid"]["eta"])
ells = Float64.(cfg["geometry"]["ell"]); gridN = Int(cfg["geometry"]["grid_N"])
nseed = Int(cfg["replicates"]["layout_seeds"])

const DET  = ["B1_constant", "B2_central", "B3_periodic", "B7_cantor"]
const RAND = ["B4_random", "B5_shuffled", "B6_center_anchored"]

"""All controller instances for a given n: deterministic ones once, random ones ×seeds."""
function instances(n, E0)
    out = Tuple{String,Int,BarrierLayout}[]
    for f in DET;  push!(out, (f, 0, build_layout(f, n, E0))); end
    for f in RAND, s in 1:nseed
        push!(out, (f, s, build_layout(f, n, E0; rng = Xoshiro(hash((f, n, s)) % 10^9))))
    end
    out
end

# ---------------------------------------------------------------- Theorems A/B
println("="^92)
println("THEOREM A (level action = E0) and THEOREM B (peak ratio = 3/2)")
println("EVERY width-matched family satisfies BOTH — they are properties of the")
println("scale-compensation scheme, NOT of the Cantor arrangement.")
println("="^92)
th = DataFrame()
for n in ns
    E0 = B / n
    for (fam, s, L) in instances(n, E0)
        s > 1 && continue                       # one representative per family here
        acts = [level_action(L, k; N = 8000) for k in 1:n]
        pks  = [peak_of_level(L, k) for k in 1:n]
        push!(th, (family = fam, n = n, E0 = E0, seed = s,
                   min_level_action = minimum(acts), max_level_action = maximum(acts),
                   max_rel_dev_from_E0 = maximum(abs.(acts .- E0)) / E0,
                   total_action = sum(acts), total_action_target = B,
                   peak_ratio_min = n ≥ 2 ? minimum(pks[2:end] ./ pks[1:end-1]) : NaN,
                   peak_ratio_max = n ≥ 2 ? maximum(pks[2:end] ./ pks[1:end-1]) : NaN,
                   deltaV = barrier_potential(L, 1.0) - barrier_potential(L, 0.0)))
    end
end
@printf("  max relative deviation of any level action from E0 : %.3e\n",
        maximum(th.max_rel_dev_from_E0))
@printf("  peak ratio across all families and n               : [%.12f, %.12f]\n",
        minimum(skipmissing(th.peak_ratio_min)), maximum(skipmissing(th.peak_ratio_max)))
@printf("  total action vs B_total                            : max |Δ| = %.3e\n",
        maximum(abs.(th.total_action .- B)))
writev2(th, "theory_theorems.csv"; meta = Dict("phase" => "V2-theory"))

# -------------------------------------------------- Proposition D : P(l), W(l)
println("\n" * "="^92)
println("PROPOSITION D — guaranteed coverage P(ℓ) and window energy W(ℓ)")
println("predicted to be essentially ORDERING-INVARIANT")
println("="^92)
geo = DataFrame()
for n in ns, (fam, s, L) in instances(n, B / n), ℓ in ells
    push!(geo, (family = fam, n = n, seed = s, ell = ℓ,
                P = guaranteed_peak(L, ℓ; N = gridN),
                W = min_window_energy(L, ℓ; N = gridN)))
end
writev2(geo, "theory_geometry.csv"; meta = Dict("phase" => "V2-theory"))
for n in (5, 8)
    println("\n  n = $n   P(ℓ):")
    @printf("  %-22s", "family"); for ℓ in ells; @printf("%10.4f", ℓ); end; println()
    for fam in vcat(DET, RAND)
        @printf("  %-22s", fam)
        for ℓ in ells
            v = geo[(geo.n .== n) .& (geo.family .== fam) .& (geo.ell .== ℓ), :P]
            @printf("%10.4f", isempty(v) ? NaN : median(v))
        end
        println()
    end
    c = [only(geo[(geo.n .== n) .& (geo.family .== "B7_cantor") .& (geo.ell .== ℓ), :P]) for ℓ in ells]
    b = [maximum(geo[(geo.n .== n) .& (geo.family .== "B5_shuffled") .& (geo.ell .== ℓ), :P]) for ℓ in ells]
    @printf("  ⇒ best shuffle / cantor = %s  (≈1 ⇒ Proposition D holds)\n",
            join([@sprintf("%.3f", b[i]/c[i]) for i in eachindex(c)], " "))
end

# ------------------------------------------ Proposition E : worst displacement
println("\n" * "="^92)
println("PROPOSITION E — worst-case displacement D(k*)  (lower = more robust)")
println("="^92)
disp = DataFrame()
for n in ns, (fam, s, L) in instances(n, B / n), k in 1:n
    push!(disp, (family = fam, n = n, seed = s, kstar = k,
                 D = worst_displacement(L, k), peak_k = peak_of_level(L, k)))
end
writev2(disp, "theory_displacement.csv"; meta = Dict("phase" => "V2-theory"))
for n in (6, 8)
    println("\n  n = $n")
    @printf("  %4s %10s %10s %10s %12s %12s %12s\n", "k*", "cantor", "periodic",
            "central", "shuf med", "canch med", "shuf worst")
    for k in 1:n
        gv(f, agg) = (v = disp[(disp.n .== n) .& (disp.family .== f) .& (disp.kstar .== k), :D];
                      isempty(v) ? NaN : agg(v))
        @printf("  %4d %10.5f %10.5f %10.5f %12.5f %12.5f %12.5f\n", k,
                gv("B7_cantor", first), gv("B3_periodic", first), gv("B2_central", first),
                gv("B5_shuffled", median), gv("B6_center_anchored", median),
                gv("B5_shuffled", maximum))
    end
end

# ------------------------------------------------- Theorem C : containment r*(A)
println("\n" * "="^92)
println("THEOREM C — exact constant-attack containment r*(A) (no integration)")
println("="^92)
cc = cfg["containment"]
As = 10 .^ range(Float64(cc["logA_min"]), Float64(cc["logA_max"]); length = Int(cc["n_A"]))
FIELDS = Dict("linear" => linear_field(), "cubic" => cubic_field(), "bistable" => bistable_field())
con = DataFrame()
for n in ns, fname in String.(cc["fields"])
    F = FIELDS[fname]
    for (fam, s, L) in instances(n, B / n)
        rs = containment_curve(L, F, As; η = η, r0 = Float64(cc["r0"]), N = 200_000)
        for (i, A) in enumerate(As)
            push!(con, (family = fam, n = n, seed = s, field = fname,
                        A = A, logA = log10(A), r_star = rs[i]))
        end
    end
    # the uncontrolled reference
    for (i, A) in enumerate(As)
        push!(con, (family = "B0_none", n = n, seed = 0, field = fname, A = A,
                    logA = log10(A),
                    r_star = containment_point(no_layout(n, B/n), F, A; η = η,
                                               r0 = Float64(cc["r0"]), N = 200_000)))
    end
end
writev2(con, "theory_containment.csv"; meta = Dict("phase" => "V2-theory",
        "note" => "exact Theorem-C containment; no ODE integration"))

println("\n  bistable field, n = 6 — mean r*(A) over the log-A sweep (lower is better)")
sub = con[(con.n .== 6) .& (con.field .== "bistable"), :]
@printf("  %-22s %12s %12s %12s\n", "family", "mean r*", "AUC_log(1-r*)", "worst r*")
for fam in vcat(["B0_none"], DET, RAND)
    v = sub[sub.family .== fam, :]
    isempty(v) && continue
    m = combine(groupby(v, :A), :r_star => mean => :r)
    sort!(m, :A)
    auc = sum((1 .- m.r) .* [i == 1 ? 0.0 : log10(m.A[i]/m.A[i-1]) for i in 1:nrow(m)])
    @printf("  %-22s %12.5f %12.5f %12.5f\n", fam, mean(m.r), auc, maximum(m.r))
end
println("\ndone.")
