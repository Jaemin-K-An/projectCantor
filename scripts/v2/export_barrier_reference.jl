# Export a Julia reference table of (r, V(r), V'(r)) so the Python port can be
# checked against it (llm/tests/test_barrier_matches_julia.py). CSV rather than
# JSON to avoid adding a dependency to the V1 environment.
using CSV, DataFrames
include(joinpath(@__DIR__, "..", "..", "src", "v2", "CantorBarrier.jl"))
using .CantorBarrier
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
rs = collect(range(0, 1; length = 20_001))
df = DataFrame(family=String[], n=Int[], E0=Float64[], r=Float64[],
               field=Float64[], potential=Float64[])
for (fam, n) in [("B1_constant",5), ("B2_central",5), ("B3_periodic",5),
                 ("B7_cantor",4), ("B7_cantor",5), ("B7_cantor",6), ("B7_cantor",7)]
    E0 = 1.0 / n
    L = build_layout(fam, n, E0)
    for r in rs
        push!(df, (fam, n, E0, r, barrier_field(L, r), barrier_potential(L, r)))
    end
end
out = joinpath(ROOT, "results", "v2", "raw", "barrier_reference.csv")
mkpath(dirname(out)); CSV.write(out, df)
println("wrote $out  ($(nrow(df)) rows)")
