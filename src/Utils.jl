# ============================================================================
# Utils.jl — configuration, provenance and result I/O.
#
# REPRODUCIBILITY RULES ENFORCED HERE
#  * every experiment reads its parameters from a TOML file in configs/
#  * every raw result table carries a provenance block (git commit, Julia
#    version, package versions, timestamp, config hash, solver settings)
#  * `write_raw` REFUSES to overwrite an existing raw table unless
#    `overwrite=true` is passed explicitly, so a code change cannot silently
#    replace results that a report already cites
# ============================================================================

using TOML, CSV, DataFrames, Dates, Random, Printf

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
cfgpath(name)   = joinpath(PROJECT_ROOT, "configs", name)
rawpath(name)   = joinpath(PROJECT_ROOT, "results", "raw", name)
procpath(name)  = joinpath(PROJECT_ROOT, "results", "processed", name)
tabpath(name)   = joinpath(PROJECT_ROOT, "results", "tables", name)
figpath(name)   = joinpath(PROJECT_ROOT, "figures", name)
logpath(name)   = joinpath(PROJECT_ROOT, "logs", name)

"""    load_config(name) -> Dict — read `configs/<name>`."""
load_config(name::AbstractString) = TOML.parsefile(cfgpath(name))

"""
    git_commit() -> String

Short commit hash of the working tree, or `"nogit"` / `"dirty:<hash>"`. Recorded
in every result file so a table can always be traced back to the code that made it.
"""
function git_commit()
    try
        h = strip(read(`git -C $PROJECT_ROOT rev-parse --short HEAD`, String))
        dirty = !isempty(strip(read(`git -C $PROJECT_ROOT status --porcelain`, String)))
        return dirty ? "dirty:$h" : h
    catch
        return "nogit"
    end
end

"""
    provenance(; extra...) -> Dict

The metadata block attached to every raw result table.
"""
function provenance(; extra...)
    d = Dict{String,Any}(
        "timestamp"      => string(now()),
        "git_commit"     => git_commit(),
        "julia_version"  => string(VERSION),
        "nthreads"       => Threads.nthreads(),
        "hostname"       => gethostname(),
    )
    for (k, v) in pairs(extra)
        d[string(k)] = v isa Union{Number,AbstractString,Bool} ? v : string(v)
    end
    return d
end

"""
    write_raw(df, name; overwrite=false, meta=Dict())

Write a raw result table to `results/raw/<name>` plus a sidecar
`<name>.meta.toml`. Errors if the file exists and `overwrite` is not set —
raw results are append-only by policy.
"""
function write_raw(df::DataFrame, name::AbstractString;
                   overwrite::Bool = false, meta::AbstractDict = Dict())
    p = rawpath(name)
    if isfile(p) && !overwrite
        error("raw result $p already exists; pass overwrite=true deliberately " *
              "or write to a new versioned filename")
    end
    mkpath(dirname(p))
    CSV.write(p, df)
    open(p * ".meta.toml", "w") do io
        TOML.print(io, merge(Dict{String,Any}(provenance()),
                             Dict{String,Any}(string(k) => v for (k, v) in meta)))
    end
    @info "wrote $(nrow(df))×$(ncol(df)) rows → $p"
    return p
end

"""    write_table(df, name) — a derived/processed table; overwriting is allowed."""
function write_table(df::DataFrame, name::AbstractString)
    p = tabpath(name); mkpath(dirname(p)); CSV.write(p, df)
    @info "wrote table → $p"; return p
end

"""
    seed_for(parts...) -> Int

Deterministic per-experiment seed derived from a stable hash of its identifying
parts. Avoids the "one global seed" anti-pattern: re-running a single condition
reproduces exactly the same random layout regardless of iteration order.
"""
seed_for(parts...) = Int(hash(parts) % 2_000_000_000) + 1

"""
    phase_checkpoint(phase, io_lines) — append a phase report to logs/PROGRESS.md.
"""
function phase_checkpoint(phase::AbstractString, lines::AbstractVector{<:AbstractString})
    p = logpath("PROGRESS.md"); mkpath(dirname(p))
    open(p, "a") do io
        println(io, "\n## [$phase]  $(now())\n")
        for l in lines; println(io, l); end
    end
    return p
end

"""    fmtf(x, d=4) — fixed-width float formatting for markdown tables."""
fmtf(x, d::Int = 4) = isfinite(x) ? string(round(x; digits = d)) : (x > 0 ? "Inf" : "-Inf")
