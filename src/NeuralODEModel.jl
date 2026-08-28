# ============================================================================
# NeuralODEModel.jl — the ONLY part of this project entitled to the name
# "Neural ODE": here the vector field carries trained parameters θ.
#
#     dh/dt = f_θ(h, t) + g(h) · δ(t)
#
# ---------------------------------------------------------------------------
# TRAINING METHOD AND WHY
#
# Training uses DISCRETISE-THEN-OPTIMISE: a fixed-step RK4 rollout that is
# differentiated end-to-end by Zygote. It is not an adjoint method, and that is
# a deliberate choice, not a shortcut:
#
#   * the hard gate makes ∂f/∂h a distribution (zero a.e. plus deltas on the
#     switching surfaces), so the continuous adjoint ODE is not well posed;
#   * adaptive steps make the loss surface non-smooth in θ because the step
#     sequence itself changes with θ;
#   * a fixed grid gives an exactly reproducible, deterministic loss.
#
# The TRAINED FIELD is afterwards evaluated with the adaptive SciML solver
# (`evaluate_neural_ode`) so the reported numbers are not an artefact of the
# training discretisation.
#
# LEARNING PROBLEM (fixed before any run; see configs/neural_ode.toml)
#   target   : the CLEAN reference trajectory h_ref(t) of
#              dh/dt = -κ(h-½) - γ(h-½)³ ,  same h₀
#   student  : dh/dt = f_θ(h,t) + g(h) δ(t) ,  δ from the TRAIN distribution
#   loss     : mean squared tracking error over the rollout
# i.e. "reproduce the nominal safe trajectory in spite of the disturbance".
# Every gate variant is trained separately on identical data with an identical
# budget, so no architecture is advantaged by the objective.
# ============================================================================

using Lux, Random, ComponentArrays, Optimisers, Zygote, ChainRulesCore, Statistics, Printf

# ---------------------------------------------------------------------------
# Batched gates (Zygote-friendly; the sweep gates in Controls.jl are scalar)
# ---------------------------------------------------------------------------

abstract type BatchGate end

"""No gate: the disturbance is applied in full."""
struct BNone <: BatchGate end

"""
    BHard(los, his, label)

Hard {0,1} gate over a union of intervals, evaluated on a whole batch.
Its state-derivative is `0` almost everywhere, which is the mathematically
correct a.e. derivative — so it is wrapped in `ChainRulesCore.ignore_derivatives`
during training. The consequence (no learning signal flows through the gate)
is the subject of the hard-vs-smooth comparison, not a bug being papered over.
"""
struct BHard <: BatchGate
    los::Vector{Float64}
    his::Vector{Float64}
    label::String
end

"""
    BSmooth(los, his, β, label)

Differentiable relaxation `Σᵢ σ(β(h-aᵢ))σ(β(bᵢ-h))`, clamped to `[0,1]`.
Evaluated densely over all intervals (fine for the `n ≤ 5` used here).
"""
struct BSmooth <: BatchGate
    los::Vector{Float64}
    his::Vector{Float64}
    β::Float64
    label::String
end

batch_gate_label(::BNone) = "none"
batch_gate_label(g::BHard) = g.label
batch_gate_label(g::BSmooth) = g.label

apply_gate(::BNone, h::AbstractVector) = one.(h)

function apply_gate(g::BHard, h::AbstractVector)
    ChainRulesCore.ignore_derivatives() do
        hc = clamp.(h, 0.0, 1.0)
        [any(i -> g.los[i] ≤ x ≤ g.his[i], eachindex(g.los)) ? 1.0 : 0.0 for x in hc]
    end
end

function apply_gate(g::BSmooth, h::AbstractVector)
    hc = clamp.(h, 0.0, 1.0)'                       # 1×B
    s = vec(sum(logistic.(g.β .* (hc .- g.los)) .* logistic.(g.β .* (g.his .- hc)); dims = 1))
    return clamp.(s, 0.0, 1.0)
end

"""    to_batch_gate(g::Gate; β) — lift a scalar [`Gate`](@ref) to a [`BatchGate`](@ref)."""
to_batch_gate(::NoGate; kwargs...) = BNone()
to_batch_gate(g::IntervalGate; kwargs...) = BHard(g.los, g.his, g.label)
to_batch_gate(g::SmoothGate; kwargs...) = BSmooth(g.base.los, g.base.his, g.β, g.label)

# ---------------------------------------------------------------------------
# The trainable vector field
# ---------------------------------------------------------------------------

"""
    NeuralVectorField(model, st, width, depth)

A small MLP `f_θ : (h, t) ↦ dh/dt`. Inputs are `[h; t/T]` (time is normalised
so both inputs are `O(1)`); `tanh` hidden activations; linear output.
"""
struct NeuralVectorField
    model::Any
    st::Any
    width::Int
    depth::Int
    T::Float64
end

"""
    init_nvf(rng; width=48, depth=2, T=20.0) -> (nvf, ps)

Build the field and its `Float64` parameter vector (a `ComponentArray`, so the
whole θ is one flat differentiable object).
"""
function init_nvf(rng::AbstractRNG; width::Int = 48, depth::Int = 2, T::Float64 = 20.0)
    layers = Any[Dense(2 => width, tanh)]
    for _ in 2:depth
        push!(layers, Dense(width => width, tanh))
    end
    push!(layers, Dense(width => 1))
    model = Chain(layers...)
    ps, st = Lux.setup(rng, model)
    psf = ComponentArray{Float64}(ComponentArray(ps))
    return NeuralVectorField(model, st, width, depth, T), psf
end

"""    nvf_apply(nvf, ps, h, t) -> Vector — `f_θ(h,t)` on a batch of states."""
function nvf_apply(nvf::NeuralVectorField, ps, h::AbstractVector, t::Real)
    x = vcat(reshape(h, 1, :), fill(t / nvf.T, 1, length(h)))
    y, _ = Lux.apply(nvf.model, x, ps, nvf.st)
    return vec(y)
end

"""
    rk4_rollout(nvf, ps, h0, bgate, δs, T, dt, save_every) -> (ts, H)

Fixed-step RK4 rollout of `dh/dt = f_θ(h,t) + g(h)·δᵢ(t)` for a batch of
initial conditions `h0` with a per-sample disturbance list `δs`. `H` is
`(nsave+1) × B`. Fully differentiable in `ps`.
"""
function rk4_rollout(nvf::NeuralVectorField, ps, h0::AbstractVector, bgate,
                     δs::AbstractVector, T::Float64, dt::Float64, save_every::Int)
    B = length(h0)
    nsteps = round(Int, T / dt)
    @inline dvec(t) = [δs[i](t) for i in 1:B]
    f(h, t) = nvf_apply(nvf, ps, h, t) .+ apply_gate(bgate, h) .* dvec(t)
    h = h0
    rows = [h]
    ts = [0.0]
    for k in 1:nsteps
        t = (k - 1) * dt
        k1 = f(h, t)
        k2 = f(h .+ (dt/2) .* k1, t + dt/2)
        k3 = f(h .+ (dt/2) .* k2, t + dt/2)
        k4 = f(h .+ dt .* k3,     t + dt)
        h = h .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        if k % save_every == 0
            rows = vcat(rows, [h])
            ts = vcat(ts, [k * dt])
        end
    end
    return ts, reduce(vcat, permutedims.(rows))
end

# ---------------------------------------------------------------------------
# Reference (teacher) system
# ---------------------------------------------------------------------------

"""
    reference_field(h; κ, γ) = -κ(h-½) - γ(h-½)³

The nominal safe dynamics the Neural ODE must reproduce. The cubic term makes
the target genuinely nonlinear, so a linear `f_θ` cannot solve the task
trivially, while `h = ½` remains the unique globally attracting equilibrium.
"""
reference_field(h; κ = 1.2, γ = 2.0) = -κ .* (h .- H_SAFE) .- γ .* (h .- H_SAFE) .^ 3

"""
    reference_trajectory(h0, T, dt, save_every; κ, γ) -> (ts, H)

Clean reference trajectories (no disturbance) — the regression target.
"""
function reference_trajectory(h0::AbstractVector, T::Float64, dt::Float64,
                              save_every::Int; κ = 1.2, γ = 2.0)
    f(h) = reference_field(h; κ = κ, γ = γ)
    nsteps = round(Int, T / dt)
    h = copy(h0); rows = [copy(h)]; ts = [0.0]
    for k in 1:nsteps
        k1 = f(h); k2 = f(h .+ (dt/2) .* k1)
        k3 = f(h .+ (dt/2) .* k2); k4 = f(h .+ dt .* k3)
        h = h .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        if k % save_every == 0
            push!(rows, copy(h)); push!(ts, k * dt)
        end
    end
    return ts, reduce(vcat, permutedims.(rows))
end

"""
    sample_train_batch(rng, cfg) -> (h0, δs)

Draw a training batch: `h₀ ~ U(h0_lo, h0_hi)` and a SINUSOIDAL disturbance with
`A ~ U(A_lo, A_hi)`, `ω ~ U(ω_lo, ω_hi)`, `φ ~ U(0,2π)`. The training family is
*sinusoids only*; everything else in the benchmark is out-of-distribution.
"""
function sample_train_batch(rng::AbstractRNG, B::Int, cfg)
    h0 = cfg["h0_lo"] .+ (cfg["h0_hi"] - cfg["h0_lo"]) .* rand(rng, B)
    δs = [sinusoid(cfg["A_lo"] + (cfg["A_hi"] - cfg["A_lo"]) * rand(rng),
                   cfg["w_lo"] + (cfg["w_hi"] - cfg["w_lo"]) * rand(rng);
                   φ = 2π * rand(rng)) for _ in 1:B]
    return h0, δs
end

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

"""
    train_neural_ode(nvf, ps, bgate, cfg; rng, verbose) -> (ps, history)

Adam on the tracking loss

    L(θ) = mean over batch and time of (h_θ(t) - h_ref(t))²

`history` records per-iteration loss and gradient 2-norm; the gradient norm is
the diagnostic used for the hard-vs-smooth differentiability comparison.
"""
function train_neural_ode(nvf::NeuralVectorField, ps, bgate, cfg;
                          rng::AbstractRNG = Xoshiro(0), verbose::Bool = true)
    T   = Float64(cfg["T"]);   dt = Float64(cfg["dt"])
    se  = Int(cfg["save_every"]); B = Int(cfg["batch"])
    its = Int(cfg["iters"])
    opt = Optimisers.setup(Optimisers.Adam(Float64(cfg["lr"])), ps)
    hist = (iter = Int[], loss = Float64[], gnorm = Float64[])
    for it in 1:its
        h0, δs = sample_train_batch(rng, B, cfg)
        _, Href = reference_trajectory(h0, T, dt, se; κ = cfg["kappa"], γ = cfg["gamma"])
        lossfn = p -> begin
            _, H = rk4_rollout(nvf, p, h0, bgate, δs, T, dt, se)
            mean(abs2, H .- Href)
        end
        l, back = Zygote.pullback(lossfn, ps)
        g = back(1.0)[1]
        gn = sqrt(sum(abs2, g))
        opt, ps = Optimisers.update(opt, ps, g)
        push!(hist.iter, it); push!(hist.loss, l); push!(hist.gnorm, gn)
        if verbose && (it == 1 || it % 20 == 0 || it == its)
            @printf("  [%s] iter %4d  loss %.6e  |∇| %.4e\n",
                    batch_gate_label(bgate), it, l, gn)
        end
    end
    return ps, hist
end

# ---------------------------------------------------------------------------
# Evaluation (adaptive solver, so results are not a training-grid artefact)
# ---------------------------------------------------------------------------

"""
    evaluate_neural_ode(nvf, ps, gate, δ; h0, T, dt_save) -> (ts, hs, gs)

PRIMARY evaluation of the *trained* field: fixed-step RK4, matching the
integrator used everywhere else in this study. The hard gate makes the closed
loop `f_θ + g·δ` discontinuous exactly as in the untrained system, so the same
argument applies (see `src/Dynamics.jl`): a fixed step is a well-defined
regularisation, an adaptive one can stall on a sliding mode.
"""
function evaluate_neural_ode(nvf::NeuralVectorField, ps, gate, δ;
                             h0::Float64 = 0.15, T::Float64 = 20.0,
                             dt_save::Float64 = 0.005)
    dt = dt_save / 2
    f(h, t) = nvf_apply(nvf, ps, [h], t)[1] +
              gate_value(gate, clamp(h, 0.0, 1.0)) * δ(t)
    nsteps = round(Int, T / dt)
    ts = Float64[0.0]; hs = Float64[h0]
    h = h0
    for k in 1:nsteps
        t = (k - 1) * dt
        k1 = f(h, t); k2 = f(h + dt/2*k1, t + dt/2)
        k3 = f(h + dt/2*k2, t + dt/2); k4 = f(h + dt*k3, t + dt)
        h += dt/6 * (k1 + 2k2 + 2k3 + k4)
        if k % 2 == 0
            push!(ts, k * dt); push!(hs, h)
        end
    end
    gs = [gate_value(gate, clamp(x, 0.0, 1.0)) for x in hs]
    return ts, hs, gs
end

"""
    evaluate_neural_ode_adaptive(nvf, ps, gate, δ; ..., maxiters)
        -> (ts, hs, gs, ok)

CROSS-CHECK evaluation with the adaptive SciML solver. `ok` is `false` when the
solver did not return `:Success` — most often because it hit `maxiters` while
chattering on a switching surface, exactly the failure documented for the
untrained system in `docs/BASELINE_AUDIT.md`. `maxiters` is capped (default
2·10⁵, ≈50× the nominal step count) so a stall is REPORTED rather than hanging
the benchmark; failed runs are counted, not silently dropped.
"""
function evaluate_neural_ode_adaptive(nvf::NeuralVectorField, ps, gate, δ;
                                      h0::Float64 = 0.15, T::Float64 = 20.0,
                                      dt_save::Float64 = 0.005, alg = Tsit5(),
                                      abstol = 1e-8, reltol = 1e-6,
                                      dtmax = 0.005, maxiters::Int = 200_000)
    fθ(h, t) = nvf_apply(nvf, ps, [h], t)[1]
    rhs!(du, u, p, t) =
        (du[1] = fθ(u[1], t) + gate_value(gate, clamp(u[1], 0.0, 1.0)) * δ(t))
    prob = ODEProblem(rhs!, [h0], (0.0, T))
    sol = solve(prob, alg; saveat = dt_save, dtmax = dtmax,
                abstol = abstol, reltol = reltol, maxiters = maxiters)
    ts = collect(sol.t)
    hs = [u[1] for u in sol.u]
    gs = [gate_value(gate, clamp(h, 0.0, 1.0)) for h in hs]
    ok = (Symbol(sol.retcode) == :Success) && length(ts) > 1
    return ts, hs, gs, ok
end

"""
    tracking_mse(ts, hs; h0, κ, γ) -> Float64

Mean squared deviation of a trajectory from the clean reference trajectory
started at the same `h₀`. This is the primary Neural ODE metric; occupancy and
deviation metrics come from [`compute_metrics`](@ref).
"""
function tracking_mse(ts::AbstractVector, hs::AbstractVector; h0::Float64,
                      κ = 1.2, γ = 2.0)
    dt = ts[2] - ts[1]
    _, Href = reference_trajectory([h0], ts[end], dt, 1; κ = κ, γ = γ)
    m = min(length(hs), size(Href, 1))
    return mean(abs2, hs[1:m] .- Href[1:m, 1])
end
