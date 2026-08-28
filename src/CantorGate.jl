"""
    CantorGate

Research code for *"Robustness Analysis of Cantor-Gated Dynamical Systems and
Neural ODEs"* — the follow-up study to the calculus investigation
"칸토어 계단 함수의 n차 근사를 활용한 Neural ODE 기반 윤리적 강건성 필터 설계".

Layer map:

| file                  | contents                                              |
|-----------------------|-------------------------------------------------------|
| `CantorCore.jl`       | `C_n`, `C'_n`, exact intervals, flat-interval lookup   |
| `Controls.jl`         | gate families G0–G6 (the ablation)                     |
| `Perturbations.jl`    | disturbance library δ(t), amplitude-matched            |
| `Metrics.jl`          | the six trajectory metrics                             |
| `Dynamics.jl`         | the gated ODE, RK4 + SciML integrators, analytic soln  |
| `NumericalAnalysis.jl`| quadrature of `C'_n` and its failure modes             |
| `NeuralODEModel.jl`   | trainable `f_θ` (this is the only Neural ODE)          |
| `Utils.jl`            | config, provenance, result I/O                         |
"""
module CantorGate

include("CantorCore.jl")
include("Controls.jl")
include("Perturbations.jl")
include("Metrics.jl")
include("Dynamics.jl")
include("NumericalAnalysis.jl")
include("Utils.jl")
include("NeuralODEModel.jl")

# CantorCore
export cantor_staircase, cantor_derivative, cantor_gate, in_cantor_set,
       in_cantor_set_intervals, cantor_intervals, cantor_flat_interval,
       cantor_gap_widths, pass_measure, flat_measure, nonflat_derivative,
       in_cantor_set_stable, endpoint_instability,
       analytic_integral
# Controls
export Gate, NoGate, IntervalGate, SmoothGate, gate_value, gate_label,
       gate_family, pass_measure_of, cantor_interval_gate, random_matched_gate,
       periodic_gate, central_gate, shuffled_multiscale_gate, build_gate,
       GATE_FAMILIES
# Perturbations
export Perturb, sinusoid, square_wave, chirp, impulse_train, multifreq,
       piecewise_random, ou_noise, zero_perturbation
# Metrics
export TrajectoryMetrics, compute_metrics, metrics_tuple, in_safe,
       first_hitting_time, recovery_time, trapz_mean,
       SAFE_LO, SAFE_HI, H_SAFE, METRIC_COLS
# Dynamics
export make_rhs, simulate_rk4, simulate_adaptive, run_case,
       analytic_no_filter, analytic_no_filter_occupancy, Tsit5, Vern9
# NumericalAnalysis
export riemann_integral_left, riemann_integral_midpoint,
       exact_interval_integral, adaptive_integral, node_hit_statistics
# NeuralODEModel
export NeuralVectorField, init_nvf, nvf_apply, rk4_rollout, train_neural_ode,
       BatchGate, BNone, BHard, BSmooth, apply_gate, to_batch_gate,
       batch_gate_label, reference_trajectory, sample_train_batch, tracking_mse,
       reference_field, evaluate_neural_ode, evaluate_neural_ode_adaptive
# Utils
export load_config, provenance, git_commit, write_raw, write_table, seed_for,
       phase_checkpoint, cfgpath, rawpath, procpath, tabpath, figpath, logpath, fmtf

end # module
