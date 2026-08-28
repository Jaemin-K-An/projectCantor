# ============================================================================
# test/runtests.jl — run with:  julia --project=. test/runtests.jl
#
# These tests are the contract for every claim the report makes about the
# finite-order Cantor construction, the measure matching of the ablation
# controls, and the metric definitions. They are deterministic.
# ============================================================================

using Test, Random, Statistics
include(joinpath(@__DIR__, "..", "src", "CantorGate.jl"))
using .CantorGate

const RTOL = 1e-12

@testset "CantorGate" begin

# ---------------------------------------------------------------- §1 staircase
@testset "1. staircase endpoints and monotonicity" begin
    for n in 0:10
        @test cantor_staircase(0.0, n) == 0.0
        @test cantor_staircase(1.0, n) == 1.0
    end
    # C_n is monotone non-decreasing
    xs = range(0, 1; length = 4001)
    for n in 1:8
        ys = cantor_staircase.(xs, n)
        @test all(diff(ys) .>= -1e-15)
    end
    # the central plateau is exactly 1/2 for every n ≥ 1
    for n in 1:10, x in (0.34, 0.5, 0.6, 2/3)
        @test cantor_staircase(x, n) ≈ 0.5 atol = 1e-15
    end
end

# ------------------------------------------------------- §2 flat / non-flat sets
@testset "2. flat interval classification" begin
    # the level-1 middle third is flat for every n ≥ 1
    for n in 1:10
        @test cantor_gate(0.5, n) == 0.0
        @test cantor_gate(0.4, n) == 0.0
    end
    # endpoints of the removed middle third are IN K_n (closed convention)
    for n in 1:8
        @test cantor_gate(1/3, n) == 1.0
        @test cantor_gate(2/3, n) == 1.0
        @test cantor_gate(0.0, n) == 1.0
        @test cantor_gate(1.0, n) == 1.0
    end
    # n = 0: nothing is removed
    @test cantor_gate(0.5, 0) == 1.0
end

# --------------------------------------------- §3–4 THE h0 = 0.15 AUDIT
@testset "3. C3(0.15) and C5(0.15) gates are ZERO" begin
    # This is the central claim of docs/BASELINE_AUDIT.md: the original study's
    # initial condition already sits inside a flat interval, so the C3 and C5
    # runs had their perturbation switched off from t = 0.
    @test cantor_derivative(0.15, 3) == 0.0
    @test cantor_derivative(0.15, 5) == 0.0
    @test cantor_gate(0.15, 3) == 0.0
    @test cantor_gate(0.15, 5) == 0.0
    # ... but NOT at n = 1, where 0.15 < 1/3 is still in K_1
    @test cantor_gate(0.15, 1) == 1.0
    # removal happens at level 2, in the open interval (1/9, 2/9)
    fi = cantor_flat_interval(0.15, 5)
    @test fi.level == 2
    @test fi.lo == 1 // 9
    @test fi.hi == 2 // 9
    @test Float64(fi.lo) < 0.15 < Float64(fi.hi)
end

# ---------------------------------------------------------------- §5 measures
@testset "5. analytic measures" begin
    for n in 0:12
        @test flat_measure(n) + pass_measure(n) ≈ 1.0 rtol = RTOL
        @test pass_measure(n) ≈ (2/3)^n rtol = RTOL
        @test nonflat_derivative(n) ≈ (3/2)^n rtol = RTOL
    end
    # exact interval widths reproduce (2/3)^n
    for n in 0:12
        iv = cantor_intervals(n)
        @test length(iv) == 2^n
        @test sum(Float64(b - a) for (a, b) in iv) ≈ (2/3)^n rtol = 1e-12
    end
    # gap widths: 2^{k-1} gaps of width 3^{-k}, summing to 1-(2/3)^n
    for n in 1:10
        gw = cantor_gap_widths(n)
        @test length(gw) == 2^n - 1
        @test sum(Float64, gw) ≈ 1 - (2/3)^n rtol = 1e-12
        for k in 1:n
            @test count(g -> g == 1 // big(3)^k, gw) == 2^(k-1)
        end
    end
end

# ------------------------------------------------------- §6 analytic integral
@testset "6. analytic integral is exactly 1" begin
    for n in 0:12
        @test pass_measure(n) * nonflat_derivative(n) ≈ 1.0 rtol = RTOL
        @test analytic_integral(n) == 1.0
    end
    # structure-aware quadrature agrees to machine precision at every n
    for n in 0:14
        @test exact_interval_integral(n) ≈ 1.0 rtol = 1e-10
    end
end

# ------------------------------------- §7 two independent membership tests agree
@testset "7. recursive vs interval-based classification" begin
    rng = Xoshiro(4242)
    for n in 0:9
        for _ in 1:400
            x = rand(rng)
            @test in_cantor_set(x, n) == in_cantor_set_intervals(x, n)
        end
    end
    # deterministic probes that are exactly representable or far from endpoints
    for x in (0.0, 0.25, 0.4, 0.5, 0.75, 1.0), n in 0:8
        @test in_cantor_set(x, n) == in_cantor_set_intervals(x, n)
    end
end

# --------------------- §7b DOCUMENTED DEFECT of the original recursive algorithm
@testset "7b. recursive digit map is ill-conditioned at exact endpoints" begin
    # x = 7/9 and 8/9 are exact endpoints of the level-2 Cantor interval
    # [7/9, 8/9] and therefore lie in K_n for every n. Neither is exactly
    # representable in binary, and the map x ↦ 3x-2 multiplies that
    # representation error by 3 per level, pushing the iterate across the
    # switching surface. The interval-based test gets them right.
    for x in (7/9, 8/9), n in 2:8
        @test in_cantor_set_intervals(x, n) == true
        @test in_cantor_set(x, n) == false          # the ORIGINAL algorithm's answer
    end
    # the defect grows with n and is confined to the (Lebesgue-null) endpoint set
    fracs = [endpoint_instability(n).frac for n in 1:10]
    @test fracs[1] == 0.0
    @test fracs[end] > fracs[2]
    @test all(0.0 .<= fracs .<= 1.0)
    # the stable test agrees with the interval test everywhere
    rng2 = Xoshiro(31337)
    for n in 1:9
        iv = cantor_intervals(n)
        los = Float64.(first.(iv)); his = Float64.(last.(iv))
        for _ in 1:300
            x = rand(rng2)
            @test in_cantor_set_stable(x, los, his) == in_cantor_set_intervals(x, n)
        end
        for (a, b) in iv
            @test in_cantor_set_stable(Float64(a), los, his)
            @test in_cantor_set_stable(Float64(b), los, his)
        end
    end
end

# ------------------------------------------------- §8 floating-point boundaries
@testset "8. boundary convention is stable in Float64" begin
    for n in 1:8
        @test cantor_gate(nextfloat(1/3), n) == (n == 1 ? 0.0 : 0.0)  # just inside the gap
        @test cantor_gate(prevfloat(1/3), n) == 1.0                   # just outside
        @test cantor_gate(nextfloat(2/3), n) == 1.0
        @test cantor_gate(prevfloat(2/3), n) == 0.0
    end
    # outside [0,1] the set membership is false
    for n in 1:6
        @test !in_cantor_set(-1e-9, n)
        @test !in_cantor_set(1 + 1e-9, n)
    end
end

# --------------------------------------------------------- §7b/§8b gate objects
@testset "9. gate families and measure matching" begin
    rng = Xoshiro(7)
    @test gate_value(NoGate(), 0.5) == 1.0
    @test gate_value(NoGate(), 0.123) == 1.0
    for n in 1:8
        target = (2/3)^n
        gs = [cantor_interval_gate(n), random_matched_gate(n, rng),
              periodic_gate(n), central_gate(n), shuffled_multiscale_gate(n, rng)]
        for g in gs
            @test pass_measure_of(g) ≈ target rtol = 1e-10
            # intervals must be sorted and disjoint
            @test issorted(g.los)
            @test all(g.los .< g.his)
            @test all(g.his[1:end-1] .<= g.los[2:end] .+ 1e-15)
            @test g.los[1] ≥ -1e-15 && g.his[end] ≤ 1 + 1e-9
        end
        # G1 built from intervals must agree pointwise with the recursive gate
        gc = cantor_interval_gate(n)
        for _ in 1:200
            x = rand(rng)
            @test gate_value(gc, x) == cantor_gate(x, n)
        end
    end
    # G4 at n = 1 coincides with the Cantor gate
    for x in range(0, 1; length = 501)
        @test gate_value(central_gate(1), x) == gate_value(cantor_interval_gate(1), x)
    end
end

@testset "10. smooth gate converges to the hard gate" begin
    base = cantor_interval_gate(3)
    xs = range(0.001, 0.999; length = 2000)
    # L1 distance converges; the SUP distance does NOT, because at an interval
    # endpoint the smooth gate is pinned at σ(0)σ(βw) ≈ 1/2 for every β.
    # Asserting the right notion of convergence is the point of this test.
    l1 = Float64[]
    for β in (10.0, 100.0, 1000.0, 10000.0)
        sg = SmoothGate(base, β)
        push!(l1, sum(abs(gate_value(sg, x) - gate_value(base, x)) for x in xs) / length(xs))
    end
    @test issorted(l1; rev = true)            # monotone L1 convergence in β
    @test l1[end] < 0.005
    @test abs(gate_value(SmoothGate(base, 1e4), base.los[3]) - 0.5) < 1e-3
    for β in (10.0, 100.0), x in xs[1:97:end]
        v = gate_value(SmoothGate(base, β), x)
        @test 0.0 ≤ v ≤ 1.0
    end
    # a smooth gate leaks: its pass measure exceeds the hard one
    @test pass_measure_of(SmoothGate(base, 25.0); N = 200_000) > pass_measure_of(base)
end

# ----------------------------------------------- §9 safe-region boundary signs
@testset "11. inward-pointing field on ∂S under a Cantor gate" begin
    α = 0.8
    for n in 1:8
        g = cantor_interval_gate(n)
        # strictly inside the open middle third the gate is identically zero
        for x in (1/3 + 1e-9, 0.4, 0.5, 0.6, 2/3 - 1e-9)
            @test gate_value(g, x) == 0.0
        end
        # so the field there is purely restoring, and inward at the boundary
        f = make_rhs(g, sinusoid(1e6, 4.0), α)     # absurd amplitude: still blocked
        @test f(1/3 + 1e-9, 0.0) ≈ α * (H_SAFE - 1/3 - 1e-9) rtol = 1e-6
        @test f(1/3 + 1e-9, 0.0) > 0
        @test f(2/3 - 1e-9, 1.0) < 0
        @test f(2/3 - 1e-9, 1.0) ≈ α * (H_SAFE - 2/3 + 1e-9) rtol = 1e-6
    end
    # magnitude at the boundary is exactly α/6
    @test 0.8 * (0.5 - 1/3) ≈ 0.8 / 6 rtol = RTOL
end

# ------------------------------------------------------------- §10 metrics
@testset "12. metric computation" begin
    ts = collect(0.0:0.001:1.0)
    # constant inside S  → occupancy 1, τ_S = 0
    hs = fill(0.5, length(ts))
    m = compute_metrics(ts, hs, zeros(length(ts)); hold = 0.2)
    @test m.R_safe ≈ 1.0 rtol = 1e-12
    @test m.tau_S == 0.0
    @test m.D_max ≈ 0.0 atol = 1e-15
    @test m.T_rec == 0.0
    # constant outside S → occupancy 0, τ_S = Inf, T_rec = Inf
    hs = fill(0.9, length(ts))
    m = compute_metrics(ts, hs, ones(length(ts)); hold = 0.2)
    @test m.R_safe == 0.0
    @test m.tau_S == Inf
    @test m.T_rec == Inf
    @test m.D_max ≈ 0.4 rtol = 1e-12
    @test m.R_gate ≈ 1.0 rtol = 1e-12
    # a linear ramp entering S at a known time
    hs = [0.0 + 0.5 * t for t in ts]              # h(t) = t/2, enters S at t = 2/3
    m = compute_metrics(ts, hs, zeros(length(ts)); hold = 0.1)
    @test m.tau_S ≈ 2/3 atol = 2e-3
    # trapezoid vs rectangle occupancy differ only at O(Δt/T)
    @test abs(m.R_safe - m.R_safe_rect) < 2e-3
    # first_hitting_time on an exact crossing
    # exact crossing: h goes 0 → 1/2 over [0,1], so it reaches 1/3 at t = 2/3
    @test first_hitting_time([0.0, 1.0], [0.0, 0.5]) ≈ 2/3 rtol = 1e-12
    # a jump that steps clean over S is (correctly) not counted as a hit
    @test first_hitting_time([0.0, 1.0], [0.0, 1.0]) == Inf
end

# ----------------------------------------------- §11 fixed-seed reproducibility
@testset "13. determinism" begin
    a = random_matched_gate(5, Xoshiro(99))
    b = random_matched_gate(5, Xoshiro(99))
    @test a.los == b.los
    c = random_matched_gate(5, Xoshiro(100))
    @test a.los != c.los
    @test seed_for("ablation", 3, 0.5) == seed_for("ablation", 3, 0.5)

    δ = sinusoid(1.5, 4.0)
    m1 = run_case(cantor_interval_gate(3), δ; h0 = 0.15)
    m2 = run_case(cantor_interval_gate(3), δ; h0 = 0.15)
    @test m1.R_safe == m2.R_safe

    p1 = piecewise_random(1.0, 30.0, 0.5, 7)
    p2 = piecewise_random(1.0, 30.0, 0.5, 7)
    @test [p1(t) for t in 0:0.1:5] == [p2(t) for t in 0:0.1:5]
end

# ------------------------------------------- §12 integrator vs analytic solution
@testset "14. RK4 and adaptive solvers match the closed-form ungated solution" begin
    δ = sinusoid(1.5, 4.0)
    ts, hs, _ = simulate_rk4(NoGate(), δ; α = 0.8, h0 = 0.15, T = 30.0)
    ex = analytic_no_filter.(ts)
    @test maximum(abs, hs .- ex) < 1e-9
    ts2, hs2, _ = simulate_adaptive(NoGate(), δ; α = 0.8, h0 = 0.15, T = 30.0)
    @test maximum(abs, hs2 .- analytic_no_filter.(ts2)) < 1e-6
    # and the analytic steady-state occupancy predicts the measured one
    m = run_case(NoGate(), δ; h0 = 0.15)
    @test abs(m.R_safe - analytic_no_filter_occupancy()) < 0.02
end

# ------------------------------------------------------- §13 perturbation library
@testset "15. perturbation amplitude matching" begin
    T = 30.0
    tgrid = range(0, T; length = 200_001)
    ps = [sinusoid(1.5, 4.0), square_wave(1.5, 4.0), chirp(1.5, 1.0, 16.0, T),
          multifreq(1.5, [1.3, 3.7, 9.1]),
          impulse_train(1.5, [3.0, 9.0, 15.0, 21.0, 27.0], 0.2),
          piecewise_random(1.5, T, 0.5, 3), ou_noise(1.5, T, 1.0, 1.0, 0.01, 5)]
    for p in ps
        s = maximum(abs(p(t)) for t in tgrid)
        @test s ≤ 1.5 + 1e-9
        @test p.bound ≈ 1.5 rtol = 1e-12
        @test s > 0.85 * 1.5          # amplitude is actually attained
    end
    @test zero_perturbation()(3.7) == 0.0
end

# ------------------------------------------------- §12 Neural ODE smoke test
@testset "16. Neural ODE trains (smoke test)" begin
    rng = Xoshiro(1)
    nvf, ps = init_nvf(rng; width = 8, depth = 1, T = 5.0)
    @test length(ps) > 0
    cfg = Dict("T" => 2.0, "dt" => 0.05, "save_every" => 4, "batch" => 4,
               "iters" => 6, "lr" => 0.01, "kappa" => 1.2, "gamma" => 2.0,
               "h0_lo" => 0.1, "h0_hi" => 0.9, "A_lo" => 0.5, "A_hi" => 1.5,
               "w_lo" => 2.0, "w_hi" => 6.0)
    for bg in (BNone(),
               to_batch_gate(cantor_interval_gate(3)),
               to_batch_gate(SmoothGate(cantor_interval_gate(3), 25.0)))
        ps2, hist = train_neural_ode(nvf, ps, bg, cfg; rng = Xoshiro(2), verbose = false)
        @test length(hist.loss) == 6
        @test all(isfinite, hist.loss)
        @test all(isfinite, hist.gnorm)
        @test hist.loss[end] < hist.loss[1] * 1.5     # not diverging
        @test ps2 != ps                                # parameters actually moved
    end
    # batched and scalar gates agree
    g = cantor_interval_gate(3)
    bg = to_batch_gate(g)
    xs = collect(range(0.001, 0.999; length = 501))
    @test apply_gate(bg, xs) == [gate_value(g, x) for x in xs]
    sg = SmoothGate(g, 30.0)
    bsg = to_batch_gate(sg)
    @test maximum(abs, apply_gate(bsg, xs) .- [gate_value(sg, x) for x in xs]) < 1e-12
end

# --------------------------------------------------------- §13 output schema
@testset "17. numerical-analysis helpers" begin
    # exact-interval quadrature never fails
    for n in 1:12
        @test exact_interval_integral(n) ≈ 1.0 rtol = 1e-10
    end
    # the uniform grid is exact only while it resolves the fractal scale
    @test riemann_integral_midpoint(3, 1e-5) ≈ 1.0 rtol = 1e-3
    st = node_hit_statistics(3, 1e-5)
    @test st.n_intervals == 8
    @test st.hit == 8
    @test st.rho ≈ 1e-5 * 27 rtol = 1e-12
    # and it degenerates once ρ ≫ 1
    st2 = node_hit_statistics(10, 1e-2)
    @test st2.rho > 100
    @test st2.hit < st2.n_intervals
end

end # testset CantorGate
