# ============================================================================
# test/v3_1/runtests_v31.jl — V3.1 Julia contract.
#   julia --project=. test/v3_1/runtests_v31.jl
# Reported SEPARATELY from the V1 and V2 suites.
# ============================================================================
using Test, Random, Statistics
R = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(R, "src", "v2", "CantorBarrier.jl"))
include(joinpath(R, "src", "v3_1", "V31Controllers.jl"))
include(joinpath(R, "src", "v3_1", "SelfSimilarityV31.jl"))
include(joinpath(R, "src", "v3", "CantorSelfSimilarity.jl"))
using .CantorBarrier, .V31Controllers, .SelfSimilarityV31, .CantorSelfSimilarity

@testset "V3.1" begin

@testset "1. TRUE constant is exactly state-independent" begin
    for n in 2:7
        C = build_v31("S1_true_constant", n, 1.0/n)
        @test C isa TrueConstant
        v0 = ctrl_field(C, 0.0)
        for r in range(0, 1; length = 2001)
            @test ctrl_field(C, r) === v0            # bitwise identical
        end
        @test sup_field_derivative(C) == 0.0
        @test analytic_action(C) ≈ 1.0 rtol = 1e-12
        # invariant under any boundary shift Δ
        for Δ in (-0.5, -0.1, 0.0, 0.1, 0.5)
            @test ctrl_field(C, clamp(0.37 + Δ, 0, 1)) === v0
        end
    end
end

@testset "2. global_smooth is NOT constant (the V3 defect)" begin
    G = build_v31("S2_global_smooth", 5, 1/5)
    vals = [ctrl_field(G, r) for r in range(0, 1; length = 4001)]
    @test maximum(vals) - minimum(vals) > 1.0        # V3 called this 'constant'
    @test ctrl_field(G, 0.0) ≈ 0.0 atol = 1e-12
    @test ctrl_field(G, 0.5) ≈ 1.5 rtol = 1e-9
    @test sup_field_derivative(G) ≈ 6.0 rtol = 1e-9
    @test analytic_action(G) ≈ analytic_action(build_v31("S1_true_constant", 5, 1/5)) rtol = 1e-12
end

@testset "3. Theorem S derivative identity" begin
    for n in 3:8
        rel, _ = selfsim_residual(cantor_layout(n, 1.0/n), cantor_layout(n-1, 1.0/n))
        @test rel < 1e-10
    end
end

@testset "4. Theorem S potential identity NEEDS the additive offset" begin
    n, E0 = 5, 1/5
    Ln, Lm = cantor_layout(n, E0), cantor_layout(n-1, E0)
    for T in (r -> r/3, r -> 2/3 + r/3)
        off = barrier_potential(Ln, T(0.0))
        for r in range(0, 1; length = 501)
            @test barrier_potential(Ln, T(r)) - off ≈ 0.5*barrier_potential(Lm, r) atol = 1e-9
        end
    end
    # and WITHOUT the offset it fails on the right copy (V3 report's typo)
    off2 = barrier_potential(Ln, 2/3)
    @test off2 > 1e-3
    @test !isapprox(barrier_potential(Ln, 2/3 + 0.5/3), 0.5*barrier_potential(Lm, 0.5); atol = 1e-6)
end

@testset "5. Corollary S.1' local — holds; global version does NOT" begin
    for n in 4:6
        rs = local_corollary_ratio(n, 1.0/n, [0.02, 0.05]; N = 20_000)
        @test all(x -> isnan(x) || abs(x - 1) < 0.02, rs)
    end
    # global claim is false: ratio far from 1
    for n in 5:6
        Ln, Lm = cantor_layout(n, 1.0/n), cantor_layout(n-1, 1.0/n)
        g = coverage_curve(Ln, [0.02/3]; N = 60_000)[1] /
            (1.5 * coverage_curve(Lm, [0.02]; N = 60_000)[1])
        @test g < 0.5                                 # nowhere near 1
    end
end

@testset "6. Theorem T — ||u'||_inf = 12 E0 (9/2)^n, exactly" begin
    for n in 2:8
        L = build_v31("S9_cantor", n, 1.0/n)
        @test sup_field_derivative(L) ≈ 12*(1.0/n)*4.5^n rtol = 1e-9
    end
    # ordering-invariant: every width-matched family has the SAME value
    n, E0 = 5, 1/5
    ref = sup_field_derivative(build_v31("S9_cantor", n, E0))
    for fam in ("S5_periodic", "S6_random", "S7_shuffled", "S8_center_anchored")
        for s in 1:4
            @test sup_field_derivative(build_v31(fam, n, E0; rng = Xoshiro(s))) ≈ ref rtol = 1e-9
        end
    end
    # and the ordering that matters for robustness is the OPPOSITE of budget
    @test sup_field_derivative(build_v31("S1_true_constant", n, E0)) == 0.0
    @test sup_field_derivative(build_v31("S4_wide_central", n, E0)) <
          sup_field_derivative(build_v31("S3_narrow_central", n, E0)) < ref
end

@testset "7-8. width and energy multisets preserved by every control" begin
    n, E0 = 6, 1/6
    base = build_v31("S9_cantor", n, E0)
    w = sort([g.b - g.a for g in base.gaps]); e = sort(base.est)
    for fam in ("S5_periodic", "S6_random", "S7_shuffled", "S8_center_anchored"), s in 1:5
        L = build_v31(fam, n, E0; rng = Xoshiro(s))
        @test sort([g.b - g.a for g in L.gaps]) ≈ w rtol = 1e-9
        @test sort(L.est) ≈ e rtol = 1e-12
        @test analytic_action(L) ≈ analytic_action(base) rtol = 1e-12
    end
end

@testset "12. analytic action identical across ALL families" begin
    n, E0 = 5, 1/5
    for fam in V31_FAMILIES
        fam == "S0_none" && continue
        C = build_v31(fam, n, E0; rng = Xoshiro(2))
        @test analytic_action(C) ≈ 1.0 rtol = 1e-9
    end
    @test analytic_action(build_v31("S0_none", n, E0)) == 0.0
end

end
