# ============================================================================
# test/v2/runtests_v2.jl — V2 barrier controller contract.
#   julia --project=. test/v2/runtests_v2.jl
#
# Items 1-8 of the harness test list (§49). The LLM-side tests live in
# llm/tests/ and run under pytest.
# ============================================================================

using Test, Random, Statistics
include(joinpath(@__DIR__, "..", "..", "src", "v2", "CantorBarrier.jl"))
using .CantorBarrier

@testset "CantorBarrier V2" begin

@testset "0. smoothstep contract" begin
    @test smoothstep(0.0) == 0.0
    @test smoothstep(1.0) == 1.0
    @test smoothstep(-1.0) == 0.0 && smoothstep(2.0) == 1.0
    @test dsmoothstep(0.0) == 0.0 && dsmoothstep(1.0) == 0.0
    @test dsmoothstep(0.5) ≈ 1.5 rtol = 1e-14        # Φ'_max = 3/2
    us = range(0, 1; length = 2001)
    @test all(dsmoothstep.(us) .≥ -1e-15)             # monotone
    @test issorted(smoothstep.(us))
    # Φ' integrates to Φ(1)-Φ(0) = 1
    @test sum(dsmoothstep, us) * step(us) ≈ 1.0 rtol = 1e-6   # ∫Φ' = Φ(1)-Φ(0)
end

@testset "1. gap count N_k = 2^(k-1)" begin
    for n in 1:12
        gs = cantor_gap_list(n)
        @test length(gs) == 2^n - 1
        for k in 1:n
            @test count(g -> g.level == k, gs) == 2^(k-1)
        end
    end
end

@testset "2. gap width w_k = 3^-k" begin
    # exact ground truth first: widths are EXACTLY 3^-k as rationals
    for n in 1:10
        ex = cantor_gap_list_exact(n)
        for (k, a, b) in ex
            @test b - a == 1 // big(3)^k
        end
        @test sum(b - a for (_, a, b) in ex) == 1 - Rational{BigInt}(2, 3)^n
    end
    # then the Float64 construction against that ground truth
    for n in 1:12
        gs = cantor_gap_list(n); ex = cantor_gap_list_exact(n)
        @test length(gs) == length(ex)
        for (g, (k, a, b)) in zip(gs, ex)
            @test g.level == k
            @test g.a ≈ Float64(a) atol = 1e-15
            @test (g.b - g.a) ≈ 3.0^(-g.level) rtol = 1e-9
        end
        @test sum(g -> g.b - g.a, gs) ≈ 1 - (2/3)^n rtol = 1e-9
    end
    # gaps are pairwise disjoint and inside [0,1]
    for n in 1:10
        gs = cantor_gap_list(n)
        @test issorted([g.a for g in gs])
        @test all(gs[i].b ≤ gs[i+1].a + 1e-15 for i in 1:length(gs)-1)
        @test gs[1].a ≥ 0 && gs[end].b ≤ 1 + 1e-15
    end
end

@testset "3. per-gap energy e_k = E0 / 2^(k-1)" begin
    E0 = 0.25
    for n in 1:8
        L = cantor_layout(n, E0)
        for k in 1:n
            @test level_energy(L, k) ≈ E0 / 2.0^(k-1) rtol = 1e-14
        end
    end
end

@testset "4. THEOREM A — level total action = E0, for EVERY family" begin
    E0 = 0.3
    rng = Xoshiro(11)
    for n in 2:7
        for fam in ("B3_periodic", "B4_random", "B5_shuffled",
                    "B6_center_anchored", "B7_cantor")
            L = build_layout(fam, n, E0; rng = Xoshiro(hash((fam, n)) % 10^6))
            for k in 1:n
                @test level_action(L, k; N = 4000) ≈ E0 rtol = 1e-6
            end
        end
    end
end

@testset "5. THEOREM B — peak scaling ratio is exactly 3/2" begin
    E0 = 0.7
    for n in 2:9
        L = cantor_layout(n, E0)
        for k in 2:n
            @test peak_of_level(L, k) / peak_of_level(L, k-1) ≈ 1.5 rtol = 1e-14
        end
        @test peak_of_level(L, 1) ≈ 3 * E0 * 1.5 rtol = 1e-14
        # the analytic peak is actually attained by the implemented field
        for k in 1:min(n, 6)
            g = first(filter(g -> g.level == k, L.gaps))
            mid = (g.a + g.b) / 2
            @test barrier_field(L, mid) ≈ peak_of_level(L, k) rtol = 1e-10
        end
        # and the field vanishes on the surviving set
        @test barrier_field(L, 0.0) == 0.0
        @test barrier_field(L, 1.0) == 0.0
    end
end

@testset "6. total budget = n·E0, and = B_total under normalisation" begin
    B_total = 1.0
    for n in 2:7
        E0 = B_total / n
        for fam in ("B3_periodic", "B5_shuffled", "B6_center_anchored", "B7_cantor")
            L = build_layout(fam, n, E0; rng = Xoshiro(hash((fam, n)) % 10^6))
            @test total_action(L; N = 4000) ≈ B_total rtol = 1e-5
            # potential rise across [0,1] equals the total action (V is monotone)
            @test barrier_potential(L, 1.0) - barrier_potential(L, 0.0) ≈ B_total rtol = 1e-6
        end
        # B1/B2 are given the same total action by construction
        for fam in ("B1_constant", "B2_central")
            L = build_layout(fam, n, E0)
            @test barrier_potential(L, 1.0) - barrier_potential(L, 0.0) ≈ B_total rtol = 1e-6
        end
        @test barrier_potential(no_layout(n, E0), 1.0) == 0.0
    end
end

@testset "7. shuffled controls preserve widths, levels and energies EXACTLY" begin
    n, E0 = 7, 1.0 / 7
    base = cantor_layout(n, E0)
    ref_w = sort([g.b - g.a for g in base.gaps])
    ref_l = sort([g.level for g in base.gaps])
    for fam in ("B3_periodic", "B4_random", "B5_shuffled", "B6_center_anchored")
        for s in 1:6
            L = build_layout(fam, n, E0; rng = Xoshiro(s))
            @test length(L.gaps) == length(base.gaps)
            @test sort([g.b - g.a for g in L.gaps]) ≈ ref_w rtol = 1e-9
            @test sort([g.level for g in L.gaps]) == ref_l
            @test issorted([g.a for g in L.gaps])
            @test all(L.gaps[i].b ≤ L.gaps[i+1].a + 1e-9 for i in 1:length(L.gaps)-1)
            @test L.gaps[1].a ≥ -1e-12 && L.gaps[end].b ≤ 1 + 1e-9
            # ⇒ Theorems A and B hold identically (the ablation is ordering-only)
            @test total_action(L; N = 3000) ≈ total_action(base; N = 3000) rtol = 1e-5
        end
    end
end

@testset "8. center-anchored shuffle keeps the level-1 gap on the boundary" begin
    n, E0 = 7, 1.0 / 7
    for s in 1:10
        L = center_anchored_shuffled_layout(n, E0, Xoshiro(s))
        g1 = only(filter(g -> g.level == 1, L.gaps))
        @test (g1.a + g1.b) / 2 ≈ 0.5 atol = 1e-9   # centred on the decision boundary
        @test g1.a < 0.5 < g1.b
        @test (g1.b - g1.a) ≈ 1/3 rtol = 1e-12
        # the fully shuffled control generally does NOT do this
    end
    off = [abs((first(filter(g -> g.level == 1, shuffled_layout(7, E0, Xoshiro(s)).gaps)).a +
                first(filter(g -> g.level == 1, shuffled_layout(7, E0, Xoshiro(s)).gaps)).b)/2 - 0.5)
           for s in 1:10]
    @test maximum(off) > 0.05     # shuffling really does move the big barrier
end

@testset "9. field/potential consistency" begin
    n, E0 = 6, 1.0/6
    for fam in ("B2_central", "B3_periodic", "B5_shuffled", "B7_cantor")
        L = build_layout(fam, n, E0; rng = Xoshiro(3))
        # V' ≥ 0 everywhere ⇒ controller never pushes toward danger
        rs = range(0, 1; length = 20_001)
        @test all(barrier_field(L, r) ≥ -1e-15 for r in rs)
        # V monotone non-decreasing
        Vs = [barrier_potential(L, r) for r in rs]
        @test all(diff(Vs) .≥ -1e-12)
        # numerically ∫V' = ΔV
        dr = step(rs)
        quad = sum(barrier_field(L, r) for r in rs) * dr
        @test quad ≈ (Vs[end] - Vs[1]) rtol = 2e-3
    end
end

@testset "10. Proposition D — guaranteed peak is nearly ordering-invariant" begin
    n, E0 = 5, 1.0/5
    ℓ = 0.05
    pc = guaranteed_peak(cantor_layout(n, E0), ℓ; N = 60_000)
    ps = [guaranteed_peak(shuffled_layout(n, E0, Xoshiro(s)), ℓ; N = 60_000) for s in 1:16]
    @test pc > 0
    # Proposition D claims NO material Cantor edge on this statistic: the best
    # shuffle must come within 1 % of Cantor. (It is a prediction about size,
    # so the test asserts size, not a strict inequality.)
    @test maximum(ps) ≥ 0.99 * pc
end

@testset "11. Proposition E — worst displacement, Cantor vs controls" begin
    n, E0 = 8, 1.0/8
    C = cantor_layout(n, E0)
    for kstar in 1:n
        dc = worst_displacement(C, kstar)
        dp = worst_displacement(periodic_layout(n, E0), kstar)
        ds = [worst_displacement(shuffled_layout(n, E0, Xoshiro(s)), kstar) for s in 1:20]
        @test 0 < dc ≤ 1
        @test dp ≥ dc - 1e-9                # periodic is never better
        @test median(ds) ≥ dc - 1e-9        # Cantor at least matches the median shuffle
    end
    # the floor: for k* ≥ 2 the width-1/3 level-1 gap is an unavoidable desert
    @test worst_displacement(C, 2) ≈ 1/3 atol = 0.01
end

end # testset
