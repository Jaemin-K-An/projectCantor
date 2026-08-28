# ============================================================================
# ORIGINAL CODE — transcribed verbatim from the appendix of
# "미적분 탐구 보고서.pdf" (칸토어 계단 함수의 n차 근사를 활용한
#  Neural ODE 기반 윤리적 강건성 필터 설계), 학번 30720 안재민.
#
# PRESERVED FOR REPRODUCTION ONLY. DO NOT EDIT.
# The follow-up study reproduces this file's numbers in
# scripts/reproduce_baseline.jl and audits them in docs/BASELINE_AUDIT.md.
#
# NOTE ON TRANSCRIPTION: the PDF text layer was extracted with pypdf.
# Figure numbering inside the appendix (plot_fig1..plot_fig7) does not match
# the figure numbering used in the report body; that is an artifact of the
# original document, reproduced here as-is.
# ============================================================================

using DifferentialEquations
using Plots
using LaTeXStrings

gr()

const OUTDIR = joinpath(@__DIR__, "figures")
mkpath(OUTDIR)

# ─────────────────────────────────────────────
# §1 핵심 함수 정의
# ─────────────────────────────────────────────

"""
    cantor_staircase(x, n) → Float64

C_n(x) = ½ C_{n-1}(3x)          (0 ≤ x < 1/3)
         ½                       (1/3 ≤ x ≤ 2/3)
         ½ + ½ C_{n-1}(3x - 2)   (2/3 < x ≤ 1)
"""
function cantor_staircase(x::Real, n::Int)::Float64
    n == 0 && return Float64(x)
    x ≤ 0 && return 0.0
    x ≥ 1 && return 1.0
    if x < 1/3
        return 0.5 * cantor_staircase(3x, n - 1)
    elseif x ≤ 2/3
        return 0.5
    else
        return 0.5 + 0.5 * cantor_staircase(3x - 2, n - 1)
    end
end

"""
    cantor_derivative(x, n) → Float64

C_n'(x) = (3/2)^n   (비평탄 구간, 총 측도 (2/3)^n)
          0         (평탄 구간, 총 측도 1-(2/3)^n)
"""
function cantor_derivative(x::Real, n::Int)::Float64
    (x ≤ 0 || x ≥ 1) && return 0.0
    xf = Float64(x)
    for _ in 1:n
        if 1/3 < xf < 2/3
            return 0.0
        elseif xf ≤ 1/3
            xf = 3xf
        else
            xf = 3xf - 2
        end
    end
    return (3/2)^n
end

"""
    kl_divergence(μ; σ=1.0) → Float64
KL(N(μ, σ²) ‖ N(0, 1))
"""
kl_divergence(μ; σ=1.0) = -log(σ) + (σ^2 + μ^2) / 2 - 0.5

"""
    kl_derivative_mu(μ) → Float64
∂/∂μ KL(N(μ,1) ‖ N(0,1)) = μ
"""
kl_derivative_mu(μ) = μ

# ─────────────────────────────────────────────
# §2 ODE 설정
# ─────────────────────────────────────────────

const H_SAFE = 0.5
const α      = 0.8
const A_PERT = 1.5
const ω      = 4.0
const TSPAN  = (0.0, 30.0)
const H0     = [0.15]

function ode_no_filter!(dh, h, p, t)
    dh[1] = -α * (h[1] - H_SAFE) + A_PERT * sin(ω * t)
end

function make_ode_cantor(n::Int)
    function ode_cantor!(dh, h, p, t)
        hc   = clamp(h[1], 0.001, 0.999)
        cd   = cantor_derivative(hc, n)
        gate = cd / (3/2)^n          # 정규화 → [0, 1]
        dh[1] = -α * (h[1] - H_SAFE) + gate * A_PERT * sin(ω * t)
    end
    return ode_cantor!
end

# ─────────────────────────────────────────────
# §3 궤적 비교 (원문 [그림 6])
# ─────────────────────────────────────────────

function plot_fig5()
    prob_no = ODEProblem(ode_no_filter!,    H0, TSPAN)
    prob_c3 = ODEProblem(make_ode_cantor(3), H0, TSPAN)
    prob_c5 = ODEProblem(make_ode_cantor(5), H0, TSPAN)

    sol_no = solve(prob_no, Tsit5(); saveat=0.005, dtmax=0.005)
    sol_c3 = solve(prob_c3, Tsit5(); saveat=0.005, dtmax=0.005)
    sol_c5 = solve(prob_c5, Tsit5(); saveat=0.005, dtmax=0.005)

    safe_lo, safe_hi = 1/3, 2/3

    function make_panel(sol, title_str, color, bg_color)
        h = sol[1, :]
        pct = round(100 * mean(safe_lo .≤ h .≤ safe_hi); digits=1)
        p = plot(sol.t, h;
            color = color, linewidth = 0.9, ylabel = L"h(t)",
            title = title_str, ylims = (-0.1, 1.1),
            legend = :topright, label = "", grid = true, gridalpha = 0.3,
        )
        hspan!(p, [safe_lo, safe_hi]; color=:palegreen, alpha=0.3,
               label=L"C_1\textrm{\ flat\ region\ }[1/3,\,2/3]")
        hline!(p, [H_SAFE]; color=:gray, linewidth=0.5, linestyle=:dot, label="")
        annotate!(p, sol.t[end] * 0.02, 0.05,
                  text("Time in safe region: $(pct)%", 10, :left))
        return p
    end

    mean(x) = sum(x) / length(x)

    p1 = make_panel(sol_no, "(A) No Cantor Filter", :red3, :mistyrose)
    p2 = make_panel(sol_c3, "(B) With Cantor Filter C_3 (flat measure=70.4%)",
                    :slateblue, :lavender)
    p3 = make_panel(sol_c5, "(C) With Cantor Filter C_5 (flat measure=86.8%)",
                    :royalblue4, :honeydew)
    plot!(p3; xlabel=L"\textrm{Time\ }t")

    combined = plot(p1, p2, p3; layout=(3,1), size=(1200,850), dpi=180)
    savefig(combined, joinpath(OUTDIR, "fig5_ode_comparison.png"))
    println("✓ Figure 5 saved")
    return (sol_no, sol_c3, sol_c5)
end
