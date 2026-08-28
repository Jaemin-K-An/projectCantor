# 칸토어 게이트 동역학계와 Neural ODE의 강건성 분석

> **Robustness Analysis of Cantor-Gated Dynamical Systems and Neural ODEs**
>
> 미적분 탐구 *"칸토어 계단 함수의 n차 근사를 활용한 Neural ODE 기반 윤리적
> 강건성 필터 설계"* 의 후속 연구. 원 보고서는 `original/`에 보존되어 있다.

---

## 한 문단 요약

원 탐구는 `dh/dt = −α(h−½) + g_n(h)·A sin(ωt)` 에서 칸토어 게이트가 안전영역
체류비율을 29.8 % → 98.2 %로 올린다고 보고했다. 본 연구는 그 수치를 0.05 %p
이내로 **재현**한 뒤, 그 수치가 뒷받침하는 주장이 무엇인지 검증했다.
결론은 다음과 같다.

* 원 실험의 초기조건 `h₀ = 0.15`는 `(1/9, 2/9)` — **이미 평탄구간 안**이라
  C₃·C₅ 실험은 30초 중 **0.17 %** 시간만 게이트가 열린 상태로 진행되었다.
* 안전집합 `S = [1/3, 2/3]`은 칸토어 게이트의 레벨-1 평탄구간 **그 자체**이므로
  `S°`의 전방불변성은 시뮬레이션 결과가 아니라 **항등식**이다 (정리 1).
* 196 020회 sweep에서 `R_safe`는 `n`에 대해 **단조가 아니며** `n = 1`에서
  최대, `n ≈ 7`에서 최소인 U자형이다.
* 113 625회 ablation에서 칸토어는 **동일 측도 대조군과 통계적으로 구분되지
  않으며**, 위상만 섞은 대조군(G5)에는 오히려 조금 뒤지고, 자명한
  "중앙 차단" 게이트(G4)에는 큰 `n`에서 진다.
* 실제로 학습되는 Neural ODE를 구현하고 ID/OOD를 나누어 평가했다.

**즉 본 연구의 주요 결과는 negative result이며, 그대로 보고한다.**
자세한 내용은 [`docs/FINAL_REPORT_KO.md`](docs/FINAL_REPORT_KO.md).

---

## 요구 환경

* Julia ≥ 1.10 (개발·검증: 1.12.6, macOS arm64)
* 디스크 약 300 MB (원자료 CSV 포함), 메모리 4 GB 이상 권장

## Fresh clone 재현 절차

```bash
git clone <repo> && cd Cantor
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml`이 정확한 패키지 버전을 고정한다. 그 뒤 아래를 **순서대로**
실행한다 (괄호는 8코어 기준 실측 소요시간).

```bash
julia --project=. test/runtests.jl                        # (~50 s) 11 687 assertions
julia --project=. scripts/reproduce_baseline.jl           # (~3 min) PHASE A
julia --project=. scripts/run_numerical_analysis.jl       # (~1 min) PHASE C
julia --project=. -t auto scripts/run_parameter_sweep.jl  # (~5 min) PHASE D
julia --project=. -t auto scripts/run_ablation.jl         # (~3 min) PHASE F
julia --project=. scripts/run_ablation_stats.jl           # (~2 min) PHASE F'
julia --project=. scripts/run_math_analysis.jl            # (~2 min) PHASE G
julia --project=. scripts/run_neural_ode.jl               # (~60 min) PHASE H  (training)
julia --project=. scripts/run_neural_benchmark.jl         # (~11 min) PHASE I  (ID/OOD)
julia --project=. scripts/run_posthoc_hitting.jl          # (~1 min) POST-HOC
julia --project=. scripts/generate_all_figures.jl         # (~2 min) 모든 그림
```

또는 전부 한 번에:

```bash
julia --project=. scripts/run_all.jl
```

`scripts/run_neural_ode.jl`은 학습된 파라미터를
`results/processed/trained_params.jls`에 저장하므로, 벤치마크만 다시 돌릴 때는
`run_neural_benchmark.jl`만 실행하면 된다 (재학습 불필요).

`scripts/reproduce_baseline.jl`은 PHASE A에서 의도적으로 76초짜리 Vern9 실패를
포함한다 — 이는 버그가 아니라 보고 대상 결과다 (`docs/BASELINE_AUDIT.md`).
같은 이유로 테스트와 벤치마크 실행 중에 `maxiters` 경고가 출력되는데, 이는
가드가 채터링을 **정상적으로 검출**하고 있다는 뜻이다.

### 결과가 이미 있는 상태에서 다시 돌리기

`results/raw/`는 append-only 정책이다. 각 스크립트는 `overwrite = true`로
명시적으로 덮어쓰도록 작성되어 있으므로 재실행하면 갱신된다. 기존 결과를
보존하려면 `results/raw`를 먼저 다른 이름으로 옮겨라.

## 디렉터리 구조

```
configs/    실행 전 고정된 실험 설정 (TOML)
src/        라이브러리 — 아래 표 참조
scripts/    각 PHASE의 실행 스크립트
test/       11 687개 assertion
results/raw/        원자료 CSV + .meta.toml provenance 사이드카
results/tables/     파생 표 (CSV/Markdown)
figures/            모든 그림 + CAPTIONS.md
docs/               감사·수학·프로토콜·결과·최종보고서
logs/               실행 로그와 PROGRESS.md 체크포인트
original/           원 PDF와 원 Julia 코드 (편집 금지)
```

| `src/` 파일 | 내용 |
|---|---|
| `CantorCore.jl` | `C_n`, `C'_n`, 정확한 구간 표현, 평탄구간 추적, 끝점 조건수 |
| `Controls.jl` | 게이트족 G0–G6 (ablation의 핵심) |
| `Perturbations.jl` | 섭동 라이브러리 δ(t), 진폭 정합 |
| `Metrics.jl` | 6개 지표 + 2개 진단값 |
| `Dynamics.jl` | 게이트 ODE, RK4 및 SciML 적분기, 무필터 닫힌해 |
| `NumericalAnalysis.jl` | `∫C'_n` 구적과 그 실패 양상 |
| `NeuralODEModel.jl` | 학습되는 `f_θ` — **여기서부터만 Neural ODE** |
| `Utils.jl` | 설정, provenance, 결과 I/O |

## 용어에 관한 주의

`f`에 학습 파라미터가 없는 단계는 **Cantor-gated dynamical system**이라 부른다.
"Neural ODE"는 `src/NeuralODEModel.jl`에서 `f_θ`를 실제로 학습한 이후에만
사용한다. 원 보고서의 명명은 이 점에서 부정확했다 (`docs/BASELINE_AUDIT.md`).

## 패키지 선택에 관한 주의

ODE 솔버는 `DifferentialEquations.jl` 전체 대신 유지보수되는 하위 패키지
`OrdinaryDiffEqTsit5`, `OrdinaryDiffEqVerner`, `SciMLBase`를 쓴다 (8 GB 환경의
설치·선컴파일 비용). ODE에 대해 `DifferentialEquations`가 재수출하는 바로 그
솔버들이다. Neural ODE는 `Lux` + `Zygote` + `Optimisers` + `ComponentArrays`로
구성하고, **adjoint 대신 고정스텝 RK4 롤아웃을 end-to-end로 미분**한다.
그 이유는 `src/NeuralODEModel.jl` 머리말에 있다 (하드 게이트에서 연속 adjoint가
잘 정의되지 않는다).

## 라이선스 / 출처

원 탐구 보고서의 저작자는 안재민(학번 30720)이며 `original/`의 내용은
후속 연구를 위해 원형 그대로 보존한다.
