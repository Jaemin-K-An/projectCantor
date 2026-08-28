# 실험 프로토콜 (Experiment Protocol)

이 문서는 **무엇을 언제 고정했는가**를 기록한다. 결과를 본 뒤 조건을 바꾼
경우는 예외 없이 "POST-HOC"으로 표시한다.

## 1. 사전등록 원칙

1. 모든 실험조건은 `configs/*.toml`에 **실행 전** 기록한다.
2. 결과를 본 뒤 파라미터 범위를 좁히거나 넓혀 특정 결론을 강화하지 않는다.
3. 사후에 추가한 분석은 별도 스크립트(`run_posthoc_*.jl`)로 분리하고
   파일 머리말에 POST-HOC임을 명시한다.
4. `results/raw/`는 append-only다. `write_raw`는 기존 파일이 있으면
   `overwrite=true`를 명시적으로 넘기지 않는 한 **에러를 낸다**.
5. 모든 원자료에 provenance(커밋 해시, Julia 버전, 스레드 수, 타임스탬프,
   설정 파일명, 솔버 설정)를 `.meta.toml` 사이드카로 저장한다.

## 2. 실행 전 설계 변경 이력

| 시점 | 변경 | 사유 | 결과를 보았는가 |
|---|---|---|---|
| sweep 실행 전 | 3분할 factorial slice → **완전 factorial 격자** (11×99×6×6×5 = 196 020) | 적분기 실측 1.42 ms/case로 완전격자가 5분 이내에 가능함을 확인 | **아니오** — 어떤 sweep 결과도 보기 전 |
| baseline 실행 전 | `DifferentialEquations.jl` → `OrdinaryDiffEqTsit5`/`OrdinaryDiffEqVerner` + `SciMLBase` | 8 GB 메모리 환경에서 설치·선컴파일 비용. 동일 SciML 스택의 유지보수되는 하위 패키지이며 `DifferentialEquations`가 ODE에 대해 재수출하는 바로 그 솔버들 | **아니오** |
| baseline 실행 전 | 주 적분기를 **고정스텝 RK4**로 결정 | §7 MATHEMATICAL_ANALYSIS: 불연속 우변에서 적응형 오차제어는 의미가 없고 슬라이딩 모드에서 실패한다 | **아니오** (이후 A단계에서 Vern9 실패가 이 결정을 사후 정당화했다) |

그 외의 격자·시드·지표 정의는 최초 작성 이후 변경하지 않았다.

## 3. 단계별 실행 순서와 산출물

| PHASE | 스크립트 | 설정 | 산출 원자료 |
|---|---|---|---|
| B (테스트) | `test/runtests.jl` | — | 11 687 assertions |
| A (재현·감사) | `reproduce_baseline.jl` | `baseline.toml` | `baseline_{reproduction,h0_audit,solver_check}.csv` |
| C (수치해석) | `run_numerical_analysis.jl` | `numerical.toml` | `numerical_{integration,structure_aware,endpoint_conditioning}.csv` |
| D (sweep) | `run_parameter_sweep.jl` | `sweep.toml` | `sweep_full.csv` (196 020행) |
| F (ablation) | `run_ablation.jl` | `ablation.toml` | `ablation_{main,smooth}.csv` (113 625 + 3 150행) |
| F′ (통계) | `run_ablation_stats.jl` | — | `tables/ablation_{stats,delta_grid}.csv` |
| G (수학 검증) | `run_math_analysis.jl` | — | `math_{invariance_check,smooth_invariance}.csv` |
| H+I (Neural ODE) | `run_neural_ode.jl` | `neural_ode.toml` | `neural_{training,benchmark}.csv` |
| POST-HOC | `run_posthoc_hitting.jl` | — | `tables/posthoc_hitting.csv` |
| 그림 | `generate_all_figures.jl` | — | `figures/*.png`, `figures/CAPTIONS.md` |

## 4. 계산비용 관리

지시된 순서(단위테스트 → smoke → tiny sweep → sanity check → coarse full sweep)를
따랐다. 실측 처리량은 8스레드에서 **1.4 ms/궤적**(T = 30, RK4 Δt = 10⁻³)이었고,
sweep 278 s / ablation 155 s로 축소가 불필요했다. 축소한 실험은 없다.

Neural ODE만 예외적으로 비싸다 (Zygote를 통한 500스텝 RK4 롤아웃의 역전파,
변형당 약 7분 × 8변형). 반복수 300, 배치 24, `T_train = 10`은 사전 고정값이며
결과를 본 뒤 조정하지 않았다.

## 5. 난수와 재현성

* 전역 시드 하나에 의존하지 않는다. 각 무작위 게이트 배치는
  `seed_for(family, n, h0, A, ω, α, replicate)`로 **조건별 결정적 시드**를 얻는다
  (`src/Utils.jl`). 조건 하나만 다시 돌려도 동일한 배치가 재생성된다.
* 무작위 게이트족 G2·G5는 조건당 **30개 시드**. 칸토어는 결코 단일 추첨과
  비교하지 않는다.
* 확률적 섭동(P2 OU, P7 piecewise-random)은 적분 **이전에** 경로를 표본화하고
  선형보간하므로, 주어진 시드에서 완전히 결정적인 ODE다. SDE가 아니며 문서 전체에서
  "randomised ODE"로 표기한다.
* 부동소수점 재현성: 모든 sweep은 스레드 인덱스에 독립적으로 결과를 배열에
  기록하므로 스레드 수를 바꿔도 동일한 CSV가 나온다.

## 6. train/test 분리 (Neural ODE)

`configs/neural_ode.toml`의 `[train]`과 `[eval]`은 **첫 학습 실행 전에**
작성되었다.

* TRAIN: 정현파만, `A ∈ [0.5, 1.5]`, `ω ∈ [2, 6]`, `h₀ ∈ [0.05, 0.95]`
* TEST-ID: 같은 분포에서 새로 추첨한 24개 조건 × 6개 h₀
* TEST-OOD: square / chirp / multifreq / impulse / OU / piecewise-random /
  진폭 외삽 `A = 3` / 주파수 외삽 `ω = 16` / adversarial random search

**테스트 조건을 보고 학습 설정을 바꾸지 않았다.** 모든 게이트 변형은
동일한 신경망 초기화(`seed = 20260828`), 동일한 데이터 스트림, 동일한 예산으로
학습한다.

공정성을 위해 OOD 파형의 진폭은 별도 명시가 없으면 **학습 진폭의 중앙값
`A = 1.0`**으로 고정한다. 그렇지 않으면 "다른 파형"과 "다른 에너지"가 교락된다.

## 7. 게이트 공정성

G1…G5는 통과측도가 **정확히** `(2/3)ⁿ`이다 (`test/runtests.jl` §9,
`rtol = 10⁻¹⁰`). 따라서 두 게이트의 성능 차이를 "칸토어가 더 많이 막아서"로
설명할 수 없다. G0(무필터)와 G6(매끄러운 게이트)는 이 정합을 의도적으로 깨며
별도로 보고한다 — G6의 통과측도는 누설 때문에 항상 하드 게이트보다 크다
(`pass_measure_of(SmoothGate)`로 측정하여 기록).

## 8. 지표 정의의 고정

`src/Metrics.jl`에 6개 지표 + 2개 진단값을 고정했다. 특히

* `R_safe`는 사다리꼴 법칙, `R_safe_rect`는 원 연구의 사각형 법칙 — **둘 다** 저장.
* `τ_S`, `T_rec`는 도달하지 못하면 `Inf`. `Inf`를 버리지 않고 그 비율을 보고한다.
* `T_rec`는 `hold = 5`(Neural ODE는 3) 동안 연속 체류를 요구하는 보수적 규약이며,
  마지막 `hold` 시간 안에 안정화된 궤적은 `Inf`가 된다. 규약을 명시하고 그대로 쓴다.

## 9. 솔버 검증

* 주 적분기: 고정스텝 RK4, `Δt = 10⁻³`, 저장 간격 `2×10⁻³`.
* 수렴 확인: `Δt = 10⁻⁴`와 4자리 일치 (`baseline_solver_check.csv`).
* 교차검증: Tsit5 (atol 10⁻⁸/rtol 10⁻⁶), Vern9 (atol 10⁻¹²/rtol 10⁻¹⁰).
* **Vern9는 슬라이딩 모드에서 실패한다** — 이 실패를 삭제하지 않고 원자료에
  남기고 `docs/BASELINE_AUDIT.md`와 `docs/MATHEMATICAL_ANALYSIS.md` §7에서 분석한다.
* 무필터 계는 닫힌해가 있으므로 두 적분기 모두 해석해와 대조한다
  (`test/runtests.jl` §14: RK4 오차 < 10⁻⁹).

## 10. 보고 규칙

각 결과에 대해 다음 네 질문에 답한다 (`docs/RESULTS.md`).

1. 무엇을 관찰했는가?
2. 어떤 수학적·동역학적 이유로 설명되는가?
3. 대안 설명은 무엇인가?
4. 어느 RQ에 답하는가?

"성능이 좋아졌다" 형태의 서술만으로는 결과를 보고하지 않는다.
