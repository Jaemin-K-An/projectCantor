# 칸토어 게이트 동역학계·Neural ODE·LLM 잔차 안전 제어의 강건성 분석

> **V3.1 (closure) 추가됨** — `docs/v3_1/FINAL_REPORT_KO.md` (branch `cantor-guard-v3.1`)
> V3의 코드 결함 2건을 수정하고(진짜 상수 컨트롤러, V2 calibration 재구성),
> **정리 T**(`‖u′‖ = 12E₀(9/2)^k`)를 증명했으며, **실제 LLM에서 실현 개입
> 예산을 ±3 %로 맞춘 direct controller test**를 수행했다.
> 동결된 분류기 판정: **`C_PRACTICALLY_EQUIVALENT`** — 다척도 개입은
> 무제어(+0.029)와 진짜 상수(+0.018)보다 낫지만, **폭-정합 대조군과
> 칸토어는 사전등록 SESOI 이내로 동등하다.**
>
> **V3 추가됨** — `docs/v3/FINAL_REPORT_KO.md` (branch `cantor-guard-v3`)
> LLM 안전 판정 경계가 **정상적이지 않음**을 측정하고(중앙값 0.34 σ, 최대
> 1.77 σ), 경계 불확실성 하에서 칸토어 구조를 재검정했다. **정리 S**로
> 칸토어 고유의 수학적 성질을 처음 확보했으나, 그 성질은 강건성과
> **무관**했다(r = +0.147). 자세한 내용은 아래 §V3.
>
> **V2 (CantorGuard) 추가됨** — `docs/v2/FINAL_REPORT_KO.md`
> V1이 정규화로 버린 척도 정보를 실제 제어장에 복원하고, 실제 open-weight
> LLM의 refusal 잔차에 적용하여 재검증했다. **두 설계 모두에서 칸토어 배치
> 고유의 이점은 확인되지 않았다.** 자세한 내용은 아래 §V2.

# V1: 칸토어 게이트 동역학계와 Neural ODE의 강건성 분석

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


---

# V2 — CantorGuard: 척도 보상 칸토어 장벽

**브랜치:** `cantor-guard-v2` · **최종 보고서:** [`docs/v2/FINAL_REPORT_KO.md`](docs/v2/FINAL_REPORT_KO.md)

## 한 문단 요약

V1은 칸토어 도함수를 `(3/2)ⁿ`으로 정규화해 **이진 마스크** `1_{K_n}`만
남겼고, 프랙탈 이점을 찾지 못했다. V2는 그 정규화가 버린
**레벨별 측도 집중과 `(3/2)^k` 크기 법칙**을 실제 제어장으로 복원했다.
각 칸토어 gap에 smoothstep 장벽을 놓되 level마다 동일 에너지를 준다.

* **증명:** 정리 A(레벨별 총 작용량 = `E₀`), 정리 B(peak = `3E₀(3/2)^k`,
  비 정확히 `3/2`), 정리 C(차단 조건).
* **그러나** 세 정리 모두 **폭 정합 대조군이 똑같이 만족**한다 — 배치와 무관.
* **합성 55,680 sim:** cantor는 무제어·상수·주기를 이기지만
  **중앙 단일장벽과 shuffled에는 진다.** center-anchored와는 `d_z = 0.049`.
* **실제 LLM(Qwen2.5-0.5B):** refusal direction이 **인과적**임을 확인
  (`+v` 0.625→0.875, `−v` →0.375). 동결 held-out 시험 10,800행에서
  **사전등록 기준 7개 중 6개 FAIL.**
* **실용적 기여:** 개입이 활성 노름의 ~15 %를 넘으면 출력 coherence가 무너지고
  **lexical 안전 지표가 그 붕괴를 "안전"으로 오독**한다.

## V2 재현 절차

```bash
julia --project=. test/v2/runtests_v2.jl                    # 27 305 assertions
julia --project=. scripts/v2/export_barrier_reference.jl    # 교차언어 기준표
julia --project=. scripts/v2/run_theory_validation.jl       # 정리 A/B/C, 명제 D/E
julia --project=. -t auto scripts/v2/run_synthetic_barrier.jl   # 55 680 sim (~10분)
PYTHONPATH=llm/src python3 scripts/v2/analyse_synthetic.py
```

LLM 단계 (Python 3.9+, torch 2.8, transformers 4.57; MPS/CUDA/CPU):

```bash
cd llm && PYTHONPATH=src python3 -m pytest tests/ -q && cd ..
PYTHONPATH=llm/src python3 scripts/v2/refusal_smoke.py qwen2.5-0.5b-instruct
PYTHONPATH=llm/src python3 scripts/v2/extract_and_fit_direction.py --model qwen2.5-0.5b-instruct
PYTHONPATH=llm/src python3 scripts/v2/tune_controllers.py --model qwen2.5-0.5b-instruct
PYTHONPATH=llm/src python3 scripts/v2/run_llm_test.py --model qwen2.5-0.5b-instruct   # 동결, ~50분
PYTHONPATH=llm/src python3 scripts/v2/run_statistics.py
PYTHONPATH=llm/src python3 scripts/v2/run_posthoc_degradation.py     # POST-HOC
PYTHONPATH=llm/src python3 scripts/v2/generate_v2_figures.py
```

`run_llm_test.py`는 `configs/v2/llm_test.yaml`의 SHA-256이
사전등록 값과 다르면 **실행을 거부한다.** 중간에 끊겨도 이어서 실행된다.

## V2 디렉터리

```
src/v2/          CantorBarrier.jl, BarrierDynamics.jl
llm/src/cantor_guard/   Python 패키지 (models, hooks, probes, threat_coordinate,
                        cantor_barrier, control_baselines, attacks, datasets,
                        safety_eval, harm_detector, statistics, io, generation)
llm/tests/       33 pytest assertions (배리어 수학 · 교차언어 · LLM hook)
configs/v2/      theory.toml, synthetic.toml, llm_test.yaml (FROZEN)
scripts/v2/      각 단계 실행 스크립트
results/v2/raw/  원자료 + .meta 사이드카 (유해 텍스트 없음)
results/v2/private/  유해 completion (gitignored, 커밋 금지)
figures/v2/      9개 그림 + CAPTIONS.md
docs/v2/         V1_TO_V2_RATIONALE, V1_ERRATA, MATHEMATICAL_THEORY,
                 LITERATURE_REVIEW, LLM_METHOD, PREREGISTRATION, RESULTS,
                 FINAL_REPORT_KO
```

## 안전 취급

유해 프롬프트에 대한 모델 출력은 **저장소에 커밋하지 않는다.**
추적 테이블에는 prompt hash와 스칼라 점수만 들어가며,
`assert_no_raw_completions`가 이를 강제한다(테스트 포함).


---

# V3 — 안전 판정 경계 불확실성 하의 다중척도 제어

**브랜치:** `cantor-guard-v3` · **최종 보고서:** [`docs/v3/FINAL_REPORT_KO.md`](docs/v3/FINAL_REPORT_KO.md)

## 한 문단 요약

V2는 **판정 경계가 정확히 알려져 있다**고 가정했다. V3는 그 가정을 실제
LLM에서 검증했고 — **거짓이었다.** 안전 경계는 층·토큰 위치·생성 단계·공격
계열에 따라 중앙값 **0.34 σ**, q95 **1.19 σ**, 최대 **1.77 σ** 이동한다
(layer 4에서 prefill → 첫 생성 토큰 사이 **1.65 σ**). 이는 threat 좌표로
칸토어 level-1 gap 폭의 **절반**이며, V2의 고정 calibration이 무시한 값이다.

* **정리 S (증명):** `V′_n(T_i(r)) = (3/2)V′_{n−1}(r)`.
  V2의 정리 A·B와 달리 **배치 불변이 아니다** — 폭-정합 대조군 잔차 0.82–0.96,
  칸토어 10⁻¹³. **칸토어 고유 성질을 처음 확보.**
* **동적 calibration:** 토큰-bin별 임계값이 held-out 경계오차를
  0.299 σ → **0.213 σ** (−28.9 %). 그러나 없애지는 못한다.
* **합성 Δ×ε 41,472 sim** (실현 예산 **±2 %** 정합):
  budget 0.60에서 `R_worst` = constant **0.524**, minimax 0.524,
  wide-central 0.439, shuffled 0.104, center-anchored 0.055,
  **cantor 0.050**, none 0.044. **칸토어가 최악.**
* **메커니즘 (82 배치):** 자기유사성 r = **+0.147** (기각);
  최장 무방비 구간 r = **−0.680**. 칸토어는 자기유사성 1.000이면서
  대조군 분포의 **35.8 백분위**.

> **정리 S는 참이지만 강건성과 무관하다. 경계 위치가 불확실할 때
> 최선의 귀납 편향은 다중척도 구조가 아니라 상태 무관성이다.**

**미완:** LLM에서 11개 controller의 Δ×ε 직접 시험은 수행하지 못했다
(합성 모형에서만). 한계 §32에 기록.

## V3 재현 절차

```bash
julia --project=. -e '
  include("src/v2/CantorBarrier.jl"); include("src/v3/CantorSelfSimilarity.jl")'   # 정리 S
julia --project=. -t auto scripts/v3/run_synthetic_uncertainty.jl   # 41 472 sim (~3분)
julia --project=. -t auto scripts/v3/analyse_mechanism.jl           # 메커니즘 (~7분)
PYTHONPATH=llm/src python3 scripts/v3/measure_boundary_shift.py     # 경계 이동 (~2분)
PYTHONPATH=llm/src python3 scripts/v3/fit_dynamic_calibration.py    # calibration (~4분)
PYTHONPATH=llm/src python3 scripts/v3/generate_v3_figures.py
```

## V3 디렉터리

```
src/v3/            CantorSelfSimilarity.jl (정리 S), RobustDynamics.jl,
                   V3Controllers.jl (L0–L10, wide-central·minimax 포함)
llm/src/cantor_guard_v3/  boundary_uncertainty.py, calibration.py, io3.py
configs/v3/        synthetic.toml (지표 도달가능성 근거 포함)
scripts/v3/        measure_boundary_shift, fit_dynamic_calibration,
                   run_synthetic_uncertainty, analyse_mechanism, generate_v3_figures
results/v3/        원자료 + provenance 사이드카 (유해 텍스트 없음)
figures/v3/        4개 그림 + CAPTIONS.md
docs/v3/           V2_LLM_AUDIT, V2_TO_V3_RATIONALE, MATHEMATICAL_THEORY,
                   BOUNDARY_SHIFT_ANALYSIS, RESULTS, FINAL_REPORT_KO
logs/v3/PROGRESS.md   phase checkpoint (미완 항목 포함)
```


---

# V3.1 — 방법론적 종결 (closure)

**브랜치:** `cantor-guard-v3.1` · **최종 보고서:** [`docs/v3_1/FINAL_REPORT_KO.md`](docs/v3_1/FINAL_REPORT_KO.md)

## 무엇을 고쳤는가

| V3 결함 | V3.1 |
|---|---|
| `L1_constant`가 실제로는 광폭 smoothstep 장벽 (range 1.500) | **진짜 상수** `S1_true_constant` 구현, 옛것은 `S2_global_smooth`로 보존 |
| `C0_fixed`가 V2 calibration이 아니라 pooled | 4종 분리 (`C_V2_LAST_PROMPT` 등) |
| `τ_mid`를 "safety boundary"라 호칭 | **projection midpoint**로 용어 후퇴 |
| 정리 S 퍼텐셜 offset 누락 | 정정 |
| 전역 Corollary S.1 (비 0.225–0.282) | **삭제**, 증명된 국소판 S.1′ (비 0.999–1.002) |
| L9가 스크립트마다 재적합 | `configs/v3_1/l9_frozen_weights.toml` 동결 |
| LLM direct test 부재 | **수행** |

## 새 정리 T

```
‖u′_k‖_∞ = 6·e_k/w_k² = 12·E₀·(9/2)^k        (측정/예측 비 1.000000)
S_u(Δ) ≤ ‖u′‖_∞·|Δ|,   진짜 상수는 ‖u′‖ = 0 ⇒ S ≡ 0
```

같은 예산에서 다척도 장벽(4428.7)은 광폭 장벽(6.0)보다 **738배** 민감하다.
**단, 정리 T도 배치 무관**이다 — 다척도가 취약한 이유는 설명하지만
칸토어가 shuffle과 다른 이유는 설명하지 않는다.

## 결과

**합성** (42,768 sim, 실현 예산 ±2 %): 최적은 상수(0.2606)가 아니라
**광폭 shaped barrier**(0.5244). 칸토어 0.0500으로 하위권.
→ **V3의 "상태 무관성이 최적" 주장 기각.**

**LLM direct test** (6,240행, 실현 예산 ±3 %, 지표 게이트 통과):

| 대조군 | Δ 안전도 | 95 % CI |
|---|---|---|
| shuffled | +0.0014 | [−0.0061, +0.0097] |
| center-anchored | +0.0028 | [−0.0027, +0.0097] |
| 무제어 | **+0.0285** | [+0.0144, +0.0451] |
| 진짜 상수 | **+0.0181** | [+0.0040, +0.0336] |

**판정: `C_PRACTICALLY_EQUIVALENT`** (자동, 동결 분류기).

## V3.1 재현

```bash
julia --project=. test/v3_1/runtests_v31.jl                    # 13 171 assertions
julia --project=. scripts/v3_1/fit_l9_frozen.jl                # L9 동결
julia --project=. -t auto scripts/v3_1/run_synthetic_v31.jl    # 42 768 sim
PYTHONPATH=llm/src python3 scripts/v3_1/run_llm_direct.py      # direct test (~3h)
PYTHONPATH=llm/src python3 scripts/v3_1/final_claim_check.py   # 자동 판정
PYTHONPATH=llm/src python3 scripts/v3_1/generate_v31_figures.py
```

## 미완 (정직하게)

모델 복제(TinyLlama) · behavioral boundary 추정 · mechanism 전체 격자 재실행.
[`FINAL_REPORT_KO.md`](docs/v3_1/FINAL_REPORT_KO.md) §31에 기록.
