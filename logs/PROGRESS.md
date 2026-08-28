# PROGRESS — phase checkpoints

각 PHASE 종료 시 §34의 항목을 기록한다.

---

## [PHASE 0] Repository survey

**목표:** 기존 저장소 조사 및 보존.

**완료:** 저장소에는 `미적분 탐구 보고서.pdf`(27쪽) 한 개만 있었다. 텍스트를
추출하여 `original/original_report_text.txt`로, 부록의 Julia 코드를
`original/original_code.jl`로 보존했다. 원 PDF는 `original/`에 복사하고
루트에도 남겼다. **원 파일은 삭제하거나 편집하지 않았다.**

**관찰:** 원 코드 전문을 확보했으므로 재현이 추측이 아니라 재실행이 되었다.

**예상과 다른 점:** 저장소에 코드가 전혀 없어 "기존 구조와 통합"할 대상이
없었다. 전부 신규 작성.

**실행 명령:** `pip3 install pypdf`, 텍스트 추출 스크립트.

---

## [PHASE B] Cantor core + tests  *(코드 작성이므로 A보다 먼저 수행)*

**목표:** 유한 칸토어 구성의 성질을 assertion으로 고정.

**완료:** `src/CantorCore.jl`, `src/Controls.jl`, `src/Perturbations.jl`,
`src/Metrics.jl`, `src/Dynamics.jl`, `src/NumericalAnalysis.jl`,
`src/Utils.jl`, `src/NeuralODEModel.jl`. `test/runtests.jl` **11 687 assertions
전부 통과**.

**실패 항목:** 최초 작성 시 3건 실패 → 전부 **테스트 쪽이 틀렸음**을 확인:
1. `first_hitting_time([0,1],[0,1])` — h가 `S`를 뛰어넘으므로 `Inf`가 정답.
2. 매끄러운 게이트의 **sup 거리는 β→∞에서 수렴하지 않는다** (끝점에서 ½에
   고정). L1 거리로 수렴을 서술하도록 정정.
3. 재귀 판정과 구간 판정이 `7/9`, `8/9`에서 불일치.

**새롭게 발견된 문제:** (3)은 테스트 버그가 아니라 **원 알고리즘의 실제
조건수 결함**이었다. `endpoint_instability(n)`을 추가하여 정량화하고
`in_cantor_set_stable`을 도입, 실험은 안정판을 쓰고 재귀판은 재현용으로 보존.

**다음 단계 수정:** 없음.

**생성 파일:** `src/*.jl`, `test/runtests.jl`, `configs/*.toml`.

**실행 명령:** `julia --project=. test/runtests.jl` (~50 s)

---

## [PHASE A] Baseline reproduction + audit

**목표:** 원 결과 재현 및 감사.

**완료:** 29.80 / 98.17 / 97.43 % (원: 29.8 / 98.2 / 97.4). h₀ = 0.15가
`(1/9, 2/9)` 안에 있음을 프로그램으로 추적. 게이트 개방 시간비율
C₃ 0.167 %, C₅ 0.133 %. 솔버 4종 교차검증.

**실패 항목:** `git_commit()`이 최초 실행 시 커밋이 없어 `nogit`을 기록했다
(이후 커밋 생성으로 해결).

**관찰:** 무필터 29.8 %가 `(2/π)asin(1/(6R))`의 닫힌형과 일치.

**예상과 다른 점:** **Vern9(atol 10⁻¹²)가 h₀ = 0.75에서 76초 후 실패**하고
정확히 `7/9`에서 멈췄다. 예상하지 못한 결과였고, 고정스텝 RK4를 주 적분기로
택한 사전 결정을 사후 정당화했다.

**새롭게 발견된 문제:** 슬라이딩 모드 / Filippov 해 (→ MATHEMATICAL_ANALYSIS §7).

**생성 파일:** `results/raw/baseline_{reproduction,h0_audit,solver_check}.csv`,
`docs/BASELINE_AUDIT.md`, `results/tables/baseline_summary.md`.

**실행 명령:** `julia --project=. scripts/reproduce_baseline.jl` (~3 min)

---

## [PHASE C] Numerical analysis

**목표:** `∫C'_n` 구적 실패의 원인 규명.

**완료:** 84개 `(n, Δx)` 셀 × 4개 구적법. `ρ = Δx·3ⁿ` 붕괴 확인.
끝점 조건수 측정.

**관찰:** 오차는 **셈 오차**이며 부호가 규칙에 의존한다 (좌끝점 0.830 vs
중점 1.246, 같은 셀).

**예상과 다른 점:** **적응형 심프슨이 균일격자보다 훨씬 나쁘다**
(n = 10에서 0.0547). 예상하지 못했으나 사후에 명확히 설명된다.
또 끝점 오분류율이 `1/2`로 수렴하는 것이 증폭이 아니라 반올림 방향의
동전던지기임을 확인 (초기 가설이 틀렸다).

**생성 파일:** `results/raw/numerical_{integration,structure_aware,endpoint_conditioning}.csv`.

**실행 명령:** `julia --project=. scripts/run_numerical_analysis.jl` (~1 min)

---

## [PHASE D] Parameter sweep

**목표:** 단일 h₀ 실험을 전 상태공간 sweep으로 대체.

**완료:** 196 020회 (11 × 99 × 6 × 6 × 5), 8스레드 278.5 s, 1.42 ms/case.

**관찰:** `R_safe`의 `n` 의존이 **U자형** — n=1 최대(0.9676), n=7 최소(0.9543).
평탄측도는 단조증가함에도 그렇다.

**예상과 다른 점:** 비단조성 자체는 가설 H1이 예측했지만, **n=1이 전 구간에서
최대**이고 큰 n에서 다시 회복한다는 U자 형태는 예상하지 못했다.

**새롭게 발견된 문제:** 왜 회복하는가 → POST-HOC 도달시간 분석으로 규명.

**생성 파일:** `results/raw/sweep_full.csv` (35 MB).

**실행 명령:** `julia --project=. -t auto scripts/run_parameter_sweep.jl` (~5 min)

---

## [PHASE F] Ablation + statistics

**목표:** 칸토어 배치 자체의 이득을 측도 효과와 분리.

**완료:** 113 625회 (G0–G5, n = 1…8, 25 h₀ × 3 A × 3 ω, G2·G5는 30시드),
155 s. 매끄러운 게이트 3 150회. 부트스트랩 CI 4 000 resample.

**관찰:** `n ≥ 2`에서 G1의 백분위 순위 41–52 (G2 대비) / 14–36 (G5 대비).
**G4_central이 큰 n에서 최고**(0.9820 at n=8).

**실패 항목(가설):** **H4 기각.** 칸토어의 프랙탈 배치가 측도정합 대조군보다
일관된 이득을 주지 않는다.

**예상과 다른 점:** G5(위상만 섞은 대조군)가 **칸토어를 이긴다**. 가장
엄격한 대조군이 가장 좋은 성적을 낸 것은 예상 밖이었다.

**생성 파일:** `results/raw/ablation_{main,smooth}.csv`,
`results/tables/ablation_{stats,delta_grid}.csv`.

**실행 명령:** `julia --project=. -t auto scripts/run_ablation.jl` +
`run_ablation_stats.jl` (~5 min)

---

## [PHASE G] Mathematical analysis

**목표:** 시뮬레이션과 독립적으로 불변성을 증명.

**완료:** 정리 1 (S°의 전방불변성 + 지수수렴, δ 무관), 정리 2 (일반 게이트의
나구모 조건 `B·g(∂S) ≤ α/6`), 따름정리 2.1 (매끄러운 게이트의 `B ≤ α/3` 한계),
명제 3·4 (도달시간 상·하계). 4 704 + 128 구성으로 회귀검증.

**관찰:** `max|h − 선형해| = 2.8×10⁻¹⁴`, A = 1000에서도 동일.

**예상과 다른 점:** 닫힌집합 `S`가 아니라 **열린집합 `S°`**에 대해서만
불변성을 서술해야 함을 발견 (닫힘 규약에서 `g(1/3) = 1`). 처음에는 `S`에
대해 쓰려다 반례를 만나 정정.

**생성 파일:** `docs/MATHEMATICAL_ANALYSIS.md`,
`results/raw/math_{invariance_check,smooth_invariance}.csv`.

**실행 명령:** `julia --project=. scripts/run_math_analysis.jl` (~2 min)

---

## [POST-HOC] Hitting-time analysis

**⚠ 이 분석은 사후(post-hoc)다.** PHASE D의 비단조성과 PHASE F의 G4 우위를
**본 뒤에** 설계했다. 사전등록된 어떤 결론도 이 분석에 의존하지 않으며,
목적은 관측을 **설명**하는 것이지 확립하는 것이 아니다.

**완료:** 명제 3·4의 상·하계와 측정 `τ_S` 비교. 726개 격자점에서
**하한 위반 0건**.

**관찰:** `τ_S(n=0) ≈ τ_S(n=1)`; median `τ_S`가 n=7에서 최대 1.301.

**생성 파일:** `results/tables/posthoc_hitting.{csv,md}`.

---

## [PHASE H+I] Neural ODE + OOD benchmark

**목표:** 학습 파라미터가 있는 실제 Neural ODE를 구현하고 사전 고정된 ID/OOD로
평가.

**완료:** `f_θ` = MLP(2 545 파라미터), 8개 게이트 변형을 **동일 초기화·동일
데이터·동일 예산**으로 학습. 손실 231–1132배 감소. ID 144조건 + OOD 145조건
+ adversarial 200회 탐색 × 8모델. RK4 vs 적응형 교차검증 192건.

**실패 항목:**
1. 1차 벤치마크가 `maxiters = 10⁸`으로 90분 이상 정지 → **중단**.
   학습 파라미터는 이미 직렬화되어 있어 재학습 없이 복구.
   `run_neural_benchmark.jl`을 분리 작성하고 적분기를 RK4로 교체, 적응형은
   상한을 둔 교차검증으로 강등. **이 변경은 사후이며
   `docs/EXPERIMENT_PROTOCOL.md` §2.1에 명시했다.**
2. `verbose = false`가 OrdinaryDiffEq v7에서 제거됨 → 제거.
3. `groupedbar`는 Plots가 아니라 StatsPlots의 함수 → 의존성 추가 대신
   `grouped2` 헬퍼를 직접 구현.

**관찰:** 게이트는 **진폭 외삽 OOD 하나에서만** 효과가 있다 (0.754 → 0.989).
파형 OOD에서는 무게이트가 근소 우위. 칸토어는 무작위 시드 3개의 한가운데.

**예상과 다른 점:**
1. **하드 게이트가 학습을 방해하지 않았다.** §21의 예상과 반대로 기울기 노름이
   전 변형에서 동일(5.7–6.0e−2). 게이트가 학습 대상이 아니기 때문이다.
2. **게이트 기하가 수치적 적격성을 좌우한다.** `random_n3_s1`의 스위칭면
   0.492·0.529가 학습된 끌개 0.4996을 감싸 적응형 솔버 실패율 92 %.
   칸토어는 정리 1에 의해 `S` 내부에 스위칭면이 **있을 수 없어** 1/24.
   사전에 측정 계획조차 없던 양이다.

**새롭게 발견된 문제:** 없음 (남은 것은 §17 한계로 이관).

**생성 파일:** `results/raw/neural_{training,benchmark,solver_crosscheck}.csv`,
`results/processed/trained_params.jls`, `results/tables/neural_summary.{csv,md}`.

**실행 명령:** `julia --project=. scripts/run_neural_ode.jl` (57 min) →
`julia --project=. scripts/run_neural_benchmark.jl` (11 min)

---

## [FIGURES / 최종]

**완료:** 16개 그림 + `figures/CAPTIONS.md`. 테스트 **11 794 assertions** 통과
(스키마 검증과 Neural ODE 평가기 검증 추가). `docs/FINAL_REPORT_KO.md` 작성.

**실행 명령:** `julia --project=. scripts/generate_all_figures.jl`
