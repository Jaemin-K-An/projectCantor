# V3 PROGRESS

## [PHASE 0-1] repository audit + V2 LLM methodological audit
**Objective:** V1/V2 이해, V2 LLM의 방법론적 결함 규명.
**Completed:** `docs/v3/V2_LLM_AUDIT.md`. 결함 A(ASR 구조적 퇴화),
B(실현 예산 6배 불일치), C(calibration 이동), D(출력 붕괴), E(단일 모델).
**Unexpected:** V2 LLM 실험이 **어느 방향으로도** 결론을 지지하지 않음을 확인.
**Alters preregistered design?** 예 — V3는 V2 LLM 결과를 "미결"로 취급.
**Files:** `docs/v3/V2_LLM_AUDIT.md`  **Runtime:** —

## [PHASE 5] Cantor self-similarity mathematics
**Objective:** 배치 불변이 **아닌** 칸토어 고유 성질을 찾는다.
**Completed:** **정리 S** 유도·증명·검증. `V′_n(T_i(r)) = (3/2)V′_{n−1}(r)`,
잔차 `3.6e−15 … 7.1e−13` (n=3..8).
**New finding:** 폭-정합 대조군 잔차 0.82–0.96 → **정리 S는 칸토어 고유.**
V2의 정리 A·B와 결정적으로 다르다.
**Files:** `src/v3/CantorSelfSimilarity.jl`  **Runtime:** ~1 min

## [PHASE 3] boundary non-stationarity measurement
**Objective:** §37 — 경계가 실제로 움직이는가?
**Completed:** 180 조건(6 layer × 5 template × 6 token-bin), 19,200 사영.
**Result:** `|Δ_norm|` median **0.339 σ**, q95 **1.187**, max **1.773**.
layer 4에서 prefill(−0.603) → gen_1(+1.046) = **1.65 σ 도약**.
**New finding:** V2 defect C가 경험적으로 확인됨. 1 σ ≈ threat 좌표 0.17 ≈
level-1 gap 폭의 절반.
**Alters design?** 아니오 — 오히려 V3 전제를 확증.
**Files:** `scripts/v3/measure_boundary_shift.py`,
`llm/src/cantor_guard_v3/boundary_uncertainty.py`,
`docs/v3/BOUNDARY_SHIFT_ANALYSIS.md`  **Runtime:** 103 s

## [PHASE 4] dynamic calibration
**Objective:** §38 — 더 나은 calibration이 기하보다 중요한가?
**Completed:** C0/C1/C2 적합, held-out DEV 평가. 19,200 사영.
**Result:** C0 0.299 σ → C2_token_bin **0.213 σ** (−28.9 %).
in-sample 0.000 vs DEV 0.213 → 일반화 간극 실재.
**Files:** `llm/src/cantor_guard_v3/calibration.py`,
`scripts/v3/fit_dynamic_calibration.py`  **Runtime:** 250 s

## [PHASE 6] synthetic Δ×ε joint uncertainty
**Objective:** 경계·공격 합동 불확실성에서 minimax 강건성 비교.
**Completed:** 41,472 sim. 11 controller × 9 Δ × 6 ε × 4 attack × 2 x₀ × 3 예산.
**실현 예산 ±2 % 정합** (V2 defect B 수정).
**Failed:** 이진 주 종점이 퇴화(45 % 셀이 0, 40 %가 1) → **사전 검출**,
graded `safe_frac`으로 교체하고 근거를 config에 기록.
**Result:** budget 0.60에서 constant 0.524, minimax 0.524, wide-central 0.439,
… center-anchored 0.055, **cantor 0.050**, none 0.044.
**Unexpected:** Cantor가 **최악**. 경계 불확실성이 다척도를 돕지 않고 처벌.
**Files:** `src/v3/RobustDynamics.jl`, `src/v3/V3Controllers.jl`,
`configs/v3/synthetic.toml`, `scripts/v3/run_synthetic_uncertainty.jl`
**Runtime:** 151 s (8 threads)

## [PHASE 6b] mechanism analysis
**Objective:** §45·§46 — 자기유사성인가 anti-clustering인가?
**Completed:** 82 배치 회귀 + 강화된 L9 탐색(137 후보 + 좌표 정제).
**Result:** self-similarity r = **+0.147** (기각), max_weak_run r = **−0.680**,
mean_nn −0.608. Cantor 자기유사성 1.000이나 백분위 **35.8**.
L9 비균일 최적해 0.5225 (균일 0.4613) → 초기 탐색이 약했음을 수정.
**New finding:** **정리 S는 참이지만 강건성과 무관하다.**
**Files:** `scripts/v3/analyse_mechanism.jl`  **Runtime:** ~7 min

## [PHASE 14-15] figures + report
**Completed:** 4개 그림 + CAPTIONS.md, `docs/v3/{RESULTS,FINAL_REPORT_KO}.md`.

## NOT COMPLETED — 정직하게 기록
* **PHASE 10–13 (LLM controller DEV/freeze/test/statistics).** V3는 LLM에서
  **경계 이동과 calibration만** 측정했고, 11개 controller의 Δ×ε 비교는
  **합성 모형에서만** 수행했다. 합성의 Δ 범위를 LLM 실측 q90에 맞췄으나
  **LLM 직접 검정은 아니다.** `docs/v3/FINAL_REPORT_KO.md` §32 한계 1에 기록.
* **model-family replication.** TinyLlama-1.1B를 내려받았으나 미수행.
