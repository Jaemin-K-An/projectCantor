# V3.2 사전분석계획 (PRE-ANALYSIS PLAN)

이 문서와 여기 지정된 파일들은 **`D_test` 데이터를 한 줄도 생성하기 전에**
확정되고 SHA-256으로 봉인된다. 실행 스크립트는 봉인 해시와 대조하며,
불일치하면 `D_test`를 건드리지 않고 중단한다.

---

## 1. 연구 질문

> 동일한 **실현** 개입 예산에서,
> **(Q1)** 다척도 상태의존 개입이 무개입 및 진짜 상수보다 안전에 기여하는가?
> **(Q2)** 칸토어의 **재귀적 배치**가 폭·에너지 정합 대조군보다 **추가** 이점을 주는가?

**Q2가 주된 질문이다.** V1–V3.1 전체가 답하려 한 것이 이것이다.

## 2. 주요 종점

`safe_lex32` — `metrics32.safe_score32`, [0,1].
검증: control set 정확도 1.000 (in-sample), degenerate 게이트 통과.

**보조 종점** `safe_ext` — `protectai/distilroberta-base-rejection-v1`
(제3자, out-of-sample 정확도 0.750). **primary로 승격하지 않는다.**

## 3. 추론 단위 — **goal**

모든 신뢰구간은 **goal 군집 부트스트랩** (20,000회). 셀 단위 부트스트랩은
감사 목적으로만 병기하며 어떤 주장에도 쓰지 않는다.
근거: [`STATISTICAL_CORRECTION.md`](STATISTICAL_CORRECTION.md).

## 4. 비교 집합

**결정적(정합) 대조군** — 배치만 다르고 폭·에너지는 같다:
`T5_shuffled`, `T6_center_anchored`, `T4_periodic`, `T3_wide_central`

**약한 baseline** — 구조 자체의 효과:
`T0_none`, `T1_true_constant`, `T2_global_smooth`, `T8_minimax`

## 5. SESOI 및 동등성

**δ = 0.03** — V3.1에서 데이터를 보기 전에 정한 값, **변경하지 않는다.**
동등성 판정: 군집 95 % CI가 `[−0.03, +0.03]`에 **완전히 포함**될 때만.
(통상 TOST의 90 % CI보다 보수적이다.)

## 6. 검정력 — 사전 계산

측정된 군집간 SD로부터 (`results/v3_2/tables/power_plan.json`):

| 층 | SD | `n=50` 반폭 | 검정력 |
|---|---|---|---|
| 정합 대조군 | 0.0199 | **0.0055** | +0.010 검출 **94.5 %** |
| 약한 baseline | 0.0498 | 0.0138 | +0.030 검출 **98.9 %** |

**사전 명시된 한계:** baseline **+0.018** 검출에는 61 goal이 필요하고
`n = 50`에서 검정력은 **72 %**다. 해당 비교의 null은
**INCONCLUSIVE(검정력 부족)**로 보고하며 **동등성의 증거로 쓰지 않는다.**

> 분류기는 이 규칙을 코드로 강제한다: 사전 계산된 SD로 검정력 < 80 %인
> 비교는 `equivalent` 판정을 낼 수 없고 `E_INCONCLUSIVE`가 된다.

## 7. 공정성 제약

`η`는 **`D_budget`에서만** 적합해 목표 `C_rms = 0.02`(허용 ±3 %)에 맞춘다.
`q_cap = 0.05`. 최종 시험은 `η`를 **읽기만** 한다.
`D_test`에서 측정된 `C_rms`는 **일반화 결과**로 보고하며 **재조정하지 않는다.**
±3 %를 벗어난 계열은 정합 비교에서 **제외**하고 그 사실을 보고한다.

## 8. 자동 판정 규칙 (`scripts/v3_2/final_claim_check_v32.py`, 봉인됨)

우선순위 순으로 적용한다:

1. 정합 대조군 중 **하나라도** 칸토어를 이기고 CI가 0을 배제 → **D_CANTOR_INFERIOR**
2. 칸토어가 **모든** 정합 대조군을 이기고 CI가 0을 배제 → **A_CANTOR_SPECIFIC_POSITIVE**
3. 모든 정합 CI가 ±δ 안 **이지만** 검정력 부족 → **E_INCONCLUSIVE**
4. 모든 정합 CI가 ±δ 안 **이고** 검정력 충분:
   * 무개입·진짜 상수 **둘 다** 이김 → **B_MULTISCALE_BUT_NOT_CANTOR_SPECIFIC**
   * 그 외 → **C_PRACTICALLY_EQUIVALENT**
5. 그 외 → **E_INCONCLUSIVE**

**채점기 강건성:** 두 채점기의 판정이 다르면 최종 판정은
**`E_INCONCLUSIVE_SCORER_DEPENDENT`**로 강등한다.

## 9. 복제

Model A `qwen2.5-0.5b-instruct` (float32), Model B `olmo2-1b-instruct` (float16).
Model B는 사전심사 5개 게이트를 모두 통과했다
([`MODEL_PRESCREEN.md`](MODEL_PRESCREEN.md)). 각 모델은 **자기 방향·자기 교정·
자기 이득**을 쓴다. 복제 성공 = 두 모델의 판정 범주가 같음.

## 10. 하지 않을 것 (금지)

* 최종 시험 후 SESOI·임계값·판정 규칙 변경
* positive가 나올 때까지 시험 반복
* negative result 삭제 또는 축소 서술
* `D_test`에서 `η` 재조정
* raw harmful prompt/completion을 저장소에 커밋
* V1/V2/V3/V3.1 history rewrite 또는 force push

## 11. 봉인 대상

`configs/v3_2/PRE_ANALYSIS_FREEZE.json`에 다음의 SHA-256을 기록한다:
`split.json` · `frozen_qwen2.5-0.5b-instruct.json` ·
`frozen_olmo2-1b-instruct.json` · `evaluator_control_set.json` ·
`final_claim_check_v32.py` · `cluster_stats.py` · `metrics32.py` ·
`run_final_test.py` · 이 문서.
