# V3.3 사전분석계획

benchmark 실행 **전에** 확정하고 SHA-256으로 봉인한다.

---

## 1. 무엇을 뒤집지 않는가

**V3.2의 negative robustness 결과를 보존한다.**

> H1 (`R_C > R_M`): **NOT SUPPORTED.**
> Model A `C_PRACTICALLY_EQUIVALENT`(두 채점기), Model B
> `E_INCONCLUSIVE_SCORER_DEPENDENT`. 13개 정합 비교 중 칸토어 승리 **0개**.

V3.3은 이 결론을 재검정하지 않는다. 새 positive를 찾기 위해 조건을
바꾸지 않는다. **"칸토어가 LLM safety를 더 높였다"고 쓰지 않는다.**

## 2. V3.3의 질문

동일 safety에서 **구조적** 이득이 있는가 — 표현 복잡도, scale 확장,
검증 비용.

## 3. 복잡도 다섯 축 (절대 합산 금지)

`C1` 명시 파라미터 수 · `C2` 재귀/절차 생성기 기술 비트 · `C3` 직렬화 바이트
· `C4` 구성/확장 비용 · `C5` 검증 의무 수.

**Kolmogorov 복잡도를 측정했다고 주장하지 않는다.** 고정 codec 하의
**description length**다 (`configs/v3_3/encoding.json`, 모든 family 동일).

## 4. PRIMARY 구조 지표 3개

| | 지표 |
|---|---|
| **M1** | canonical symbolic description bits |
| **M2** | certificate proof obligations |
| **M3** | point-query resident words |

SECONDARY: serialized/gzip bytes, AST nodes, construction/verification time,
materialised words. **gzip 단독 결론 금지** (STOP A).

## 5. 필수 대조군

| 대조군 | 이유 |
|---|---|
| `shuffled_seeded` | seed 하나로 재생성 가능 → 압축은 Cantor 고유가 아님 (**STOP B**) |
| `center_anchored_seeded` | 동일 |
| `periodic_procedural` | 닫힌 형식 주소지정 가능 |
| **`recursive_non_cantor`** | 이득이 **재귀 일반**인지 Cantor 고유인지 분리 (**STOP C**) |
| `shuffled_explicit` | D1 baseline |
| `learned_minimax_explicit` | D1 baseline |

## 6. 검증 의무 계수 규칙 — 대조군에 공정하게

**seeded shuffle은 Cantor gap multiset의 치환이고, 치환은 multiset을
보존한다.** 따라서 multiset에만 의존하는 성질

`P1 에너지 보존` · `P2 폭 법칙` · `P3 peak 한계` · `P4 slope 한계` · `P5 방향성`

은 **shuffle에서도 동일하게 싸게** 증명된다. 이들을 shuffle에 비싸게
계상하지 않는다. 분리되는 것은 **위치 의존** 성질뿐:

* `P6` cross-scale identity — 비재귀 layout에서는 **비싼 것이 아니라 거짓**
* `P8` address soundness — 성분 간 구조 관계가 없으면 전수 방문 필요

## 7. Lower bound 모델 (제한 명시)

**black-box interval oracle**에서만: adversary가 임의 성분 1개를 손상시킬 수
있고 P가 성분별로 검사되어야 하면 worst-case **Ω(N_n)** 질의.
**procedural control에는 적용하지 않는다** (STOP G).

## 8. Pareto 규칙

A dominates B iff `R_A ≥ R_B − δ_R` **그리고** `C_A < C_B`, 최소 하나 strict.
`δ_R` = **SESOI 0.03** (V3.1부터 불변). **안전도 부족은 어떤 구조적 저렴함으로도
보상되지 않는다.** `J = R − λC`는 secondary only — λ를 사후 선택하면 원하는
controller를 이기게 만들 수 있다 (§57).

## 9. 판정 규칙 (`final_claim_check_v33.py`, 봉인)

우선순위:

1. safety 동등 미성립 → **F_INCONCLUSIVE** (STOP F)
2. seeded 대조군 없음 → **F** (STOP B) / recursive non-Cantor 없음 → **F** (STOP C)
3. Cantor가 **모든** strong control을 primary 지표에서 strict하게 이기고,
   depth 증가에도 유지되며, recursive non-Cantor가 이를 따라오지 **못하면**
   → **A_CANTOR_SPECIFIC_STRUCTURAL_ADVANTAGE**
4. Cantor와 recursive non-Cantor가 **함께** 비재귀 layout을 이기면
   → **B_RECURSIVE_SELF_SIMILARITY_ADVANTAGE**
5. 인증 비용만 → **D** / 기술 길이만 → **C**
6. 그 외 → **E_NO_STRUCTURAL_ADVANTAGE**

strict 우위 판정 margin **5 %** (동률 방지).

## 10. Safety gate 상속 조건 (STOP D)

symbolic evaluator가 explicit과 **수치 조건수 한계 내에서 일치**해야
V3.2의 safety 결과를 상속한다. 불일치하면 소규모 LLM 확인 run을 수행한다.

측정된 오차 법칙: 모든 경로가 `C·3^n`으로 증가하며
`C ≈ 2e-16`(symbolic vs explicit), `4e-15`(정리 S), `6e-16`(general IFS).
exact Fraction 기준에서 symbolic이 explicit보다 **나쁘지 않다**
(n=18에서 4.5e-10 vs 4.9e-10).

## 11. 하지 않을 것

* 결과를 본 뒤 지표·margin·판정 규칙 변경
* Cantor에 유리하도록 지표 설계 (ICR은 secondary로만)
* gzip 단독 근거
* V3.2 negative result 삭제·약화
* materialised와 symbolic 복잡도 혼동 (§29)
