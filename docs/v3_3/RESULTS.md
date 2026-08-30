# V3.3 결과

> V1/V2/V3/V3.1/V3.2의 결과는 삭제하거나 약화하지 않았다.
> **H1 (`R_C > R_M`)은 NOT SUPPORTED로 고정**되어 있고 V3.3은 이를 재검정하지 않았다.

---

## 1. Safety gate — V3.2 결과를 상속할 수 있는가 (STOP D)

구조적 주장이 "성능 중립"이려면 기호적 제어기가 **LLM 시험이 실제로 적용한
그 제어**를 계산해야 한다.

LLM 시험이 쓴 차수는 `n = 5`. 200,000점에서:

| n | mutual rel. error | explicit vs exact | symbolic vs exact | `V(1)−nE₀` |
|---|---|---|---|---|
| 3 | 6.2e−15 | 6.1e−15 | 1.8e−15 | **0** |
| **5** | **1.9e−14** | 4.9e−14 | 5.0e−14 | **0** |
| 8 | 1.2e−12 | 1.4e−12 | 1.3e−12 | **0** |
| 12 | 7.9e−11 | 6.6e−11 | 6.8e−11 | **0** |
| 15 | 1.1e−09 | 1.7e−10 | 1.2e−10 | **0** |

`n = 5`에서 두 구현은 **1.9e−14** 상대오차로 일치한다 — 실현 개입량
(잔차 노름의 약 2 %)보다 **12자릿수 아래**다. 따라서
**V3.2 Model A의 `C_PRACTICALLY_EQUIVALENT`를 상속한다.**

**한계 명시:** `n ≥ 12`에서는 두 부동소수 경로 어느 쪽도 정확하지 않다
(§MATHEMATICAL_THEORY §4). 어떤 안전 주장도 그 깊이에 의존하지 않는다.

## 2. PRIMARY 구조 지표 (n = 20)

| family | **M1** bits | **M2** 의무 (봉인) | **M2** (post-hoc) | **M3** words |
|---|---:|---:|---:|---:|
| `cantor_recursive` | 208 | **125** | 125 | **4** |
| `recursive_non_cantor` | 208 | **125** | 125 | **4** |
| `periodic_procedural` | **112** | 3,145,805 | **104** | **4** |
| `shuffled_seeded` | **184** | 3,145,805 | 3,145,805 | 3,145,725 |
| `center_anchored_seeded` | **184** | 3,145,805 | 3,145,805 | 3,145,725 |
| `shuffled_explicit` | 201,326,536 | 3,145,805 | 3,145,805 | 3,145,725 |
| `learned_minimax_explicit` | 648 | 7,340,024 | 7,340,024 | 3,145,725 |

### M1 — **칸토어가 진다**

`periodic`(112 bits)과 `shuffled_seeded`(184)가 칸토어(208)보다 **짧다.**
seeded shuffle은 *알고리즘 + seed + n*으로 재생성되므로 압축된다.

> **STOP B가 예견한 그대로다.** "칸토어가 압축된다"는 관찰은
> Cantor-specific 결과가 **아니다.** 절차적이면 압축된다.

### M3 — **동률**

칸토어·재귀 비칸토어·periodic 모두 `O(1)`(4 words). 명시적/seeded shuffle만
`Θ(2^n)`. 이것은 **닫힌 형식 주소지정 가능성**의 이득이지 칸토어의 이득이 아니다.

### M2 — 봉인 계수에서는 칸토어 우세, **정정 후 소멸**

봉인된 인증 모델은 `periodic`을 열거적 분기에 넣어 `Θ(2^n)` 의무를 부과하면서
동시에 `O(1)` point query를 인정했다 — **모순이고, 대조군을 과소평가하는
(즉 칸토어에 유리한) 오류였다.**

`L3_periodic`의 폭 누적합은 닫힌 형식이므로 주소 사상이 `O(n)`이다.
정정하면 **periodic 104 < 칸토어 125.**

## 3. 척도 전이 — 유일하게 실제로 분리되는 축

| family | max `E_scale` | 새 파라미터 | zero-shot |
|---|---:|---:|---|
| **`cantor_recursive`** | **6.06e−09** | **0** | **YES** |
| **`recursive_non_cantor`** | **7.18e−10** | **0** | **YES** |
| `periodic_procedural` | 4.50 | 24,573 | no |
| `shuffled_seeded` | 3.87e+02 | 24,573 | no |
| `center_anchored_seeded` | 5.62e+02 | 24,573 | no |

**정확한 척도 전이는 재귀 family에만 존재하고, `periodic`도 실패한다**
(상대오차 4.5 = 450 %).

그러나 **칸토어가 재귀 비칸토어보다 낫지 않다.** 오히려 `ρ = 0.28` 쪽이
조건수가 한 자릿수 좋다 (7.2e−10 vs 6.1e−09) — 칸토어의 `3^n` 증폭 때문이다.

## 4. 판정

```
봉인된 판정   : B_RECURSIVE_SELF_SIMILARITY_ADVANTAGE
POST-HOC 정정 : E_NO_STRUCTURAL_ADVANTAGE
```

**post-hoc 정정이 더 정확한 계수이므로 그쪽을 최종 문구로 채택한다.**
정정은 칸토어에 **불리한** 방향이며, 유리한 방향의 정정이었다면 채택하지
않았을 것이다.

### 사전등록 3개 지표 기준

| 지표 | 칸토어가 모든 strong control을 이기는가 |
|---|---|
| M1 canonical bits | **아니오** (periodic·seeded shuffle에 짐) |
| M2 certificate obligations (정정) | **아니오** (periodic이 더 쌈) |
| M3 point-query words | **아니오** (periodic과 동률) |

## 5. 증거 행렬

| family | 짧은 기술 | 싼 인증서 | `O(1)` 질의 | **정확한 척도 전이** |
|---|---|---|---|---|
| `cantor_recursive` | YES | YES | YES | **YES** |
| `recursive_non_cantor` | YES | YES | YES | **YES** |
| `periodic_procedural` | YES | YES | YES | **no** |
| `shuffled_seeded` | YES | no | no | no |
| `shuffled_explicit` | no | no | no | no |

**칸토어 행과 재귀 비칸토어 행이 모든 열에서 동일하다.**
칸토어 고유 이득은 존재하지 않는다.

## 6. 부수 결과 — 기호적 평가기는 실제로 유용하다

칸토어 고유는 아니지만, 재귀 표현 자체는 실질적 이득이 있다:

* 저장: `O(1)` vs `Θ(2^n)` — `n=20`에서 4 words vs 3,145,725 words
* 인증: 125 의무 vs 3,145,805 (구조 없는 layout 대비 **25,000배**)
* 척도 확장: 재학습·재최적화·새 파라미터 **없이** `n → n+1`
* 조건수: 정수 삼진 주소 유지로 명시적 구현보다 **나쁘지 않음**

이는 **비구조적 layout 대비** 이득이며, 절차적·닫힌형식 layout 대비가 아니다.
