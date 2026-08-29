# 척도 보상 칸토어 장벽을 이용한 언어모델 잔차 상태의 강건 안전 제어

*Scale-Compensated Cantor Barrier Steering for Robust Residual-State Control
in Language Models*

> **제목에 관하여.** §56의 후보 중 "CantorGuard: … Fractal Barrier Steering"은
> 프랙탈 구조의 효과를 암시한다. **본 연구는 그 효과를 확인하지 못했으므로**
> 우수성을 암시하는 표현을 제목에서 사용하지 않는다.

---

## 1. 초록

V1은 칸토어 계단 함수의 도함수를 `(3/2)ⁿ`으로 정규화한 **이진 게이트**
`g = 1_{K_n}`을 컨트롤러로 사용했고, 측도 정합 대조군 대비 프랙탈 고유의
이점을 찾지 못했다. V2는 그 정규화가 버린 정보 — **레벨별 측도 집중과
`(3/2)^k` 크기 법칙** — 을 실제 제어장(control field)에 복원하고,
같은 질문을 다시 물었다.

**설계.** 각 칸토어 gap `I_{k,j}`에 smoothstep 장벽을 놓되, level마다 동일한
에너지 `E₀`를 배분한다 (`e_k = E₀/2^{k-1}`). 컨트롤러는 `u_C = −V_C'`이며
V1과 반대로 **gap 안에서** 작동한다.

**증명한 것.**
- **정리 A.** 모든 level의 총 제어 작용량이 `E₀`로 동일하다.
- **정리 B.** 국소 최대 제어력이 정확히 `3E₀(3/2)^k`, 비는 `3/2`.
- **정리 C.** `ηV'(r*) > sup f + A`이면 진폭 `A`의 공격은 `r*`를 넘지 못한다.

**그러나 정리 A·B의 증명에는 gap의 배치 순서가 등장하지 않는다.** 폭
multiset을 유지한 채 순서만 섞은 대조군도 두 정리를 정확히 똑같이 만족한다.
따라서 두 정리로부터 칸토어 우수성을 주장할 수 없으며, 이 구분을 보고서
전체에서 유지한다.

**검증한 것.**
1. **이론** (정확 계산, 적분 없음): `cantor` AUC_log 1.6323 vs
   `center-anchored` 1.6317 — **차이 0.04 %**. 이를 근거로
   **H4 실패를 사전등록**했다 (P6).
2. **합성 동역학** (55,680 시뮬레이션): cantor는 무제어(+0.239)·상수(+0.057)·
   주기(+0.077)를 이기지만, **중앙 단일장벽(−0.081)과 shuffled(−0.038)에는
   진다.** center-anchored와는 `d_z = 0.049`로 사실상 무차이.
3. **실제 LLM** (Qwen2.5-0.5B-Instruct): refusal direction이 **인과적**임을
   확인했다 (layer 13, `+v` 0.625→0.875, `−v` →0.375). 동결된 held-out 시험
   10,800행에서 **사전등록 기준 7개 중 6개 FAIL** — Cantor 특이적 positive를
   주장할 수 없다.

**결론.** V1(이진 마스크)과 V2(척도 보상 장벽)라는 **완전히 다른 두 설계**가
같은 답에 도달했다: **강건성을 결정하는 것은 판정 경계를 무엇이 지키는가이며,
칸토어의 자기유사 배치 자체는 측정 가능한 기여를 하지 않는다.**

---

## 2. V1 연구 요약과 negative result

V1은 `ḣ = −α(h−½) + g_n(h)δ(t)`에서 319,703회 적분과 11,794 assertion으로:

* 원 결과 재현 (29.80 / 98.17 / 97.43 %)
* `h₀ = 0.15 ∈ (1/9, 2/9)` — 게이트가 30초 중 0.17 %만 열림
* 정리 1: `S° ⊂ F_n`이므로 `S°`의 불변성은 **δ와 무관한 항등식**
* `R_safe`는 `n`에 대해 비단조 (n=1 최대, n≈7 최소)
* **측도 정합 대조군 대비 백분위 41–52 — 프랙탈 이점 없음**

**V1의 negative result는 유효하며 V2가 뒤집지 않는다.**

---

## 3. 왜 V1에서 Cantor effect가 사라졌는가

정규화

    g_n(x) = C'_n(x)/(3/2)ⁿ = 1_{K_n}(x)

의 **그 한 줄**에서 derivative magnitude `(3/2)^k`, removal level `k`,
gap width `3^{-k}`, level별 gap 수 `2^{k-1}`, 척도별 질량 재분배가 전부
소실된다. 남는 것은 **집합 멤버십**뿐이다.

더 결정적으로, V1 정리 1의 증명은 `g(S°) = 0` **하나만** 사용한다. 즉 V1의
설계에서는 "안전집합을 덮는가"가 정의상 전부이고, 나머지 구조가 정리에
등장할 여지가 없었다. **V1이 프랙탈 이점을 찾지 못한 것은 실험의 실패가
아니라 컨트롤러 설계의 논리적 귀결이다.**

전문: [`V1_TO_V2_RATIONALE.md`](V1_TO_V2_RATIONALE.md).

---

## 4. 연구 질문·가설과 판정

| | 질문 | 결과 |
|---|---|---|
| **RQ1** | 척도 보상 컨트롤러를 구성할 수 있는가? | **예** (정리 A·B) |
| **RQ2** | 측도인가 배치인가? | **경계를 덮는가가 전부** |
| **RQ3** | 불변성·도달조건을 설명할 수 있는가? | **예** (정리 C, 명제 D·E) |
| **RQ4** | LLM refusal direction을 안전좌표로 쓸 수 있는가? | **예** (인과 검증 통과) |
| **RQ5** | 동일 예산에서 더 높은 강건성을 주는가? | **아니오** (지배당함) |
| **RQ6** | 두 model family로 일반화되는가? | **검증 못 함** (§8 한계) |

| 가설 | 판정 |
|---|---|
| **H1** 척도 보상 + `(3/2)^k` | **증명됨** (정리 A·B) |
| **H2** cantor > single-scale | **부분 지지**: constant·periodic은 이기나 **central에는 진다** |
| **H3** cantor > fully shuffled | **기각** (Δ = −0.038, CI가 0 배제) |
| **H4** cantor > center-anchored | **실패** (`d_z` 0.049; LLM 시험 기준 전부 FAIL) |
| **H5** 동일 utility에서 더 낮은 ASR | **검증 불가** (주 종점 퇴화, §7) |
| **H6** unseen family/연속 척도로 유지 | **기각** (기준 5·6 FAIL) |

---

## 5. 수학적 정리 (전문: [`MATHEMATICAL_THEORY.md`](MATHEMATICAL_THEORY.md))

**정리 A.** `Σ_j ∫_{I_{k,j}} |V'_{k,j}| dr = N_k e_k = E₀`, 모든 `k`.

**정리 B.** `‖V'_{k,j}‖_∞ = 3E₀(3/2)^k`, 비 `3/2` 정확.
→ V1이 정규화로 제거한 `(3/2)^k`가 **제어력의 진폭으로 복원**된다.

**정리 C (차단).** `ηV_C'(r*) > sup f + A`이면 `r(t₀) < r*`인 해는 영원히
`r(t) < r*`. (1차원 나구모)

**명제 D (negative).** minimax coverage `P_L(ℓ)`는 가장 넓은 gap의 가장자리에
지배되어 **배치에 무관**하다. 측정: best shuffle / cantor = **1.000**.

**명제 E.** 최악 변위 `D_L(k*)`는 배치에 의존하며 cantor가 최소지만,
shuffle 대비 **0.07–1.3 %**(periodic 대비 188 %).

> **정리 A·B·C는 모두 폭 정합 대조군이 똑같이 만족한다.**
> Cantor 특이적 주장은 오직 명제 E의 1 % 규모 여유에만 기댈 수 있었고,
> 그 여유는 동역학·LLM 잡음에서 살아남지 못했다.

---

## 6. 합성 실험 (전문: [`RESULTS.md`](RESULTS.md) V2-R5)

55,680 시뮬레이션. 1,920 matched condition paired bootstrap, `Δ>0` = cantor 우수:

| 대조군 | Δ P(safe) | 95 % CI | 판정 |
|---|---|---|---|
| B0 none | +0.2385 | [+0.2198, +0.2583] | cantor 우수 |
| B1 constant | +0.0573 | [+0.0443, +0.0708] | cantor 우수 |
| B3 periodic | +0.0771 | [+0.0641, +0.0906] | cantor 우수 |
| **B2 central** | **−0.0813** | [−0.0938, −0.0693] | **대조군 우수** |
| **B5 shuffled** | **−0.0383** | [−0.0463, −0.0307] | **대조군 우수** |
| B6 c-anchored | +0.0028 | [+0.0003, +0.0053] | `d_z`=0.049, 무차이 |

5개 attack family 전부에서 방향 동일. [FIG V2-05/06](../../figures/v2/figV2_05_06_synthetic.png).

---

## 7. LLM 실험

### 7.1 refusal direction의 인과 검증 (RQ4)

Qwen2.5-0.5B-Instruct layer 13, `ε` = 활성 노름 8 %:
harmful 거부 `−v` 0.375 ← 무개입 0.625 → `+v` 0.875.
**양방향 인과효과 ±0.25.** [FIG V2-07/08](../../figures/v2/figV2_07_08_direction.png).

방향은 benign에도 같은 방향으로 작용한다 — **유해성 탐지기가 아니라
refusal 손잡이**다. 그래서 동결된 harm detector로 게이팅한다.

### 7.2 사전등록과 동결

`configs/v2/llm_test.yaml`,
SHA-256 `42854a0f…f70f`, [`PREREGISTRATION.md`](PREREGISTRATION.md).
모든 family가 **동일한 12개 config 격자**에서 DEV 점수로 선택되었다.

동결 **전에** 두 개의 설계 결함을 DEV에서 발견하고 고쳤다.
1. `L1_constant`를 단일 광폭 장벽으로 구현했더니 `Φ'(0)=Φ'(1)=0` 때문에
   **가장 위험한 지점에서 힘이 0**이었다(측정: 개입 0.000). 진짜 상수 조종으로
   교체 — **baseline을 강화**하는 방향의 수정이다.
2. oracle harm gate에서는 benign 비용이 정의상 0이라 Pareto를 말할 수 없다.
   calibration에 적합한 동결 logistic detector를 추가했다.

### 7.3 동결 시험 결과 — 주 종점 퇴화 (§V2-R7)

10,800행. **모든 controller에서 ASR = 0**, 무개입 포함.
원인은 지표 결함이다: `compliance_score ≤ min(1, 24/60) = 0.4 < 0.5`이므로
**24 토큰 생성에서 ASR 임계값 도달이 원천 불가능**했다.

**사전등록 기준: 3만 PASS, 1·2·4·5·6·8 FAIL.
→ "Cantor-specific positive" 주장 불가.**

시험을 다시 열어 임계값을 조정하지 **않았다.**

### 7.4 POST-HOC 진단 — 거부인가 붕괴인가 (§V2-R8)

layer 13의 평균 활성 노름은 **16.09**인데 DEV가 고른 `η`에서 실현 개입은
활성 노름의 **60–75 %**였다.

| family | 개입/‖h‖ | coherent 거부 | coherence | benign 오거부 |
|---|---|---|---|---|
| L0 none | 0.000 | 0.157 | 0.993 | 0.271 |
| **L3 periodic** | **0.166** | **0.457** | **0.987** | 0.500 |
| L6 c-anchored | 0.600 | 0.043 | 0.764 | 0.100 |
| **L7 cantor** | **0.743** | 0.486 | 0.864 | 0.643 |
| **L1 constant** | 0.746 | **0.557** | 0.903 | **1.000** |

**메커니즘:** 장벽 컨트롤러의 개입은 상태 의존적이고 **토큰마다 급변**한다.
활성 노름의 60–75 %에 이르는 급변 push는 선형 조종의 유효 영역을 벗어나
출력을 붕괴시킨다(비-ASCII 표류·반복). 붕괴한 출력은 refusal 조각을 우연히
포함하되 실질 응답은 되지 못하므로 lexical 지표에 **"높은 거부 + ASR 0"**으로
나타난다. 상수 조종은 같은 크기여도 **균일한 오프셋**이라 모델이 흡수한다.

**coherence 보정 Pareto에서 어떤 장벽 컨트롤러도 지배적이지 않다.**
개입량 대비 안전성 최고는 **`L3_periodic`(폭 정합 대조군)**이며,
`L7_cantor`는 그 **4.5배 개입**으로 거부 +0.03, 오거부 +0.14 — 지배당한다.
[FIG V2-18](../../figures/v2/figV2_17_18_posthoc.png).

---

## 8. 실패한 가설과 예상 밖의 결과

1. **H3 기각, H4 실패.** 이론(0.04 % 차이)에서 예측했고 합성·LLM에서 확인.
2. **가장 단순한 컨트롤러가 이긴다.** 합성에서 `B2_central`,
   LLM에서 `L3_periodic`. V1의 `G4_central`과 같은 패턴.
3. **`L1_constant`가 광폭 장벽으로는 baseline이 되지 못했다.**
   `Φ'(0)=Φ'(1)=0`이 가장 위험한 지점에서 힘을 0으로 만든다. 동결 전 수정.
4. **주 종점이 퇴화했다.** 24 토큰과 ASR 임계값 0.5의 비호환. 내 사전등록 결함.
5. **큰 개입이 모델을 부순다.** 장벽 컨트롤러는 **차단(blocking)** 컨트롤러라
   경계에서만 힘을 몰아쓰고, 그 급변이 상수 조종보다 훨씬 파괴적이다.
6. **SmolLM2-360M은 거부 행동이 아예 없다** (harmful 6/6에서 0회).
   두 번째 model family 확보 실패의 직접 원인.
7. **MPS SDPA + left padding이 NaN을 만든다** (2-프롬프트 배치에서 980,224개).
   `attn_implementation="eager"`로 해결. 회귀 테스트 추가.

---

## 9. 한계

1. **모델 규모.** 8 GB / 3.5 GB 환경에서 0.5B가 상한이었다. 큰 모델에서
   결론이 달라질 수 있다(특히 큰 모델은 큰 개입에 더 강건할 수 있다).
2. **두 번째 model family 미확보.** TinyLlama-1.1B를 내려받았으나 시간 안에
   복제 실험을 완료하지 못했다. **RQ6는 미해결.**
3. **주 종점 퇴화.** ASR이 원천 불가능했다(§7.3). 대체 지표는 전부 post-hoc.
4. **LLM judge 미사용.** StrongREJECT 방식 rubric autograder를 돌릴 자원이
   없어 lexical + 실질성 + coherence 조합을 썼다. 절대 ASR 수준은 신뢰할 수
   없고, **controller 간 비교만** 의미가 있다.
5. **1차원 refusal 가정.** Wollschläger et al. (2025)은 refusal이 concept
   cone일 수 있음을 보인다. 본 연구는 1차원 좌표를 가정했다.
6. **개입 크기.** DEV가 고른 `η`가 선형 조종 영역을 벗어났다. 더 작은 `η`
   격자에서는 다른 그림이 나올 수 있으나, **동결 후에는 바꾸지 않았다.**
7. **adversarial 약함.** 기울기 기반 최악외란 합성을 하지 않았다.
8. **합성 스텝 수렴.** bistable separatrix 근처에서 소수 궤적이 스텝 크기에
   따라 basin을 뒤집는다(중앙값 차이 0, 99 백분위 0.41). 이진 `crossed`
   지표와 55k 평균은 영향받지 않는다.
9. **깊이 방향 Cantor schedule(§44) 미수행.** core 실험 우선.

---

## 10. LLM safety에 대한 함의

* **refusal direction은 실제로 인과적이며 매우 저렴한 안전 손잡이다.**
  0.5B 모델에서도 ±0.25의 거부율 변화를 만든다 — 문헌과 일치.
* **그러나 "얼마나 미느냐"가 "어디서 미느냐"보다 훨씬 중요하다.**
  본 연구가 정교하게 설계한 척도 보상 다척도 장벽은, 같은 예산의 상수 조종이나
  단순 주기 배치보다 나은 점이 없었다.
* **개입 크기에는 실용적 상한이 있다.** 활성 노름의 ~15 %를 넘어서면
  출력 coherence가 무너지기 시작하고, lexical 안전 지표는 그 붕괴를
  "안전"으로 오독한다. **coherence를 함께 보고하지 않는 activation-steering
  안전 평가는 신뢰할 수 없다.** 이것이 본 연구의 가장 실용적인 기여다.

---

## 11. 결론

V1은 칸토어를 **이진 마스크**로 축약했고 프랙탈 이점을 찾지 못했다.
V2는 그 축약이 버린 **측도 집중 법칙과 removal hierarchy**를 실제 제어장에
복원했다. 정리 A·B로 척도별 총 작용량 불변성과 `(3/2)^k` 증폭을 **증명**했고,
정리 C로 차단 조건을 세웠다.

그러나 세 단계 전부에서 같은 답이 나왔다.

* **이론**: 정리 A·B·C는 폭 정합 대조군이 **똑같이** 만족한다.
  봉쇄 계산에서 cantor와 center-anchored의 차이는 **0.04 %**.
* **합성**: cantor는 shuffled에 **−0.038**로 지고 center-anchored와
  `d_z = 0.049`로 무차이.
* **LLM**: 동결 시험에서 사전등록 기준 **6/7 FAIL**.

> **따라서 본 보고서는 "칸토어 구조가 강건성을 만든다"고 쓰지 않는다.**
>
> 확인된 것은 이것이다: **척도 보상은 잘 정의된 수학적 구성이고 정리 A·B는
> 참이지만, 그 정리들은 배치에 무관하며, 배치가 만드는 차이는 1 % 규모여서
> 실제 동역학과 LLM에서 다른 요인들에 압도된다. 강건성을 결정하는 것은
> 판정 경계를 무엇이 얼마나 세게 지키는가이다.**

V1과 V2는 컨트롤러 설계를 **정반대로**(마스크 → 장벽, `K_n` 위 → gap 안) 바꾸고도
같은 결론에 도달했다. 두 번 독립적으로 같은 답이 나왔다는 것이,
이 negative result에 대한 가장 강한 증거다.

---

## 참고문헌

* Arditi, A., Obeso, O. B., Syed, A., Paleka, D., Panickssery, N., Gurnee, W.,
  Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single
  Direction.* NeurIPS 2024. arXiv:2406.11717
* Zou, A., Phan, L., Chen, S., et al. (2023). *Representation Engineering:
  A Top-Down Approach to AI Transparency.* arXiv:2310.01405
* Turner, A. M., Thiergart, L., Leech, G., et al. (2023). *Activation Addition:
  Steering Language Models Without Optimization.* arXiv:2308.10248
* Rimsky, N., Gabrieli, N., Schulz, J., et al. (2024). *Steering Llama 2 via
  Contrastive Activation Addition.* ACL 2024. arXiv:2312.06681
* Chao, P., Debenedetti, E., Robey, A., et al. (2024). *JailbreakBench.*
  NeurIPS 2024 Datasets & Benchmarks.
* Souly, A., Lu, Q., Bowen, D., et al. (2024). *A StrongREJECT for Empty
  Jailbreaks.* NeurIPS 2024 Datasets & Benchmarks.
* Röttger, P., Kirk, H. R., Vidgen, B., et al. (2024). *XSTest.* NAACL 2024.
  arXiv:2308.01263
* Wollschläger, T., et al. (2025). *The Geometry of Refusal in Large Language
  Models.* arXiv:2502.17420
* Filippov, A. F. (1988). *Differential Equations with Discontinuous Righthand
  Sides.* Kluwer.
* 안재민 (2025). *칸토어 계단 함수의 n차 근사를 활용한 Neural ODE 기반 윤리적
  강건성 필터 설계.* (V1의 선행 탐구, `original/`)
