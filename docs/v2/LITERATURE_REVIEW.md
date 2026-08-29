# 문헌 조사 (Literature Review)

V2의 LLM 구현 이전에 조사했다. **블로그만으로 핵심 주장을 뒷받침하지 않으며**,
arXiv/conference 원문을 우선한다. 각 항목에 본 연구와의 직접 관계를 적는다.

---

## 1. Refusal direction — V2 threat coordinate의 근거

**Arditi, A., Obeso, O. B., Syed, A., Paleka, D., Panickssery, N., Gurnee, W.,
Nanda, N. (2024). "Refusal in Language Models Is Mediated by a Single
Direction." NeurIPS 2024.**
arXiv:2406.11717 · [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf) · [code](https://github.com/andyrdt/refusal_direction)

* **핵심 결과.** 13개 open-source chat model에서 refusal이 잔차 스트림의
  **1차원 부분공간**에 의해 매개된다. 그 방향을 제거(ablate)하면 유해 요청에
  대한 거부가 사라지고, 더하면 무해 요청에도 거부가 유도된다. 즉 그 방향은
  refusal에 대해 **필요충분**하다. 저자들은 이를 white-box jailbreak로도
  전환하고, adversarial suffix가 이 방향의 전파를 억제한다는 것을 기계적으로
  분석한다.
* **본 연구와의 관계.** V2의 전제 그 자체다. `r = sigmoid(−γ m)`의 margin
  `m`은 이 refusal direction 위로의 사영이다. 방법론(difference-in-means로
  방향을 뽑고 **causal ablation/addition으로 검증**)을 그대로 따르며,
  §19의 `+εv / −εv` causal validation은 이 논문의 검증 절차에 대응한다.
  또한 latent attack `h ← h − εv_ref`는 이 논문의 white-box jailbreak를
  **연속 강도로 파라미터화한 것**이다.

**Wollschläger, T. et al. (2025). "The Geometry of Refusal in Large Language
Models: Concept Cones and Representational Independence."**
arXiv:2502.17420

* refusal이 단일 방향이 아니라 **concept cone**(다차원)일 수 있으며 서로
  표현적으로 독립인 여러 방향이 존재함을 보인다.
* **관계.** V2의 1차원 threat coordinate 가정에 대한 **가장 강한 반론**이다.
  §17 한계에 명시하고, 1차원 가정이 성립하는 범위(causal validation의 효과
  크기)를 실제로 측정하여 보고한다.

---

## 2. Activation steering / representation engineering — 개입 방식의 근거

**Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X.,
Mazeika, M., Dombrowski, A.-K., et al. (2023). "Representation Engineering:
A Top-Down Approach to AI Transparency."** arXiv:2310.01405

* 정직성·무해성 같은 고수준 개념이 표현공간의 선형 방향으로 읽고 제어된다는
  것을 체계화한다. training-free 개입으로 행동을 조절.
* **관계.** V2의 개입 형태 `h ← h + c(r)·v_ref`가 이 계열이다. V2가 더하는
  것은 **개입 강도를 상태에 의존하는 척도 보상 장벽으로 만드는 것**이다.

**Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U.,
MacDiarmid, M. (2023). "Activation Addition: Steering Language Models Without
Optimization."** arXiv:2308.10248

**Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A. M.
(2024). "Steering Llama 2 via Contrastive Activation Addition." ACL 2024.**
arXiv:2312.06681

* contrastive pair의 평균 차이로 steering vector를 만들고 잔차에 더한다.
* **관계.** V2의 baseline **L1 (constant refusal steering)**이 정확히 이
  방법이다. 즉 문헌의 표준 기법이 본 연구의 대조군으로 들어간다.

**Braun, J. et al. (2025). "SteeringSafety: A Systematic Safety Evaluation
Framework of Representation Steering in LLMs."** arXiv:2509.13450

* steering이 목표 행동은 바꾸지만 **부작용(side-effect)**이 크다는 것을
  체계적으로 평가.
* **관계.** V2가 safety만이 아니라 **benign utility / over-refusal / KL**을
  반드시 함께 보고하는 이유(§32, Pareto)의 근거.

**Tan, D. et al. / follow-ups (2024–2026). "Understanding Unreliability of
Steering Vectors" (arXiv:2602.17881), "What Can We Actually Steer?"
(arXiv:2511.18284)**

* steering vector의 신뢰성이 행동·모델에 따라 크게 달라지며 선형 근사가
  깨지는 영역이 있음.
* **관계.** V2가 방향을 뽑자마자 쓰지 않고 **causal validation을 통과해야만**
  진행하는 이유(STOP CONDITION A).

---

## 3. Jailbreak benchmark — 평가 기준

**Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F.,
Sehwag, V., Dobriban, E., Flammarion, N., Pappas, G. J., Tramèr, F.,
Hassani, H., Wong, E. (2024). "JailbreakBench: An Open Robustness Benchmark
for Jailbreaking Large Language Models." NeurIPS 2024 Datasets & Benchmarks.**
[repo](https://github.com/JailbreakBench/jailbreakbench) ·
dataset `JailbreakBench/JBB-Behaviors` (ungated, 확인함)

* 100개 policy-violating behavior, 표준 위협모델·채점, 재현 가능한 workflow.
* **관계.** V2의 **benchmark family 1**. behavior별 goal ID가 있어 §29의
  goal 단위 split이 가능하다.

**Souly, A., Lu, Q., Bowen, D., Trinh, T., Hsieh, E., Pandey, S., Abbeel, P.,
Svegliato, J., Emmons, S., Watkins, O., Toyer, S. (2024). "A StrongREJECT for
Empty Jailbreaks." NeurIPS 2024 Datasets & Benchmarks.**
[proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/e2e06adf560b0706d3b1ddfca9f29756-Paper-Datasets_and_Benchmarks_Track.pdf) ·
[code](https://github.com/dsbowen/strong_reject)

* 기존 jailbreak 성공률이 **과대추정**되어 있음을 보이고, 응답의 유용성까지
  채점하는 rubric 기반 autograder와 연속 harm score를 제시.
* **관계.** V2가 **단순 키워드 refusal detector를 primary metric으로 쓰지 않는**
  이유(§26). 본 연구의 refusal 판정은 키워드 + 응답 실질성 확인을 결합한다.

**Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., Sakhaee, E.,
Li, N., Basart, S., Li, B., Forsyth, D., Hendrycks, D. (2024). "HarmBench:
A Standardized Evaluation Framework for Automated Red Teaming and Robust
Refusal." ICML 2024.** arXiv:2402.04249

* 400개 텍스트 행동 + 표준 평가 파이프라인.
* **관계.** benchmark family 2 후보. HF 미러(`walledai/HarmBench`)는
  gated=auto이므로 접근 가능성을 실행 시점에 확인한다.

**Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., Fredrikson, M.
(2023). "Universal and Transferable Adversarial Attacks on Aligned Language
Models."** arXiv:2307.15043 (AdvBench)

* GCG suffix 공격과 AdvBench harmful-behaviors 데이터셋의 출처.
* **관계.** suffix 계열 공격의 표준. V2 calibration의 제한된 suffix 공격.

**Röttger, P., Kirk, H. R., Vidgen, B., Attanasio, G., Bianchi, F.,
Hovy, D. (2024). "XSTest: A Test Suite for Identifying Exaggerated Safety
Behaviours in Large Language Models." NAACL 2024.** arXiv:2308.01263

* **과잉거부(over-refusal)** 측정용 250 safe + 200 unsafe 프롬프트.
* **관계.** V2 §20·§32의 핵심. refusal 방향으로 무조건 밀면 trivial
  over-refusal controller가 되므로, XSTest류로 반드시 벌점을 매긴다.

---

## 4. 본 연구가 문헌에서 가져오지 않는 것 / 새로 하는 것

| 문헌 | 개입 강도 |
|---|---|
| Turner 2023 / Rimsky 2024 | **상수** (또는 프롬프트 유형별 상수) |
| Zou 2023 RepE | 상수 또는 선형 판별 기반 |
| Arditi 2024 | 방향 **제거**(ablation) 또는 상수 추가 |
| **본 연구 (CantorGuard)** | **상태 의존 다척도 장벽** `c(r) = η·V_C'(r)` |

즉 V2의 기여 후보는 "새 방향"이나 "새 벤치마크"가 아니라
**개입 강도를 threat coordinate의 함수로 만들되, 그 함수를 척도별 총 작용량이
일정한 다척도 장벽으로 설계하는 것**이다. 그리고 그 설계가 **동일 예산의
상수/단일척도/무작위 대조군보다 나은지**를 검증한다.

**중요.** `docs/v2/MATHEMATICAL_THEORY.md` §7.1은 이론 단계에서 이미
"칸토어 배치 자체의 이점은 center-anchored 대조군과 구별되지 않는다"고
예측한다. 따라서 본 연구가 지지할 수 있는 최대 주장은
**"다척도 장벽 > 단일척도/상수"**이지 **"칸토어 특이적 우수성"**이 아닐
가능성이 높다. 이 구분은 최종 보고서에서 유지된다.

---

## 5. 조사했으나 본 연구에 직접 쓰지 않은 것

* **Sparse autoencoder 기반 feature steering** — 별도의 SAE 학습이 필요하고
  8 GB/3.5 GB 환경에서 비현실적. 한계에 기록.
* **Circuit-level mechanistic interpretability** — refusal 회로 자체의 규명은
  본 연구 범위 밖. threat coordinate가 causal하다는 것만 요구한다.
* **Latent adversarial training (LAT)** — 학습 기반 방어. V2는 **training-free
  inference-time 개입**만 다룬다 (모델 가중치를 바꾸지 않음).
