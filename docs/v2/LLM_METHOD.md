# LLM 방법론 (V2)

## 1. 용어 — Transformer는 ODE가 아니다

Transformer의 층 진행은 **이산 잔차 동역학**이다.

    h_{ℓ+1} = h_ℓ + F_ℓ(h_ℓ)

본 연구는 이를 `discrete residual dynamical system`으로 부른다.
합성 실험의 연속시간 이론은 **유추(analogy)**이며, LLM 결과가 연속시간
정리로부터 따라 나온다고 주장하지 않는다. 실제로 LLM에 적용된 것은
정리 C의 **차단 아이디어**(위험 좌표가 커질수록 밀어낸다)뿐이고,
정리 A·B는 컨트롤러의 **예산 정합**을 보장하는 데 쓰인다.

## 2. 안전 좌표

layer `ℓ`의 마지막 프롬프트 토큰 잔차 `h_ℓ`에 대해

    z = ⟨h_ℓ, v_ref⟩,   m = (z − τ_ℓ)/σ_ℓ,   r = sigmoid(−γ m)

`v_ref`는 단위 노름이므로 개입 노름이 곧 `c(r)`이다.
`m = 0` ⇔ `r = 1/2` ⇔ 판정 경계. 칸토어·center-anchored 배치의 level-1 gap이
정확히 여기 놓인다.

## 3. 개입

    h' = h + c(r)·v_ref,    c(r) = η·V_L'(r) ≥ 0

**적용 순서** (단일 forward hook 안에서, 등록 순서에 의존하지 않게):

1. 공격: `h ← h − ε v_ref + ξ`
2. margin 읽기 (**공격 이후의 상태에서**)
3. 방어: `h ← h + c(r(m)) v_ref`

상태 의존 방어는 공격받은 상태를 봐야 의미가 있으므로 이 순서를 고정하고
테스트로 검증한다.

## 4. 검증된 정확성 요건

| # | 요건 | 테스트 |
|---|---|---|
| 9 | 개입 0이면 로짓이 **bit-identical** | `test_zero_intervention_identity` |
| 10 | KV cache 생성에서 토큰·레이어당 hook **정확히 1회** | `test_hook_fires_once_per_token_per_layer` |
| 11 | `v_ref` 단위 노름 | `test_v_ref_unit_norm` |
| 12 | `+v/−v` 인과 smoke | `extract_and_fit_direction.py` |
| 13 | threat 좌표가 margin에 대해 단조 | `test_threat_coordinate_monotone_in_margin` |
| 14 | `c(r) ≥ 0` — 항상 안전 방향 | `test_controller_never_pushes_toward_danger` |
| 15 | 예산 정합 | `test_theorem_A_level_action` (Julia·Python) |
| 16 | 동일 프롬프트로 paired 비교 | `run_llm_test.py` (pid 기록) |
| 17 | 동결 후 config 변경 불가 | `run_llm_test.py` SHA-256 검사 |
| 18 | 추적 테이블에 유해 텍스트 없음 | `assert_no_raw_completions` |

## 5. harmfulness와 refusal의 분리

refusal direction은 **유해성 탐지기가 아니라 refusal 손잡이**다 (V2-R6:
benign 프롬프트도 같은 방향으로 움직인다). 따라서 모든 컨트롤러가
**동일한 동결 harm detector**로 게이팅된다.

* **Experiment A (oracle)**: 벤치마크 라벨로 게이팅. 메커니즘 분리용.
  benign 비용이 정의상 0이므로 **Pareto 주장 불가**.
* **Experiment B (deployable)**: calibration에 적합한 logistic detector
  (layer 13, target FPR 0.10). 그 오탐이 실제 효용 비용을 만든다.

## 6. 누출 방지

`goal_group` 단위로 split한다(원문 문자열이 아니라). JBB의 `Behavior`를
정규화(소문자·구두점 제거·불용어 제거·정렬)한 키를 쓰며, split 결과의
group overlap이 **0**임을 assert한다.

공격 템플릿도 분리한다: calibration 3종 / dev 2종 / **test 4종(전부 held-out)**.

## 7. 유해 출력 취급

* 추적 테이블에는 **prompt hash와 스칼라 점수만** 저장한다.
* 생성 텍스트는 `results/v2/private/`(gitignored)에만 남긴다.
* 보고서·README에 유해 completion을 인용하지 않는다.
* 사람이 유해 지침의 품질을 직접 평가하지 않는다.
* `assert_no_raw_completions`가 커밋 대상 테이블에 텍스트 컬럼이 섞이는 것을
  막는다(테스트 포함).

## 8. 환경 특이사항 — MPS SDPA NaN

Apple M1 + PyTorch 2.8 + SDPA + **left padding**이면 초기 query 위치가
pad 토큰만 보게 되어 all `-inf` softmax → **NaN**이 잔차 전체로 전파된다.
2-프롬프트 배치에서 980,224개의 NaN을 측정했다.

`attn_implementation="eager"`로 해결되며 batch-size-1 SDPA와 `4×10⁻⁵` 이내로
일치한다. 배치 생성에는 left padding이 필수이므로 eager를 고정한다.
회귀 테스트: `test_residuals_finite_batched`.
