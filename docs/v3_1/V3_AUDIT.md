# V3 감사 (V3.1 PHASE 0)

**V3의 파일·결과는 수정하지 않는다.** 발견된 결함을 기록하고, 수정은
`src/v3_1/`, `results/v3_1/` 등 분리된 namespace에서 수행한다.

---

## D1. **[치명적]** `L1_constant`가 상수 컨트롤러가 아니다

`src/v3/V3Controllers.jl:59`

```julia
family == "L1_constant" && return piecewise_layout(ones(1), n, E0, "constant", family)
```

`piecewise_layout(ones(1))`은 `[0,1]` 전체를 덮는 **gap 1개**를 만들고
smoothstep 장벽을 놓는다. 따라서 장은

    u(r) ∝ Φ'(r) = 6r(1−r)

측정값:

| r | 0.00 | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| u(r) | 0.000 | 0.540 | 1.125 | **1.500** | 1.125 | 0.540 | 0.000 |

**range = 1.500.** 이것은 상태 의존적인 **광폭 smooth 장벽**이지 상수가 아니다.

### 결과에 미치는 영향 — **V3의 헤드라인 주장이 무효**

V3는 다음을 주장했다:

> "상수 조종은 Δ에 완전히 무감각하다(어디서나 같은 힘). …
>  경계 위치가 불확실할 때 최선의 귀납 편향은 상태 무관성이다."

그러나 측정된 컨트롤러는 `∂u/∂r ≠ 0`이므로 **Δ에 무감각하지 않다.**
`R_worst = 0.524`라는 수치 자체는 유효하지만, 그것이 *상수 조종*의 성능이라는
**해석은 근거가 없다.**

**V3.1 대응.** 두 컨트롤러로 분리한다.

* `S1_true_constant`: `u(r) ≡ −c`, `∂u/∂r = ∂u/∂Δ = 0` (기계정밀도 테스트)
* `S2_global_smooth`: 기존 `piecewise_layout(ones(1))`을 이 이름으로 **보존**

"state-independence"라는 표현은 `S1`의 결과에만 사용한다.

---

## D2. **[치명적]** `C0_fixed`가 V2의 calibration이 아니다

`llm/src/cantor_guard_v3/calibration.py`의 `fit_calibrator`는

```python
for l, gl in proj_df.groupby("layer"):      # 위치 필터 없음
    c.tau_global[int(l)] = 0.5 * (zh.mean() + zb.mean())
```

즉 **prefill과 모든 생성 토큰을 pooling**한다. V2가 실제로 쓴 값은
`results/v2/cache/*_dirs.npz`의 `diff_means_tau`(마지막 프롬프트 토큰에서
추정)이며 **완전히 다른 객체**다.

### 결과에 미치는 영향

> "C0_fixed(V2 방식) 0.299 σ → C2_token_bin 0.213 σ (−28.9 %)"

이 문장의 "V2 방식"은 거짓이다. 실제로 비교된 것은
**pooled-global** baseline이다. 28.9 %라는 수치는
"pooled 대비 개선"이며 **"V2 대비 개선"이 아니다.**

**V3.1 대응.** 네 가지를 분리하여 재계산한다.

| 이름 | 정의 |
|---|---|
| `C_V2_LAST_PROMPT` | npz에 저장된 V2의 실제 `τ_ℓ, σ_ℓ` |
| `C_GLOBAL_POOLED` | V3의 현재 pooled global |
| `C_PHASE` | prefill / generation 분리 |
| `C_TOKEN_BIN` | 토큰 구간별 |

---

## D3. **[중대]** 측정한 것은 behavioral boundary가 아니다

V3가 측정한 것은

    τ_mid = ½(mean z_harm + mean z_harmless)

즉 **사영 분포의 중점**이다. 이것은 "모델이 거부하기 시작하는 지점"이 아니다.
V3 보고서·README가 이를 "안전 판정 경계"라 부른 것은 **과잉 해석**이다.

**V3.1 대응.** `P(refusal | z)`를 적합해

    τ_beh :  P(refusal | z = τ_beh) = 0.5

를 별도로 추정하고, `τ_mid`와의 관계를 보고한다. behavioral 적합이 불안정하면
**"projection midpoint non-stationarity"**라는 보수적 용어로 후퇴한다.

---

## D4. **[오류]** 정리 S의 potential 식에 가법 상수 누락

정확한 형태는

    V_n(T_i(r)) − V_n(T_i(0)) = ½·V_{n−1}(r)

이다. `T₂` branch에서는 `V_n(T₂(0)) = V_n(2/3) ≠ 0`이므로 offset이 실재한다.
`src/v3/CantorSelfSimilarity.jl`의 주석과 `docs/v3/MATHEMATICAL_THEORY.md`
§1의 정리 서술에는 offset이 있으나, `docs/v3/FINAL_REPORT_KO.md` §12–13의
요약 표기는 offset 없이 `V_n(T_i(r)) = ½V_{n−1}(r)`로 적혀 있다.
**미분 항등식은 영향받지 않는다.**

---

## D5. **[오류]** Corollary S.1 (전역 coverage 멱법칙)이 거짓이다

V3는 `M_n(Δ) = min_r max_{|u−r|≤Δ} V'_n(u)`에 대해
`M_n(Δ/3) = (3/2)M_{n−1}(Δ)`가 정리 S에서 따라온다고 서술했다.
직접 계산하면 비는 **1.0이 아니다**:

| n | Δ = 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|
| 4 | — | — | 0.2316 | 0.2628 |
| 5 | — | 0.2280 | 0.2373 | 0.2760 |
| 6 | 0.2254 | 0.2302 | 0.2405 | 0.2822 |

이유: **전역 최소값을 실현하는 창이 좌/우 복사본·중앙 gap·구간 경계를
넘나들 수 있으므로** 자기유사성이 국소적으로만 적용된다.

**V3.1 대응.** 전역 따름정리를 **삭제**하고, 각 복사본 내부로 제한한
국소 버전만 증명하여 유지한다.

---

## D6. **[방법론]** mechanism 분석이 축소 격자에서 수행됨

`scripts/v3/analyse_mechanism.jl`은 attack 2종 · budget 0.30 · x₀ 2종만
사용했다. 그 결과(자기유사성 r = +0.147, max_weak_run r = −0.680)를
주 결론으로 쓰기에는 격자가 좁다.

**V3.1 대응.** 전체 Δ×ε 격자 · 전체 attack family · 3 budget에서
150개 이상 배치로 재실행하고, coverage 통계를 통제한 **다변량 회귀**로
자기유사성의 독립 설명력을 검정한다.

---

## D7. **[방법론]** L9가 main sweep과 분리되어 재학습됨

`run_synthetic_uncertainty.jl`이 매 실행마다 L9를 다시 적합하고,
`analyse_mechanism.jl`은 **다른** 탐색으로 다른 가중치를 얻었다
(0.4613 vs 0.5225). 두 결과가 같은 컨트롤러를 가리키지 않는다.

**V3.1 대응.** DEV에서 한 번 적합 → `configs/v3_1/l9_frozen_weights.toml`에
동결 → main sweep은 **읽기만** 한다.

---

## D8. **[미완]** LLM direct controller test 부재

V3는 LLM에서 경계 이동과 calibration만 측정했고, controller 비교는
합성 모형에서만 수행했다. V3 보고서 §32 한계 1에 기록되어 있다.

**V3.1 대응.** PHASE 10–17에서 실제 LLM direct test를 수행한다.

---

## 요약

| # | 결함 | 심각도 | V3 결론에 미치는 영향 |
|---|---|---|---|
| D1 | `L1_constant`가 상수 아님 | **치명적** | **헤드라인 해석 무효** |
| D2 | `C0_fixed` ≠ V2 calibration | **치명적** | "V2 대비 28.9 %" 무효 |
| D3 | τ_mid ≠ behavioral boundary | 중대 | "안전 경계" 용어 과잉 |
| D4 | potential offset 누락 | 오류 | 미분 항등식은 무영향 |
| D5 | Corollary S.1 거짓 | 오류 | 삭제 필요 |
| D6 | mechanism 축소 격자 | 방법론 | 재실행 필요 |
| D7 | L9 미동결 | 방법론 | 재현성 결함 |
| D8 | LLM direct test 부재 | 미완 | 최종 결론 불가 |

**D1·D2가 V3의 두 헤드라인 문장을 무효화한다.** 수치는 유효하지만
그 수치가 무엇의 성능인지에 대한 서술이 틀렸다.
