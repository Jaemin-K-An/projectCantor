# V3.3.1 실증 브리지 감사 — 다섯 개 결함 전부 확인됨

수학(LEVEL 1)은 **그대로 유지**한다. 무너진 것은 **LLM 연결부**다.

## A. PROMPT calibration은 정상이었다
`calibrate(ah_c, al_c, ...)` — harmful/harmless **두 class 중점**. ✓

## B–C. GENERATION calibration은 calibration이 아니었다

```python
_, st = generate32(b, P_attacked_harmful, dirs=dirs_prompt, ...)
tau_g = tau_p + m_mean * sig_p      # "generation-phase midpoint"
sig_g = m_std * sig_p
```

`m_mean`은 **공격된 harmful 생성문 한 class의 평균 margin**이다.
경계(τ)는 정의상 **두 class의 중점**인데, harmless를 한 번도 보지 않았다.
**harmless를 보지 않은 추정량은 두 class 사이의 경계를 찾을 수 없다.**
`sig_g` 역시 pooled std가 아니라 그 한 class의 산포다.

부차 결함: 공격 템플릿을 **class 정의**에 사용했다. 공격은 평가 섭동이지
class 정의가 아니다.

## D. hook은 위상을 구분하지 않았다

```python
dirs = RefusalDirections([LAYER], v, z["tau_gen"], z["sigma_gen"], ...)
```

`generation32`의 hook은 **단일 τ/σ를 모든 forward에** 적용한다.
"phase-specific"이라 불렀지만 실제로는 **어느 단일 calibration을 쓸지만
바꾼 것**이고, prefill에도 generation 값이 적용되었다.
**코드에 위상 인식이 존재하지 않았다.**

## E. `eps_cal`은 추정량 불확실성이 아니었다

```python
r_prompt = sig_r(zs, tau_p, sig_p)
r_gen    = sig_r(zs, tau_g, sig_g)
eps      = |r_prompt - r_gen|
```

**같은** prompt 사영 `zs`에 **두 좌표변환**을 적용한 차이다.
표본 변동성이 **0**인 결정론적 offset이며, 정의상 estimator uncertainty가
아니다.

## 따라서 철회하는 주장

| V3.3.1 주장 | 상태 |
|---|---|
| `eps_cal q50 = 0.111` | **철회** — 추정량 불확실성이 아님 |
| `n_max = 1~2` | **철회** — 위 값에 의존 |
| "V1–V3.2 negative의 메커니즘 설명" | **철회** — `M_PLAUSIBLE`로 강등 |
| `δ ≈ 1/9`라는 우연의 일치 | **인공물** — 올바른 추정에서 재현되지 않음 |

## 유지되는 것

정리 G · BGR · P · R, 반례 전부, 합성 검증(`κ=β`, corr 0.9992).
단 합성 결과의 명칭을 **SCALE-INVARIANT SYNTHETIC MODEL**로 바꾸고,
`corr = 0.9992`는 **구현 검증**이지 LLM 불확실성 검증이 아님을 명시한다.
