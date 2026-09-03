# V3.3.5 결과

## 1. 정리 CP — affine이 **강제**된다

φ가 연속·강단조이고 하나의 위치 무관 상수 `c`로
`|φ(z₂)−φ(z₁)| = c|z₂−z₁|`이면 φ는 affine이다.
(강단조로 부호 고정 ⟹ `φ(z₂)−φ(z₁) = s·c(z₂−z₁)` ⟹ `φ(z)=φ(z₀)+s·c(z−z₀)`.) ∎

> **ρ=1/3이 이기도록 좌표를 고른 것이 아니다.** "Cantor의 Euclidean 여유를
> 잔차 계량에 위치 무관 상수배로 보존한다"는 **설계 요구만으로** affine이
> 유일하게 강제된다. 로지스틱은 이 가설을 만족하지 않는다
> (측정: ±3 구간 기울기 비 2.93, ±6에서 23.6; affine은 정확히 1).

`r_aff(z) = ½ + s(z−τ)/(2W)` — 끝점 정확히 0/1, τ 정확히 ½,
**`|dr/dz| = 1/(2W)` 정확히 상수**. 창 밖은 **clipping이 아니라 OUTSIDE 상태**.

## 2. 정리 AZ / AZ3 / AH — exact 인증서

`|Δr| = |Δz|/(2W)` **정확** ⟹
```
ε_z^A(ρ,n) = 2W·ρ^{n−1}(1−2ρ)          (하한이 아니라 정확한 환산)
ε_h^A = ε_z^A                           (‖v‖=1, Cauchy–Schwarz)
```
`2W`가 ρ 무관 ⟹ **`argmax ε_z^A(·,3) = 1/3`**, `ε_C = 2W/27`.

| ρ | 0.25 | 0.28 | 0.30 | **1/3** | 0.36 | 0.40 | 0.44 |
|---|---|---|---|---|---|---|---|
| Cantor 이득 | +18.5 % | +7.4 % | +2.9 % | **0** | +2.1 % | +15.7 % | +59.4 % |

깊이별 exact 최적: **n=2 → 0.25, n=3 → 0.3333, n=5 → 0.40** = `(n−1)/(2n)` ✓
**V3.3.4의 로지스틱 최적 0.296은 historical control로 보존한다.**

## 3. 인증서 검증 — **실제 forward 공격**

D_attack_dev, G1에 순수 방향 공격 주입:

| | 결과 |
|---|---|
| `|Δz| − ε` 최대 오차 | **4.5e−07** (순수 방향 공격이므로 정확히 ε) |
| below-certificate 설정 | 42 |
| **direct switch 위반** | **0** |
| λ=1.01에서 | switch 0, **guard capture 41** |
| 인증서 순위 | **Cantor #1** |
| 창 coverage (calibration) | 0.975 |

> λ=1.01에서 상태가 leaf를 벗어나지만 **다른 leaf가 아니라 guard로** 간다 —
> guard가 설계대로 정확히 작동한다.

## 4. **결정적 발견 — G1 인과 경계는 식별 불가**

G1-only dose (prefill 0, G1만, G2+ 0; 위상 추적 검증 prefill 1 / G1 1 / G2+ 46):

| dose | −100 | −80 | −60 | −45 | **−30** | −20 | −14 | −8 | 0 | +4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 거부율 (confirm) | 0.183 | 0.117 | 0.233 | 0.183 | **0.500** | 0.617 | 0.667 | 0.750 | 0.733 | 0.733 |

| | DEV | CONFIRM |
|---|---|---|
| τ_G1 점추정 | −24.73 | −28.19 |
| isotonic | −28.67 | −28.67 |
| **slope** | 0.0367 | **0.0354** (게이트 ≥ 0.05) ✗ |
| **CI 폭** | 28.4 σ | **24.7 σ** (게이트 ≤ 3 σ) ✗ |

**`TAU_G1_UNIDENTIFIABLE` — 독립 확증 분할에서 재현.**
τ_mid·global τ_beh 대체는 **금지**이므로 하지 않았다.

**의미:** 거부율을 0.5 아래로 내리려면 G1에서 **약 30 σ**의 섭동이 필요하다.
V3.3.3이 찾은 경계(CI 1.38 σ)는 **모든 forward에 dose를 준** 결과였다.
> **인증서가 다루는 단일 G1 상태는 그 자체로 거부 결정을 인과적으로 좌우하지
> 않는다.**

## 5. D_final_335 — **쓰지 않았다**

행동적 중심이 없으므로 최종 생성은 실행하지 않았다.
**프로토콜의 게이트가 최종 집합을 소모하기 전에 실험을 멈춘 것**이며,
`PRE_ANALYSIS_FREEZE.json`에 `D_final_335_touched: false`로 기록했다.

## 6. 판정

```
MATH        M1_CP_AND_M2_CANTOR_EXACT_MAXIMIN_PROVED
BEHAVIOR    B2_G1_BOUNDARY_UNIDENTIFIABLE
CERTIFICATE C1_EXACT_AFFINE_CERTIFICATE_VALIDATED  (기하학적 anchor — 공개)
GENERATION  G5_INCONCLUSIVE (미실행, D_final 미접촉)
UTILITY     U_NOT_RUN
OVERALL     E_NO_APPLICABLE_BEHAVIORAL_CONTROLLER
```
