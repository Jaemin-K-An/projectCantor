# V3.3.4 결과

## 1. V3.3.3 감사 — 네 결함 전부 확인

| # | 결함 | 확인 |
|---|---|---|
| C | System A가 `τ_beh`를 쓰지 않음 | DECODE `τ=+0.9887`(중점), 측정된 `τ_beh=−2.6263` — **3.90 σ 보정을 측정만 하고 적용 안 함** |
| D | `τ_beh`가 phase-local 아님 | dose가 **모든 forward**에 주입됨 → global |
| E | D_final 예산 미감사 | **ρ=0.40(−5.6 %)·0.44(−6.1 %)가 D_final에서 실패**했는데 D_budget 플래그로 통과 — **0.44는 V3.3.3의 primary alternative였다** |
| F | `d_cross`가 exact 아님 | `step=(hi−lo)·1e−3`, `max_steps=4096` |

## 2. 정리 CR / CR.1 — 증명 및 검증

`M_n(ρ) = ρ^{n−1}(1−2ρ)` (**infimum**, leaf 내부가 열린 집합이므로),
`M_3′(ρ) = 2ρ(1−3ρ)` ⟹ **ρ=1/3 유일 최대, `M_3(1/3)=1/27`**.
`ρ_max(n)=(n−1)/(2n)` → 1/4, **1/3**, 3/8, 2/5.

수치 검증: 실측 하한이 `M_n`을 **위에서** 접근 (비 1.0000–1.0013).

## 3. 인증서

| 인증서 | 공식 | argmax |
|---|---|---|
| Lipschitz `ε_z_L` | `(4σ/γ)M_n(ρ)` | **1/3** ✓ |
| 전체 L2 `ε_h` | 동일 (Cauchy–Schwarz) | **1/3** ✓ |
| **정확(역-로짓)** | `min (σ/γ)|logit b − logit a|` | **0.296** ✗ |

칸토어 인증서 `= 4σ/(27γ) = 0.1959` (사영 단위).

## 4. 인증서 검증 — **위반 0건**

468개 below-certificate 설정에서 직접 policy switch **0건**.

> 다만 이 상태들에서 Lipschitz 상계는 **6.3× 보수적**이다
> (국소 `|dr/dz|` 중앙값 0.0301 vs 최대 0.1890). λ=1.01에서도 전환이 없어
> **tightness는 이 데이터로 입증되지 않는다.**

## 5. Depth-shift — **지지되지 않음**

첫 시도는 **순환적**이었다: 인증서로 정규화한 격자에서 전환이 전혀 없으면
"실측 임계"가 `λ_max·ε_cert(ρ)`가 되어 argmax가 `ρ_theory`로 **구성상** 나온다.
**공통 절대 격자**로 다시 측정하니:

| n | 실측 peak | 이론 `(n−1)/(2n)` |
|---|---|---|
| 2 | 0.44 | 0.25 |
| 3 | 0.20 | 0.333 |
| 5 | 0.44 | 0.40 |

**corr = −0.064, 중앙값 오차 0.133 → 기각.**

**원인:** 실제 G1 상태는 `τ_beh` 중심 좌표에서 `r ≈ 0.041`, 즉 로지스틱의
포화 꼬리에 몰려 있다. **전역 maximin 여유가 이들의 국소 강건성을 지배하지
않는다.**

## 6. 행동 중심 ablation

| 중심 | 상태 `r` 중앙값 | 국소 `|dr/dz|` | 실측 robustness peak |
|---|---|---|---|
| `τ_mid` (+0.9887) | 0.43 (반응 영역) | — | **0.36** (1/3에 가까움) |
| `τ_beh` (−2.6263) | **0.041 (포화 꼬리)** | 0.0301 | 0.25 |

> **행동적으로 옳은 중심이 상태를 제어기가 덜 구별하는 영역에 놓는다.**

## 7. 생성 (D_final_334, 90 goal, 미접촉)

**D_FINAL 예산 게이트: 6/6 정합** (−1.6 %…+0.9 %) — V3.3.3 결함 수정 확인.

| ρ (τ_beh) | 0.28 | 0.30 | **1/3** | 0.36 | 0.40 |
|---|---|---|---|---|---|
| AUC | 0.20349 | 0.20357 | **0.20349** | 0.19907 | 0.20133 |

max-T 동시 신뢰대: **4/4 모두 SESOI ±0.02 안 (equivalent), 유의 0건.**
AUC 확산 0.0058 ≪ SESOI. 중심 ablation `τ_beh−τ_mid = +0.0057`,
CI [−0.0111, +0.0228], 비유의. benign utility 전 설정 동일 (false refusal 0.5).

## 8. 판정

```
MATH        M1_CANTOR_DEPTH3_MAXIMIN_PROVED
CERTIFICATE C1_CERTIFICATE_VALIDATED
COORDINATE  R2_PARTIAL_COORDINATE_TRANSFER
GENERATION  G3_PRACTICALLY_EQUIVALENT
UTILITY     U_PASS
OVERALL     C_GEOMETRIC_CERTIFICATE_VALID_NO_BEHAVIORAL_TRANSFER
```
