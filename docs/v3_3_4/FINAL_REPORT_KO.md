# 중삼분 칸토어 기반 인증형 LLM 잔차 안전 제어 (V3.3.4)

## 1. 최종 연구 질문과 답

> 중삼분 Cantor recursive partition을 실제 LLM의 행동적 안전 좌표에
> 적용했을 때, 동일한 depth-3 이진 재귀 제어기 중 adversarial policy
> transition에 대한 **certified residual robustness radius를 최대화하는가**?
> 그리고 그 certificate가 실제 잔차 공격과 생성 안전성에서도 관찰되는가?

**앞부분은 예. 뒷부분은 부분적으로만.**

칸토어는 이번에 **실제 추론 시점 안전 제어기**(`CantorGuardedPolicy`)로
구현되어 LLM 생성 안에서 동작했고, 그 인증서는 **한 번도 위반되지 않았다.**
그러나 인증서가 근거하는 **전역 maximin 여유는 모델이 실제로 만들어내는
상태들의 강건성을 지배하지 않았고**, 생성 수준 이득은 관측되지 않았다.

## 2. 여섯 판정 — 절대 합치지 않는다

```
MATH        M1_CANTOR_DEPTH3_MAXIMIN_PROVED
CERTIFICATE C1_CERTIFICATE_VALIDATED
COORDINATE  R2_PARTIAL_COORDINATE_TRANSFER
GENERATION  G3_PRACTICALLY_EQUIVALENT
UTILITY     U_PASS
OVERALL     C_GEOMETRIC_CERTIFICATE_VALID_NO_BEHAVIORAL_TRANSFER
```

## 3. 수학 (M1)

깊이 `n`의 terminal leaf **내부**에서 다른 leaf로 가는 최소 좌표 섭동:
```
M_n(ρ) = inf_{r∈L_n°} d_leaf(r) = ρ^{n−1}(1−2ρ)
```
**infimum이지 minimum이 아니다** — leaf 내부가 열린 집합이므로 도달되지 않는다.
실측 하한이 항상 위에서 접근함을 확인했다 (비 1.0000–1.0013).

```
M_3′(ρ) = 2ρ(1−3ρ)  ⟹  ρ = 1/3 유일 최대,  M_3(1/3) = 1/27
ρ_max(n) = (n−1)/(2n)  →  1/4, 1/3, 3/8, 2/5
```
> **1/3은 깊이 3에서만 특별하다.** 다른 깊이에서 최적은 1/3이 아니다.

## 4. 인증서 (C1)

`|dr/dz| ≤ γ/(4σ)` 이므로 직접 전환을 막는 충분조건:
```
ε_z_cert(ρ,n) = (4σ/γ)·M_n(ρ)          ε_h_cert = ε_z_cert  (Cauchy–Schwarz)
```
`4σ/γ`가 ρ와 무관하므로 **`argmax ε_cert(·,3) = 1/3`**, 칸토어 값 `4σ/(27γ)`.

**검증: 468개 below-certificate 설정에서 직접 policy switch 0건.**

**정직한 단서 둘:**
1. **정확한 역-로짓 인증서의 최적점은 0.296이지 1/3이 아니다.** 로지스틱
   뒤틀림이 최적을 옮긴다. 칸토어의 정확한 최적성은 `r`-공간 여유와
   **Lipschitz** 인증서에 한정된다.
2. 이 상태들에서 Lipschitz 상계는 **6.3× 보수적**이며, λ=1.01에서도 전환이
   없어 **tightness는 입증되지 않았다.**

## 5. 좌표 전이 (R2) — depth law는 지지되지 않았다

첫 depth-shift 검정은 **순환적**이었다(인증서 정규화 격자에서 전환이 없으면
argmax가 구성상 `ρ_theory`가 된다). **공통 절대 격자**로 다시 하니
**corr = −0.064, 중앙값 오차 0.133 → 기각.**

**원인이 분명하다:** `τ_beh` 중심 좌표에서 실제 G1 상태는 `r ≈ 0.041`,
로지스틱의 **포화 꼬리**에 몰려 있다. 국소 기울기가 최대의 1/6이므로
**전역 maximin 여유는 이 상태들의 국소 강건성과 무관하다.**

## 6. 중심 ablation — V3.3.3의 직접 수정

| 중심 | 상태 `r` | 실측 robustness peak |
|---|---|---|
| `τ_mid` | 0.43 (반응 영역) | **0.36** |
| `τ_beh` | **0.041 (포화)** | 0.25 |

> **행동적으로 옳은 중심이 상태를 제어기가 덜 구별하는 영역에 놓는다.**
> V3.3.3이 τ_beh를 적용하지 않은 것은 오류였지만, 적용해도 생성 이득은
> 생기지 않았다 (`+0.0057`, CI [−0.0111, +0.0228], 비유의).

## 7. 생성 (G3) — 실질적 동등

미접촉 `D_final_334` 90 goal, **D_FINAL 예산 6/6 정합**(V3.3.3 결함 수정).
max-T 동시 신뢰대에서 **4/4 전부 SESOI ±0.02 안, 유의 0건**,
AUC 확산 0.0058. benign utility 전 설정 동일.

## 8. 최종 결론

> **중삼분 칸토어 구조를 LLM의 행동적 거부 경계를 중심으로 하는 잔차 안전
> 좌표 위의 추론 시점 guard 제어기로 구현하였다. 깊이 3의 이진 재귀 제어에서
> terminal policy 간 직접 전환에 필요한 worst-case 좌표 여유는
> `M_3(ρ)=ρ²(1−2ρ)`이고 `M_3′=2ρ(1−3ρ)`로부터 `ρ=1/3`에서 유일 최대임을
> 증명했으며, 좌표 변환의 Lipschitz 상계로부터
> `ε_cert=(4σ/γ)ρ²(1−2ρ)`라는 잔차 공간 인증서를 도출했다. 따라서 중삼분
> 제어기는 동일 depth-3 족 중 **가장 큰 certified residual attack radius**를
> 제공하며, 실제 잔차 공격에서 인증서 이하 구간의 직접 policy switch는
> **한 건도 관측되지 않았다.**
>
> **그러나 이 1차원 policy-transition 인증서는 텍스트 수준의 semantic
> safety 이득으로 이어지지 않았다.** 정확한 역-로짓 인증서의 최적은 1/3이
> 아니고, depth law는 실제 상태에 전이되지 않았으며, 동일 예산 생성 비교는
> 실질적 동등이었다.

**즉 칸토어는 이번에 실제로 LLM 안전 제어에 적용되었고 인증 가능한 강건성
성질을 제공했다. semantic 생성 이득은 별개의 문제로 남는다.**

## 9. 한계

1. Model A 단독. 두 번째 모델 복제 미실행 → 외적 타당성 제한.
2. `D_final_334`는 AdvBench 계열로 **JBB·XSTest와 다른 세 번째 모집단**이다.
3. `τ_beh`는 V3.3.3의 **global** dose 추정치를 재사용했다. phase-local G1
   재추정은 미실행 — 이는 남은 결함이며 결론을 그만큼 제한한다.
4. 인증서 tightness 미입증(6.3× 보수적, λ=1.01에서도 전환 없음).
5. 안전 채점은 coherence-gated lexical 지표다. semantic evaluator 검증
   미실행 → **refusal-proxy 수준**으로 제한한다.
6. depth-shift 첫 시도가 순환적이었고, 수정판은 DEV에서만 돌았다.

## 10. V1 → V3.3.4

```
V1–V3.2   raw safety 우월성 없음
V3.3      구조적 우월성 없음
V3.3.1    균형 κ=1에서 ρ=1/3 유일 최적 (수학)
V3.3.2    측정된 불확실성은 ρ≈0.46 선택
V3.3.3    행동 경계가 사영 중점과 3.90 σ 어긋남
V3.3.4    깊이 3 maximin 정리 + 잔차 인증서 증명·검증
          → Cantor가 실제 제어기로 동작, 인증서 위반 0건
          → 그러나 생성 이득 없음, depth law 전이 실패
```

**칸토어에 대한 최종 답: 기하와 그 인증서는 정확히 성립하고, 행동으로의
전이는 성립하지 않는다.**
