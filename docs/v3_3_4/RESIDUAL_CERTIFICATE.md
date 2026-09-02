# 잔차 강건성 인증서 — 범위를 먼저 못박는다

## 인증서가 말하는 것 / 말하지 않는 것

**허용:** certified residual policy-switch margin · certified safety-coordinate
transition radius · certified one-dimensional residual robustness

**금지:** certified LLM safety · guaranteed jailbreak immunity ·
mathematically guaranteed harmless generation · universal safety proof

인증서는 오직 **이 층·이 거부 방향·이 행동적 중심·이 1차원 제어기**에서
**직접 terminal-policy 전환**에 대한 것이다.

## 정리 L — 사영 인증서

`r(z) = σ(−γ(z−τ_beh)/σ)` ⟹ `dr/dz = −(γ/σ)r(1−r)`, `r(1−r) ≤ 1/4` ⟹
```
|Δr| ≤ (γ/4σ)|Δz|
```
직접 전환에는 `|Δr| ≥ M_n(ρ)`가 필요하므로 충분조건:
```
ε_z_cert(ρ,n) = (4σ/γ)·M_n(ρ) = (4σ/γ)·ρ^{n−1}(1−2ρ)
```

## 정리 H — 전체 잔차 L2 인증서

`Δz = ⟨Δh, v⟩`, `‖v‖=1` ⟹ Cauchy–Schwarz `|Δz| ≤ ‖Δh‖₂` ⟹
```
‖Δh‖₂ < ε_z_cert  ⟹  직접 전환 불가
```
**동일 반경이 임의 잔차 섭동에 대한 충분조건이다.**

## 따름정리 L.1 — 칸토어

`4σ/γ`는 ρ와 무관하므로 `argmax_ρ ε_z_cert(ρ,3) = 1/3`, 그리고
```
ε_cert,Cantor = 4σ / (27γ)
```
수치 확인: Lipschitz 인증서의 argmax = **0.3333** ✓

## 정확한 역-로짓 인증서 — **1/3이 아니다**

변환이 강단조이므로 `z(r) = τ_beh − (σ/γ)logit(r)`, guard `[a,b]`를 완전히
건너는 정확한 사영 변위는 `(σ/γ)|logit(b) − logit(a)|`이고
`ε_z_exact = min over 분리 guards`.

> **측정: `argmax ε_z_exact ≈ 0.296`, 1/3이 아니다.**
> 로지스틱 뒤틀림이 최적점을 옮긴다. 이는 harness §9가 예상한 결과 (B)이며
> 그대로 보고한다. **칸토어의 정확한 최적성은 `r`-공간 여유와 Lipschitz
> 인증서에 대해서만 성립한다.**

`ε_exact ≥ ε_Lipschitz` 는 모든 ρ에서 성립 (Lipschitz는 보수적).
`ε_exact`는 `τ`와 무관하다 — z-공간 **폭**만 관여하기 때문.
