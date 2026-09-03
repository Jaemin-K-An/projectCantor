# V3.3.4 감사 — 아홉 항목, 전부 코드로 확인

> **V3.3.4의 negative 결과는 하나도 삭제·수정하지 않는다.**
> V3.3.5는 "V3.3.4가 틀렸다"가 아니라 **비선형 좌표 왜곡과 위상 불일치를
> 제거한 새로운 prospective 설계**다.

| # | 항목 | 확인 |
|---|---|---|
| A | `M_n(ρ)=ρ^{n−1}(1−2ρ)`, `argmax M_3 = 1/3` | 성립 (보존) |
| B | Lipschitz 인증서 `(4σ/γ)M_n`이 1/3에서 최대 | 성립 |
| **C** | **exact 역-로짓 인증서의 최적 ≈ 0.296** | **성립 — 보존** |
| **D** | controller가 G1-only가 아님 | `phase_generation.py:94` `if active:` — forward index 게이트 없음 ⟹ PREFILL·G1·G2+ 전부 개입 |
| **E** | cross-phase z 재구성 버그 | `guarded_policy.py` `z = m*sigma + tau_beh` — hook은 PREFILL calibration으로 m을 계산 |
| **F** | `τ_beh`가 global dose 결과 재사용 | G1-local 인과 경계 아님 |
| **G** | 인증서 공격이 offline 산술 | `z1 = z0 + sgn*eps` — forward에 주입 안 함 |
| **H** | safety primary가 `safe_score32` | semantic harmful-compliance 아님 |
| **I** | **D_final 결과로 비교 대상 사후 선택** | `analyse_generation.py:66` `best = min(rows, key=...)` — §51 위반 |

## 왜 1/3이 exact z-space 최적이 아니었나

Lipschitz 인증서는 `M_n`의 **상수배**이므로 argmax를 그대로 물려받는다.
그러나 exact 폭 `(σ/γ)|logit b − logit a|`는 상수배가 아니다 —
`|dr/dz| = (γ/σ)r(1−r)`가 **위치에 따라 변하기** 때문이다.
서로 다른 `r`에 놓인 guard들이 `z`에서 다르게 늘어나므로 최적이 옮겨간다.
**V3.3.5는 이 왜곡 자체를 제거한다.**
