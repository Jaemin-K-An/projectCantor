# V3.3.5c 사후 용어 정정 (additive erratum)

이 문서는 V3.4.0 작업 중 발견된 **용어상의** 두 가지 정정만 추가한다.
V3.3.5c의 원 데이터, 표, 판정, 보고서는 하나도 수정하지 않는다.

## 정정 1 — "경계 실패"의 범위

V3.3.5c가 보인 것은 다음 두 가지이며, 서로 다르다.

- **재현된 것**: 정규화된 P0 개입의 **인과 효과**. CONFIRM 거부율이
  `0.117 → 0.700`으로 움직였고 Spearman p = 1.4e-17, coherence 0.998,
  degeneration 0.26 %, major reversal 0건, 0.5 전이 bracket됨.
- **재현되지 않은 것**: **동일한 조향 방향 위의 전역 절대 사영**
  `z = <h, v>`에 대한 정밀한 공유 행동 문턱.

따라서 정확한 서술은 "P0에 행동 경계가 없다"가 아니라
**"조향 방향 v 위의 전역 절대 사영이 안정된 공유 문턱을 주지 않았다"**이다.

## 정정 2 — `C3_WINDOW_APPLICABILITY_FAILURE`

`final_verdict.json`의 CERTIFICATE 필드는 `C3_WINDOW_APPLICABILITY_FAILURE`로
기록되어 있다. 이 라벨은 **경험적 window 실패를 뜻하지 않는다.** window·인증서
단계는 **행동 gate 실패로 실행되지 않았다**.

V3.4.0 이후 이 상태를 회고적으로 인용할 때는

    C0_NOT_RUN_BEHAVIORAL_GATE

를 사용한다. 원본 JSON은 역사적 기록으로 그대로 둔다.

## 이 문서가 하지 않는 것

- 실패한 경계 결과를 삭제하거나 긍정으로 재해석하지 않는다.
- 미사용 `D_final_P0_335c` 90개를 증거로 전환하지 않는다.
- V3.3.5c의 `OVERALL = E_P0_BEHAVIORAL_ANCHOR_NOT_REPLICATED`는 유효하다.
