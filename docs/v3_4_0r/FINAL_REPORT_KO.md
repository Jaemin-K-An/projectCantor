# Project Cantor V3.4.0R 최종 보고서

## 결론

V3.4.0R은 외부 확인 실험의 첫 번째 절반만 통과했다. V3.4.0의 frozen refusal-state
sensor는 `LLM-LAT/harmful-dataset`에서 AUROC 0.8992, 95% CI
[0.8291, 0.9562], `d0=0` balanced accuracy 0.7348을 기록해 판별력은 transport했다.

그러나 동일한 frozen coordinate를 controller에 배치하는 데 필요한 window
applicability는 통과하지 못했다. 역사적 `W=2.2805212277347544` 안에 든 상태는
150개 중 130개(0.8667)로 사전 기준 0.90보다 작았다. sensor를 재학습하거나 W를
키우면 frozen V3.4.0 architecture의 외부 transport를 시험한다는 질문 자체가 바뀐다.
따라서 이를 하지 않고 `ST3_WINDOW_SHIFT`에서 중단했다.

## 재개 감사와 무효화

GitHub의 최신 공개 실험 브랜치는 V3.4.0이었고, 로컬 V3.4.0R에는 미공개 선행 작업이
남아 있었다. 그 작업은 두 가지 핵심 고정값을 어겼다. W를 `2.2805459…`로 복사했고,
사양에 없는 clipping≤10% 조건을 넣어 q target을 0.03에서 0.025로 낮췄다. 더구나
fixed-W gate를 실행하지 않은 채 budget/evaluator/certificate/final로 진행했다.

잘못 생성된 harmful final 11,600행은 삭제하지 않고 audit-only 파일로 격리했다.
그 출력의 refusal이나 controller 결과는 보지 않고, 기계적 최종 판정에서도 읽지
않는다. formal freeze는 `NOT_FROZEN_EXTERNAL_WINDOW_SHIFT`로 수정했고 canonical
final output은 존재하지 않는다.

## 과학적 해석

허용되는 결론은 다음과 같다.

> The frozen refusal-state sensor retained strong discrimination on the external
> harmful-prompt population, but the frozen affine window did not meet its
> preregistered applicability requirement. Controller efficacy and Cantor-specific
> behavioural value were therefore not tested confirmatorily.

최종 판정은 **`E_EXTERNAL_SENSOR_TRANSPORT_FAILURE`**다. 여기서 “failure”는 sensor
AUROC 실패가 아니라 state machine이 ST3 window shift를 같은 external transport
실패 family로 분류한다는 뜻이다.

구조적 결과는 별도로 남는다. 고정 W에서
`epsilon_h(rho)=2W rho^2(1-2rho)`는 `rho=1/3`에서 유일 최대다. 이는 residual
policy-transition certificate이지 경험적 행동 최적이나 semantic safety 보증이 아니다.
