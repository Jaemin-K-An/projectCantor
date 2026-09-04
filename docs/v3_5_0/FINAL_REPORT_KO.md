# Project Cantor V3.5.0 최종 보고서

## 결론

V3.5.0은 one-sided risk geometry 자체는 성공적으로 구현·검증했지만, 명세에 고정된 `q_target_rms=.03`과 `q_cap=.05`, 그리고 안전측 exact zero-action을 동시에 만족하는 attacked-state budget이 존재하지 않았다. 판정은 `BUD2_MISMATCH`, 전체 결과는 `INCONCLUSIVE`다. 최종 controller efficacy 실험은 수행하지 않았다.

## 무엇이 성립했는가

V3.4.0R의 frozen refusal sensor와 causal actuator는 hash까지 그대로 상속됐다. 새 좌표 `x=max(0,-d)`는 positive safe tail을 정확히 0으로 접어 대칭 window의 safe-side overflow를 제거한다. 1-Lipschitz 정리와 depth-3 certificate

`epsilon_R(rho)=W_R rho²(1-2rho)`

를 구현했고, `rho=1/3`의 유일한 구조적 최대와 0건의 구현 위반을 확인했다. Fresh 300개 calibration에서 `W_R=0.388282882`, `epsilon_R,C≈0.01438085`를 얻었다.

## 왜 중단했는가

새 architecture의 핵심인 `d>=0 → q=0` 때문에 실제 attacked calibration distribution의 약 74%는 Cantor 팔에서 positive action을 받을 수 없다. `q_cap=.05`에서 middle-third arm의 가능한 최대 RMS는 약 0.025388이며, linear arm도 약 0.025464에 그친다. 목표 0.03은 어떤 eta로도 도달하지 못한다.

V3.4.0R의 invalid `.025` artifact가 `.03` 가능성을 시사했다는 전제는 새 architecture에는 그대로 옮겨지지 않는다. 과거 대칭 controller는 safe-side outside-window 상태에도 maximum action을 주었지만, V3.5.0은 바로 그 동작을 금지하기 때문이다. 즉 safe-side pathology를 제거한 결정이 전체-population RMS의 상한도 함께 낮췄다.

명세는 `.03`이 불가능하면 중단하고 다른 target을 찾지 말라고 명시한다. 따라서 target을 낮추거나, large-epsilon 조건의 가중치를 늘리거나, safe state에 작은 action을 주거나, grid를 budget 결과에 맞춰 수정하지 않았다. Formal freeze 상태는 `NOT_FROZEN_BUDGET_IMPOSSIBLE`이며 final prompt output은 한 번도 생성되지 않았다.

## 해석

이번 결과는 one-sided Cantor geometry의 수학적 실패가 아니다. 또한 controller가 행동적으로 inert하거나 harmful하다는 증거도 아니다. Frozen 설계 제약들의 조합이 equal-budget confirmatory experiment를 정의하지 못했다는 feasibility 결과다.

따라서 다음 문장은 주장할 수 없다: “Cantor controller가 attack-only보다 개선했다.” 반대로 “Cantor controller가 실패했다”도 주장할 수 없다. 허용되는 결론은 다음과 같다.

> 외부 검증된 refusal sensor를 one-sided risk coordinate로 변환하고 depth-3 Cantor policy 및 residual certificate를 구현했다. 그러나 안전측 zero-action과 `.05` hard cap 아래에서 사전 고정한 `.03` attacked-state RMS budget은 수학적으로 달성 불가능했으므로 행동 효능 실험은 수행되지 않았다.

SUCCESS-A, SUCCESS-B, SUCCESS-C는 모두 `false`다.
