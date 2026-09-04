# One-sided risk 가설

V3.5.0의 새 prospective 가설은 refusal sensor의 안전측 양의 꼬리를 controller metric에 배정하지 않고, 위험 half-space만 직접 제어하면 Cantor policy를 실제 inference-time controller로 사용할 수 있다는 것이다.

Frozen signed distance `d(h)`는 클수록 safe/refusal이며 작을수록 compliance/risk이다. 새 좌표는

`x(h)=max(0,-d(h))`

이다. 따라서 `d>=0`인 모든 상태는 `x=0`과 정확한 zero action을 받는다. `d<0`에서만 위험 깊이를 양수로 측정한다. `x>W_R`은 clipping하거나 undefined 처리하지 않고 maximum frozen safety action을 적용한다.

이 변경은 sensor, actuator, layer, token, model 또는 rho를 재탐색하지 않는다. 변경되는 것은 대칭 domain이라는 단 하나의 architecture assumption이다.
