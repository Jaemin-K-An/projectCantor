# Controller 설계

파이프라인은 attacked P0 residual을 직접 소비한다.

`h_attacked → d(h_attacked) → x=max(0,-d) → r_R=x/W_R → depth-3 Cantor cell → q_ctrl ||h_attacked|| v_safe`

동작 schedule은 `[0,1/7,2/7,3/7,4/7,5/7,6/7,1]`이고 guard는 인접 terminal policy의 더 큰 동작을 사용한다. `d>=0`이면 정확히 0, `x>W_R`이면 1이다. 최종 개입은 `q_ctrl=min(eta*a(cell),0.05)`로 statewise cap을 적용한다.

Matched linear baseline은 같은 `x`, `W_R`, sensor, actuator, cap을 사용하되 window 내부에서 `a_linear=r_R`만 사용한다. Harmful final에는 `ATTACK_ONLY` arm이 별도로 존재하도록 구현되어 있다. 두 final runner는 `PRE_ANALYSIS_FROZEN`이 아니면 모델을 로드하기 전에 중단한다.

이번 실행에서는 budget hard stop 때문에 eta가 존재하지 않아 final runner가 실행되지 않았다.
