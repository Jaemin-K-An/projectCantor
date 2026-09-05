# V3.6.0 certificate의 정확한 범위

인증 대상은 frozen signed Euclidean sensor를 사용하는 깊이 3 대칭·동일 rho 재귀
라우터의 서로 다른 terminal leaf 사이 전환이다. 허용 공격은 고정 P0 residual에
대한 임의의 additive L2 perturbation이다. 자연 언어 prompt로 그 perturbation을
실현할 수 있다는 주장 또는 모델 manifold 위의 최적 공격이라는 주장은 아니다.

safe side, guard, outside는 terminal policy와 다른 route type이다. guard가 보수적
인접 action을 적용하더라도 guard 진입은 terminal switch가 아니다. action이 같아도
route label이 달라질 수 있고, route가 같아도 residual과 logits는 달라질 수 있다.
이를 action 안정성, 전체 출력 불변성 또는 거부 행동의 보장으로 해석해서는 안 된다.

`D_leave`, `D_guard`, `D_terminal_switch`를 따로 저장한다. raw floating-point
violation과 solver error를 분리하며, error tolerance를 위반 은폐에 사용하지 않는다.
guard 경계를 guard에 배정하므로 certificate와 같은 norm까지 terminal-to-terminal
switch가 없지만, infimum은 일반적으로 특정 source/target 쌍에서 달성되지 않는다.

허용 주장은 “depth-3 symmetric self-similar recursive family 내부에서 middle-third가
최소 terminal separation을 유일하게 최대화한다”이다. 모든 partition 최적성,
semantic safety, linear 대비 응답 우월성, 모든 depth의 rho=1/3 최적성은 금지한다.
V3.5.1의 negative behavioral result는 보존한다.

실제 residual의 관측 최소 거리 검정은 B에 따로 반영한다. 투영 probe를 자연 표본으로
섞거나 B가 실패했을 때 이론적 envelope로 대체하지 않는다. secondary는 과거 사용한
prompt의 경계 투영 출력 실험이므로 C가 양성이어도 자연 분포 행동 우월성 주장이 아니다.
