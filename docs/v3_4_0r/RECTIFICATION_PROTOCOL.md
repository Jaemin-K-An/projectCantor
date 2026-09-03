# V3.4.0R 교정 프로토콜

V3.4.0R은 새 Cantor 가설이 아니라 V3.4.0의 verdict/budget/baseline/censoring 결함을
고치는 외부 확인 실험이다. 역사적 `results/configs/docs/v3_4_0`은 수정하지 않는다.

동결값은 sensor hash `f16942ce…f402f1fe`, actuator hash
`c22957e2…d480172a`, `W=2.2805212277347544`, κ≈0.31022973, depth 3,
rho `{.25,.28,.30,1/3,.36,.40,.44}`, 기존 actions와 attack grid다.
`d0=0`은 class-weighted classifier hyperplane이지 자연 모집단 50% 행동 경계가 아니다.

수리 항목은 statewise `q<=.05`, attacked-state `q_rms=.03` calibration,
`ATTACK_ONLY`, 비재귀 `LINEAR(a(r)=r)`, 올바른 위험 방향(큰 leaf index), 우측 절단과
reversion을 보존하는 실패 분석, 그리고 budget mismatch가 동등성보다 우선하는
판정 기계다.

외부 transport gate와 fixed-W gate는 controller 실험보다 먼저 실행한다. 결과는
sensor discrimination `ST1_PASS`, fixed-window `ST3_WINDOW_SHIFT`다. 후자가 hard
stop이므로 budget/freeze/final을 확증 단계로 실행하지 않는다. 앞선 로컬 작업이
이 순서를 어기고 만든 q=.025 및 final 산출물은 모두 audit-only이며 최종 판정에
쓰지 않는다.
