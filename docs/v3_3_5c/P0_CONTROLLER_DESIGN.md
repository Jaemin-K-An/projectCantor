# P0 Cantor controller 설계

`P0CantorSafetyController`는 정규화 margin을 받지 않고 실제 P0 residual `h`를
직접 받는다. 내부 순서는 `z=<h,v>` → affine coordinate → Cantor cell → 동결
policy action → `h_corrected=h+delta_h_controller`이다.

깊이는 3, rho family는
`{0.25,0.28,0.30,1/3,0.36,0.40,0.44}`이다. 각 분할은 8 terminal leaves와
7 guards를 갖는다. risk coordinate 순서의 leaf action은
`[0,1/7,2/7,3/7,4/7,5/7,6/7,1]`로 모든 rho에서 같다. guard는 양옆 leaf 중
더 강한 safety action을 취하므로 `r=0.5`를 포함하는 central guard도 보수적으로
개입한다. window 밖에서는 affine를 extrapolate/clip하지 않고 `OUTSIDE_WINDOW`와
action 1을 반환한다.

벡터는 하나뿐이며 correction은 동결된 `s_safe*v` 방향이다. q는
`||delta_h_controller||/||h_P0||`; eta는 budget split에서 rho별로 맞추되 모든
rho가 같은 RMS target을 쓴다. 실제 final RMS가 target의 ±3% 밖이면 해당 rho는
equal-budget 비교에서 제외된다.

모델 hook의 순서는 clean residual → unsafe projection attack → attacked residual을
controller가 관찰 → 실제 correction → token-1 logits → full decode이다. hook은
prefill의 mask-indexed last prompt token에서 한 번만 작동하고 G1 이후에는 residual을
바꾸지 않는다.

구현과 단위/integration test는 완료했지만 corrected behavioral gate가 B3이므로
실제 배포 controller의 tau/W/eta는 동결하지 않았다. 이 설계를 final generation에
적용하지 않았으며, config 상태는 `NOT_DEPLOYABLE_BEHAVIORAL_GATE_FAILED`이다.
