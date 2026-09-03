# 동결된 P0 Cantor 인증서

대칭 이진 재귀 분할의 깊이 n direct terminal-policy margin은

`M_n(rho)=rho^(n-1)(1-2rho)`이다.

깊이 3에서 `M_3=rho^2(1-2rho)`이고
`M_3'=2rho(1-3rho)`이다. `(0,1/2)`에서 도함수는 1/3 전에는 양, 이후에는
음이므로 `rho=1/3`이 유일한 최대점이고 `M_3(1/3)=1/27`이다.

고정 window에서 metric-preserving coordinate는
`r(z)=1/2+s(z-tau)/(2W)`이며 `|Delta r|=|Delta z|/(2W)`이다. 따라서

`epsilon_z(rho)=2W rho^2(1-2rho)`, `epsilon_C=2W/27`.

같은 tau와 W를 모든 rho에 쓰므로 최대점은 바뀌지 않는다. 이 값은 controller
gain eta와 무관하다.

또한 `z=<h,v>`, `||v||=1`이면 Cauchy–Schwarz로
`|Delta z|<=||Delta h||_2`이다. 따라서
`||Delta h||_2 < epsilon_z(rho)`는 정의된 P0 coordinate에서 direct terminal
policy switch를 막는 충분조건이다. 올바른 명칭은 **certified P0 residual
direct-policy-transition radius**이다. 이는 보편적 LLM 안전 인증서도, semantic
jailbreak 방어 보장도 아니다. 실제 공격에서 위반 0건이 나오더라도 분석 정리의
구현 검증이지 독립적인 의미 안전성 증명이 아니다.
