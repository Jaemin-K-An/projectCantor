# V3.6.0 정리와 증명

## 자기유사 정책군의 최소 간격

각 level에서 좌우 자식 길이를 rho배로 유지하고 중앙 guard를 제거한다. 깊이 k에서
새 guard 길이는 `rho^(k-1)(1-2rho)`이다. `0<rho<1/2`이므로 가장 짧은 guard는
깊이 n에서 생성된다. 서로 다른 인접 terminal 정책 사이의 최소 간격은 따라서

`M_n(rho)=rho^(n-1)(1-2rho)`.

이 식과 일반 depth 최적점은 이미 V3.3.2의 AG.1에서 다룬 결과다. V3.6.0은 그 정리를
다시 증명하고 exact residual switch solver와 실제 구현 검증에 연결한다.

## 최적점과 n=1의 예외

n>=2일 때

`M_n'(rho)=rho^(n-2)[(n-1)-2n*rho]`.

양의 인자 rho^(n-2)를 제외하면 기울기는 `(n-1)/(2n)` 왼쪽에서 양수, 오른쪽에서
음수다. 양 끝에서 M_n은 0으로 수렴하므로 이 점은 열린 구간의 유일한 전역 최댓값이다.
`rho_n*=(n-1)/(2n)`이고 `M_n(rho_n*)=((n-1)/(2n))^(n-1)/n`이다.
특히 n=3에서 rho*=1/3, M3=1/27이다. n=1은 별도로 M1=1-2rho, M1'=-2이며,
rho->0+에서 supremum 1을 가지지만 열린 구간 안에는 maximizer가 없다.

## residual 공간으로의 정확한 이전

`d(h)=(w^T h+b)/||w||`, `u=w/||w||`라 하면
`d(h+delta)-d(h)=u^T delta`이다. Cauchy–Schwarz로 거리 변화는 `||delta||` 이하이고,
센서 법선 방향에서 등호가 성립한다. `x=max(0,-d)`도 1-Lipschitz 합성이다.

terminal source는 d<0이다. target terminal interval J는 양의 위험 좌표에 있으므로
정확한 거리는 `dist(x(h),closure(J))`다. lower bound는 Lipschitz로 얻고,
`delta=-(x_target-x(h))*u`로 임의의 target 내부 점을 실현해 upper infimum을 얻는다.
따라서 단순한 보수적 bound가 아니라 unrestricted additive residual 공간의 exact infimum이다.

`D_terminal_switch(h)=min_{J != current leaf} dist(x(h),closure(J))`.

현재 leaf의 끝까지 거리가 D_leave이며, 모든 guard closure까지 거리의 최솟값이
D_guard다. 첫/마지막 leaf에서는 safe/outside가 더 가까울 수 있어 두 양은 다를 수 있다.
`D_terminal_switch>=W_R*M_n`이지만 D_leave나 D_guard는 임의로 0에 가까울 수 있다.

## 경계와 infimum

기존 V3.5.1처럼 내부 끝점은 guard에 속한다. 따라서 leaf 출발점과 다른 leaf
도착점 사이에는 guard 폭보다 엄격히 큰 거리가 필요하다. `||delta||<=W_R*M_n`에서
direct terminal switch가 없다는 해석은 이 경계 규칙 아래 성립한다. 전체 source의
infimum은 guard 양끝에 내부에서 임의로 가까이 접근해 달성하는 극한값이다.
끝점을 양쪽 terminal에 포함하는 다른 규칙에서는 등호에서 switch가 가능하며
인증은 엄격한 `<`로 표현해야 한다. finite precision 구현의 경계 오차는 별도 보고한다.

## family 밖의 반례

8개 leaf의 총 길이를 8/27로 유지하면 guard 총 길이는 19/27이다. 재귀 제약이 없을 때
7개 guard의 최소 길이는 평균 19/189보다 클 수 없고, 균등 배분으로 이를 달성한다.
이는 Cantor의 1/27보다 19/7배 크다. 그러므로 본 최적성은 모든 partition에 대한
주장이 아니며 same-rho symmetric self-similar recursive family에 한정된다.

또한 유한 자연 표본의 `min_h D_terminal_switch(h)`는 전체 영역의 lower envelope가
아니다. 표본의 leaf 내부 위치와 rho별 terminal eligibility가 다르므로 그 순위는
M_n의 순위와 달라질 수 있다. 이 정리는 empirical switch curve의 전역 순위도 보장하지 않는다.
