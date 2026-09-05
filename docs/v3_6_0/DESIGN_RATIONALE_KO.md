# V3.6.0 설계 근거 — Cantor-Certified Recursive Risk Routing

V3.5.1의 `CTRL4_INCONCLUSIVE`, `LIN2_EQUIVALENT`, `RHO2_EQUIVALENT`를 계승한다.
V3.6.0의 primary endpoint는 다른 terminal policy에 도달하는 residual-L2 거리의
infimum이다. 행동 우월성을 다시 확인하기 위한 연구가 아니다.

## 정책군의 정의

`F_n`은 [0,1]에서 매 node와 level마다 같은 rho를 사용하는 대칭 이진 재귀 분할이다.
부모 [a,b]의 두 자식은 [a,a+rho(b-a)]와 [b-rho(b-a),b]이고, 그 사이가 guard다.
내부 경계는 guard 소유로 한다. terminal 내부에는 이진 주소와 개입 수준을 부여한다.
깊이 k+1의 retained set은 깊이 k의 retained set에 포함되며, 각 자식 주소의 prefix가
부모 주소다. 정책 라벨 자체가 같은 채로 집합이 포함된다는 뜻은 아니다. 기존 guard는
세분화 후에도 유지되고, 부모 정책은 더 세밀한 자식 정책들로 정제된다.

이 제약은 일반 decision tree와 다르다. 일반 tree는 node마다 임계값, 좌우 길이,
깊이가 달라도 된다. arbitrary threshold와 ordinary bin은 인접 영역이 닿을 수 있고,
공통 축척의 재귀 refinement 또는 명시적 guard를 요구하지 않는다. 일반 tree도 이
구조를 구현할 수 있지만, tree라는 자료구조만으로 이 보증이 따라오는 것은 아니다.

## 깊이와 좌표의 한계

깊이 3은 8개 정책을 주고 guard 계층을 해석하기 쉬운 설계 절충이다. 깊이 1은
2개, 깊이 2는 4개, 깊이 4 이상은 16개 이상의 정책을 주지만 최소 간격이 작아진다.
깊이 3의 경험적 최적성을 주장하지 않는다. depth별 최적 rho와 residual radius는
`depth_optima.json`에 수치로 기록한다.

V3.3.2 `U_EST`의 q50=0.02527649, q95=0.07218206은 이전 사영의 정규화 좌표다.
현재 frozen V3.4.0 센서의 거리 오차가 아니다. 이를 W_R로 곱한 값은 단위 환산의
예시일 뿐, 새로운 sensor uncertainty bound나 depth 선택의 경험적 증거가 아니다.

## 검증과 대조 설계

고정 자산은 Qwen2.5-0.5B, layer 14 P0, V3.4.0 sensor, V3.3.5a actuator와
V3.5.1 W_R=1.3727408729684236이다. 새로 학습하거나 보정하지 않는다.
기존 clean residual 2,092개를 자연 상태 집합으로 재사용한다. 이들은 새로운
확증 행동 표본이 아니며, 고정 좌표의 수학적 구현 점검에 사용한다.

모든 200개 risk calibration anchor의 센서 직교 성분을 유지하고, 각 rho의
8개 leaf 내부 10개 위치로 투영해 112,000개 state-boundary trial을 만든다.
이 자료를 112,000개의 독립적인 자연 LLM 상태라고 부르지 않는다. 자연 상태의
관측 최소 거리, 투영 probe의 관측 최소 거리, 전체 영역의 이론적 infimum을 구분한다.

NO_GUARD는 8개 균등 bin이다. NON_RECURSIVE_GUARD는 middle-third와 동일한
8개 leaf 및 총 retained 길이 8/27을 가지지만, 19/27의 guard 길이를 7개에 균등
배분한다. 이 대조군은 재귀 제약을 풀면 Cantor보다 큰 최소 간격이 가능함을 검증한다.

## 통계와 성공 판정

정리는 미분과 유리수 계산으로 판정하며 p-value를 붙이지 않는다. solver의 절대
오차 허용값은 1e-9, 상대 오차는 1e-6이다. 위반 횟수는 이 tolerance로 보정하지 않고
원시 값으로 센다. guard 진입, terminal 전환, outside/safe 전환을 분리한다.

A는 정리와 solver·certificate 검증, B는 A와 자연 terminal 상태의 관측 최소 거리
엄격 우위, C는 B와 secondary association의 사전 고정된 음의 slope 구간으로 정의한다.
이론은 B의 유한 표본 순위 또는 C의 행동 결과를 보장하지 않는다.

secondary는 고정 risk calibration 순서의 최초 12개 prompt에 대해 boundary projection을
적용한다. 0.9/1.1/2.0배 자체 certificate와 공통 absolute 1.1배 middle-third
certificate에서 routed/locked policy를 비교한다. 기존 eta, q cap 0.05를 계승하고,
greedy 32 token을 생성한다. 같은 perturbed residual에서 원래 policy를 고정한
대조를 두어 입력 섭동 효과와 policy action 차이를 구별한다. 20,000회 prompt-cluster
bootstrap과 max-T를 사용한다. historically used prompt이므로 명시적으로 exploratory다.
