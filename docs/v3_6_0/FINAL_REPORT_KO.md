# Project Cantor V3.6.0 — Cantor-Certified Recursive Risk Routing

## 연구 질문과 V3.5.1의 계승

V3.6.0의 primary contribution은 Cantor-specific certified policy stability다.
V3.5.1의 Cantor 1/3 대 ATTACK_ONLY AUC 차이 +0.020695, 대 LINEAR 차이
-0.001876, `CTRL4_INCONCLUSIVE / LIN2_EQUIVALENT / RHO2_EQUIVALENT` 결과를
그대로 보존한다. 이번 연구는 행동 우월성에 관한 그 결론을 뒤집지 않는다.

기준 HEAD는 `9e525e886fcf68bff61a8735b2e56c1d8d145908`, 연구 브랜치는
`cantor-guard-v3.6.0`이다. Qwen2.5-0.5B-Instruct의 고정 revision, layer 14,
padding-safe P0, V3.4.0 sensor, V3.3.5a actuator, W_R=1.3727408729684236을
계승했다. 모델·센서·액추에이터·W_R·기존 eta를 재학습하거나 보정하지 않았다.

## 정책군과 정리

`F_n`은 시작 구간 [0,1]에서 모든 node와 level에 동일한 rho를 쓰는 대칭 이진
자기유사 재귀 정책군이다. retained interval은 terminal policy, 제거된 interval은
guard가 된다. 자식의 이진 주소는 부모 주소를 prefix로 가져 계층적 refinement를
유지한다. guard는 보수적 인접 action을 사용하고 내부 끝점도 guard에 배정한다.

최소 terminal separation은 `M_n(rho)=rho^(n-1)(1-2rho)`이며 n>=2에서
`M_n'=rho^(n-2)[(n-1)-2n*rho]`다. 유일한 최적점은 `(n-1)/(2n)`이다.
깊이 3에서는 정확히 `rho*=1/3`, `M3*=1/27`이다. n=1은 예외로,
rho->0+의 supremum 1만 존재하고 열린 구간 내부의 maximizer는 없다.
500,001점 rho grid에서도 1/3이 유일한 최대였다.

이 일반 depth 식은 V3.3.2에서도 이미 증명했다. 이번 버전의 기여는 해당 정리를
명시적 routing semantics, exact residual switch solver, adversarial 구현 검증에
연결하고 보증의 범위를 구분한 것이다. 모든 partition의 보편적 최적성은 주장하지 않는다.

## depth=3 선택의 근거

| 깊이 | 정책 수 | 최적 rho | 최대 normalized margin | residual-L2 radius |
|---:|---:|---:|---:|---:|
| 1 | 2 | 없음, 0+ 극한 | 1, supremum | 1.372741, supremum |
| 2 | 4 | 0.25 | 0.125000 | 0.171593 |
| 3 | 8 | 1/3 | 0.037037 | 0.050842 |
| 4 | 16 | 0.375 | 0.013184 | 0.018098 |
| 5 | 32 | 0.40 | 0.005120 | 0.007028 |
| 6 | 64 | 5/12 | 0.002093 | 0.002873 |

깊이 3은 8개 대응 수준과 해석 가능한 guard 계층을 제공하는 사전 설계 절충이다.
깊이 4 이상은 정책 해상도를 늘리면서 가장 작은 간격을 빠르게 줄인다. 현재 센서에서
깊이 3이 경험적으로 최적이라고 검증한 것은 아니다.

V3.3.2 추정량 불확실성 U_EST의 normalized q50=0.02527649, q95=0.07218206은
이전 사영 좌표에서 측정됐다. 현재 W_R와 단순 곱한 q50=0.03469807은 단위 환산의
예시이며, V3.4.0 frozen sensor의 실제 residual 거리 오차로 사용할 수 없다.
동일 센서의 불확실성을 새로 측정하지 않았다는 한계를 명시한다.

## exact distance와 인증 범위

`d=(w^T h+b)/||w||`, `x=max(0,-d)`는 residual L2에 대해 1-Lipschitz다.
terminal source와 target interval J에 대해 거리의 정확한 infimum은
`dist(x(h),closure(J))`다. sensor normal 방향으로 target 내부에 접근하면 이
하한을 임의로 가까이 실현할 수 있다. 따라서 고정 센서 아래 unrestricted additive
residual 공격의 exact 값이며, 자연어 prompt 공격의 최소 거리는 아니다.

기록한 세 양은 다음과 같다.

- D_leave: 현재 terminal interval을 벗어나는 거리의 infimum.
- D_guard: 가장 가까운 guard에 도달하는 거리의 infimum.
- D_terminal_switch: 다른 terminal policy에 도달하는 거리의 infimum.

현재 상태가 safe/guard/outside이면 terminal-switch 거리를 결측으로 두고,
안정적인 terminal 상태로 계산하지 않았다. guard 진입은 매우 작은 섭동으로도
가능하다. primary theorem은 D_terminal_switch의 global lower envelope에 적용된다.
내부 끝점이 guard 소유이므로 certificate 이하에서 direct terminal switch가
없다. 그 infimum은 일반적으로 특정 terminal 쌍에서 달성되는 minimum이 아니다.

## rho별 이론과 실제 상태 결과

| rho | CRE_abs: 이론적 하한 | 자연 terminal 표본 수 | 자연 상태 관측 최소 D_switch | 투영 probe 최소 D_switch |
|---:|---:|---:|---:|---:|
| 0.25 | 0.04289815 | 26 | 0.04301971 | 0.04290030 |
| 0.28 | 0.04735407 | 40 | 0.04822217 | 0.04735708 |
| 0.30 | 0.04941867 | 45 | 0.05087186 | 0.04942238 |
| **1/3** | **0.05084225** | **79** | **0.05104488** | **0.05084734** |
| 0.36 | 0.04981402 | 97 | 0.05071126 | 0.04982043 |
| 0.40 | 0.04392771 | 117 | 0.04540791 | 0.04393649 |
| 0.44 | 0.03189152 | 160 | 0.03193958 | 0.03190321 |

unmodified 자연 상태는 기존 clean residual 2,092개다. rho마다 terminal mask가
달라 표본 수가 다르며, 7개 집계의 564개 terminal state-rho 관측은 독립 prompt
564개라는 뜻이 아니다. 자연 상태의 관측 minimum은 global envelope보다 크다.
이번 표본에서는 middle-third minimum이 비교 rho 모두보다 엄격히 컸다. 이 순위는
유한 표본에서 확인한 결과이며 이론만으로 보장되는 사항은 아니다.

경계 검증용 112,000개 trial은 실제 risk calibration anchor 200개를 7 rho ×
8 leaf × 10개 내부 위치에 센서 법선 방향으로 투영한 counterfactual residual이다.
직교 성분은 실제 LLM 상태에서 가져왔으나 112,000개의 자연 LLM 상태로 표현하지 않는다.
최소 projected slack은 2.14491e-6, 자연 상태 최소 slack은 4.80615e-5였다.

## solver와 stress test

analytic interval-distance solver와 실제 route predicate를 검사하는 sensor-normal
bisection solver를 비교했다. 최종 interior witness는 전체 896차원 residual 벡터에
실제로 적용해 target leaf 진입을 검증했다. 독립 SLSQP 제약 최적화 단위 테스트도 통과했다.

- 총 solver trial: 112,564회, interior witness 실패 0.
- projected trial 평균 절대 오차: 6.44033e-17.
- 최대 절대 오차: 1.80411e-16, 사전 허용값 1e-9.
- 최대 상대 오차: 3.94116e-15, 사전 허용값 1e-6.
- 10,000개 무작위 Lipschitz 검증: 위반 0.
- 100개 tight-direction 검증: 최대 절대 오차 1.66533e-15.
- certificate 이하 terminal-source 섭동 평가 4,823,460회: direct switch 0, 최대 위반 0.

stress test는 sensor-normal, actuator line, isotropic random, sensor-orthogonal,
mixed sensor-actuator의 5개 family와 10개 norm factor, 자체 certificate 및 공통
absolute certificate의 2개 scaling을 모두 포함한다. 전체 상태-방향-norm 평가 수는
12,664,400이고 terminal-source 평가는 11,256,400이다. 방향의 896차원 내적과
정확한 affine sensor 변화 및 실제 라우터를 계산했으며, 이 횟수만큼 모델을 생성한
것은 아니다. random direction과 prompt 재사용 때문에 독립 표본 수도 아니다.

공통 absolute sensor-normal grid에서 projected probe의 최초 전환 norm은 rho 순서대로
0.0457580, 0.0503338, 0.0503338, **0.0513507**, 0.0503338, 0.0457580,
0.0381317이었다. 자연 상태에서는 0.30, 1/3, 0.36이 같은 0.0513507 grid에서 처음
관측돼 coarse grid가 세 onset을 구분하지 못했다. middle-third가 모든 norm에서
낮은 switch rate를 가진다고 주장하지 않는다.

이 곡선은 주어진 크기로 이동한 **끝점**이 다른 terminal에 있는지를 센다. 더 큰
섭동이 다시 guard로 들어가면 rate가 내려갈 수 있어 누적 “한 번이라도 전환” 곡선이
아니다. 정확히 경계에 가까운 probe의 above-certificate rate는 부동소수점 반올림에
민감하다. raw 값을 보존했고 위반을 제거하기 위해 tolerance를 늘리지 않았다.

## ablation: 무엇이 1/3의 필요성을 만드는가

| 구조 | 정책 수 | 총 leaf 길이 | CRE_abs |
|---|---:|---:|---:|
| NO_GUARD 균등 bin | 8 | 1 | 0 |
| NON_RECURSIVE_GUARD 균등 gap | 8 | 8/27 | **0.13800041** |
| MIDDLE_THIRD_CANTOR | 8 | 8/27 | 0.05084225 |

같은 총 leaf 길이를 유지해도 재귀 제약을 풀면 최소 간격을 **19/7배** 크게 만들 수
있었다. 이는 7개 gap의 평균 길이 19/189를 모든 gap에 동일하게 배분해 얻는 전역
최댓값이다. explicit guard가 양의 separation을 만들고, 같은 rho의 재귀 제약과
depth=3을 함께 적용했을 때 1/3이 유일해가 된다. Cantor가 모든 partition보다
낫거나 모든 routing 문제에 필요하다는 주장은 이 ablation으로 반박된다.

## behavioral secondary와 최종 판정

고정된 risk calibration 순서의 최초 12개 prompt에 대해 총 756행을 MPS float32
eager, greedy 32 token으로 생성했다. 각 rho에서 leaf 2의 오른쪽 끝보다 자체
certificate의 0.01배 안쪽인 residual로 투영했다. 이 projected baseline에서
0.9/1.1/2.0배 자체 certificate 및 공통 absolute 1.1배 middle-third certificate의
sensor-normal 섭동을 가하고, routed/locked-original-policy를 쌍으로 비교했다.

token-1 KL은 작지만 0이 아니었다. routed perturbation 336개 조건 중 direct terminal
switch는 216개(64.2857%)였다. 자체 0.9배에서는 모두 guard에 있고 terminal 전환은
없었지만, 보수적 guard action이 원래 leaf action과 달라 policy-isolated KL은
평균 약 1.55e-4--1.64e-4였다. 이는 terminal certificate가 guard 진입이나 action
불변성을 보장하지 않는다는 직접적인 예다.

모든 비교에서 token-1 top-1 변화율, refusal label 변화율, normalized generation
edit distance는 **0**이었다. 평균 coherence proxy는 1, degeneration proxy는 0이다.
즉 정책이 바뀌어도 greedy 출력이 반드시 바뀌지는 않았다. 기존 eta를 계승한 최대 q는
0.02426259로 cap 0.05 이하였고, float32 좌표 roundoff 최대값은 4.52575e-8이었다.
출력 진단의 float32 오차를 primary float64 solver 오차와 혼합하지 않았다.

공통 absolute perturbation에서 각 prompt별로 7개 rho의 certificate와
`KL(locked || routed)` 사이 slope를 구했다. 20,000회 prompt-cluster bootstrap의
평균 slope는 **-0.0002816975**, 95% CI는
**[-0.0003665333, -0.0001965808]**로 사전 조건인 upper bound < 0을 만족했다.
따라서 기계적 C 판정도 true다. 이 결과의 이름은 exploratory behavioral alignment이며,
새로운 natural prompt에 대한 behavioral confirmation이나 실질 개선 주장이 아니다.

또한 middle-third의 KL이 모든 rho보다 낮은 것은 아니다. 공통 absolute 조건의
max-T paired 대비는 1/3 - 0.25 = +3.9220e-6, 1/3 - 0.28 = +2.5239e-6,
1/3 - 0.30 = +1.5670e-6로 오히려 더 컸다. 반대로 0.36/0.40/0.44보다
1.2958e-6/3.5711e-6/5.3816e-6 작았다. 6개 simultaneous interval은 각각 같은
부호였다. 전체 rho에 대한 음의 slope가 middle-third의 행동 최적성을 뜻하지 않는다.
rho별 projected baseline 위치와 inherited eta도 함께 달라지므로 인증 반경 자체가
KL 감소의 원인이라는 인과 해석은 할 수 없다.

| 판정 축 | 결과 |
|---|---|
| M3 정리·유일 최적점 | PASS |
| residual 좌표 Lipschitz | PASS |
| exact solver·인증 반경 검증 | PASS |
| 자연 표본 최소 거리의 middle-third 엄격 우위 | PASS |
| SUCCESS_A_CERTIFIED_CANTOR | true |
| SUCCESS_B_IMPLEMENTED_CANTOR | true |
| SUCCESS_C_BEHAVIORAL_ALIGNMENT | true, 위에 정의한 탐색적 KL association에 한정 |
| OVERALL | SUCCESS_C |
| semantic safety / behavioral superiority | 주장하지 않음 |

## 연구 무결성과 재현

사전 코드 커밋 `7aef039`, formal freeze 커밋 `4d756eb` 이후 고정 코드와 입력
26개 파일의 SHA256을 검증한다. sensor SHA256은
`f16942ce8c6f89d2eaee2679da4778156450cd44fe1b9ac3529f3434f402f1fe`, actuator는
`c22957e2fe05e9fa3bc158853dbb5c88965b62a98c2aefd63f11fa73d480172a`다.

freeze 전 개발 테스트에서 SLSQP finite-difference 수렴 실패 3건이 발생했다.
정확한 목적함수·제약 gradient를 제공해 해결했고 종료 tolerance와 테스트를 유지했다.
실패는 `results/v3_6_0/development/pre_freeze_test_failures.json`에 보존했다.

검증 명령 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=llm/src pytest -q
test llm/tests`는 **711 passed, 2 warnings**였다. 새 V3.6.0 테스트 26개를 포함하며,
경고는 기존 pandas GroupBy 및 NumPy trapz API의 deprecation 알림이다.

core 계산은 `run_primary.py`, secondary는 `run_behavioral_secondary.py`, 판정은
`finalize.py`, 그림은 `generate_figures.py`가 재현한다. `PRE_ANALYSIS_FREEZE.json`의
코드·자산 해시가 다르면 실행을 거부한다. secondary 생성은 MPS float32 eager,
primary exact routing은 float64로 실행한다. completion과 logits는 private 경로에
보존하고 버전 관리에는 수치 결과와 해시만 포함한다.

## 최종 주장의 경계

기하학적 policy stability와 model behavior 사이에는 별도 경험적 검증이 필요하다.
이번 연구는 frozen sensor의 오류, 새로운 sensor calibration, 자연어 공격의 실현 가능성,
다른 모델이나 층, semantic safety를 검증하지 않는다. 다음 연구는 다른 depth·모델·센서의
자연 상태 coverage와 guard 비용을 독립적으로 조사해야 한다.

V3.6.0은 Cantor가 일반적인 LLM 행동 성능을 우월하게 만든다고 주장하지 않는다.
대신 동일한 재귀 정책 제약 아래에서 middle-third Cantor가 최악의 정책 전환 여유를
유일하게 최대화하며, 이 구조적 보증이 실제 LLM residual space에서 정확한
L2 policy-switch certificate로 구현됨을 검증한다.
