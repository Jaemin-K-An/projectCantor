# Project Cantor V3.5.1 최종 보고서

## Risk-Conditional One-Sided Cantor Controller

V3.5.1은 새로운 Cantor 이론을 제안하는 연구가 아니다. V3.5.0에서 보존된
one-sided risk geometry의 calibration domain, action domain, budget domain을 모두
동일한 위험 반공간 `R={h:d(h)<0}`에 맞춘 뒤, 동결된 외부 거부 평가기로 실제
controller 효과를 끝까지 검정한 확증 연구다. 결론부터 말하면 기하, 보정, 예산,
효용 gate는 통과했지만 사전등록된 실질효과 기준은 통과하지 못했다. 최종 판정은
`CONFIRMATORY_CRITERIA_NOT_MET`이며 SUCCESS A/B/C는 모두 false다.

## 1. V3.5.0에서 왜 중단되었는가

V3.5.0은 비교 생성 전에 `BUD2_MISMATCH`로 정직하게 중단됐다. 기존 risk
calibration 표본 300개 중 `x=max(0,-d)=0`인 비율은 0.903333이었는데, 전체
`x` 분포의 95% 순서통계량으로 `W_R=0.3882828819`를 정했다. 실제 위험 상태는
29개뿐이었고 이 W_R의 위험 조건부 경험 coverage는 0.517241에 불과했다.

예산도 전체 attacked population에서 RMS 0.03을 요구했다. 그러나 attacked
state의 0.740641은 `d>=0`인 safe side여서 action이 정확히 0이어야 했다.
`q_cap=0.05`일 때 가능한 global RMS 상한은
`0.05*sqrt(0.259359)=0.02546365`로, 목표 0.03보다 작다. 즉 controller를 어떻게
조정해도 사전 gate를 통과할 수 없는 정의였다. 재현 수치와 코드 근거는
`V350_FAILURE_AUDIT_KO.md`와
`results/v3_5_1/tables/v350_failure_audit.json`에 기록했다.

## 2. 그것이 왜 Cantor geometry failure가 아니었는가

실패는 Cantor 집합의 구조나 certificate의 위반이 아니었다. V3.5.0에서도
`x=max(0,-d)`의 one-sided folding, residual-L2 1-Lipschitz 성질, depth 3
certificate 식과 수치 검증은 유지됐다. 문제는 위험 상태에서만 작동하는 sparse
controller를 safe-side zero mass가 다수인 전체 분포에 맞추려 했다는 domain
mismatch였다. V3.5.1은 sensor, actuator, depth, rho family, q cap을 바꾸지 않고
측정 모집단의 정의만 이 architecture와 일치시켰다.

## 3. unconditional calibration이 왜 one-sided risk space와 불일치했는가

`d>=0`인 모든 상태는 folding 후 같은 점 `x=0`이 된다. 따라서 unconditional
분포에서 `P(x<=W_R)`를 95%로 맞추면 safe-side point mass가 대부분의 coverage를
선점한다. 그 결과 얻은 W_R은 위험 반공간 내부의 폭을 나타내지 않는다. 실제로
V3.5.0 표본에서 unconditional coverage는 목표를 만족했지만, 위험 조건부 coverage는
약 51.7%였다. 이 차이는 표본오차가 아니라 서로 다른 모집단을 묻는 두 확률
`P(x<=W_R)`와 `P(x<=W_R | d<0)`의 차이다.

## 4. risk-conditional calibration을 어떻게 정의했는가

historical-touched prompt와 final prompt를 제외하고, 초기 후보 3,148개에서 발견된
near-duplicate 2개를 endpoint 관찰 전에 제거했다. 고정 seed로 정한 후보 3,146개의
순서를 따라 frozen sensor residual만 계산해 최초 200개의 `d<0` clean state를
선택했다. 200개 확보까지 1,792개를 스캔했으며 위험 prevalence는 0.117746이었다.

`x=-d | d<0`, `alpha=0.05`, `k=ceil((200+1)*0.95)=191`로 계산한 조건부
conformal window는 다음과 같다.

- `n_risk=200`
- `W_R=1.3727408729684236`
- 조건부 경험 coverage `0.955`
- 스캔된 전체 상태에서의 비조건부 coverage `0.994978`

고정 W_R의 탐색적 20,000회 비모수 bootstrap 95% 구간은
`[1.2525166611, 1.7765351909]`였다. 이 구간으로 W_R을 재조정하지 않았다.
leakage audit은 exact, normalized, token-Jaccard>=0.9 overlap이 모두 0인
`LEAK1_PASS`다.

## 5. Cantor certificate가 어떻게 유지되는가

frozen sensor distance와 `max(0,-d)`의 합성에 대해
`|x(h+Delta h)-x(h)| <= ||Delta h||_2`가 성립한다. 10,000개 무작위 섭동과
100개 tight-direction test에서 위반은 0이었다. depth 3의 direct terminal
risk-policy-switch certificate는 그대로

`epsilon_R(rho)=W_R*rho^2*(1-2*rho)`

이며 도함수 `2*rho*(1-3*rho)`와 500,000점 grid가 모두 `rho=1/3`의 유일한
최댓값을 확인했다. 새 middle-third 값은
`epsilon_R_C=W_R/27=0.05084225455438606`으로 V3.5.0의 0.01438085보다
3.5354배 커졌다. 모든 rho에서 certificate 위반은 0이다. 이 결과의 범위는
오직 “residual-L2 perturbation에 대한 direct terminal risk-policy switch
certificate”이며 의미론적 안전 증명이 아니다.

위험 leaf action은 사전에 `[1/8,2/8,...,1]`로 동결했다. `d>=0`에서는 정확히
0, guard에서는 인접 leaf 중 더 큰 action, `x>W_R`에서는 1을 사용했다.

## 6. risk-conditional equal-budget이 왜 더 적합한 비교인가

공통 attacked residual에서 controller 적용 전에 `E={d_attacked<0}`를 만들고,
모든 arm이 같은 eligibility mask를 사용했다. final audit에서 risk prevalence는
0.244464였고 mask hash는 모든 controller에서 같았다. primary matching target은
`sqrt(mean(q_ctrl^2 | E))=0.03`, cap은 0.05다.

모든 controller가 허용오차 안에서 통과했다. risk RMS 범위는
0.0300158--0.0302504였고 LINEAR는 0.0302106, CANTOR 1/3은 0.0301076이었다.
q max 범위는 0.0386663--0.0409628로 cap보다 작았으며 clipping과 safe-side
intervention은 모두 0이었다. 반면 global RMS는 0.0148408--0.0149568에 불과했다.
이는 실패가 아니라 safe-side zero action을 포함한 배치 부담이 위험 조건부
개입 강도보다 작다는 뜻이다. 이 정의로 V3.5.0의 수학적 불가능성이 제거됐다.

## 7. controller efficacy 결과

formal freeze 후 inherited untouched harmful prompt 200개에 대해 2 attack family,
14 epsilon, 9 arm의 완전 요인 50,400행을 생성했다. prompt가 통계 단위이며 모든
대비에 같은 20,000개 prompt bootstrap matrix와 max-T simultaneous interval을
사용했다. endpoint는 frozen external refusal evaluator의 refusal robustness AUC,
SESOI는 +0.03이다.

primary ATTACK_V에서 CANTOR 1/3의 평균 AUC는 0.797843, LINEAR는 0.799720이었다.
CANTOR 1/3 대 ATTACK_ONLY 차이는 양수였지만 simultaneous lower bound가 SESOI를
넘지 못해 `CTRL4_INCONCLUSIVE`였다. secondary ATTACK_W는
`CTRL2_PRACTICALLY_INERT`였다. 따라서 이 데이터는 controller가 큰 실질 개선을
냈다고 확증하지 못한다.

## 8. attack-only 비교

- ATTACK_V, CANTOR 1/3 - ATTACK_ONLY: `+0.0206951`, max-T simultaneous 95% CI
  `[+0.0088945, +0.0324957]`
- ATTACK_W, CANTOR 1/3 - ATTACK_ONLY: `+0.0100286`, CI
  `[-0.0037357, +0.0237929]`

ATTACK_V의 구간은 0보다 위지만 사전 성공 기준은 lower bound가 `+0.03`보다
커야 한다. 관측된 lower bound 0.0088945는 그 기준에 못 미친다. 유의한 양의
신호를 SUCCESS로 재정의하지 않았다.

## 9. linear 비교

같은 sensor, actuator, W_R, cap, prompt, eligibility와 risk RMS를 사용하는
LINEAR를 구조 대조군으로 삼았다.

- ATTACK_V, CANTOR 1/3 - LINEAR: `-0.0018764`, CI
  `[-0.0056293, +0.0018764]`
- ATTACK_W, CANTOR 1/3 - LINEAR: `-0.0012529`, CI
  `[-0.0037791, +0.0012734]`

두 family 모두 `LIN2_EQUIVALENT`다. 위험 상태에서 두 mapping의 mean absolute
action difference는 키로 정확히 맞춘 1,369쌍에서 0.081238이었지만, 이 구조적
차이가 refusal AUC의 추가 이득으로 이어지지 않았다.

## 10. rho family 비교

전체 판정은 `RHO2_EQUIVALENT`다. primary ATTACK_V에서 1/3 대 0.30, 0.36,
0.40의 AUC 차이와 simultaneous interval은 모두 정확히 0이었다. secondary
controls 중 1/3 대 0.25 및 0.28은 `-0.0009354`, CI
`[-0.0028061,+0.0009354]`; 1/3 대 0.44는 0이었다. ATTACK_W에서도 모든
차이가 0 또는 약 `-0.0012529`였고 interval은 0을 포함했다.

즉 `rho=1/3`이 certificate margin을 유일하게 최대화한다는 수학적 결과는
유지되지만, 이번 frozen refusal endpoint에서 인접 rho보다 우수하다는 경험적
결과는 없다.

## 11. benign utility

untouched benign prompt 80개, 9 arm, 총 720행에서 utility gate는 `U1_PASS`다.
무제어 false refusal은 0.0625, 모든 controller는 0.075로 증가량은 0.0125였고
허용 한계 0.05보다 작았다. 모든 controller의 평균 coherence는 0.999032,
degeneration rate는 0, safe-side intervention은 0이었다.

다만 benign prompt의 `d<0` 비율이 1.0이었다. 따라서 false refusal 0.075는 전부
sensor가 risk side로 분류한 benign 상태에서 발생했고, `d>=0` 조건부 utility는
표본이 없어 추정할 수 없다. safe-side zero-intervention 검사는 구현상 통과하지만
이 benign split에서는 해당 상태가 없어 경험적 검증력이 없다.

## 12. 실패 또는 성공 판정

최종 다축 판정은 다음과 같다.

| 축 | 판정 |
|---|---|
| GEOMETRY | PASS |
| RISK_CALIBRATION | PASS |
| SENSOR | PASS, frozen inherited |
| ACTUATOR | PASS, frozen inherited |
| BUDGET | PASS |
| CONTROLLER | CTRL4_INCONCLUSIVE |
| LINEAR | LIN2_EQUIVALENT |
| RHO | RHO2_EQUIVALENT |
| UTILITY | PASS |
| SEMANTIC_SCOPE | REFUSAL_ONLY |
| OVERALL | CONFIRMATORY_CRITERIA_NOT_MET |

`SUCCESS_A_CANTOR_CONTROLLER=false`, `SUCCESS_B_CANTOR_ADDS_VALUE=false`,
`SUCCESS_C_MIDDLE_THIRD_EMPIRICAL_OPTIMUM=false`다. A가 실패한 이유는 ATTACK_V의
Cantor 대 attack-only simultaneous lower bound가 +0.03을 넘지 못했기 때문이다.
B와 C의 전제도 충족되지 않았고, 별도 대비 역시 Cantor 또는 middle-third의
추가 이득을 보이지 않았다.

실패 artifact는 삭제하지 않았다. 첫 후보 풀에서 near-duplicate 2개를 발견한
감사는 `INVALIDATED_initial_historical_leakage_audit.json`으로 보존했다. final 생성
도중 CPU 체크포인트를 MPS로 이어붙여 공통 `d_attacked`가 최대 3.36e-5 달라진
4,000행은 외부 endpoint 평가 전에 무효화하고, 동일 동결 설정의 MPS 결과로
정확히 재생성했다. 혼합 실행 원자료와 실패한 BUD2 감사도 `INVALIDATED_*`로
보존했다. 재생성 때문에 바뀐 CSV 행 순서를 키 조인 없이 사용한 탐색 진단 한
항목은 원본을 무효화한 뒤 0.224145에서 0.081238로 정정했다. 어느 경우에도
prompt, seed, grid, rho, action, eta, threshold 또는 endpoint 판정 기준을 바꾸지
않았다. W_R bootstrap과 후처리 진단은 명시적으로 exploratory다.

## 13. semantic-safety claim의 한계

외부 평가기는 `LibrAI/longformer-action-ro`의 고정 revision
`bb1f0a07dcb55ae0e9af5c5431ea8075f9a92c92`이며 action class 0/1을 behavioral
refusal로 해석한다. harmful 50,400행에서 669개, benign 720행에서 92개의 고유
prompt-completion 쌍을 평가했다. 이 proxy는 거부 표현을 판별할 뿐 응답 내용의
실질적 무해성, 사실성, 우회 가능성 또는 공격자 적응을 판정하지 않는다.
따라서 본 연구는 refusal robustness만 보고하며 semantic safety improved,
safe model, harmfulness eliminated 같은 주장을 하지 않는다.

frozen sensor도 거부 방향을 측정한다. `d_attacked`와 refusal label의 상관은
0.4939였고 correction 후에는 0.4903이었다. sensor-actuator coupling은
`kappa_safe=0.31023`으로 방향은 일치했지만, 이것만으로 의미론적 안전이나
행동 효과를 보장하지 않는다.

## 14. 다음 연구 질문

가장 큰 과학적 한계는 controller가 residual sensor 좌표를 원하는 방향으로
움직이더라도 그 이동이 frozen refusal 행동의 실질효과로 충분히 전달되지 않는다는
점이다. 다음 연구는 이번 결과를 재튜닝 데이터로 사용하지 않는 새 분할에서
(1) risk-side benign을 포함해 sensor specificity를 재검증하고, (2) 더 다양한
모델·공격·semantic evaluator에서 refusal proxy와 실제 내용 품질을 분리하고,
(3) Cantor의 불연속 mapping이 연속형 controller와 달라질 수 있는 개입 regime을
사전등록해 독립 복제해야 한다. 새 가설은 V3.5.1의 SUCCESS 실패를 뒤집기 위한
사후 설정 변경이 아니라 별도 연구로 다뤄야 한다.

---

재현의 기준점은 `configs/v3_5_1/PRE_ANALYSIS_FREEZE.json`이며, 핵심 표는
`results/v3_5_1/tables/`, 그림 7개와 설명은 `figures/v3_5_1/`에 있다. frozen
sensor SHA256은 `f16942ce...f402f1fe`, actuator SHA256은
`c22957e2...d480172a`로 V3.5.0과 동일하다.
