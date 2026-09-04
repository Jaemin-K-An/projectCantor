# V3.5.0 pre-analysis plan

Primary 질문은 `ATTACK_V`에서 middle-third Cantor controller가 true `ATTACK_ONLY`보다 refusal robustness AUC를 개선하는지다. Prompt가 독립 단위이며 20,000회 shared prompt bootstrap과 max-T simultaneous interval을 사용한다. SESOI는 0.03이다.

성공 단계는 다음과 같이 고정했다.

1. SUCCESS-A: `GEO1 + BUD1 + CTRL1 + U1`.
2. SUCCESS-B: SUCCESS-A와 `LIN1`.
3. SUCCESS-C: SUCCESS-B와 `RHO1`.

Fresh split은 risk-window 300, budget 300, attack DEV 50, final harmful 200, final benign 80이다. 총 930개 prompt는 서로 겹치지 않으며 과거 registry 3,655개와 exact, normalized, token-Jaccard≥0.9 중복이 모두 0이었다.

Attack grid는 endpoint 전에 `W_R/27`의 `{.5,.9,.99,1.01,1.25,2,4}`배와 causal regime의 사전 고정 절대값 `{0.1013565,0.405426,0.92065487,1.84130973,3.68261946,7.38213168}`의 합집합으로 정했다.

Hard stop은 sensor/actuator 손상, 수학 구현 오류, prompt leakage, controller 미실행, 또는 `.03` budget의 수학적 불가능성에 한정한다. 이번 실행은 마지막 조건에서 중단되었으므로 formal pre-analysis freeze와 final experiment로 진행하지 않았다.
