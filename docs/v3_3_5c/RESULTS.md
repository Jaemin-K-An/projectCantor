# V3.3.5c 결과

## 완료된 단계

- V3.3.5b audit 완료.
- 기존 generation을 재사용한 공통-index 20,000회 max-T 정정 완료:
  critical value 3.017031, `T1_P0_CONCENTRATION_SUPPORTED`.
- TD1의 “같은 분산 schedule이 P0와 G1을 모두 이김” 조건 수정 완료.
- 새 harmful/benign split 생성: 과거 1,570 hash 및 V3.3.5c 내부 overlap 0.
- frozen affine/Cantor/controller/attack-order 구현과 단위 검증 완료.
- 의미 평가기 감사: `S2_PROXY_ONLY`.

## 새 P0 행동 결과

13개 대칭 정규화 dose 전부 DEV 비퇴화 gate를 통과했다. DEV/CONFIRM 모두
0.5 refusal 전이를 bracket했고 slope 방향과 slope CI, Spearman, coherence,
degeneration, major adjacent reversal 조건을 통과했다. 이는 정규화 P0 개입이
실제 행동 레버리지를 가진다는 V3.3.5b 단서를 독립 재현한다.

그러나 clean P0 prompt 분산을 기준으로 경계 불확실성이 너무 컸다. CONFIRM
logistic tau는 3.0707이나 CI 폭이 18.26 sigma_P0, isotonic crossing은 -0.3371이고
bootstrap CI 폭이 7.83 sigma_P0였다. 두 crossing도 1 sigma 규칙을 실패했다.
DEV도 각각 17.46 및 11.32 sigma_P0로 실패했다. 최종 행동 판정은
**`B3_BOUNDARY_UNIDENTIFIABLE`**이다.

## 사전 freeze 감사에서 발견한 오류

최초 fit은 실수로 clean `z_clean` prompt 표준편차가 아니라 dose로 넓어진 전체
`z_after` 표준편차를 sigma_P0에 사용했다. 이 때문에 provisional B1, W, budget,
attack DEV까지 진행했다. formal freeze와 D_final 전에 오류를 발견해 분석과 테스트를
고쳤고, 뒤 단계 산출물은 `POST_GATE_INVALIDATION.json`에 비확증 자료로 명시했다.
그 결과를 final claim에 사용하지 않는다.

## 최종 미실행 항목

경계 실패 stop rule에 따라 protocol freeze, D_final certificate validation,
D_final generation, final budget audit, semantic scoring, failure threshold,
first-token final analysis와 benign utility를 실행하지 않았다. `D_final_P0_335c`
90개는 untouched이다. 의미 평가기도 `S2_PROXY_ONLY`이므로 의미 안전 주장은
독립적으로도 차단된다.

```
TEMPORAL_CORRECTION  T1_P0_CONCENTRATION_SUPPORTED
BEHAVIORAL           B3_BOUNDARY_UNIDENTIFIABLE
SEMANTIC_EVALUATOR   S2_PROXY_ONLY
CERTIFICATE          C3_WINDOW_APPLICABILITY_FAILURE
GENERATION           G6_NOT_RUN_BEHAVIORAL_GATE
UTILITY              U3_NOT_RUN
OVERALL              E_P0_BEHAVIORAL_ANCHOR_NOT_REPLICATED
```
