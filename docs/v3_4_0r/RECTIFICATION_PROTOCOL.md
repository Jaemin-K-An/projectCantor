# V3.4.0R 교정 프로토콜

## 무엇을 바꾸지 않는가

모델 `qwen2.5-0.5b-instruct`, layer 14, P0 정의, sensor `w`와 `b`, actuator
`v_safe`, `W = 2.2805`, 깊이 3, rho family, leaf action 스케줄, guard 규칙,
window 정책, 공격 격자, Cantor 수학. 새 sensor도 새 actuator도 적합하지 않는다.
layer·token·rho 탐색을 하지 않는다.

## 무엇을 바꾸는가

**제어 법칙의 예산 규율**과 **실험 설계**뿐이다.

1. `q_cap`을 실제 강제한다: `q_ctrl = min(eta·a(cell), q_cap)`.
2. `eta`를 **attacked 상태 분포**에서 보정한다.
3. `ATTACK_ONLY`(공격만, controller 없음) 기준선을 추가한다.
4. `LINEAR`(같은 sensor·actuator·예산, 재귀 분할 없음) 기준선을 추가한다.
5. 실패 분석을 절단 인지 생존분석으로 바꾸고 위험 leaf 방향을 고친다.
6. 판정 기계에서 예산 유효성을 **경성 선행조건**으로 만든다.

## 모집단 변경 (강제된 것)

HarmfulQA는 **고갈되었다**: 미사용 28개가 남았고 V3.4.0R은 800개 이상이
필요하다. 따라서 harmful 프롬프트를 `LLM-LAT/harmful-dataset`에서 뽑는다
(신선 4,768개). benign은 alpaca(51,852개 신선).

이것은 **동결된 sensor를 훈련 모집단 밖에서 시험**한다는 뜻이며, 프로토콜 수리와
교란된다. 그래서 `D_sensor_transfer_r`(150개) 이전 gate를 두었다.

**결과**: 동결 sensor는 아무것도 재적합하지 않고 새 모집단에서 AUROC
**0.8992 [0.8291, 0.9562]**, `d=0`에서 balanced accuracy 0.7348을 냈다.
모집단은 훈련 대비 `+0.69` 훈련 SD만큼 안전 쪽으로 이동했고 안전율은
0.661 → 0.860으로 올랐다. gate 통과.

## 중단 규칙

sensor transfer 실패 → 중단. 예산 설계 불가능 → 중단.
최종 예산 실패 → rho 추론 차단(재보정 금지).

## 동결 시점

`configs/v3_4_0r/PRE_ANALYSIS_FREEZE.json`을 예산 보정과 기준선 인스턴스화
직후, `D_final_r`을 열기 **전에** 커밋했다 (`712a9e1`).
