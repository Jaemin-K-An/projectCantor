# Semantic evaluator 상태

올바른 실행 순서에서는 external fixed-W gate 실패 직후 중단하므로 V3.4.0R의
confirmatory semantic-evaluator 단계는 열리지 않았다. 따라서 최종 상태는 보수적으로
`SEM3_PROXY_ONLY`이며 semantic safety 주장을 허용하지 않는다.

로컬 선행 작업은 gate 순서를 건너뛰고 target-model output 150개(유해 순응/거부/
무해 순응 각 50)를 만든 뒤 후보 평가기를 검사했다. 최고 balanced accuracy는
0.765로 고정 기준 0.85에 미달했다. 그 산출물은 `POST_GATE_INVALIDATION.json`에
audit-only로 표시했다. 설령 참고하더라도 의미 평가기 통과를 뒷받침하지 않는다.

Sensor 자체의 범위도 변하지 않는다. frozen `w`는 refusal/compliance state를
구분하도록 학습되었지 harmfulness를 감지하도록 학습되지 않았다.
