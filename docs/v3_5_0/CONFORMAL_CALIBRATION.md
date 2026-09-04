# Conformal risk-window calibration

Pinned `LLM-LAT/harmful-dataset` revision `8bfba31bc6d93a5b71808fee5275ef4b6330ed91`에서 과거에 사용한 모든 prompt를 제외한 새 `D_risk_window_cal` 300개를 사용했다. 모델 출력, refusal label, attack outcome 또는 final endpoint는 사용하지 않았다.

- `alpha = 0.05`
- finite-sample index: `ceil((300+1)(1-.05)) = 286`
- `W_R = 0.388282882`
- calibration empirical coverage: `286/300 = 0.953333`

`W_R`은 clean P0에서 얻은 `x=max(0,-d)`의 286번째 order statistic이다. 이후 enlarge하거나 final data로 조정하지 않았다. `x>W_R` 상태도 controller가 maximum action으로 명시적으로 처리하므로 coverage는 진단량이지 controller 정의 여부를 결정하는 gate가 아니다.
