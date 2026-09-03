# 외부 sensor transport

V3.4.0의 `w`, `b`, norm을 그대로 사용했다. 새 출력으로 계수·threshold·W를
재적합하지 않았다. `d0=0`은 class-weighted refusal classifier hyperplane이다.

## Discrimination gate

`D_sensor_transfer_r` 150개 clean/no-controller 출력에서 frozen refusal labeller를
기준으로 측정했다.

| metric | result | gate |
|---|---:|---:|
| AUROC | 0.8992 | ≥0.70 |
| AUROC bootstrap 95% CI | [0.8291, 0.9562] | lower>0.60 |
| balanced accuracy at d0=0 | 0.7348 | ≥0.65 |
| PR-AUC | 0.9820 | report |
| Brier | 0.1038 | report |
| calibration intercept / slope | +0.5364 / 2.0500 | report |

따라서 refusal-state ranking은 외부 모집단으로 transport했다: `ST1_PASS`.
다만 이는 semantic safety sensor의 transport가 아니다.

## Frozen-window applicability

역사적 `W=2.2805212277347544`를 바꾸지 않고 `|d0|<=W`를 계산하자 130/150,
즉 0.8667만 window 안에 있었다. 사전 기준 0.90에 미달하므로
`ST3_WINDOW_SHIFT`다. W를 늘리거나 sensor를 재학습하지 않고 controller final을
중단했다. 판별력 transport와 배치 coordinate applicability는 별개라는 것이 핵심이다.
