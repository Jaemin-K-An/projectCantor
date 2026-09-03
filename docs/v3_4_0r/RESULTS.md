# V3.4.0R 결과

## 유효한 선행 결과

- V3.4.0 역사적 산출물 49개 hash 보존.
- sensor/actuator hash 일치, `||v_safe||=1`, κ≈0.31022973.
- 외부 데이터 고정 및 leakage: 과거/internal exact·normalized·near overlap 모두 0.
- frozen refusal sensor: AUROC 0.8992 [0.8291, 0.9562], balanced accuracy 0.7348.
- frozen W applicability: 130/150=0.8667 < 0.90.

## Hard stop

Transport discrimination은 `ST1_PASS`지만 fixed affine window가 새 모집단에 충분히
적용되지 않아 `ST3_WINDOW_SHIFT`다. 프로토콜은 이 경우 controller final testing을
중단하고 W를 늘리지 않도록 정했다.

재개 감사에서는 선행 로컬 구현이 W를 잘못 복사하고 q target을 허용 없이 .025로
낮춘 뒤 final 11,600행을 생성했음을 발견했다. 해당 numeric 결과는 hash와 함께
`results/v3_4_0r/invalidated/`에 보존했지만 어떤 endpoint 분석도 하지 않았고 최종
주장에 사용하지 않는다. canonical final path는 비웠다.

```
SENSOR_TRANSPORT  ST3_WINDOW_SHIFT
SENSOR_SCOPE      SENS2_REFUSAL_SENSOR_ONLY
ACTUATOR          ACT1_FROZEN_REPLICATED
CERTIFICATE       NOT_RUN_EXTERNAL_TRANSPORT_GATE
BUDGET            NOT_RUN_EXTERNAL_TRANSPORT_GATE
CONTROLLER        CTRL4_INCONCLUSIVE
BASELINE          BASE4_INCONCLUSIVE
RHO               RHO4_INCONCLUSIVE
SEMANTIC          SEM3_PROXY_ONLY
UTILITY           NOT_RUN_EXTERNAL_TRANSPORT_GATE
OVERALL           E_EXTERNAL_SENSOR_TRANSPORT_FAILURE
```

수학적 `rho=1/3` maximin 정리는 유지되지만 tested external population에 controller를
배치할 고정-window applicability가 확보되지 않았다.
