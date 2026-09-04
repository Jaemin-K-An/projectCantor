# V3.4.0R 불변성 감사

V3.5.0은 커밋 `8d04b67933aa95b1fc452b27d6a2c77517486332`의 `cantor-guard-v3.4.0r`에서 분기했다. 다음 여섯 historical tree는 base tree object와 현재 HEAD의 tree object가 모두 일치한다.

- `results/v3_4_0/`, `results/v3_4_0r/`
- `configs/v3_4_0/`, `configs/v3_4_0r/`
- `docs/v3_4_0/`, `docs/v3_4_0r/`

Frozen sensor SHA-256은 `f16942ce8c6f89d2eaee2679da4778156450cd44fe1b9ac3529f3434f402f1fe`, actuator SHA-256은 `c22957e2fe05e9fa3bc158853dbb5c88965b62a98c2aefd63f11fa73d480172a`로 정확히 일치했다. 기계 판정은 `AUDIT1_HISTORICAL_IMMUTABLE`이다.

V3.4.0R의 결론도 그대로 유지한다. 외부 refusal sensor discrimination은 통과했지만 대칭 fixed-window coverage는 0.8667로 실패했고, 당시 controller efficacy는 confirmatory하게 시험되지 않았다.
