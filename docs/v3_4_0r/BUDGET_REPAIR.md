# 예산 수리 설계와 중단 상태

V3.4.0은 clean state 점유율로 `eta`를 맞췄지만 controller는 attacked state를
관측했다. 그 결과 final `q_rms`가 목표 0.03보다 14–25% 컸고, 선언된
`q_cap=0.05`도 실제로 강제되지 않았다.

V3.4.0R의 고정 수리는 다음뿐이다.

```
q_raw  = eta_rho * a_rho(attacked_state)
q_ctrl = min(q_raw, 0.05)
Q_rho  = sqrt(mean_Omega(q_ctrl^2))
target = 0.03
```

`Omega`는 `D_budget_attacked_r`의 prompt × `{ATTACK_W, ATTACK_V}` × 고정 절대
epsilon 격자다. 출력 생성·refusal label·semantic outcome은 적합에 사용하지 않는다.
각 arm은 `|Q_rho/.03-1|<=.01`과 `q_max<=.05`를 만족해야 한다. clipping 비율은
보고 항목이지 target을 바꾸는 선택 조건이 아니다.

## 재개 감사에서 발견한 사양 이탈

로컬 선행 작업은 사양에 없는 `clipping<=10%` 조건을 추가해 target을 0.025로
내렸다. 이는 “q_target=.03 고정, 불가능하면 중단” 규칙 위반이다. 기존 계산표에서
0.03은 모든 일곱 rho와 LINEAR가 1% 이내로 도달하고 `q_max<=.05`를 지켰다.
LINEAR의 clipping 18%는 공개해야 하지만 불가능 판정의 근거가 아니다.

다만 올바른 실행 순서에서 budget 단계에 도달하지 못했다. 앞선 fixed-W gate가
coverage 0.8667로 실패했으므로 V3.4.0R의 confirmatory budget은 **미실행**이며
`eta_per_arm=null`이다. 이전 q=.025 표와 생성은 `POST_GATE_INVALIDATION.json`에
비확증 자료로 격리했다.
