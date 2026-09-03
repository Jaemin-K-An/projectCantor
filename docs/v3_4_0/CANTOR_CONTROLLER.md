# Sensor–Cantor–Actuator controller

## 실제 순서 (모델 forward 안에서)

```
clean h_P0
   -> + attack delta                (ATTACK-W 또는 ATTACK-V)
   -> attacked h                    <- controller가 관찰하는 상태
   -> d = (w^T h + b)/||w||
   -> r = 1/2 - d/(2W)
   -> Cantor cell 분류 (깊이 3)
   -> 동결된 action a(cell)
   -> h' = h + eta * a * ||h|| * v_safe
   -> token-1 logits
   -> 전체 디코딩
```

controller는 **실제 잔차 벡터**를 받는다. 미리 계산된 margin을 넘기고 나중에
`z`를 복원하는 경로는 없앴다 — 그것이 과거 cross-calibration 버그의 원인이었다.
hook은 prefill의 mask-indexed 마지막 프롬프트 토큰에서 **한 번만** 작동하고
G1 이후 잔차는 건드리지 않는다.

## 정책 구조

깊이 3에서 모든 rho가 **8 terminal leaf, 7 guard**를 갖는다.

- leaf action: `[0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1]`, 위험 좌표 순서, 모든 rho 동일
- guard action: 인접 두 leaf 중 **더 보수적인** 쪽
- `r = 0.5`(즉 `d = 0`)을 포함하는 level-1 중앙 guard는 따라서 항상 보수적
- window 밖: 외삽·clip 없이 `OUTSIDE_WINDOW` + action 1.0

`r = 1/2`이 중앙 guard 안에 놓인다는 것은 rho와 무관하게 성립한다. 즉
**행동 경계는 Cantor 첫 단계 guard의 중심에 구조적으로 놓인다.**

## rho마다 다른 것은 rho뿐

동일: `w`, `b`, `v_safe`, `W`, 깊이, action 스케줄, guard 규칙, window 정책,
공격 격자, 디코딩, 라벨러, 예산 목표.

## 예산

`q_ctrl = ||delta_h_ctrl|| / ||h_P0||`이고 `q_ctrl = eta * a(cell)`이므로
`eta`를 조절하면 어떤 목표든 도달한다. 따라서 진짜 질문은 목표가 **의미
있는가**이다. 동결된 규칙은 인증서 자체를 잣대로 쓴다:

    q >= epsilon_C / (median||h|| * |kappa|) = 0.1689 / (18.441 * 0.3102) = 0.0295

후보 격자 `{0.01, 0.02, 0.03, 0.05}` 중 이를 만족하고 `q_cap = 0.05` 이하인
가장 작은 값은 **0.03**이다. 어떤 rho의 성적도 참조하지 않는다.

`D_controller_budget`에서 적합한 `eta` (모두 `q_rms = 0.0300`):

| rho | `eta` | guard % | leaf % | outside % |
|---:|---:|---:|---:|---:|
| 0.25 | 0.0500 | 0.88 | 0.00 | 0.12 |
| 0.28 | 0.0507 | 0.76 | 0.12 | 0.12 |
| 0.30 | 0.0508 | 0.76 | 0.12 | 0.12 |
| 1/3 | 0.0516 | 0.60 | 0.28 | 0.12 |
| 0.36 | 0.0524 | 0.56 | 0.32 | 0.12 |
| 0.40 | 0.0529 | 0.56 | 0.32 | 0.12 |
| 0.44 | 0.0554 | 0.36 | 0.52 | 0.12 |

guard 점유율이 rho와 함께 감소하는 것은 기하의 직접적 결과다 — rho가 크면
leaf가 넓고 guard가 좁다.

## 닫힌 형태 복원은 primary가 아니다

`dd = eta * a * kappa`이므로 원하는 sensor 거리로 정확히 되돌리는 진폭을
계산할 수 있다. 그러나 rho마다 다른 적응적 복원 목표를 primary controller로
쓰면 rho 비교가 무의미해진다. primary는 모든 rho 공통의 동결 스케줄이며,
닫힌 형태 복원은 보조 진단으로만 남긴다.
