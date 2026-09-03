# 기준선 설계

## V3.4.0이 답할 수 없었던 것

V3.4.0은 rho controller들을 **서로하고만** 비교했다. 따라서 다음 두 문장을
구별할 수 없었다.

1. controller가 도움이 되지만 rho끼리는 같다.
2. controller가 아무 일도 하지 않으므로 rho끼리 같은 것이 당연하다.

`GEN2`(동등)를 붙이든 "무력하다"고 부르든, 둘 다 이 설계로는 지지되지 않았다.

## V3.4.0R의 arm

| arm | 공격 | controller | 답하는 질문 |
|---|---|---|---|
| `CLEAN` | 없음 | 없음 | 기저 |
| `ATTACK_ONLY` | 있음 | **없음** | 공격이 실제로 해를 끼치는가 |
| `LINEAR` | 있음 | sensor–actuator, 재귀 분할 **없음** | 아키텍처가 돕는가 |
| `0.25 … 0.44` | 있음 | Cantor depth-3 | 재귀 분할이 더 돕는가 |

`ATTACK_W`와 `ATTACK_V` 각각에 대해, 동결 격자의 모든 epsilon에서 동일한
프롬프트·디코딩으로 돌린다.

## 두 질문의 분리

**질문 A — controller 효능**

    Cantor 1/3   vs   ATTACK_ONLY

**질문 B — Cantor 특이성**

    Cantor 1/3   vs   LINEAR

`LINEAR`는 같은 `w`, 같은 `v_safe`, 같은 `W`, 같은 예산, 같은 window 정책을 쓰고
`a = clip(r, 0, 1)`만 다르다. 즉 **유일한 차이가 재귀 분할**이다. 단위 테스트가
두 controller의 감지가 동일하고 구동 방향이 동일하며 action만 다름을 확인한다.

## 무력함 판정 규칙

`CTRL2_CONTROLLER_PRACTICALLY_INERT`는 **Cantor 1/3 대 `ATTACK_ONLY`의 동시구간이
효능 SESOI 안에 완전히 들어갈 때에만** 붙는다.

- rho끼리 비슷하다는 사실은 **충분하지 않다**.
- 0을 포함하지만 SESOI보다 넓은 구간은 `CTRL3_INCONCLUSIVE`이지 무력함이 아니다.
