# V3.4.0 사후 감사 (additive erratum)

이 문서는 **추가만** 한다. `results/v3_4_0/`의 어떤 JSON도, 기존 `docs/v3_4_0/`의
어떤 보고서도 수정하지 않는다. V3.4.0은 역사적 실험으로 그대로 보존된다.

## 1. `GEN2`는 분류기 상태 버그였다

`results/v3_4_0/tables/final_verdict.json`은 다음을 담고 있다.

```
GENERATION   GEN2_RHO_FAMILY_PRACTICALLY_EQUIVALENT
```

그러나 같은 실행의 다른 표는 이렇게 기록한다.

```
generation_analysis.json : confirmatory_comparison_blocked = true
final_budget_audit.json  : all_matched                     = false
```

즉 **모든 rho가 예산 유효성 검사에서 탈락한 상태**였다. 분류기는 그 사실을
읽지 않고 SESOI 폭만 보고 "실질적 동등"을 선언했다. 예산이 정합되지 않은
arm들 사이의 동등성은 아무 의미가 없다.

## 2. 올바른 해석

올바른 라벨은 다음이며, V3.4.0R이 이를 도입한다.

```
GEN6_EQUAL_BUDGET_COMPARISON_BLOCKED
```

규칙: `confirmatory_comparison_blocked` 또는 `all_matched == false`이면
`all_within_sesoi`·`any_favours_cantor`·`any_favours_other`와 **무관하게**
`GEN6`이다. 재정의 여지 없음.

## 3. 서술적 결과는 그대로 보존된다

rho 사이의 출력이 거의 동일했다는 관측(출력 차이 1.5 %, 라벨 차이 0.07 %,
9,520행 중 서로 다른 완성 267개, ATTACK-W에서 7개 rho의 AUC 동일)은 유효한
**서술적** 결과로 남는다. 지우지 않는다.

## 4. 그러나 동등성 결론은 따라 나오지 않는다

이 서술적 유사성은 `CANTOR_RHO_DIFFERENTIATION_WEAK`를 지지하지만,
예산 정합 실패 때문에 **equal-budget 실질적 동등**을 주장할 수 없다.
또한 attacked no-controller 기준선이 없었으므로 `DEFENCE_EFFECT_ZERO`도
주장할 수 없다.

## 5. 전체 판정은 방향적으로 옳았다

V3.4.0의 `OVERALL = G_INCONCLUSIVE`는 이 버그에도 불구하고 적절하다.
`architecture_complete`가 예산 실패로 이미 `false`였기 때문이다.
바뀌는 것은 `GENERATION` 필드의 라벨과 그 해석 범위이지 전체 결론이 아니다.

## 6. 함께 기록되는 다섯 가지 다른 결함

| | 결함 | V3.4.0R의 수리 |
|---|---|---|
| b | `eta`를 clean cell 점유율로 보정하고 attacked 상태에 배포 | attacked 상태 분포에서 보정 |
| c | `q_cap = 0.05` 선언만 하고 강제하지 않음 (`q_max`가 0.0554까지) | statewise hard clip |
| d | attacked no-controller 기준선 부재 | 기준선 arm 추가 |
| e | 62–82 % 우측 절단인데 관측 사건만으로 중앙값 계산 | 절단 인지 생존분석 |
| f | 위험한 leaf 방향이 뒤집힘 (`r = 1/2 - d/2W`이면 index가 클수록 위험) | 방향 수정 |

## 7. "11–70배" 표현 정정

V3.4.0 보고서의 "행동 실패는 인증 반경의 11–70배에서 일어난다"는 62–82 %가
우측 절단된 상태에서 관측 사건만의 중앙값이므로 **모집단 중앙값이 아니다.**

허용되는 서술은 다음뿐이다.

> 시험 격자 안에서 행동 실패가 관측된 프롬프트에 한해, 관측된 실패 크기는
> 해석적 인증서보다 통상 훨씬 컸다. 그러나 62–82 %가 우측 절단되었으므로
> 모집단 실패 반경의 중앙값은 확립되지 않았다.

V3.4.0R이 이를 절단 인지 추정으로 대체한다.
