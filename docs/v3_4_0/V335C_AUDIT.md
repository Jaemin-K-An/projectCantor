# V3.3.5c 감사

V3.4.0은 `cantor-guard-v3.3.5c`의 `7225484`에서 분기했다. V3.3.5c의 원 데이터,
표, 판정, 보고서는 **하나도 수정하지 않았다.** 추가된 것은
`docs/v3_3_5c/POSTHOC_CLAIM_NOTE.md` 하나뿐이며, 그것도 용어 정정만 담는다.

## V3.3.5c 공식 판정 (그대로 유지)

```
TEMPORAL_CORRECTION  T1_P0_CONCENTRATION_SUPPORTED
BEHAVIORAL           B3_BOUNDARY_UNIDENTIFIABLE
SEMANTIC_EVALUATOR   S2_PROXY_ONLY
CERTIFICATE          C3_WINDOW_APPLICABILITY_FAILURE
GENERATION           G6_NOT_RUN_BEHAVIORAL_GATE
UTILITY              U3_NOT_RUN
OVERALL              E_P0_BEHAVIORAL_ANCHOR_NOT_REPLICATED
```

## 왜 P0 인과 레버리지는 재현되었는가

V3.3.5c의 CONFIRM에서 정규화 dose를 `u = -0.8 → +0.8`로 움직였을 때 거부율은
`0.117 → 0.700`으로 단조롭게 변했다. Spearman p = 1.4e-17, 평균 coherence
0.998, degeneration 0.26 %, 인접 bin의 major reversal 0건이었다. 0.5 전이는
DEV·CONFIRM 양쪽에서 bracket되었다. **개입이 행동을 바꾼다는 사실 자체는
독립적으로 재현되었다.**

## 왜 전역 v-사영 문턱은 실패했는가

같은 실험에서 경계의 **위치**는 재현되지 않았다. clean prompt 표준편차
`sigma_P0` 기준으로 CONFIRM logistic tau의 CI 폭은 18.26 sigma, isotonic은
7.83 sigma였고 두 추정치는 약 6.96 sigma 떨어져 있었다. DEV도 17.46 및 11.32
sigma로 같은 실패를 보였다.

즉 실패한 것은 "P0에 행동 경계가 있는가"가 아니라
**"조향 방향 `v` 위의 스칼라 사영 `z = <h,v>`이 프롬프트 사이에 공유되는
정밀한 문턱을 주는가"**였다.

## 왜 actuator 증거가 sensor 타당성을 함의하지 않는가

두 진술은 논리적으로 독립이다.

- **actuator**: `h`를 `v` 방향으로 밀면 행동이 바뀐다. 이는 `v`가 인과 사슬에
  들어 있다는 뜻이다.
- **sensor**: `<h, v>` 하나로 현재 행동 상태를 읽을 수 있다. 이는 `v`가 행동
  상태를 **선형적으로 완전히 요약**한다는 훨씬 강한 주장이다.

제어이론 용어로 첫째는 **가제어성**, 둘째는 **가관측성**이며, 일반적으로 한
방향이 두 역할을 동시에 수행할 이유가 없다. V3.3.5x 계열은 계속 그것을
가정했다. V3.4.0은 그 가정을 버린다.

## `C3` 라벨 정정

`C3_WINDOW_APPLICABILITY_FAILURE`는 경험적 window 실패를 뜻하지 않는다.
window·인증서 단계는 행동 gate 실패로 **실행되지 않았다**. 회고 인용 시
`C0_NOT_RUN_BEHAVIORAL_GATE`를 쓴다.

## 미사용 자료

V3.3.5c의 `D_final_P0_335c` 90개는 untouched로 남아 있으며, V3.4.0은 이를
증거로 전환하지 않는다. V3.4.0의 새 split은 V1–V3.3.5c의 프롬프트 hash
1,910개를 모두 배제했고 겹침은 0이다.
