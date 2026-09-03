# Project Cantor V3.3.5c 최종 보고서

## 결론

V3.3.5b의 반복측정 max-T를 바로잡아도 P0 집중 결론은 유지되었다. 정정된 임계값은
3.0170이고 판정은 `T1_P0_CONCENTRATION_SUPPORTED`이다. 다만 역사적 GLOBAL의
차이는 누적 하나로 식별되지 않는다.

V3.3.5c의 새 정규화 대칭 실험은 P0의 인과적 행동 레버리지를 재현했다. DEV와
CONFIRM 모두 불안전 방향에서 50% refusal 전이를 bracket했고, coherence와
degeneration gate를 통과했다. 이는 V3.3.5a의 극단 절대 dose 실패가 국소 P0
레버리지를 반증하지 않았음을 확인한다.

그러나 deployable affine anchor에 필요한 경계 위치는 재현되지 않았다. CONFIRM
logistic 경계 CI 폭은 clean sigma_P0의 18.26배, isotonic 경계 CI 폭은 7.83배였고,
두 경계 추정치는 서로 약 6.96 sigma만큼 달랐다. DEV도 같은 식별 실패를 보였다.
따라서 행동 판정은 `B3_BOUNDARY_UNIDENTIFIABLE`이다.

의미 평가기 감사도 별도의 제한을 확인했다. 기존 LEX32는 in-sample lexical refusal
proxy이고 외부 refusal classifier는 0.75로 문턱 0.85에 미달했다. 상태는
`S2_PROXY_ONLY`이며 의미 안전 주장은 불가능하다.

프로토콜 stop rule에 따라 final freeze와 D_final을 열지 않았다. provisional
window/budget/attack DEV는 freeze 전 sigma 구현 오류로 잘못 진행된 단계라 명시적으로
무효화했으며 final claim에 쓰지 않았다. 최종 판정은
**`E_P0_BEHAVIORAL_ANCHOR_NOT_REPLICATED`**이다.

## V1에서 V3.3.5c까지

- V1: Cantor derivative gate의 겉보기 이득에 confound가 존재했다.
- V2–V3.2: matched control에서 고유 raw-score 우월성이 사라졌다.
- V3.3–V3.3.2: middle-third 1/3의 재귀 maximin 수학을 도출했다.
- V3.3.3: 표현 midpoint와 행동 threshold가 다름을 확인했다.
- V3.3.4: r-space maximin은 성립하지만 logistic warp가 exact z optimum을 옮겼다.
- V3.3.5: metric-preserving affine coordinate로 exact 1/3 optimum을 복원했지만 G1
  행동 경계가 실패했다.
- V3.3.5a: P0가 token 1에 도달함을 확인했지만 절대 dose 실험은 비국소적이었다.
- V3.3.5b: residual-norm 정규화와 trajectory-L2 정합에서 P0 집중이 초기 분산보다
  강했고 정상 범위의 전이를 bracket했다.
- V3.3.5c: 그 전이를 새 자료에서 인과적으로 재현했지만, 경계 위치의 prompt-level
  불확실성이 너무 커 affine controller의 안정적 anchor로 쓰지 못했다.

## 남는 수학적 결과

`epsilon_z(rho)=2W rho^2(1-2rho)`이고 고정 affine depth-3 family에서
`rho=1/3`이 유일 최대라는 정리는 변하지 않는다. 그러나 이번 모델에는 그 인증서를
배치할 충분히 재현 가능한 P0 행동 경계가 없다. 올바른 최종 문장은 다음이다.

> V3.3.5b suggested strong P0 causal leverage under normalized interventions,
> but an independent behavioural calibration failed to produce a stable P0
> safety boundary. The Cantor maximin theorem therefore remains mathematically
> valid without a sufficiently replicable behavioural anchor for deployment in
> the tested LLM.
