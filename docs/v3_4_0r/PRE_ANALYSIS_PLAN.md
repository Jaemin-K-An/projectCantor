# V3.4.0R 사전 분석 계획

## 고정 질문

1. V3.4.0 refusal-state sensor가 새 harmful 모집단으로 transport하는가?
2. attacked-state에서 실제로 맞춘 `q_rms=.03` 예산 아래 controller가 동일 공격의
   no-controller보다 나은가?
3. 그렇다면 recursive Cantor policy가 `a(r)=r`인 LINEAR보다 나은가?
4. equal-budget rho family에서 1/3의 경험적 효과가 구조적 maximin 정리와 별도로
   존재하는가?

모델, layer 14, P0, `w`, `b`, `v_safe`, `W=2.2805212277347544`, depth 3, rho
family, actions, attack grid, `q_target=.03`, `q_cap=.05`, SESOI=.03은 검색하거나
바꾸지 않는다.

## 자료와 누수 방지

HarmfulQA는 미사용 28개뿐이라 동일 모집단 confirmatory split이 불가능하다.
`LLM-LAT/harmful-dataset@8bfba31…`을 외부 모집단으로 고정했다. 전체 과거 prompt
hash 2,560개를 제외했고 exact/normalized/고 token-overlap 감사에서 과거 및 내부
겹침은 모두 0이었다. final harmful 120과 benign 80은 별도 block이다.

## 순서와 hard stop

1. V3.4.0 audit·정오표·classifier 회귀 수정.
2. 외부 자료와 split 고정.
3. frozen sensor transport: AUROC≥.70, balanced accuracy≥.65, AUROC CI lower>.60.
4. frozen W applicability: `P(|d0|<=W)>=.90`.
5. 둘 다 통과할 때만 evaluator, attacked budget, baselines, freeze, final을 실행.
6. q=.03이 cap 아래 불가능하면 `BUD0_NOT_FEASIBLE`로 중단.
7. final budget이 벗어나면 `BUD2_MISMATCH`로 rho 추론을 차단하고 재적합하지 않는다.

실측 결과 3은 통과했지만 4는 130/150=0.8667로 실패했다. 따라서 formal
`PRE_ANALYSIS_FROZEN` 상태는 생성되지 않았고 후속 단계는 confirmatory 관점에서
열리지 않았다.

## 통계(후속 단계가 열릴 경우에만)

프롬프트가 독립 단위다. 하나의 `IDX[replicate,prompt]`를 모든 arm·epsilon·attack·
endpoint·contrast에 재사용하고 20,000회 paired max-T 동시구간을 계산한다.
효능 family는 Cantor–attack-only, linear–attack-only, Cantor–linear이고 rho
primary는 1/3–.30/.36/.40이다. p>.05는 동등성이 아니며 동등성은 전체 동시구간이
`[-.03,+.03]` 안에 있을 때만 인정한다.
