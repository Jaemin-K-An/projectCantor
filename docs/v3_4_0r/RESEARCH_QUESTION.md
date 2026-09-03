# 연구 질문

V3.4.0R은 `rho=1/3`이 이기도록 조건을 찾는 실험이 아니다. 이미 고정된 refusal
sensor와 actuator가 외부 harmful 모집단에서 작동 가능한지, 같은 attacked-state
예산에서 공격만 받은 모델보다 실제로 나은지, 그리고 재귀 Cantor policy가 연속
linear sensor–actuator보다 더하는 것이 있는지를 한 번 확인하는 교정 실험이다.

구조적 정리는 독립적이다.

```
epsilon_h(rho) = 2W rho^2(1-2rho)
argmax_(0<rho<1/2) epsilon_h(rho) = 1/3
```

이는 frozen sensor coordinate의 terminal-policy 직접 전환에 대한 충분 residual-L2
반경이다. 경험적 행동 optimum이나 semantic LLM safety guarantee가 아니다.

외부 fixed-W applicability가 사전 gate에서 실패했으므로 controller 효능·linear
비교·rho specificity 질문은 이번 실행에서 답하지 않는다.
