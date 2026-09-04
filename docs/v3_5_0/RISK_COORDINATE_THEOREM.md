# Risk coordinate 정리

Frozen signed sensor distance는 Cauchy–Schwarz에 의해

`|d(h+Δh)-d(h)| <= ||Δh||₂`

를 만족한다. `φ(z)=max(0,-z)` 또한 1-Lipschitz이므로 합성 함수 `x=φ∘d`에 대해

`|x(h+Δh)-x(h)| <= ||Δh||₂`

가 성립한다.

구현 감사에서는 무작위 residual/perturbation 10,000쌍과 `-w_hat` tight direction 100개를 검사했다. 위반은 0개였고 tight-direction 최대 절대 slack은 약 `4.77e-15`였다. 판정은 `GEO1_RISK_COORDINATE_LIPSCHITZ`이다.
