# One-sided Cantor certificate

Window 내부에서 `r_R=x/W_R`이므로

`|Δr_R| <= ||Δh||₂/W_R`.

Depth 3 recursive family의 terminal separation factor는

`M₃(ρ)=ρ²(1-2ρ)`

이고 residual certificate는

`ε_R(ρ)=W_R ρ²(1-2ρ)`

이다. 도함수는 `2W_R ρ(1-3ρ)`이므로 `ρ=1/3`이 `(0,1/2)`에서 유일한 최대점이다. Middle-third 값은

`ε_R,C=W_R/27≈0.0143808475`.

300개 calibration-independent budget residual에 대해 각 rho, 64개 below-certificate radius, 양쪽 sensor-normal 방향을 검사했다. 허용되는 same leaf, guard, safe `x=0`, outside-risk fallback을 제외한 direct terminal-leaf switch는 0건이었다. 판정은 `GEO1_ONE_SIDED_CANTOR_CERTIFICATE_VALID`이다. 이는 residual policy-transition certificate이며 semantic safety 또는 jailbreak 보증이 아니다.
