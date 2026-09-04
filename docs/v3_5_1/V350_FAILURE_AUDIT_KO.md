# V3.5.0 실패 원인 감사

## 결론

V3.5.0은 Cantor 기하가 틀려서 중단된 것이 아니다. one-sided sparse controller가 실제로 작동하는 위험 반공간 `R={h:d(h)<0}`과, 전체 모집단에 대해 정의한 scale/budget의 domain이 서로 달랐기 때문에 expensive confirmatory generation 이전의 예산 gate에서 정당하게 중단됐다. 감사 재현 결과는 `AUDIT1_V350_DOMAIN_MISMATCH_REPRODUCED`다.

## A. unconditional risk scale의 불일치

V3.5.0은 `x=max(0,-d)`를 300개 clean state 전체에서 계산하고

`k=ceil((300+1)(1-0.05))=286`

번째 순서통계량을 `W_R`로 사용했다. 그러나 300개 중 271개(90.3333%)는 `d>=0`이어서 `x=0`이었고 실제 `d<0` 상태는 29개뿐이었다. 재계산한 기존 값은 `W_R=0.3882828819`이며, 이 값은 전체 표본에서 `P(x<=W_R)`를 맞춘다. 실제 위험 상태에 조건부로 보면 기존 창의 경험적 coverage는 51.7241%에 불과하다. 같은 29개 위험 상태만 사용한 반사실적 95% 순서통계량은 `1.4943668881`이다. 작은 표본이라 새 추정치로 사용하지는 않지만, domain mismatch의 크기를 보여준다.

근거는 [audit_v350.py](../../scripts/v3_5_1/audit_v350.py), V3.5.0의 [calibrate_risk_window.py](../../scripts/v3_5_0/calibrate_risk_window.py), 그리고 [v350_failure_audit.json](../../results/v3_5_1/tables/v350_failure_audit.json)에 보존했다.

## B. unconditional RMS budget의 수학적 불가능성

V3.5.0 attacked budget state의 위험 적격 비율은 `p=0.2593589744`, safe-side 비율은 74.0641%였다. safe side에서 `q=0` exactly이고 `q<=q_cap=0.05`이면 전체 모집단 RMS에는 다음 상한이 있다.

`q_rms_global <= q_cap * sqrt(p) = 0.05*sqrt(0.2593589744) = 0.0254636493`

따라서 전체 모집단 목표 `0.03`은 어떤 `eta`로도 달성할 수 없다. 실제 V3.5.0 arm별 최대값도 약 0.02528–0.02546이었으며 결과는 `BUD2_MISMATCH`였다. 이는 safe state에 작은 action을 넣거나 목표를 낮춰 해결할 문제가 아니라, 비교하려는 intervention intensity의 모집단을 위험 반공간으로 명시해야 하는 정의 문제다.

## C. 기하 실패가 아닌 이유

V3.5.0의 `x=max(0,-d)`는 residual L2에 대해 1-Lipschitz였고, depth-3 certificate `epsilon_R(rho)=W_R*rho^2(1-2rho)` 구현 검증의 위반은 0건이었다. `rho=1/3`의 유일 최대화 성질도 유지된다. 잘못된 것은 좌표나 certificate가 아니라, safe-side zero mass가 압도적인 전체 상태를 scale과 예산의 primary domain으로 삼은 점이다.

V3.5.1은 sensor/actuator/Cantor geometry를 그대로 유지하고 다음 두 양만 `d<0`에 조건부로 정의한다.

- `W_R`: 최초 200개 clean `d<0` 상태의 `x=-d`에 대한 one-sided conformal upper order statistic
- equal budget: 공통 pre-control mask `d_attacked<0` 위의 `sqrt(mean(q^2))`

global RMS는 배포 burden의 보조 지표로 계속 보고하지만 matching target으로 사용하지 않는다.

## 보존 감사

출발 commit `4c62b9dee2df37f7c775f8b842233d248a34d9b1`과 비교해 V3.5.0의 config/result/doc/script/source tree는 모두 동일했다. frozen sensor와 actuator SHA256도 각각 `f16942ce...402f1fe`, `c22957e2...480172a`로 일치했다. V3.5.0 final freeze의 `D_final_v350_touched=false`와 harmful/benign output·label artifact 부재를 확인했으므로 prompt set만 V3.5.1 confirmatory final로 계승했다.
