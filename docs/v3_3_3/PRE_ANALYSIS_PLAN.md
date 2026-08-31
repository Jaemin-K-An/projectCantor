# V3.3.3 사전분석계획

D_final 생성 **이전**에 `configs/v3_3_3/PRE_ANALYSIS_FREEZE.json`으로 봉인.
전체 동결 내용은 `configs/v3_3_3/protocol.json`에 있다.

## 동결 대상
git SHA · D_beh/D_final ID 해시 · 모델 · 층 14 · 거부 방향 · ρ 격자 7개 ·
깊이 {2,3,5} (n=3 primary) · 공격 2종 × 강도 3단 · 목표 `C_rms=0.02` ·
허용 ±3 % · `q_cap=0.05` · primary endpoint(robust safety AUC) ·
평가기 · prompt-군집 부트스트랩 20,000 · SESOI 0.02 · max-T 다중성 ·
행동 모형 · 분위 q50/q75/q90/q95 · 분류기 해시

## n=3이 primary인 이유 (이론 주도, 결과 주도 아님)
`ρ_max(n)=(n−1)/(2n)`이 1/3이 되는 유일한 깊이가 n=3이다.
n≠3은 일반화 대조군이다.

## 연대기 공개
* 봉인 전 이미 알려진 것: V3.3.2 `U_EST_mid` 분위, V3.3.2 System B 결과
* ρ 격자는 D_final을 보기 전에 확정, 1/3을 대칭으로 감쌈
* V3.3.2의 System B는 그 자신의 봉인보다 먼저 실행되었다(기록됨)
* **D_final은 이 봉인 이전에 접근되지 않았다**
* 봉인 후 분류기 1회 수정: 출력 형식만, 판정 로직 불변
