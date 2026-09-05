# Cantor가 필요한 조건과 필요하지 않은 조건

1. **왜 single threshold가 아닌가?** 하나의 문턱은 두 대응 수준만 나눈다. 이번
   설계 요구는 8개 ordered terminal policy와 그 사이의 불확실성 영역이다.
2. **왜 arbitrary thresholds가 아닌가?** 임의 문턱도 8개 정책을 만들 수 있지만
   공통 재귀 refinement와 scale consistency를 요구하지 않는다. 같은 문제의 더 넓은
   설계 공간이며 오히려 더 큰 최소 간격을 가질 수 있다.
3. **왜 decision tree만으로 충분하지 않은가?** tree는 표현 수단이다. 각 node에
   동일한 대칭 split과 명시적 guard라는 제약이 없으면 본 정리는 적용되지 않는다.
   Cantor 라우터도 decision tree로 구현할 수 있다.
4. **왜 explicit guard가 필요한가?** 맞닿는 terminal 영역은 전역 최소 전환 거리가
   0이다. positive-width guard가 서로 다른 terminal 정책을 양의 거리로 분리한다.
   guard는 유보 또는 보수적 라우팅을 뜻하며 의미론적 불확실성을 자동 추정하지 않는다.
5. **왜 recursive nesting인가?** coarse policy를 prefix가 같은 finer policy로
   확장하면서 기존 guard를 유지하는 계층적 의미가 설계 요구일 때 유용하다.
   계층이 필요 없는 문제라면 재귀 제약을 선택할 이유가 약해진다.
6. **왜 same rho인가?** level과 node마다 같은 길이 비율을 유지해야 단일 rho의
   자기유사 family와 닫힌 형태의 M_n이 성립한다. 이는 선택한 설계 제약이지
   모든 현실 문제에 필요한 조건이 아니다.
7. **왜 depth=3에서 1/3인가?** `M3=rho^2(1-2rho)`의 기울기
   `2rho(1-3rho)`가 1/3에서만 양에서 음으로 바뀐다. 깊이 n>=2에서는
   `(n-1)/(2n)`이고, 다른 깊이에서는 일반적으로 1/3이 아니다.
8. **어떤 조건을 빼면 최적성이 사라지는가?** 깊이, 대칭성, 동일 비율, 재귀성을
   바꾸면 목적함수 또는 family가 달라진다. 같은 8개 leaf와 총 leaf 길이를 유지한
   비재귀 균등 guard는 Cantor보다 19/7배 큰 최소 간격을 만든다.
9. **정확히 무엇을 보장하는가?** 고정 family 안의 global minimum terminal
   separation과, frozen 1-Lipschitz sensor 좌표를 거친 residual-L2 direct terminal
   switch 하한이다. terminal 내부 위치별 exact distance는 별도 solver가 계산한다.
10. **무엇을 보장하지 않는가?** guard 진입 방지, action 불변성, token 불변성,
    semantic safety, refusal 우월성, natural prompt 공격의 난도, rho별 유한 표본
    switch curve의 일률적 순위를 보장하지 않는다.

지원서에 사용할 때도 Cantor가 일반적으로 필요하다고 단정하지 않는다. 연구의 성과는
“재귀성과 동일 축척이라는 제약을 명시했을 때 생기는 수학적 최적성”을 정확히 밝히고,
실제 frozen LLM residual에 그 정책 기하를 구현해 검증했다는 데 있다. 일반 depth
정리는 V3.3.2에서 이미 다룬 내용이며 V3.6.0에서 새로 발견한 정리로 포장하지 않는다.
