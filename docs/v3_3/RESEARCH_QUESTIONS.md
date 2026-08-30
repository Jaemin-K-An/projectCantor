# V3.3 연구 질문과 답

| | 질문 | 답 |
|---|---|---|
| **Q1** | 칸토어 재귀 배치가 raw LLM safety robustness를 높이는가? | **아니오** (V3.2, 재검정 안 함) |
| **Q2** | 동일 safety에서 구조적 효율을 높이는가? | **아니오** — 사전등록 3개 지표 전부에서 strong control에 지지 않거나 짐 |
| **Q3** | 그 이득은 Cantor-specific인가 재귀 일반인가? | **재귀 일반**. 칸토어와 재귀 비칸토어가 모든 지표에서 동률 |
| **Q4** | 실제 계산적 이점이 있는가? | **있으나 비구조적 layout 대비만** — 저장 `O(1)` vs `Θ(2^n)`, 인증 25,000배, zero-shot 척도 확장 |

## 가설 상태

| | 가설 | 상태 |
|---|---|---|
| H1 | `R_C > R_M` | **NOT SUPPORTED** (V3.2) |
| H2 | 기술 복잡도 우위 | **REJECTED** — periodic 112 < shuffled_seeded 184 < cantor 208 bits |
| H3 | zero-shot 척도 확장 | **SUPPORTED, 그러나 재귀 일반** — 칸토어 6.1e−09, 비칸토어 7.2e−10 |
| H4 | 인증 복잡도 우위 | **REJECTED (post-hoc 정정 후)** — periodic 104 < cantor 125 |
| H5 | Pareto advantage | **NOT ESTABLISHED** — safety는 동등하나 구조적으로 strict하게 싸지 않음 |
