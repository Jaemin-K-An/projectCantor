# V3.3 정준 부호화 명세

`configs/v3_3/encoding.json` (benchmark 전 동결).

## 1. 원칙

**Kolmogorov 복잡도를 측정했다고 주장하지 않는다** (불가능). 측정하는 것은
**고정 codec 하의 description length**이며, **모든 family에 동일한 schema**를
적용한다.

## 2. 비트 폭

| 항목 | 비트 |
|---|---|
| opcode | 8 |
| integer | 32 (fixed) |
| float | 64 (IEEE-754 binary64) |
| seed | 64 |
| enum | 8 |

좌표 정밀도 binary64. PRNG `numpy.random.Generator(PCG64)`,
shuffle은 `Generator.permutation` (Fisher–Yates).

## 3. Family별 필드

| opcode | 필드 |
|---|---|
| `EXPLICIT_LIST` | opcode, n, E0, n_components, **성분당 (a, b, e)** |
| `PROCEDURAL_SEEDED` | opcode, n, E0, energy_law, algorithm, **seed** |
| `PROCEDURAL_PERIODIC` | opcode, n, E0, energy_law |
| `RECURSIVE_IFS` | opcode, n, E0, energy_law, **b, ρ** |
| `EXPLICIT_WEIGHTS` | opcode, n, E0, n_weights, 가중치 |

**칸토어는 특별 취급을 받지 않는다** — 다른 모든 재귀 family와 같은
`RECURSIVE_IFS` opcode를 쓴다.

## 4. 별도 보고 (합산 금지)

`raw serialized bytes` · `gzip bytes` · `canonical description bits` ·
`parameter count` · `generator AST nodes`.

**gzip 결과 하나로 결론짓지 않는다** (STOP A).

## 5. materialised vs symbolic

`n=20`에서 칸토어의 **기호적** 표현은 4 words이지만 **materialise하면**
3,145,725 words다. *"칸토어 runtime은 항상 `O(n)`"* 같은 진술은 **거짓**이며
쓰지 않는다 (§29). 두 양은 언제나 분리해 보고한다.

## 6. 실측 (n = 20)

| family | canonical bits | serialized | gzip |
|---|---:|---:|---:|
| `periodic_procedural` | **112** | — | — |
| `shuffled_seeded` | **184** | — | — |
| `cantor_recursive` | 208 | — | — |
| `learned_minimax_explicit` | 648 | — | — |
| `shuffled_explicit` | 201,326,536 | — | — |

**칸토어는 가장 짧지 않다.**
