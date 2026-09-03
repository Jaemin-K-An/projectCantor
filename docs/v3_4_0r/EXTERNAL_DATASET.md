# 외부 데이터셋

## 선택과 고정

- harmful: `LLM-LAT/harmful-dataset`
- 저장소: https://huggingface.co/datasets/LLM-LAT/harmful-dataset
- revision: `8bfba31bc6d93a5b71808fee5275ef4b6330ed91`
- parquet SHA-256: `51a41eae…7a645b1`
- 원본 4,948행, 고유 prompt text 4,946개
- pinned revision의 dataset card에 명시적 license가 없어 “not specified”로 기록
- benign: `tatsu-lab/alpaca@dce01c9b08f87459cf36a430d809084718273017`

이 선택은 V3.4.0R 모델 출력을 생성하기 전에 이루어졌고 benchmark outcome으로
“쉬운” prompt를 고르지 않았다. HarmfulQA는 V3.4.0까지 약 550개를 소진해 미사용
28개만 남았으므로 transport 100+, budget 200+, evaluator, final 80+의 상호 배타적
block을 만들 수 없었다.

## 중복 제거와 split

raw text의 양끝 공백을 제거한 SHA-256 앞 16자를 exact ID로 썼다. source 내부 exact
duplicate를 제거하고 V1–V3.4.0 config/cache registry의 2,560개 hash를 제외해 harmful
4,768개가 남았다. seed 20260903으로 다음 block을 고정했다.

| block | n |
|---|---:|
| D_sensor_transfer_r | 150 |
| D_eval_val_r | 200 |
| D_budget_attacked_r | 300 |
| D_final_r_harmful | 120 |
| D_eval_val_benign_r | 60 |
| D_final_r_benign | 80 |

NFKC·소문자·구두점/공백 정규화와 unique-token Jaccard≥.90(최소 5 token) 감사 결과
과거 exact 0, 과거 normalized 0, 과거 near 0, V3.4.0R 내부 normalized/near 0이었다.
prompt text와 completions는 gitignored cache/private에만 둔다.
