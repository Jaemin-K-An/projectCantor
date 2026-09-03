# D_final clean-baseline seal

`collect_sensor_training_data.py` ran over every split in one pass, so clean P0
residuals and clean completions exist for `D_final_harmful` (80) and
`D_final_benign` (60) before the protocol freeze.

These artifacts are SEALED:

- `results/v3_4_0/raw/clean_D_final_harmful.csv`
- `results/v3_4_0/raw/clean_D_final_benign.csv`
- `results/v3_4_0/private/clean_D_final_*_completions.csv`
- `results/v3_4_0/cache/h_D_final_*.npy`

No pre-freeze decision may read them. They carry no intervention (they are the
epsilon=0 arm of the final experiment) and nothing about the sensor, window,
budget, attack grid or rho family is derived from them. Any analysis touching
them must happen only after `PRE_ANALYSIS_FREEZE.json` is committed.
