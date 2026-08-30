#!/bin/bash
# V3.2 remaining pipeline, STRICTLY SEQUENTIAL.
# Two models must never be resident at once: running the Qwen eps sweep
# alongside the OLMo-2 fit pushed this 8 GB machine to 6.6 GB of swap and the
# kernel killed both processes.
set -e
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
L=logs/v3_2
A=qwen2.5-0.5b-instruct
B=olmo2-1b-instruct

step () { echo; echo "=============== $* ==============="; date; }

if [ ! -f configs/v3_2/frozen_$A.json ]; then
  step "FIT model A ($A)"
  python3 scripts/v3_2/fit_and_freeze.py --model $A --batch 8 \
    2>&1 | tee $L/fit_qwen.log
fi

step "FREEZE model A"
python3 scripts/v3_2/freeze.py --model $A 2>&1 | tee $L/freeze_A.log

step "FINAL TEST model A"
python3 scripts/v3_2/run_final_test.py --model $A --batch 10 2>&1 | tee $L/test_qwen.log

step "EXTERNAL SCORING model A"
python3 scripts/v3_2/score_external.py --model $A 2>&1 | tee $L/score_qwen.log

step "FIT model B ($B)"
if [ ! -f configs/v3_2/frozen_$B.json ]; then
  python3 scripts/v3_2/fit_and_freeze.py --model $B --dtype float16 --batch 4 \
    --layout-seeds 2 \
    --families T0_none,T1_true_constant,T4_periodic,T5_shuffled,T6_center_anchored,T7_cantor \
    2>&1 | tee $L/fit_olmo2.log
fi

step "FREEZE model B"
python3 scripts/v3_2/freeze.py --model $B 2>&1 | tee $L/freeze_B.log

step "FINAL TEST model B"
python3 scripts/v3_2/run_final_test.py --model $B --dtype float16 --batch 4 \
  --attacks authority_test 2>&1 | tee $L/test_olmo2.log

step "EXTERNAL SCORING model B"
python3 scripts/v3_2/score_external.py --model $B 2>&1 | tee $L/score_olmo2.log

step "ANALYSIS"
python3 scripts/v3_2/analyse_v32.py --model $A 2>&1 | tee $L/analyse_qwen.log
python3 scripts/v3_2/analyse_v32.py --model $B 2>&1 | tee $L/analyse_olmo2.log

step "VERDICT"
python3 scripts/v3_2/final_claim_check_v32.py \
  --csv results/v3_2/raw/v32_final_$A.csv \
  --out results/v3_2/tables/verdict_$A.json 2>&1 | tee $L/verdict_qwen.log
python3 scripts/v3_2/final_claim_check_v32.py \
  --csv results/v3_2/raw/v32_final_$B.csv \
  --out results/v3_2/tables/verdict_$B.json 2>&1 | tee $L/verdict_olmo2.log

step "FIGURES"
python3 scripts/v3_2/generate_figures.py 2>&1 | tee $L/figures.log

step "DONE"
