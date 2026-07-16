#!/bin/bash
# Property prediction v7 — large-scale synthetic training + formula-level evaluation
# v7: SymPy-based labels, no signature rejection, 100% synthetic pre-training
# Features: Test A/B/C/D (lsr_synth 4 categories), formula-level val/test split
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

# Ensure formula-level split exists
if [ ! -f "data/llm-srbench-data/property_label/v6_split.json" ]; then
    echo "[$(date)] Preparing v6 formula split..."
    conda run -n sragent python scripts/nn_tools/prepare_v6_split.py
fi

SCRATCH_DIR="logs/nn_tools/train_property_v7/scratch"
mkdir -p "$SCRATCH_DIR"

# Target: ~100M formulas = batch_size * max_steps
# Default: 4096 * 24414 ≈ 100.0M
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_STEPS="${MAX_STEPS:-24414}"
DEVICE="${DEVICE:-cuda:7}"

echo "[$(date)] Launching v7 from-scratch training on ${DEVICE}..."
echo "  batch_size=${BATCH_SIZE}, max_steps=${MAX_STEPS}, target_formulas=$((BATCH_SIZE * MAX_STEPS))"
echo "  use_sympy_labels=True, max_per_signature=999999999 (disabled)"

conda run -n sragent python scripts/nn_tools/train_property.py \
    --mode scratch \
    --save_path "$SCRATCH_DIR" \
    --device "$DEVICE" \
    --seed 42 \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --eval_every 200 \
    --eval_batch_size 64 \
    --patience 30 \
    --num_workers 8 \
    --sample_num 200 \
    --max_var_num 5 \
    --min_depth 1 \
    --max_depth 5 \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --dim_feedforward 512 \
    --dropout 0.2 \
    --lr 3e-4 \
    --max_per_signature 999999999 \
    --srbench_mix_ratio 0.0 \
    --label_smoothing 0.05 \
    --noise_std 0.01 \
    --scale_augment \
    --permute_vars \
    --reject_trivial_prob 0.6 \
    --combo_augment_prob 0.3 \
    2>&1 | tee "$SCRATCH_DIR/console.log"

echo ""
echo "[$(date)] Evaluating v7 model (best_f1)..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$SCRATCH_DIR/best_f1.pth" \
    --device "$DEVICE" \
    --n_test 500 \
    --sample_num 200 \
    --use_sympy_labels \
    --output eval_v7_results_best_f1.json \
    2>&1 | tee "$SCRATCH_DIR/eval_best_f1.log"

echo "[$(date)] Evaluating v7 model (best_loss)..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$SCRATCH_DIR/best_loss.pth" \
    --device "$DEVICE" \
    --n_test 500 \
    --sample_num 200 \
    --use_sympy_labels \
    --output eval_v7_results_best_loss.json \
    2>&1 | tee "$SCRATCH_DIR/eval_best_loss.log"

echo "[$(date)] === v7 scratch experiment done ==="
