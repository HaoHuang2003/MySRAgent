#!/bin/bash
# Property prediction v7 — Fine-tune on reserved lsr_transform formulas
# Uses 29 reserved real-physics formulas to adapt the synthetically-pretrained model
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

SCRATCH_DIR="logs/nn_tools/train_property_v7/scratch"
FT_DIR="logs/nn_tools/train_property_v7/finetune"
mkdir -p "$FT_DIR"

DEVICE="${DEVICE:-cuda:7}"
CHECKPOINT="${SCRATCH_DIR}/best_f1.pth"

echo "[$(date)] Fine-tuning v7 on reserved lsr_transform formulas..."
echo "  checkpoint: ${CHECKPOINT}"
echo "  device: ${DEVICE}"
echo "  lr: 5e-5, batch_size: 32, real_mix_ratio: 0.7"

conda run -n sragent python scripts/nn_tools/finetune_v7.py \
    --checkpoint "$CHECKPOINT" \
    --save_path "$FT_DIR" \
    --device "$DEVICE" \
    --seed 42 \
    --max_steps 5000 \
    --batch_size 32 \
    --eval_every 50 \
    --patience 40 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --label_smoothing 0.05 \
    --real_mix_ratio 0.7 \
    --num_workers 4 \
    --sample_num 200 \
    2>&1 | tee "$FT_DIR/console.log"

echo ""
echo "[$(date)] Evaluating fine-tuned model (best_f1)..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$FT_DIR/best_f1.pth" \
    --device "$DEVICE" \
    --n_test 500 \
    --sample_num 200 \
    --use_sympy_labels \
    --output eval_v7_results_ft_best_f1.json \
    2>&1 | tee "$FT_DIR/eval_ft_best_f1.log"

echo "[$(date)] Evaluating fine-tuned model (last)..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$FT_DIR/last.pth" \
    --device "$DEVICE" \
    --n_test 500 \
    --sample_num 200 \
    --use_sympy_labels \
    --output eval_v7_results_ft_last.json \
    2>&1 | tee "$FT_DIR/eval_ft_last.log"

echo "[$(date)] === v7 fine-tuning experiment done ==="
