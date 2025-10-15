#!/bin/bash
set -e

echo "=========================================="
echo "Full Training Pipeline"
echo "=========================================="

# Step 6: Pre-training
echo ""
echo "=== Step 6: Pre-training GPT-2 model ==="
python src/modeling/pretrain_clm.py \
  --tokenizer artifacts/tokenizer \
  --corpus data/pretrain_corpus.txt \
  --out-dir artifacts/pretrain_gpt2 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256 \
  --batch-size 4 \
  --epochs 1 \
  --learning-rate 5e-4 \
  --eval-steps 500 \
  --save-steps 2000 \
  --max-samples 5000

# Step 7: Fine-tuning
echo ""
echo "=== Step 7: Fine-tuning for if-condition prediction ==="
python src/modeling/finetune_if_condition.py \
  --tokenizer artifacts/tokenizer \
  --pretrained artifacts/pretrain_gpt2 \
  --train data/finetune_train.jsonl \
  --val data/finetune_val.jsonl \
  --out-dir artifacts/ifrec_finetuned \
  --batch-size 8 \
  --epochs 3 \
  --learning-rate 3e-5 \
  --eval-steps 500 \
  --save-steps 1000

# Step 8: Generate predictions
echo ""
echo "=== Step 8: Generating predictions ==="
python src/modeling/predict.py \
  --tokenizer artifacts/tokenizer \
  --model artifacts/ifrec_finetuned \
  --test data/finetune_test.jsonl \
  --out predictions.csv

# Step 9: Evaluate
echo ""
echo "=== Step 9: Evaluating predictions ==="
python src/evaluation/score_predictions.py --csv predictions.csv

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
