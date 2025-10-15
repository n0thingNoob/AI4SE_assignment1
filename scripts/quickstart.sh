#!/bin/bash
# Quickstart script to run the complete pipeline end-to-end
# Usage: bash scripts/quickstart.sh [--smoke-test]

set -e  # Exit on error

# Parse arguments
SMOKE_TEST=false
if [[ "$1" == "--smoke-test" ]]; then
    SMOKE_TEST=true
    echo "Running in SMOKE TEST mode (reduced dataset sizes)"
fi

# Configuration
REPOS_FILE="repos.txt"
RAW_REPOS_DIR="data/raw_repos"
FUNCTIONS_FILE="data/functions.jsonl"
PRETRAIN_CORPUS="data/pretrain_corpus.txt"
FINETUNE_PREFIX="data/finetune"
TOKENIZER_DIR="artifacts/tokenizer"
PRETRAIN_MODEL_DIR="artifacts/pretrain_gpt2"
FINETUNED_MODEL_DIR="artifacts/ifrec_finetuned"
PREDICTIONS_CSV="predictions.csv"

# Smoke test parameters
if [ "$SMOKE_TEST" = true ]; then
    MAX_SAMPLES=1000
    PRETRAIN_EPOCHS=1
    FINETUNE_EPOCHS=2
    echo ""
    echo "Smoke test parameters:"
    echo "  Max samples: $MAX_SAMPLES"
    echo "  Pre-train epochs: $PRETRAIN_EPOCHS"
    echo "  Fine-tune epochs: $FINETUNE_EPOCHS"
    echo ""
else
    MAX_SAMPLES=""
    PRETRAIN_EPOCHS=3
    FINETUNE_EPOCHS=5
fi

echo "============================================================"
echo "PYTHON IF-CONDITION RECOMMENDER - FULL PIPELINE"
echo "============================================================"
echo ""

# Step 1: Mine GitHub repositories
echo "------------------------------------------------------------"
echo "Step 1/9: Mining GitHub repositories..."
echo "------------------------------------------------------------"
if [ ! -f "$REPOS_FILE" ]; then
    echo "Error: $REPOS_FILE not found. Please create it with repository URLs."
    exit 1
fi

python src/data/mine_github.py \
    --repos-file "$REPOS_FILE" \
    --out-dir "$RAW_REPOS_DIR" \
    --max-workers 8

echo ""

# Step 2: Extract functions
echo "------------------------------------------------------------"
echo "Step 2/9: Extracting Python functions..."
echo "------------------------------------------------------------"
python src/data/extract_functions.py \
    --repos-root "$RAW_REPOS_DIR" \
    --out "$FUNCTIONS_FILE" \
    --min-lines 5 \
    --max-lines 400

echo ""

# Step 3: Build pre-training corpus
echo "------------------------------------------------------------"
echo "Step 3/9: Building pre-training corpus..."
echo "------------------------------------------------------------"
python src/data/build_pretrain_corpus.py \
    --functions "$FUNCTIONS_FILE" \
    --out "$PRETRAIN_CORPUS"

echo ""

# Step 4: Build fine-tuning dataset
echo "------------------------------------------------------------"
echo "Step 4/9: Building fine-tuning dataset..."
echo "------------------------------------------------------------"
FINETUNE_ARGS="--functions $FUNCTIONS_FILE --out-prefix $FINETUNE_PREFIX --val 0.1 --test 0.1 --seed 42"
if [ "$SMOKE_TEST" = true ]; then
    FINETUNE_ARGS="$FINETUNE_ARGS --max-examples $MAX_SAMPLES"
fi

python src/data/build_finetune_dataset.py $FINETUNE_ARGS

echo ""

# Step 5: Train tokenizer from scratch
echo "------------------------------------------------------------"
echo "Step 5/9: Training tokenizer from scratch..."
echo "------------------------------------------------------------"
python src/tokenizer/train_tokenizer.py \
    --corpus "$PRETRAIN_CORPUS" \
    --out-dir "$TOKENIZER_DIR" \
    --vocab-size 52000

echo ""

# Step 6: Pre-train GPT-2 from scratch
echo "------------------------------------------------------------"
echo "Step 6/9: Pre-training GPT-2 model from scratch..."
echo "------------------------------------------------------------"
PRETRAIN_ARGS="--tokenizer $TOKENIZER_DIR --corpus $PRETRAIN_CORPUS --out-dir $PRETRAIN_MODEL_DIR --epochs $PRETRAIN_EPOCHS --batch-size 8 --block-size 512"
if [ "$SMOKE_TEST" = true ]; then
    PRETRAIN_ARGS="$PRETRAIN_ARGS --max-samples $MAX_SAMPLES"
fi

python src/modeling/pretrain_clm.py $PRETRAIN_ARGS

echo ""

# Step 7: Fine-tune for if-condition prediction
echo "------------------------------------------------------------"
echo "Step 7/9: Fine-tuning for if-condition prediction..."
echo "------------------------------------------------------------"
python src/modeling/finetune_if_condition.py \
    --tokenizer "$TOKENIZER_DIR" \
    --pretrained "$PRETRAIN_MODEL_DIR" \
    --train "${FINETUNE_PREFIX}_train.jsonl" \
    --val "${FINETUNE_PREFIX}_val.jsonl" \
    --out-dir "$FINETUNED_MODEL_DIR" \
    --epochs $FINETUNE_EPOCHS \
    --batch-size 4

echo ""

# Step 8: Generate predictions
echo "------------------------------------------------------------"
echo "Step 8/9: Generating predictions on test set..."
echo "------------------------------------------------------------"
python src/modeling/predict.py \
    --tokenizer "$TOKENIZER_DIR" \
    --model "$FINETUNED_MODEL_DIR" \
    --test "${FINETUNE_PREFIX}_test.jsonl" \
    --out "$PREDICTIONS_CSV"

echo ""

# Step 9: Evaluate predictions
echo "------------------------------------------------------------"
echo "Step 9/9: Evaluating predictions..."
echo "------------------------------------------------------------"
python src/evaluation/score_predictions.py \
    --csv "$PREDICTIONS_CSV" \
    --samples 10

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  - Tokenizer:        $TOKENIZER_DIR"
echo "  - Pre-trained model: $PRETRAIN_MODEL_DIR"
echo "  - Fine-tuned model:  $FINETUNED_MODEL_DIR"
echo "  - Predictions CSV:   $PREDICTIONS_CSV"
echo ""
echo "To run predictions on a different test set:"
echo "  python src/modeling/predict.py \\"
echo "    --tokenizer $TOKENIZER_DIR \\"
echo "    --model $FINETUNED_MODEL_DIR \\"
echo "    --test path/to/test.jsonl \\"
echo "    --out output.csv"
echo ""
