#!/bin/bash
# Check training status

echo "======================================"
echo "Training Status Check"
echo "======================================"
echo ""

# Check pre-training
if [ -d "artifacts/pretrain_gpt2" ]; then
    echo "✅ Pre-training: COMPLETED"
    echo "   Model: artifacts/pretrain_gpt2/"
    if [ -f "pretrain_output.log" ]; then
        echo "   Loss: $(grep "train_loss" pretrain_output.log | tail -1 | cut -d"'" -f8 || echo 'N/A')"
    fi
else
    echo "❌ Pre-training: NOT COMPLETED"
fi

echo ""

# Check fine-tuning
if [ -f "finetune.pid" ]; then
    PID=$(cat finetune.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "🔄 Fine-tuning: IN PROGRESS (PID: $PID)"
        echo "   Log: finetune_output.log"
        if [ -f "finetune_output.log" ]; then
            LINES=$(wc -l < finetune_output.log)
            echo "   Log lines: $LINES"
        fi
    else
        if [ -d "artifacts/ifrec_finetuned" ]; then
            echo "✅ Fine-tuning: COMPLETED"
            echo "   Model: artifacts/ifrec_finetuned/"
        else
            echo "⚠️  Fine-tuning: PROCESS ENDED (check logs)"
        fi
    fi
else
    if [ -d "artifacts/ifrec_finetuned" ]; then
        echo "✅ Fine-tuning: COMPLETED"
        echo "   Model: artifacts/ifrec_finetuned/"
    else
        echo "⏸️  Fine-tuning: NOT STARTED"
    fi
fi

echo ""

# Check predictions
if [ -f "predictions.csv" ]; then
    echo "✅ Predictions: GENERATED"
    echo "   File: predictions.csv"
    PRED_LINES=$(wc -l < predictions.csv)
    echo "   Predictions: $((PRED_LINES - 1))"
else
    echo "⏸️  Predictions: NOT GENERATED"
fi

echo ""
echo "======================================"
echo "Data Statistics"
echo "======================================"

if [ -f "data/functions.jsonl" ]; then
    FUNC_COUNT=$(wc -l < data/functions.jsonl)
    echo "Functions extracted: $FUNC_COUNT"
fi

if [ -f "data/finetune_train.jsonl" ]; then
    TRAIN_COUNT=$(wc -l < data/finetune_train.jsonl)
    echo "Training samples: $TRAIN_COUNT"
fi

if [ -f "data/finetune_test.jsonl" ]; then
    TEST_COUNT=$(wc -l < data/finetune_test.jsonl)
    echo "Test samples: $TEST_COUNT"
fi

echo ""
echo "======================================"
echo "Next Steps"
echo "======================================"

if [ ! -d "artifacts/pretrain_gpt2" ]; then
    echo "1. Start pre-training"
elif [ ! -d "artifacts/ifrec_finetuned" ]; then
    if [ -f "finetune.pid" ] && ps -p $(cat finetune.pid) > /dev/null 2>&1; then
        echo "1. Wait for fine-tuning to complete"
        echo "   Check progress: tail -f finetune_output.log"
    else
        echo "1. Start or restart fine-tuning"
    fi
elif [ ! -f "predictions.csv" ]; then
    echo "1. Generate predictions:"
    echo "   python src/modeling/predict.py --tokenizer artifacts/tokenizer --model artifacts/ifrec_finetuned --test data/finetune_test.jsonl --out predictions.csv"
else
    echo "1. Evaluate results:"
    echo "   python src/evaluation/score_predictions.py --csv predictions.csv"
fi

echo ""
