#!/usr/bin/env python3
"""
Run the full training pipeline and monitor progress.
"""
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, desc):
    """Run a command and stream output."""
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\n❌ Command failed with exit code {process.returncode}")
        return False
    
    print(f"\n✅ {desc} completed successfully!")
    return True

def main():
    print("="*60)
    print("FULL TRAINING PIPELINE - LARGER SCALE")
    print("="*60)
    print(f"Data: 5 GitHub repos, 36,803 functions")
    print(f"Model: 4-layer GPT-2, 5.47M parameters")
    print(f"Pre-training: 5,000 samples, 1 epoch")
    print(f"Fine-tuning: 24,633 train / 3,078 val / 3,078 test")
    print("="*60)
    
    steps = [
        (
            "python src/modeling/pretrain_clm.py "
            "--tokenizer artifacts/tokenizer "
            "--corpus data/pretrain_corpus.txt "
            "--out-dir artifacts/pretrain_gpt2 "
            "--n-layer 4 --n-head 4 --n-embd 256 "
            "--batch-size 4 --epochs 1 --learning-rate 5e-4 "
            "--eval-steps 500 --save-steps 2000 --max-samples 5000",
            "Step 6: Pre-training GPT-2 model"
        ),
        (
            "python src/modeling/finetune_if_condition.py "
            "--tokenizer artifacts/tokenizer "
            "--pretrained artifacts/pretrain_gpt2 "
            "--train data/finetune_train.jsonl "
            "--val data/finetune_val.jsonl "
            "--out-dir artifacts/ifrec_finetuned "
            "--batch-size 8 --epochs 3 --learning-rate 3e-5 "
            "--eval-steps 500 --save-steps 1000",
            "Step 7: Fine-tuning for if-condition prediction"
        ),
        (
            "python src/modeling/predict.py "
            "--tokenizer artifacts/tokenizer "
            "--model artifacts/ifrec_finetuned "
            "--test data/finetune_test.jsonl "
            "--out predictions.csv",
            "Step 8: Generating predictions"
        ),
        (
            "python src/evaluation/score_predictions.py --csv predictions.csv",
            "Step 9: Evaluating predictions"
        ),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n❌ Pipeline failed at: {desc}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ FULL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nResults:")
    print("  - Pre-trained model: artifacts/pretrain_gpt2/")
    print("  - Fine-tuned model: artifacts/ifrec_finetuned/")
    print("  - Predictions: predictions.csv")
    print("  - See evaluation metrics above")
    print("="*60)

if __name__ == "__main__":
    main()
