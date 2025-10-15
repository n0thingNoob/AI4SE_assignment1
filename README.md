# Python If-Condition Recommender

A complete pipeline to train a Transformer model that predicts masked Python if-conditions inside functions.

## Overview

This repository implements:
1. **Pre-training**: Train a GPT-2 style decoder-only Transformer from scratch on Python code using Causal Language Modeling (CLM)
2. **Fine-tuning**: Predict masked if-conditions in Python functions
3. **Custom tokenizer**: Byte-level BPE trained from scratch on our corpus

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── mine_github.py          # Clone GitHub repositories
│   │   ├── extract_functions.py    # Extract Python functions using AST
│   │   ├── build_pretrain_corpus.py # Build pre-training text corpus
│   │   └── build_finetune_dataset.py # Create masked if-condition pairs
│   ├── tokenizer/
│   │   └── train_tokenizer.py      # Train BPE tokenizer from scratch
│   ├── modeling/
│   │   ├── utils.py                # AST utilities for if-condition masking
│   │   ├── data_collators.py      # Custom data collators
│   │   ├── pretrain_clm.py        # Pre-train GPT-2 from scratch
│   │   ├── finetune_if_condition.py # Fine-tune for if-prediction
│   │   └── predict.py             # Generate predictions
│   └── evaluation/
│       └── score_predictions.py   # Evaluate prediction accuracy
├── scripts/
│   └── quickstart.sh              # End-to-end pipeline
├── repos.txt                      # List of GitHub repos to mine
└── report_template.md             # 3-page report template
```

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Create a `repos.txt` file** with GitHub repository URLs (one per line):
```
https://github.com/psf/requests
https://github.com/django/django
https://github.com/pallets/flask
https://github.com/scikit-learn/scikit-learn
https://github.com/pytorch/pytorch
https://github.com/pandas-dev/pandas
https://github.com/numpy/numpy
https://github.com/python/cpython
https://github.com/torvalds/linux
https://github.com/tensorflow/tensorflow
```

3. **Run the complete pipeline**:
```bash
bash scripts/quickstart.sh
```

## Full-Scale Training Commands

After testing and bug fixes, use these commands for production training with larger datasets:

```bash
# Step 1: Mine GitHub repositories (10-20 popular repos recommended)
python src/data/mine_github.py \
  --repos-file repos.txt \
  --out-dir data/raw_repos

# Step 2: Extract Python functions with if-statements
python src/data/extract_functions.py \
  --repos-root data/raw_repos \
  --out data/functions.jsonl

# Step 3: Build pre-training corpus from extracted functions
python src/data/build_pretrain_corpus.py \
  --functions data/functions.jsonl \
  --out data/pretrain_corpus.txt

# Step 4: Build fine-tuning dataset with masked if-conditions
python src/data/build_finetune_dataset.py \
  --functions data/functions.jsonl \
  --out-prefix data/finetune

# Step 5: Train custom Byte-level BPE tokenizer from scratch
python src/tokenizer/train_tokenizer.py \
  --corpus data/pretrain_corpus.txt \
  --vocab-size 10000 \
  --out-dir artifacts/tokenizer

# Step 6: Pre-train GPT-2 model from scratch using Causal Language Modeling
python src/modeling/pretrain_clm.py \
  --tokenizer artifacts/tokenizer \
  --corpus data/pretrain_corpus.txt \
  --out-dir artifacts/pretrain_gpt2 \
  --n-layer 6 \
  --n-head 8 \
  --n-embd 512 \
  --batch-size 8 \
  --epochs 5 \
  --lr 5e-4

# Step 7: Fine-tune pre-trained model for if-condition prediction
python src/modeling/finetune_if_condition.py \
  --tokenizer artifacts/tokenizer \
  --pretrained artifacts/pretrain_gpt2 \
  --train data/finetune_train.jsonl \
  --val data/finetune_val.jsonl \
  --out-dir artifacts/ifrec_finetuned \
  --batch-size 8 \
  --epochs 10 \
  --lr 3e-5

# Step 8: Generate predictions on test set
python src/modeling/predict.py \
  --tokenizer artifacts/tokenizer \
  --model artifacts/ifrec_finetuned \
  --test data/finetune_test.jsonl \
  --out predictions.csv

# Step 9: Evaluate prediction accuracy
python src/evaluation/score_predictions.py \
  --csv predictions.csv
```

**Expected Results with 10-20 Repositories:**
- ~50,000+ functions extracted
- ~100,000+ masked if-condition examples
- ~10,000 vocabulary size
- ~35-60% accuracy on test set
- Training time: 2-6 hours (depends on hardware)

**Monitoring Training:**
```bash
# Monitor training progress with TensorBoard
tensorboard --logdir artifacts/pretrain_gpt2/logs
tensorboard --logdir artifacts/ifrec_finetuned/logs
```

## Individual Steps (Alternative)

For more control, run each step separately with default parameters:

```bash
# 1. Mine GitHub repositories
python src/data/mine_github.py --repos-file repos.txt --out-dir data/raw_repos

# 2. Extract functions
python src/data/extract_functions.py --repos-root data/raw_repos --out data/functions.jsonl

# 3. Build pre-training corpus
python src/data/build_pretrain_corpus.py --functions data/functions.jsonl --out data/pretrain_corpus.txt

# 4. Build fine-tuning dataset
python src/data/build_finetune_dataset.py --functions data/functions.jsonl --out-prefix data/finetune

# 5. Train tokenizer from scratch
python src/tokenizer/train_tokenizer.py --corpus data/pretrain_corpus.txt --out-dir artifacts/tokenizer

# 6. Pre-train GPT-2 model from scratch
python src/modeling/pretrain_clm.py --tokenizer artifacts/tokenizer --corpus data/pretrain_corpus.txt --out-dir artifacts/pretrain_gpt2

# 7. Fine-tune for if-condition prediction
python src/modeling/finetune_if_condition.py --tokenizer artifacts/tokenizer --pretrained artifacts/pretrain_gpt2 --train data/finetune_train.jsonl --val data/finetune_val.jsonl --out-dir artifacts/ifrec_finetuned

# 8. Generate predictions
python src/modeling/predict.py --tokenizer artifacts/tokenizer --model artifacts/ifrec_finetuned --test data/finetune_test.jsonl --out predictions.csv

# 9. Evaluate
python src/evaluation/score_predictions.py --csv predictions.csv
```

## Dataset Format

**Pre-training corpus** (`pretrain_corpus.txt`): One function per line

**Fine-tuning dataset** (`*.jsonl`): One example per line with:
```json
{
  "input": "def foo():\n    if <IFMASK>:\n        pass",
  "target": "x > 0",
  "meta": {"file": "...", "function": "...", "original": "..."}
}
```

## Prediction Output

CSV with exact columns:
1. `Input provided to the model`
2. `Whether the prediction is correct (true/false)`
3. `Expected if condition`
4. `Predicted if condition`
5. `Prediction score (0-100)`

## Special Tokens

- `<pad>`: Padding token
- `<s>`: Beginning of sequence
- `</s>`: End of sequence
- `<unk>`: Unknown token
- `<mask>`: General mask token
- `<IFMASK>`: Masked if-condition placeholder
- `<answer>`: Answer delimiter in prompts

## Key Features

- ✅ Tokenizer trained from scratch (no pre-trained tokenizers)
- ✅ GPT-2 architecture trained from random initialization
- ✅ AST-based function extraction and if-condition masking
- ✅ Deduplication and quality filtering
- ✅ Reproducible with configurable random seeds
- ✅ Train/validation/test splits
- ✅ Exact match evaluation with normalization
- ✅ Confidence scores (0-100) based on generation probabilities

## License

For educational purposes. Mined code is not included in the repository.