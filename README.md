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
```

3. **Run the complete pipeline**:
```bash
bash scripts/quickstart.sh
```

Or run individual steps:

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