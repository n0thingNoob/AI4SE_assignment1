# AI4SE Assignment 1 - Transformer for Python If-Condition Recommendation

This repository contains a minimal pipeline for training a Transformer model on Python code.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Minimum Test

To verify the pipeline works:

```bash
python run_minimum_test.py
```

This will:
1. Generate sample Python code data
2. Train a byte-level BPE tokenizer
3. Pre-train a small CLM (Causal Language Model)

## Project Structure

```
.
├── src/
│   ├── data/           # Data generation scripts
│   ├── tokenizer/      # Tokenizer training
│   └── modeling/       # Model training scripts
├── data/
│   ├── raw/           # Raw data (generated)
│   └── processed/     # Processed data (generated)
├── models/            # Trained models (generated)
└── run_minimum_test.py  # Main test script
```

## Manual Steps

If you want to run individual steps:

### Generate Sample Data
```bash
python src/data/generate_samples.py
```

### Train Tokenizer
```bash
python src/tokenizer/train_tokenizer.py
```

### Train CLM Model
```bash
python src/modeling/pretrain_clm.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See requirements.txt for full list