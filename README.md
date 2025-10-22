# AI4SE Assignment 1: If-Condition Prediction

This project implements a neural model for predicting masked if-conditions in Python code using GPT-2 architecture.

## 🎯 Project Overview

The goal is to train a language model that can predict the missing condition in Python `if` statements, given the context of the surrounding code.

**Example:**
```python
def process_data(x):
    if <IFMASK>:  # Model should predict: x > 0
        return x * 2
    return 0
```

## 🏗️ Architecture

- **Base Model**: GPT-2 Medium (355M parameters)
- **Tokenizer**: Byte-level BPE (GPT-2 aligned)
- **Training**: Two-stage approach (pre-training + fine-tuning)
- **Evaluation**: AST-based correctness checking

## 📁 Project Structure

```
├── src/
│   ├── data/                    # Data processing scripts
│   │   ├── build_pretrain_corpus.py
│   │   ├── build_finetune_dataset.py
│   │   └── extract_functions.py
│   ├── modeling/                # Model training and inference
│   │   ├── pretrain_clm.py
│   │   ├── finetune_if_condition.py
│   │   ├── predict.py
│   │   └── utils.py
│   ├── tokenizer/               # Tokenizer training
│   │   └── train_tokenizer.py
│   └── evaluation/              # Evaluation utilities
│       └── score_predictions.py
├── data/                        # Processed datasets
├── artifacts/                   # Model artifacts
├── scripts/                     # Utility scripts
└── tests/                       # Test files
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch transformers tokenizers datasets tqdm numpy pandas
```

### 2. Run Complete Pipeline

```bash
# Run the entire training pipeline
bash run_complete_pipeline.sh
```

### 3. Individual Steps

```bash
# 1. Build pre-training corpus
python src/data/build_pretrain_corpus.py \
  --functions data/functions_v2.jsonl \
  --out data/pretrain_corpus_v3.txt \
  --augment-ratio 0.08

# 2. Train tokenizer
python src/tokenizer/train_tokenizer.py \
  --corpus data/pretrain_corpus_v3.txt \
  --out-dir artifacts/tokenizer_v6_gpt2style

# 3. Pre-train model
python src/modeling/pretrain_clm.py \
  --tokenizer artifacts/tokenizer_v6_gpt2style \
  --corpus data/pretrain_corpus_v3.txt \
  --out-dir artifacts/pretrained_model

# 4. Fine-tune for if-condition prediction
python src/modeling/finetune_if_condition.py \
  --tokenizer artifacts/tokenizer_v6_gpt2style \
  --pretrained artifacts/pretrained_model \
  --train data/finetune_v3_train_prepped.jsonl \
  --val data/finetune_v3_val_prepped.jsonl \
  --out-dir artifacts/finetuned_model

# 5. Generate predictions
python src/modeling/predict.py \
  --tokenizer artifacts/tokenizer_v6_gpt2style \
  --model artifacts/finetuned_model \
  --test data/finetune_v3_test_prepped.jsonl \
  --out predictions.csv
```

## 📊 Dataset Construction

### Pre-training Corpus
- **Source**: Python functions from GitHub repositories
- **Processing**: 
  - AST-based deduplication
  - Targeted augmentation (8% of if-containing functions)
  - Wrapped in `<CODE>...</CODE>` tags
- **Size**: 165,886 code blocks

### Fine-tuning Dataset
- **Source**: Functions containing `if` statements
- **Processing**:
  - Mask if-conditions with `<IFMASK>` token
  - Extract conditions as ground truth
  - Split: 72k train, 9k val, 9k test
- **Preprocessing**: AST validation for correctness

## 🔧 Key Features

### Special Tokens
- `<CODE>`, `</CODE>`: Code block delimiters
- `<IFMASK>`: Masked if-condition placeholder
- `<ANS>`: Answer prefix for training
- `<TASK=IF_COND>`: Task specification

### Training Optimizations
- **GPT-2 Alignment**: Byte-level BPE with `add_prefix_space=True`
- **Left Truncation**: Preserves task-specific tokens
- **Targeted Augmentation**: Only augments if-containing functions
- **AST Evaluation**: Robust correctness checking

### Model Configuration
- **Architecture**: GPT-2 Medium (355M parameters)
- **Context Length**: 512 tokens
- **Batch Size**: 4
- **Learning Rate**: 5e-5
- **Optimizer**: AdamW with warmup

## 📈 Results

The model achieves competitive performance on if-condition prediction through:
- **Pre-training**: Causal language modeling on code corpus
- **Fine-tuning**: Task-specific training on masked conditions
- **Evaluation**: AST-based semantic correctness checking

## 🛠️ Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Tokenizers
- Datasets
- NumPy, Pandas
- TQDM

## 📝 Citation

```bibtex
@misc{ai4se_assignment1_2024,
  title={If-Condition Prediction in Python Code using GPT-2},
  author={Your Name},
  year={2024},
  howpublished={GitHub Repository}
}
```

