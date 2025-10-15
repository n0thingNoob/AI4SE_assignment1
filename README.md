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
1. Generate sample Python code data (8 functions with if-conditions)
2. Train a byte-level BPE tokenizer (vocab size: ~161 tokens)
3. Pre-train a small CLM (Causal Language Model) with 433K parameters
4. Generate sample code completions

**Expected Output:**
- All steps should complete with ✓ marks
- Loss should decrease across epochs (e.g., 5.08 → 4.55)
- Model will generate code completions (may be imperfect due to minimal training)

### 3. Demo Code Generation

After training, you can test the model:

```bash
python demo_generation.py
```

This will generate code completions for sample prompts using the trained model.

## Project Structure

```
.
├── src/
│   ├── data/           # Data generation scripts
│   │   └── generate_samples.py
│   ├── tokenizer/      # Tokenizer training
│   │   └── train_tokenizer.py
│   └── modeling/       # Model training scripts
│       └── pretrain_clm.py
├── data/
│   ├── raw/           # Raw data (generated, not committed)
│   └── processed/     # Processed data (generated, not committed)
├── models/            # Trained models (generated, not committed)
├── run_minimum_test.py  # Main test script
└── demo_generation.py   # Demo code generation
```

## Manual Steps

If you want to run individual steps:

### Generate Sample Data
```bash
python src/data/generate_samples.py
```
Creates sample Python functions in `data/raw/pretrain_corpus.txt`

### Train Tokenizer
```bash
python src/tokenizer/train_tokenizer.py
```
Trains a byte-level BPE tokenizer and saves to `data/processed/tokenizer.json`

### Train CLM Model
```bash
python src/modeling/pretrain_clm.py
```
Pre-trains a GPT-2 style model and saves to `models/clm_model/`

## Technical Details

- **Model Architecture:** GPT-2 style decoder-only transformer
  - 2 layers, 2 attention heads
  - 128 embedding dimension
  - 128 max sequence length
  - ~433K parameters

- **Tokenizer:** Byte-level BPE
  - Vocabulary size: ~161 tokens (from minimal corpus)
  - Special tokens: `<pad>`, `<s>`, `</s>`, `<unk>`, `<mask>`

- **Training:**
  - Dataset: 8 sample Python functions
  - Epochs: 2-3 (configurable)
  - Batch size: 2
  - Optimizer: AdamW with learning rate 5e-4
  - Device: CPU (CUDA if available)

## Limitations

This is a **minimum test case** with:
- Very small dataset (8 samples)
- Tiny model (433K parameters)
- Minimal training (2-3 epochs)
- No evaluation metrics

For production use, you would need:
- Larger dataset (thousands of code samples)
- Bigger model (millions of parameters)
- More training epochs
- Proper validation and evaluation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See requirements.txt for full list

## Troubleshooting

**Issue:** Import errors
- **Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue:** CUDA out of memory
- **Solution:** The code uses CPU by default. For GPU, ensure your batch size is appropriate.

**Issue:** Generated text looks weird (has Ġ symbols)
- **Solution:** This is expected - these are byte-level tokens representing spaces. With more training data and epochs, generation quality improves.