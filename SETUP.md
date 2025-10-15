# Setup and Testing Guide

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Repository List

Create a `repos.txt` file with GitHub repository URLs (one per line):

```txt
https://github.com/psf/requests
https://github.com/pallets/flask
https://github.com/numpy/numpy
https://github.com/scikit-learn/scikit-learn
https://github.com/matplotlib/matplotlib
```

**Recommended**: Add 10-20 popular Python repositories for good coverage.

### 3. Run Smoke Test

Test the entire pipeline on a small dataset:

```bash
bash scripts/quickstart.sh --smoke-test
```

This will:
- Clone repositories
- Extract functions
- Train a tokenizer
- Pre-train a small model (1 epoch, 1000 samples)
- Fine-tune (2 epochs)
- Generate predictions
- Evaluate results

**Estimated time**: 15-30 minutes (depending on hardware)

### 4. Run Full Pipeline

Once the smoke test succeeds, run the full pipeline:

```bash
bash scripts/quickstart.sh
```

**Estimated time**: Several hours to days (depending on dataset size and hardware)

---

## Manual Step-by-Step Execution

If you prefer to run steps individually:

### Step 1: Mine Repositories

```bash
python src/data/mine_github.py \
    --repos-file repos.txt \
    --out-dir data/raw_repos \
    --max-workers 8
```

### Step 2: Extract Functions

```bash
python src/data/extract_functions.py \
    --repos-root data/raw_repos \
    --out data/functions.jsonl \
    --min-lines 5 \
    --max-lines 400
```

### Step 3: Build Pre-training Corpus

```bash
python src/data/build_pretrain_corpus.py \
    --functions data/functions.jsonl \
    --out data/pretrain_corpus.txt
```

### Step 4: Build Fine-tuning Dataset

```bash
python src/data/build_finetune_dataset.py \
    --functions data/functions.jsonl \
    --out-prefix data/finetune \
    --val 0.1 \
    --test 0.1 \
    --seed 42
```

For smoke test, add `--max-examples 1000`.

### Step 5: Train Tokenizer

```bash
python src/tokenizer/train_tokenizer.py \
    --corpus data/pretrain_corpus.txt \
    --out-dir artifacts/tokenizer \
    --vocab-size 52000
```

### Step 6: Pre-train Model

```bash
python src/modeling/pretrain_clm.py \
    --tokenizer artifacts/tokenizer \
    --corpus data/pretrain_corpus.txt \
    --out-dir artifacts/pretrain_gpt2 \
    --epochs 3 \
    --batch-size 8
```

For smoke test, add `--max-samples 1000 --epochs 1`.

**Hardware Recommendations**:
- GPU: Highly recommended (10-50x speedup)
- RAM: 16GB minimum, 32GB+ recommended
- Disk: 10-50GB depending on dataset size

### Step 7: Fine-tune Model

```bash
python src/modeling/finetune_if_condition.py \
    --tokenizer artifacts/tokenizer \
    --pretrained artifacts/pretrain_gpt2 \
    --train data/finetune_train.jsonl \
    --val data/finetune_val.jsonl \
    --out-dir artifacts/ifrec_finetuned \
    --epochs 5 \
    --batch-size 4
```

### Step 8: Generate Predictions

```bash
python src/modeling/predict.py \
    --tokenizer artifacts/tokenizer \
    --model artifacts/ifrec_finetuned \
    --test data/finetune_test.jsonl \
    --out predictions.csv
```

### Step 9: Evaluate

```bash
python src/evaluation/score_predictions.py \
    --csv predictions.csv \
    --samples 10
```

---

## Testing Individual Components

### Test Data Mining

```bash
# Create a small test repos file
echo "https://github.com/psf/requests" > test_repos.txt

python src/data/mine_github.py \
    --repos-file test_repos.txt \
    --out-dir data/test_repos \
    --max-workers 2
```

### Test Function Extraction

```bash
python src/data/extract_functions.py \
    --repos-root data/test_repos \
    --out data/test_functions.jsonl \
    --min-lines 5 \
    --max-lines 400
```

### Test Tokenizer Training

```bash
python src/tokenizer/train_tokenizer.py \
    --corpus data/pretrain_corpus.txt \
    --out-dir artifacts/test_tokenizer \
    --vocab-size 10000
```

### Test Utilities

```python
# Test AST utilities
python -c "
from src.modeling.utils import find_if_conditions_in_func, mask_one_if_condition

code = '''
def foo(x):
    if x > 0:
        return True
    if x < 0:
        return False
'''

conditions = find_if_conditions_in_func(code)
print(f'Found {len(conditions)} conditions:', conditions)

masked, condition = mask_one_if_condition(code, 0)
print(f'Masked code:\n{masked}')
print(f'Condition: {condition}')
"
```

---

## Common Issues and Solutions

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce batch size: `--batch-size 2` or `--batch-size 1`
2. Reduce sequence length: `--block-size 256` or `--max-length 512`
3. Use gradient accumulation (modify training args)
4. Use smaller model: `--n-layer 6 --n-embd 512 --n-head 8`

### Issue: Slow Training

**Solutions**:
1. Enable FP16 (automatic if GPU available)
2. Reduce dataset size: `--max-samples 10000`
3. Reduce epochs: `--epochs 1`
4. Use GPU if available

### Issue: Git Clone Failures

**Solutions**:
1. Check network connection
2. Try fewer workers: `--max-workers 2`
3. Remove failed repos from `repos.txt`
4. Use `git` command manually to debug

### Issue: ImportError

**Solutions**:
1. Ensure you're running from project root
2. Install all dependencies: `pip install -r requirements.txt`
3. Check Python version (3.8+ recommended)

### Issue: Tokenizer Errors

**Solutions**:
1. Ensure corpus file exists and is not empty
2. Check special tokens in tokenizer config
3. Try smaller vocab size: `--vocab-size 10000`

---

## Model Checkpoints

All models are saved in the `artifacts/` directory:

- `artifacts/tokenizer/` - Custom BPE tokenizer
- `artifacts/pretrain_gpt2/` - Pre-trained model checkpoints
- `artifacts/ifrec_finetuned/` - Fine-tuned model

You can resume training from any checkpoint by pointing to the directory.

---

## Dataset Format Reference

### functions.jsonl

Each line is a JSON object:
```json
{
  "file": "path/to/file.py",
  "function_name": "my_function",
  "lineno": 10,
  "end_lineno": 25,
  "n_lines": 15,
  "has_if": true,
  "num_ifs": 2,
  "code": "def my_function():\n    if x > 0:\n        return True",
  "hash": "abc123..."
}
```

### Fine-tuning JSONL (train/val/test)

Each line is a JSON object:
```json
{
  "input": "def foo():\n    if <IFMASK>:\n        pass",
  "target": "x > 0",
  "meta": {
    "file": "path/to/file.py",
    "function": "foo",
    "lineno": 10,
    "if_index": 0,
    "total_ifs": 1,
    "original": "def foo():\n    if x > 0:\n        pass"
  }
}
```

### Predictions CSV

Columns (exact names required):
1. `Input provided to the model`
2. `Whether the prediction is correct (true/false)`
3. `Expected if condition`
4. `Predicted if condition`
5. `Prediction score (0-100)`

---

## Performance Benchmarks

### Smoke Test (1000 samples)

- Mining: ~2-5 minutes
- Extraction: ~1-2 minutes
- Tokenizer training: ~1 minute
- Pre-training: ~5-15 minutes (GPU) / 30-60 minutes (CPU)
- Fine-tuning: ~3-10 minutes (GPU) / 15-30 minutes (CPU)
- Prediction: ~2-5 minutes

**Total**: ~15-30 minutes with GPU

### Full Pipeline (50k+ samples)

- Mining: ~10-30 minutes
- Extraction: ~10-20 minutes
- Tokenizer training: ~5-10 minutes
- Pre-training: ~2-6 hours (GPU) / 1-3 days (CPU)
- Fine-tuning: ~30-90 minutes (GPU) / 4-12 hours (CPU)
- Prediction: ~10-30 minutes

**Total**: ~3-8 hours with GPU, or 1-4 days with CPU

---

## Tips for Better Results

1. **More data is better**: Try to get 100k+ functions for pre-training
2. **Quality over quantity**: Filter out test files and generated code
3. **Balanced dataset**: Use `--max-per-file` to avoid over-representation
4. **Longer pre-training**: Try 5-10 epochs for better base model
5. **Early stopping**: Use validation set to prevent overfitting
6. **Hyperparameter tuning**: Experiment with learning rates and batch sizes
7. **Ensemble**: Train multiple models with different seeds

---

## Next Steps

After successful execution:

1. **Analyze results**: Look at prediction samples in the CSV
2. **Error analysis**: Identify common failure patterns
3. **Improve data quality**: Filter out problematic examples
4. **Tune hyperparameters**: Try different model sizes and learning rates
5. **Fill report template**: Document your findings in `report_template.md`
6. **Share results**: Commit (excluding data/artifacts) and share your work

---

## Questions?

Check the main README.md or individual script help:
```bash
python src/data/mine_github.py --help
python src/modeling/pretrain_clm.py --help
# etc.
```
