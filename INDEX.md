# Project Index

Complete reference guide for the Python If-Condition Recommender project.

## 📚 Documentation Files

| File | Purpose | When to Read |
|------|---------|--------------|
| **README.md** | Project overview, quick start, basic usage | Start here |
| **SETUP.md** | Detailed setup, testing, troubleshooting | Before running pipeline |
| **SUMMARY.md** | Comprehensive project summary | Understanding architecture |
| **ACCEPTANCE_CHECKLIST.md** | Verification of requirements | Confirming completeness |
| **EXAMPLE_WORKFLOW.md** | Step-by-step walkthrough with outputs | Running smoke test |
| **report_template.md** | 3-page technical report template | Writing final report |
| **INDEX.md** | This file - complete reference | Finding specific info |

## 🗂️ Source Code Modules

### Data Pipeline (`src/data/`)

| Script | Purpose | Key Functions | Dependencies |
|--------|---------|---------------|--------------|
| **mine_github.py** | Clone GitHub repositories | `parse_repo_url()`, `clone_repo()`, `mine_github_repos()` | git, concurrent.futures |
| **extract_functions.py** | Extract Python functions using AST | `extract_function()`, `count_if_statements()`, `compute_hash()` | ast, hashlib |
| **build_pretrain_corpus.py** | Build pre-training text corpus | `build_pretrain_corpus()` | json |
| **build_finetune_dataset.py** | Create masked if-condition pairs | `create_masked_example()`, `build_finetune_dataset()` | utils.py |

### Tokenization (`src/tokenizer/`)

| Script | Purpose | Key Functions | Dependencies |
|--------|---------|---------------|--------------|
| **train_tokenizer.py** | Train Byte-level BPE tokenizer | `train_tokenizer()` | tokenizers, transformers |

### Modeling (`src/modeling/`)

| Script | Purpose | Key Functions | Dependencies |
|--------|---------|---------------|--------------|
| **utils.py** | AST utilities for if-conditions | `find_if_conditions_in_func()`, `mask_one_if_condition()`, `normalize_condition()`, `is_correct()` | ast, astor |
| **data_collators.py** | Custom data collators | `DataCollatorForIfConditionPrediction`, `DataCollatorForCLM` | transformers |
| **pretrain_clm.py** | Pre-train GPT-2 from scratch | `create_model_from_scratch()`, `prepare_dataset()`, `pretrain_model()` | transformers, datasets |
| **finetune_if_condition.py** | Fine-tune for if-prediction | `format_example()`, `prepare_dataset()`, `finetune_model()` | transformers |
| **predict.py** | Generate predictions and CSV | `generate_prediction()`, `predict_batch()`, `save_predictions_csv()` | transformers, torch |

### Evaluation (`src/evaluation/`)

| Script | Purpose | Key Functions | Dependencies |
|--------|---------|---------------|--------------|
| **score_predictions.py** | Evaluate prediction accuracy | `compute_metrics()`, `print_metrics()` | csv, numpy |

### Scripts (`scripts/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| **quickstart.sh** | End-to-end pipeline | `bash scripts/quickstart.sh [--smoke-test]` |

## 🔧 Configuration Files

| File | Purpose | Format |
|------|---------|--------|
| **repos.txt** | GitHub repository URLs | Plain text, one URL per line |
| **requirements.txt** | Python dependencies | pip format |
| **.gitignore** | Excluded files/directories | Git ignore format |

## 📊 Data Formats

### Input Formats

#### repos.txt
```
https://github.com/owner/repo1
https://github.com/owner/repo2
# Comments start with #
```

### Intermediate Formats

#### functions.jsonl
```json
{
  "file": "relative/path/to/file.py",
  "file_abs": "/absolute/path/to/file.py",
  "function_name": "my_function",
  "lineno": 10,
  "end_lineno": 25,
  "n_lines": 15,
  "has_if": true,
  "num_ifs": 2,
  "code": "def my_function():\n    if x > 0:\n        return True",
  "hash": "abc123def456..."
}
```

#### pretrain_corpus.txt
```
def function1():
    pass

def function2():
    if x > 0:
        return True

```

#### Fine-tuning JSONL (train/val/test)
```json
{
  "input": "def foo():\n    if <IFMASK>:\n        return True",
  "target": "x > 0",
  "meta": {
    "file": "path/to/file.py",
    "function": "foo",
    "lineno": 10,
    "if_index": 0,
    "total_ifs": 1,
    "original": "def foo():\n    if x > 0:\n        return True"
  }
}
```

### Output Formats

#### predictions.csv
```csv
Input provided to the model,Whether the prediction is correct (true/false),Expected if condition,Predicted if condition,Prediction score (0-100)
"Task: Predict...",true,x > 0,x > 0,87.34
"Task: Predict...",false,len(items) > 0,items,45.67
```

## 🎯 Command Reference

### Data Pipeline

```bash
# Mine repositories
python src/data/mine_github.py \
    --repos-file repos.txt \
    --out-dir data/raw_repos \
    --max-workers 8

# Extract functions
python src/data/extract_functions.py \
    --repos-root data/raw_repos \
    --out data/functions.jsonl \
    --min-lines 5 \
    --max-lines 400

# Build pre-training corpus
python src/data/build_pretrain_corpus.py \
    --functions data/functions.jsonl \
    --out data/pretrain_corpus.txt

# Build fine-tuning dataset
python src/data/build_finetune_dataset.py \
    --functions data/functions.jsonl \
    --out-prefix data/finetune \
    --val 0.1 --test 0.1 --seed 42 \
    [--max-per-file N] [--max-examples N]
```

### Training

```bash
# Train tokenizer
python src/tokenizer/train_tokenizer.py \
    --corpus data/pretrain_corpus.txt \
    --out-dir artifacts/tokenizer \
    --vocab-size 52000

# Pre-train model
python src/modeling/pretrain_clm.py \
    --tokenizer artifacts/tokenizer \
    --corpus data/pretrain_corpus.txt \
    --out-dir artifacts/pretrain_gpt2 \
    --epochs 3 --batch-size 8 \
    [--max-samples N] [--n-layer N] [--n-embd N]

# Fine-tune model
python src/modeling/finetune_if_condition.py \
    --tokenizer artifacts/tokenizer \
    --pretrained artifacts/pretrain_gpt2 \
    --train data/finetune_train.jsonl \
    --val data/finetune_val.jsonl \
    --out-dir artifacts/ifrec_finetuned \
    --epochs 5 --batch-size 4
```

### Inference & Evaluation

```bash
# Generate predictions
python src/modeling/predict.py \
    --tokenizer artifacts/tokenizer \
    --model artifacts/ifrec_finetuned \
    --test data/finetune_test.jsonl \
    --out predictions.csv

# Evaluate predictions
python src/evaluation/score_predictions.py \
    --csv predictions.csv \
    --samples 10
```

### Full Pipeline

```bash
# Smoke test (quick verification)
bash scripts/quickstart.sh --smoke-test

# Full pipeline
bash scripts/quickstart.sh
```

## 🔍 Finding Specific Information

### "How do I...?"

| Task | See File | Section/Function |
|------|----------|------------------|
| Clone repositories | mine_github.py | `mine_github_repos()` |
| Extract functions from Python files | extract_functions.py | `extract_function()` |
| Find if-conditions in code | utils.py | `find_if_conditions_in_func()` |
| Mask an if-condition | utils.py | `mask_one_if_condition()` |
| Check if prediction is correct | utils.py | `is_correct()` |
| Train a tokenizer | train_tokenizer.py | `train_tokenizer()` |
| Create GPT-2 model | pretrain_clm.py | `create_model_from_scratch()` |
| Mask prompt tokens in labels | data_collators.py | `DataCollatorForIfConditionPrediction` |
| Generate predictions | predict.py | `generate_prediction()` |
| Compute confidence score | predict.py | `compute_score_from_logprobs()` |
| Evaluate accuracy | score_predictions.py | `compute_metrics()` |

### "Where is...?"

| Item | Location |
|------|----------|
| Special tokens definition | tokenizer/train_tokenizer.py, line ~50 |
| Prompt template | finetune_if_condition.py, line ~27 |
| Model architecture config | pretrain_clm.py, line ~45 |
| CSV column names | predict.py, line ~231 |
| Normalization logic | utils.py, `normalize_condition()` |
| Deduplication logic | extract_functions.py, `compute_hash()` |
| Directory exclusions | extract_functions.py, line ~24 |
| Training hyperparameters | pretrain_clm.py and finetune_if_condition.py, TrainingArguments |

### "What does this error mean?"

| Error | File | Solution |
|-------|------|----------|
| "Repository file not found" | mine_github.py | Create repos.txt |
| "Tokenizer not found" | Any training script | Run train_tokenizer.py first |
| "No functions with if statements" | build_finetune_dataset.py | Add more repositories |
| "Out of memory" | Training scripts | Reduce batch size or model size |
| "Answer token not found" | data_collators.py | Ensure tokenizer includes `<answer>` |
| Import errors | Any script | Run from project root, install dependencies |

## 📈 Performance Metrics

### Expected Resource Usage

| Phase | Time (GPU) | Time (CPU) | Memory | Disk |
|-------|-----------|-----------|--------|------|
| Mining (10 repos) | 5-10 min | 5-10 min | ~1GB | ~500MB |
| Extraction | 2-5 min | 2-5 min | ~2GB | ~100MB |
| Tokenizer training | 1-5 min | 5-15 min | ~4GB | ~50MB |
| Pre-training (1k samples) | 5-15 min | 30-90 min | ~8GB | ~500MB |
| Pre-training (50k samples) | 2-6 hours | 1-3 days | ~16GB | ~500MB |
| Fine-tuning (1k samples) | 3-10 min | 15-45 min | ~8GB | ~500MB |
| Fine-tuning (50k samples) | 30-90 min | 4-12 hours | ~16GB | ~500MB |
| Prediction (100 samples) | 2-5 min | 5-15 min | ~4GB | - |

### Expected Accuracy

| Dataset Size | Pre-train Epochs | Fine-tune Epochs | Typical Accuracy |
|--------------|------------------|------------------|------------------|
| 1k examples | 1 | 2 | 10-30% |
| 10k examples | 3 | 5 | 25-45% |
| 50k+ examples | 3-5 | 5-10 | 35-60% |

## 🎓 Learning Path

### For Beginners

1. Read **README.md** - understand the project
2. Read **SETUP.md** sections 1-3 - installation and setup
3. Read **EXAMPLE_WORKFLOW.md** - see complete example
4. Run smoke test - verify everything works
5. Read **SUMMARY.md** - understand architecture

### For Implementation

1. Study **utils.py** - AST manipulation
2. Study **data_collators.py** - prompt masking
3. Study **pretrain_clm.py** - model architecture
4. Study **finetune_if_condition.py** - task formulation
5. Study **predict.py** - generation and scoring

### For Extension

1. Modify **utils.py** - different masking strategies
2. Modify **pretrain_clm.py** - different architectures
3. Modify **finetune_if_condition.py** - different prompts
4. Modify **score_predictions.py** - semantic evaluation

## 🔗 Quick Links

### Essential Commands
- **Smoke test**: `bash scripts/quickstart.sh --smoke-test`
- **Full pipeline**: `bash scripts/quickstart.sh`
- **Help**: `python src/[module]/[script].py --help`

### Common Tasks
- **Add repositories**: Edit `repos.txt`
- **Change model size**: Add `--n-layer X --n-embd Y` to pretrain_clm.py
- **Reduce dataset**: Add `--max-examples N` to build_finetune_dataset.py
- **Resume training**: Point to checkpoint directory

### Troubleshooting
- **Check logs**: `artifacts/*/logs/`
- **Verify setup**: Run individual scripts with `--help`
- **Test components**: See SETUP.md "Testing Individual Components"

## 📋 Checklist for Running

### Before Starting
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create `repos.txt` with 3+ repositories
- [ ] Verify disk space: ~10GB free
- [ ] Check Python version: 3.8+

### Smoke Test
- [ ] Run: `bash scripts/quickstart.sh --smoke-test`
- [ ] Check output CSV exists
- [ ] Verify accuracy > 0%
- [ ] Check no error messages

### Full Pipeline
- [ ] Add 10-20 repositories to `repos.txt`
- [ ] Run: `bash scripts/quickstart.sh`
- [ ] Monitor training (hours to days)
- [ ] Check final accuracy
- [ ] Analyze predictions CSV

### After Completion
- [ ] Fill report template
- [ ] Analyze error patterns
- [ ] Document hyperparameters
- [ ] Save artifacts (exclude from git)

## 📞 Support

### Self-Help Resources
1. Script help: `python <script>.py --help`
2. SETUP.md troubleshooting section
3. EXAMPLE_WORKFLOW.md common issues
4. Individual script docstrings

### Debugging Steps
1. Verify installation: `python -c "import torch, transformers"`
2. Test individual components (see SETUP.md)
3. Check error messages in console
4. Review SETUP.md "Common Issues" section

## 🎉 Success Criteria

Your setup is successful when:
- ✅ Smoke test completes without errors
- ✅ predictions.csv is generated
- ✅ Accuracy > 0% (>10% for smoke test is good)
- ✅ CSV has exactly 5 columns with correct names
- ✅ Model artifacts saved in artifacts/

## 📚 Citation

If using this project:
```bibtex
@misc{ifrec2024,
  title={Python If-Condition Recommender},
  author={[Your Name]},
  year={2024},
  note={AI for Software Engineering Assignment}
}
```

---

**Last Updated**: [Current Date]  
**Version**: 1.0  
**Status**: Production Ready ✅
