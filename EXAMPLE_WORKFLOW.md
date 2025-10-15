# Example Workflow

This document shows a complete example of running the pipeline from start to finish.

## Scenario: Quick Smoke Test

You want to verify the entire pipeline works before committing to a long training run.

## Step-by-Step Example

### 1. Initial Setup

```bash
# Navigate to project directory
cd /workspaces/AI4SE_assignment1

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers, tokenizers; print('All dependencies installed!')"
```

**Expected output**: `All dependencies installed!`

### 2. Prepare Repository List

```bash
# Create a minimal repos.txt for testing
cat > repos.txt << EOF
https://github.com/psf/requests
https://github.com/pallets/flask
https://github.com/kennethreitz/requests
EOF
```

**What this does**: Creates a file with 3 popular Python repositories.

### 3. Run Smoke Test

```bash
# Run the full pipeline in smoke test mode
bash scripts/quickstart.sh --smoke-test
```

**What happens**:
1. Clones 3 repositories (~2-5 minutes)
2. Extracts Python functions (~1 minute)
3. Builds datasets (~1 minute)
4. Trains tokenizer (~1 minute)
5. Pre-trains model for 1 epoch (~5-15 minutes with GPU)
6. Fine-tunes model for 2 epochs (~3-10 minutes with GPU)
7. Generates predictions (~2-5 minutes)
8. Evaluates accuracy (~1 second)

**Total time**: 15-30 minutes with GPU, 1-2 hours with CPU

### 4. Expected Console Output

```
============================================================
PYTHON IF-CONDITION RECOMMENDER - FULL PIPELINE
============================================================

Smoke test parameters:
  Max samples: 1000
  Pre-train epochs: 1
  Fine-tune epochs: 2

------------------------------------------------------------
Step 1/9: Mining GitHub repositories...
------------------------------------------------------------
Found 3 repositories to clone.
Output directory: data/raw_repos
Using 8 worker threads.

Cloning repos: 100%|██████████| 3/3 [00:15<00:00,  5.2s/it]
✓ Cloned: psf/requests
✓ Already exists: pallets/flask
✓ Cloned: kennethreitz/requests

Cloning complete: 3/3 successful

------------------------------------------------------------
Step 2/9: Extracting Python functions...
------------------------------------------------------------
Scanning for Python files in: data/raw_repos
Found 247 Python files.
Extracting functions: 100%|██████████| 247/247 [00:12<00:00, 20.1it/s]
Extracted 3542 raw functions.
After filtering (5-400 lines): 2891 functions.
After deduplication: 2654 unique functions.

------------------------------------------------------------
Step 3/9: Building pre-training corpus...
------------------------------------------------------------
Reading functions from: data/functions.jsonl
Loaded 2654 functions.
After min-lines filter (5): 2654 functions.
Writing corpus: 100%|██████████| 2654/2654 [00:01<00:00, 1892.3it/s]
Wrote 2654 functions to: data/pretrain_corpus.txt

------------------------------------------------------------
Step 4/9: Building fine-tuning dataset...
------------------------------------------------------------
Reading functions from: data/functions.jsonl
Loaded 2654 functions.
Functions with if statements: 1842
Creating masked examples: 100%|██████████| 1842/1842 [00:05<00:00, 352.1it/s]
Generated 1000 masked examples. (capped at 1000 for smoke test)

Split sizes:
  Train: 800
  Val:   100
  Test:  100

------------------------------------------------------------
Step 5/9: Training tokenizer from scratch...
------------------------------------------------------------
Training tokenizer from scratch on: data/pretrain_corpus.txt
Vocabulary size: 52000
Output directory: artifacts/tokenizer

Special tokens: ['<pad>', '<s>', '</s>', '<unk>', '<mask>', '<IFMASK>', '<answer>']

Training tokenizer (this may take a while)...
[00:00:47] Training: 100%|██████████| 52000/52000

Tokenizer saved to: artifacts/tokenizer

Converting to HuggingFace PreTrainedTokenizerFast...
HuggingFace tokenizer saved to: artifacts/tokenizer

============================================================
Testing tokenizer:
============================================================
Original: def foo(x):
    if x > 0:
        return True
Encoded:  [50256, 376, 12345, 7, 87, 10, ...]... (18 tokens)
Decoded:  def foo(x):
    if x > 0:
        return True

With <IFMASK>: def foo(x):
    if <IFMASK>:
        return True
Encoded: [50256, 376, 12345, 7, 87, 10, 50261, ...]... (18 tokens)

Special token IDs:
  <pad>:    50256
  <s>:      50257
  </s>:     50258
  <unk>:    50259
  <mask>:   50260
  <IFMASK>: 50261
  <answer>: 50262

Vocabulary size: 52263
============================================================

------------------------------------------------------------
Step 6/9: Pre-training GPT-2 model from scratch...
------------------------------------------------------------

Creating GPT-2 model from scratch (random initialization)...
Model config:
  Layers: 12
  Embedding dim: 768
  Attention heads: 12
  FFN dim: 3072
  Max sequence length: 512
  Total parameters: 124,439,808

Loading corpus from: data/pretrain_corpus.txt
Loaded 1000 examples.
Tokenizing dataset...
Grouping texts into blocks of 512 tokens...
Final dataset size: 945 blocks
Train samples: 897
Eval samples:  48

============================================================
Starting pre-training...
============================================================

Epoch 1/1: 100%|██████████| 112/112 [07:23<00:00,  3.96s/it, loss=3.24]
Eval loss: 3.18

Saving final model...

============================================================
Pre-training complete!
Model saved to: artifacts/pretrain_gpt2
============================================================

------------------------------------------------------------
Step 7/9: Fine-tuning for if-condition prediction...
------------------------------------------------------------
Loading tokenizer from: artifacts/tokenizer
Tokenizer loaded. Vocab size: 52263
Special token IDs:
  <answer>: 50262
  <IFMASK>: 50261

Loading pre-trained model from: artifacts/pretrain_gpt2
Model loaded. Parameters: 124,439,808

Loading dataset from: data/finetune_train.jsonl
Loaded 800 examples.
Tokenizing dataset...
Dataset ready: 800 examples

Loading dataset from: data/finetune_val.jsonl
Loaded 100 examples.
Tokenizing dataset...
Dataset ready: 100 examples

Train samples: 800
Val samples:   100

============================================================
Starting fine-tuning...
============================================================

Epoch 1/2: 100%|██████████| 200/200 [04:12<00:00,  1.26s/it, loss=2.15]
Eval loss: 2.08

Epoch 2/2: 100%|██████████| 200/200 [04:09<00:00,  1.25s/it, loss=1.87]
Eval loss: 1.92

Saving final model...

============================================================
Fine-tuning complete!
Model saved to: artifacts/ifrec_finetuned
============================================================

------------------------------------------------------------
Step 8/9: Generating predictions on test set...
------------------------------------------------------------
Loading tokenizer from: artifacts/tokenizer
Loading model from: artifacts/ifrec_finetuned
Model loaded on device: cuda

Loading test data from: data/finetune_test.jsonl
Loaded 100 test examples.

============================================================
Generating predictions...
============================================================

Generating predictions: 100%|██████████| 100/100 [02:15<00:00,  1.35s/it]
Saved 100 predictions to: predictions.csv

============================================================
Prediction Summary:
  Total examples: 100
  Correct: 27
  Accuracy: 27.00%
  Average score: 64.32
============================================================

------------------------------------------------------------
Step 9/9: Evaluating predictions...
------------------------------------------------------------
Loading predictions from: predictions.csv
Loaded 100 predictions.

============================================================
EVALUATION METRICS
============================================================
Total examples:              100
Correct predictions:         27
Incorrect predictions:       73
Accuracy:                    27.00%
------------------------------------------------------------
Average prediction score:    64.32
Std dev prediction score:    15.47
Avg score (correct):         78.45
Avg score (incorrect):       59.21
============================================================

============================================================
SAMPLE PREDICTIONS (first 10)
============================================================

✓ Example 1:
  Expected:  x > 0
  Predicted: x > 0
  Score:     87.34

✗ Example 2:
  Expected:  len(items) > 0
  Predicted: items
  Score:     62.15

✓ Example 3:
  Expected:  isinstance(obj, dict)
  Predicted: isinstance(obj, dict)
  Score:     91.23

✗ Example 4:
  Expected:  not self.closed
  Predicted: self.closed
  Score:     71.54

✓ Example 5:
  Expected:  value is None
  Predicted: value is None
  Score:     88.76

... (5 more examples)

============================================================
PIPELINE COMPLETE!
============================================================

Outputs:
  - Tokenizer:        artifacts/tokenizer
  - Pre-trained model: artifacts/pretrain_gpt2
  - Fine-tuned model:  artifacts/ifrec_finetuned
  - Predictions CSV:   predictions.csv

To run predictions on a different test set:
  python src/modeling/predict.py \
    --tokenizer artifacts/tokenizer \
    --model artifacts/ifrec_finetuned \
    --test path/to/test.jsonl \
    --out output.csv
```

### 5. Examine Results

```bash
# View predictions CSV
head -n 5 predictions.csv

# View detailed stats
python src/evaluation/score_predictions.py --csv predictions.csv --samples 20

# Check file sizes
du -sh artifacts/tokenizer artifacts/pretrain_gpt2 artifacts/ifrec_finetuned
```

**Example output**:
```
Input provided to the model,Whether the prediction is correct (true/false),Expected if condition,Predicted if condition,Prediction score (0-100)
"Task: Predict the masked if condition.

Function:
def validate(x):
    if <IFMASK>:
        return True
    return False

<answer>
",true,x > 0,x > 0,87.34
...

# File sizes
47M     artifacts/tokenizer
470M    artifacts/pretrain_gpt2
472M    artifacts/ifrec_finetuned
```

### 6. Inspect Specific Predictions

```python
# Python script to analyze predictions
import csv

with open('predictions.csv', 'r') as f:
    reader = csv.DictReader(f)
    predictions = list(reader)

# Find high-confidence correct predictions
high_conf_correct = [
    p for p in predictions 
    if p['Whether the prediction is correct (true/false)'] == 'true'
    and float(p['Prediction score (0-100)']) > 85
]

print(f"High-confidence correct: {len(high_conf_correct)}")
for pred in high_conf_correct[:5]:
    print(f"  Expected: {pred['Expected if condition']}")
    print(f"  Score: {pred['Prediction score (0-100)']}")
    print()

# Find high-confidence incorrect predictions (interesting errors)
high_conf_incorrect = [
    p for p in predictions 
    if p['Whether the prediction is correct (true/false)'] == 'false'
    and float(p['Prediction score (0-100)']) > 75
]

print(f"High-confidence incorrect: {len(high_conf_incorrect)}")
for pred in high_conf_incorrect[:5]:
    print(f"  Expected: {pred['Expected if condition']}")
    print(f"  Predicted: {pred['Predicted if condition']}")
    print(f"  Score: {pred['Prediction score (0-100)']}")
    print()
```

### 7. Next Steps After Smoke Test

If the smoke test succeeds:

```bash
# 1. Add more repositories to repos.txt
cat >> repos.txt << EOF
https://github.com/django/django
https://github.com/scikit-learn/scikit-learn
https://github.com/matplotlib/matplotlib
https://github.com/pandas-dev/pandas
https://github.com/python/cpython
https://github.com/tensorflow/tensorflow
https://github.com/pytorch/pytorch
EOF

# 2. Run full pipeline (no smoke test flag)
bash scripts/quickstart.sh

# 3. Or run with custom parameters
python src/modeling/pretrain_clm.py \
    --tokenizer artifacts/tokenizer \
    --corpus data/pretrain_corpus.txt \
    --out-dir artifacts/pretrain_gpt2_large \
    --epochs 5 \
    --batch-size 16 \
    --n-layer 24 \
    --n-embd 1024 \
    --n-head 16
```

## Common Results Analysis

### Good Signs
- ✅ Training loss decreases smoothly
- ✅ Validation loss tracks training loss
- ✅ Accuracy > 20% on smoke test
- ✅ Higher scores for correct predictions
- ✅ Model generates syntactically valid conditions

### Warning Signs
- ⚠️ Training loss not decreasing → reduce learning rate
- ⚠️ Validation loss increases → overfitting, add regularization
- ⚠️ Very low accuracy (<10%) → need more data or longer training
- ⚠️ Similar scores for correct/incorrect → confidence calibration issue

### Expected Accuracy Ranges
- **Smoke test (1k examples, 1-2 epochs)**: 10-30%
- **Medium (10k examples, 3-5 epochs)**: 25-45%
- **Full (50k+ examples, 5-10 epochs)**: 35-60%

Note: Accuracy depends heavily on:
- Dataset quality and diversity
- Model size and architecture
- Training duration
- Hyperparameter tuning

## Example Error Analysis

### Common Error Types

**Type 1: Negation Errors**
```
Expected:  not x
Predicted: x
Analysis: Model struggles with negation
```

**Type 2: Comparison Direction**
```
Expected:  x > 0
Predicted: 0 > x
Analysis: Semantically equivalent, but exact match fails
```

**Type 3: Complex Conditions**
```
Expected:  len(items) > 0 and items[0] is not None
Predicted: len(items) > 0
Analysis: Model generates partial condition
```

**Type 4: Type Checks**
```
Expected:  isinstance(obj, dict)
Predicted: type(obj) == dict
Analysis: Different but equivalent expressions
```

## Troubleshooting Smoke Test Issues

### Issue: "No functions with if statements"
**Solution**: Repositories too small or no if statements found. Add more repos.

### Issue: OOM during pre-training
**Solution**: 
```bash
python src/modeling/pretrain_clm.py \
    --batch-size 2 \
    --block-size 256 \
    --max-samples 500
```

### Issue: Very slow on CPU
**Solution**: Use even smaller smoke test:
```bash
python src/data/build_finetune_dataset.py \
    --functions data/functions.jsonl \
    --out-prefix data/finetune \
    --max-examples 100
```

### Issue: Git clone failures
**Solution**:
```bash
# Clone manually to debug
git clone --depth=1 https://github.com/psf/requests data/raw_repos/psf/requests
```

## Full Pipeline Example (After Smoke Test)

Once smoke test passes, scale up:

```bash
# 1. Expand repository list (aim for 15-20 repos)
# Edit repos.txt manually or:
cat additional_repos.txt >> repos.txt

# 2. Clean previous artifacts (optional)
rm -rf data/ artifacts/

# 3. Run full pipeline
bash scripts/quickstart.sh

# This will:
# - Extract 10k-100k functions (depending on repos)
# - Train tokenizer on full corpus
# - Pre-train for 3 epochs (~2-6 hours with GPU)
# - Fine-tune for 5 epochs (~30-90 minutes with GPU)
# - Generate predictions
# - Achieve 35-60% accuracy (typical)
```

## Conclusion

This example demonstrates:
1. ✅ Complete pipeline execution
2. ✅ Expected outputs and timings
3. ✅ Result interpretation
4. ✅ Error analysis
5. ✅ Next steps for scaling

The smoke test should complete successfully in 15-30 minutes with GPU, confirming your setup is correct before committing to the full multi-hour training run.

**Ready to start?**
```bash
bash scripts/quickstart.sh --smoke-test
```
