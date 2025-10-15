# Python If-Condition Recommender: Technical Report

**Course**: AI for Software Engineering  
**Assignment**: If-Condition Prediction using Transformers  
**Date**: [Your Date]  
**Author**: [Your Name]

---

## 1. Introduction

This project implements a Transformer-based system to predict masked if-conditions in Python functions. The approach consists of two phases: (1) pre-training a GPT-2 style decoder-only model from scratch on Python code using causal language modeling (CLM), and (2) fine-tuning the model to predict masked if-conditions.

### Key Requirements Met
- ✅ Custom tokenizer trained from scratch (Byte-level BPE)
- ✅ GPT-2 model trained from random initialization (no pre-trained weights)
- ✅ AST-based function extraction and if-condition masking
- ✅ Deduplication and quality filtering
- ✅ Train/validation/test splits with exact match evaluation
- ✅ CSV output with exact required columns and confidence scores

---

## 2. Data Collection and Preprocessing

### 2.1 Repository Mining

**Approach**: We mined Python code from GitHub repositories specified in `repos.txt`. The mining process:
- Clones repositories with `depth=1` to minimize storage
- Uses multi-threading (8 workers by default) for parallel downloads
- Skips already-cloned repositories for reproducibility

**Repositories Used**:
```
[List your repositories here, e.g.:]
- https://github.com/psf/requests (22,000+ stars)
- https://github.com/pallets/flask (65,000+ stars)
- https://github.com/numpy/numpy (25,000+ stars)
[Add more as appropriate]
```

### 2.2 Function Extraction

**Method**: AST-based parsing using Python's built-in `ast` module.

**Extraction Pipeline**:
1. Walk through repository directories, excluding vendor/venv/test/build folders
2. Parse each `.py` file using `ast.parse()`
3. Extract function metadata: name, line range, source code, if-condition count
4. Apply quality filters:
   - Minimum lines: 5
   - Maximum lines: 400
   - Exclude files with syntax errors

**Deduplication**: Functions are deduplicated using SHA-256 hash of normalized source code (whitespace collapsed, comments removed).

**Statistics**:
```
[Fill in your actual numbers]
Total Python files processed:    [X]
Raw functions extracted:         [X]
After line-count filtering:      [X]
After deduplication:             [X]
Functions with if statements:    [X]
```

### 2.3 Dataset Construction

#### Pre-training Corpus
- **Format**: Plain text, one function per line (separated by double newline)
- **Size**: [X] functions, [Y] lines of code
- **Purpose**: Learn general Python syntax and patterns

#### Fine-tuning Dataset
- **Task**: Predict masked if-condition given function context
- **Masking Strategy**: For each function with N if-statements, create N examples by masking each condition once with `<IFMASK>` token
- **Prompt Format**:
  ```
  Task: Predict the masked if condition.
  
  Function:
  [function with one condition replaced by <IFMASK>]
  
  <answer>
  [target condition]
  ```

**Split Statistics**:
```
[Fill in your actual numbers]
Training examples:      [X] (80%)
Validation examples:    [Y] (10%)
Test examples:          [Z] (10%)
```

**Quality Controls**:
- Maximum examples per file: [N] (if applied)
- Random seed: 42 (for reproducibility)
- Only functions with at least one if-statement

---

## 3. Tokenization

### 3.1 Tokenizer Architecture

**Type**: Byte-level BPE (Byte Pair Encoding)

**Rationale**: 
- Handles any Unicode character (robust to variable names in different languages)
- Efficient for code with special characters and operators
- No pre-training bias from existing tokenizers

### 3.2 Training Details

**Vocabulary Size**: 52,000 tokens  
**Training Corpus**: Same as pre-training corpus  
**Minimum Frequency**: 2

**Special Tokens**:
| Token | Purpose |
|-------|---------|
| `<pad>` | Padding |
| `<s>` | Beginning of sequence |
| `</s>` | End of sequence |
| `<unk>` | Unknown token |
| `<mask>` | General masking |
| `<IFMASK>` | Masked if-condition placeholder |
| `<answer>` | Answer delimiter in prompts |

**Validation**: Tested encoding/decoding on sample code to ensure special tokens are properly handled.

---

## 4. Pre-training

### 4.1 Model Architecture

**Base**: GPT-2 (decoder-only Transformer)  
**Initialization**: Random (no pre-trained weights)

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Embedding dimension | 768 |
| Attention heads | 12 |
| FFN inner dimension | 3072 |
| Total parameters | ~[X]M |
| Max sequence length | 512 |

### 4.2 Training Configuration

**Objective**: Causal Language Modeling (CLM) - predict next token given previous context

**Training Details**:
| Setting | Value |
|---------|-------|
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 5e-4 |
| Warmup steps | 500 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| FP16 | Yes (if GPU available) |

### 4.3 Pre-training Results

```
[Fill in your actual results]
Final training loss:     [X.XX]
Final validation loss:   [X.XX]
Training time:           [X] hours
Hardware:                [GPU/CPU model]
```

**Observations**:
- [Describe loss curves, convergence behavior, etc.]
- [Any challenges or adjustments made]

---

## 5. Fine-tuning

### 5.1 Task Formulation

**Input**: Function with one if-condition replaced by `<IFMASK>`  
**Output**: The original condition expression  
**Format**: Prompt-based generation with `<answer>` token as delimiter

### 5.2 Training Configuration

| Setting | Value |
|---------|-------|
| Epochs | 5 |
| Batch size | 4 |
| Learning rate | 2e-5 |
| Warmup steps | 100 |
| Early stopping patience | 3 |
| Max sequence length | 1024 |

**Loss Masking**: The data collator masks loss on prompt tokens (sets labels to -100), so the model only trains on generating the target condition.

### 5.3 Fine-tuning Results

```
[Fill in your actual results]
Final training loss:     [X.XX]
Final validation loss:   [X.XX]
Best validation loss:    [X.XX] (epoch [Y])
Training time:           [X] minutes/hours
```

---

## 6. Evaluation

### 6.1 Methodology

**Metric**: Exact match accuracy after normalization
- Collapse whitespace
- Remove surrounding parentheses if balanced
- Case-sensitive comparison

**Confidence Score**: 
- Computed from token-level log probabilities
- Formula: `score = exp(mean_log_prob) × 100`
- Range: 0-100

### 6.2 Results

**Test Set Performance**:
```
[Fill in your actual results]
Total test examples:           [X]
Correct predictions:           [Y]
Accuracy:                      [Z]%
Average confidence score:      [W]
Avg score (correct):           [A]
Avg score (incorrect):         [B]
```

### 6.3 Error Analysis

**Common Error Types**:
1. **[Error type 1]**: [Description and examples]
2. **[Error type 2]**: [Description and examples]
3. **[Error type 3]**: [Description and examples]

**Example Predictions**:

✓ **Correct Example**:
- Expected: `x > 0`
- Predicted: `x > 0`
- Score: [XX]

✗ **Incorrect Example**:
- Expected: `len(items) > 0`
- Predicted: `items`
- Score: [YY]
- Analysis: [Why did it fail?]

---

## 7. Discussion

### 7.1 Strengths

1. **From-scratch training**: No dependency on pre-trained models ensures the system learns from our specific corpus
2. **AST-based masking**: Precise extraction of if-conditions using Python's AST
3. **Reproducibility**: Fixed random seeds and configurable parameters
4. **Scalability**: Pipeline supports streaming and sampling for large datasets

### 7.2 Limitations

1. **Dataset size**: [Discuss if your dataset was smaller than ideal and impact]
2. **Simple conditions**: Model may struggle with complex multi-line or nested conditions
3. **Context window**: 512/1024 token limit may truncate very large functions
4. **Exact match**: Evaluation doesn't credit semantically equivalent conditions (e.g., `x > 0` vs `0 < x`)
5. **Generalization**: Limited to Python; would need retraining for other languages

### 7.3 Future Work

1. **Semantic evaluation**: Use AST equivalence checking instead of string matching
2. **Larger datasets**: Mine more repositories to reach 150k+ pre-training examples
3. **Architectural improvements**: Try larger models or encoder-decoder architectures
4. **Multi-language support**: Extend to Java, JavaScript, etc.
5. **Context enhancement**: Include documentation, variable types, or cross-file context

---

## 8. Conclusion

This project successfully implements a complete pipeline for training a Transformer model from scratch to predict masked if-conditions in Python code. The system achieves [X]% accuracy on the test set, demonstrating that decoder-only models can learn useful patterns for code completion tasks. The fully reproducible pipeline provides a foundation for future research in neural code understanding and generation.

**Key Takeaways**:
- Custom tokenization is essential for code understanding
- Pre-training on code before fine-tuning improves task performance
- AST-based data extraction ensures high-quality training examples
- Exact match evaluation is strict but ensures precision

---

## 9. References

1. Vaswani et al. (2017). "Attention is All You Need." NeurIPS.
2. Radford et al. (2019). "Language Models are Unsupervised Multitask Learners." OpenAI.
3. Chen et al. (2021). "Evaluating Large Language Models Trained on Code." arXiv:2107.03374.
4. Feng et al. (2020). "CodeBERT: A Pre-Trained Model for Programming and Natural Languages." EMNLP.
5. [Add any other relevant references]

---

## 10. Appendix

### A. Repository Structure

```
[Include your actual directory tree]
```

### B. Hyperparameter Sensitivity

[Optional: If you tried different hyperparameters, report their effects]

### C. Sample Outputs

[Include additional prediction examples, particularly interesting cases]

---

**Code Repository**: [Your GitHub URL if applicable]  
**Artifact Sizes**: Tokenizer: [X] MB, Pre-trained: [Y] GB, Fine-tuned: [Z] GB
