#!/usr/bin/env python3
"""
Fixed fine-tuning script for if-condition prediction.
"""

import json
import os
import sys
import torch
import re
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Disable HF tokenizers parallelism to avoid fork warnings/deadlocks with multi-worker dataloader
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Reduce HF logging verbosity and suppress tokenization pad warnings
from transformers.utils import logging as hf_logging
# Show progress bars and warnings/info again
hf_logging.set_verbosity_info()                     
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()
import warnings
warnings.filterwarnings("ignore", module="transformers.tokenization_utils_base")

# Mitigate CUDA memory fragmentation and allow expandable segments
# Note: some torch versions only accept a single boolean arg here
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Prefer TF32 on Ampere+ to speed and reduce precision stalls
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EvalPrediction, DefaultDataCollator
)
from datasets import Dataset
import numpy as np
import ast
from src.modeling.utils import is_correct_ast

# Constants
MAX_LEN = 384
SPECIAL_TOKENS = ["<IFMASK>", "<ANS>", "<CODE>", "<TASK=IF_COND>"]

# Put this near the top (module scope)
_FIRST_CM_CALLED = False

# ==== AST eval controls (env overridable) ====
AST_EVAL_RATIO = float(os.getenv("AST_EVAL_RATIO", "0.1"))
AST_EVAL_MIN   = int(os.getenv("AST_EVAL_MIN",   "100"))
AST_EVAL_MAX   = int(os.getenv("AST_EVAL_MAX",   "500"))
AST_EVAL_EVERY = int(os.getenv("AST_EVAL_EVERY", "1"))
AST_EVAL_MAX_NEW = int(os.getenv("AST_EVAL_MAX_NEW", "64"))

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def setup_tokenizer_and_model(tokenizer_path: str, pretrained_path: str):
    """Setup tokenizer and model with proper special tokens."""
    print("=== Setting up Tokenizer and Model ===")
    
    # Load tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Keep tail where <IFMASK> and <ANS> live
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    print(f"✅ Tokenizer truncation_side: {tokenizer.truncation_side}, padding_side: {tokenizer.padding_side}")
    
    # Fix PAD token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Warning: No PAD token found, using EOS token as PAD")
    
    # Check if special tokens are already in tokenizer (from our improved training)
    existing_special_tokens = tokenizer.additional_special_tokens
    missing_tokens = [token for token in SPECIAL_TOKENS if token not in existing_special_tokens]
    
    if missing_tokens:
        print(f"Adding missing special tokens: {missing_tokens}")
        tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
    else:
        print(f"✅ All special tokens already present: {SPECIAL_TOKENS}")
    
    # Load model
    print(f"Loading pre-trained model from: {pretrained_path}")
    if pretrained_path == "gpt2":
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_path)
    
    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Set pad token in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Check mask token consistency
    mask_token_id = tokenizer.convert_tokens_to_ids("<IFMASK>")
    if mask_token_id == tokenizer.unk_token_id:
        print("WARNING: <IFMASK> token not found in tokenizer vocabulary!")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Special tokens: {tokenizer.additional_special_tokens}")
    else:
        print(f"✅ <IFMASK> token ID: {mask_token_id}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    
    return tokenizer, model

def ensure_ans(prompt: str) -> str:
    """确保 prompt 末尾恰好有一个 <ANS> 标记，避免重复。"""
    p = prompt.rstrip()

    if p.endswith("<ANS>"):
        return p + " "
    return p + " <ANS> "

def encode_if_example(prefix_ctx: str, target_if: str, tokenizer) -> Dict[str, List[int]]:
    # 保证只有一个 <ANS>，若预处理已拼接，这里不会重复
    inp = ensure_ans(prefix_ctx)
    tgt = target_if + tokenizer.eos_token

    inp_ids = tokenizer(inp, add_special_tokens=False).input_ids
    tgt_ids = tokenizer(tgt, add_special_tokens=False).input_ids

    # If target alone exceeds MAX_LEN, keep last MAX_LEN tokens of target
    if len(tgt_ids) >= MAX_LEN:
        input_ids = tgt_ids[-MAX_LEN:]
        attention_mask = [1] * len(input_ids)
        labels = input_ids[:]  # supervise all target tokens
    else:
        keep_inp = MAX_LEN - len(tgt_ids)          # trim prompt only
        inp_ids = inp_ids[-keep_inp:] if keep_inp > 0 else []
        input_ids = inp_ids + tgt_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(inp_ids) + tgt_ids

    pad = MAX_LEN - len(input_ids)
    if pad > 0:
        input_ids += [tokenizer.pad_token_id] * pad
        attention_mask += [0] * pad
        labels += [-100] * pad

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def create_finetune_dataset(jsonl_path: str, tokenizer) -> Dataset:
    """Create fine-tuning dataset with proper encoding."""
    print(f"Loading dataset from: {jsonl_path}")
    
    # Load raw data
    raw_data = load_jsonl(jsonl_path)
    print(f"Loaded {len(raw_data)} examples")
    
    # Process examples
    processed_data = []
    for example in raw_data:
        # Extract context and target - use preprocessed data if available
        input_text = example.get('input_prepped', example['input'])
        target = example.get('expected_condition', example['target'])
        
        
        # Encode example
        encoded = encode_if_example(input_text, target, tokenizer)
        
        # Add original text for evaluation
        encoded['original_input'] = input_text
        encoded['original_target'] = target
        
        processed_data.append(encoded)
    
    # Create dataset
    dataset = Dataset.from_list(processed_data)
    print(f"Dataset ready: {len(dataset)} examples")
    
    # Validate labels（稳健性：避免样本过少时断言失败）
    n_check = min(2, len(dataset))
    sample_batch = [dataset[i] for i in range(n_check)]
    valid_labels = sum(1 for item in sample_batch for label in item["labels"] if label != -100)
    print(f"Valid label tokens in sample: {valid_labels}")
    assert valid_labels > 0, "Labels are all -100; check encode_if_example logic"
    
    return dataset


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'\s+', ' ', text.strip())

def _sanitize_condition_line(s: str) -> str:
    """
    清洗生成的条件行，去除换行、冒号、行内注释、尾随分隔符，并简单配平右括号。
    这样可以降低 AST 误判。
    """
    import re
    # 只取第一行/到冒号为止
    line = re.split(r'[\r\n:]', s, maxsplit=1)[0]
    # 去掉行内注释
    line = line.split('#', 1)[0]
    # 去掉空白
    line = line.strip()
    # 去掉尾随分隔符（逗号、分号）
    while line and line[-1] in ',;':
        line = line[:-1].strip()
    # 简单右括号配平（常见漏 1 个）
    if line.count('(') > line.count(')'):
        line += ')' * (line.count('(') - line.count(')'))
    return line

@torch.no_grad()
def generate_prediction(model, tokenizer, input_text: str, max_new_tokens: int = 64) -> str:
    # 确保 prompt 末尾恰好一个 <ANS>
    input_text = ensure_ans(input_text)
    inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,              # was True; single sample doesn't need padding
        return_tensors="pt"
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=min(max_new_tokens, 32),
        do_sample=False,
        no_repeat_ngram_size=4,      # avoid "A and A and A..."
        repetition_penalty=1.1,      # light penalty
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs["input_ids"].shape[1]
    gen_tokens = generated[0][input_length:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    
    if not text or not text.strip():
        return "True"  # 合理兜底
    return _sanitize_condition_line(text) or "True"

def compute_metrics(eval_pred):
    """
    Proper token-level accuracy for causal LM:
    compare argmax(logits[..., :-1, :]) with labels[..., 1:],
    and ignore positions where labels == -100.
    """
    import numpy as np
    global _FIRST_CM_CALLED

    logits = eval_pred.predictions
    # Some HF versions return (logits, ...) tuple
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    labels = eval_pred.label_ids

    # Shift for next-token prediction
    pred_ids = np.argmax(logits, axis=-1)[..., :-1]   # [B, T-1]
    labels   = labels[..., 1:]                        # [B, T-1]
    mask = (labels != -100)

    denom = mask.sum()
    token_acc = float(((pred_ids == labels) & mask).sum() / denom) if denom > 0 else 0.0

    if not _FIRST_CM_CALLED:
        print(f"[compute_metrics] supervised tokens: {int(denom)} "
              f"(batch={labels.shape[0]}, seq={labels.shape[1]})")
        _FIRST_CM_CALLED = True

    # Only one clean metric name
    return {"token_accuracy": token_acc}

class ASTEvaluatorTrainer(Trainer):
    """Custom trainer with AST-based evaluation."""
    
    def __init__(self, *args, tokenizer=None, enable_ast_eval=True, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.enable_ast_eval = enable_ast_eval
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to use AST-based comparison."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Run evaluation
        output = self.evaluation_loop(eval_dataloader, description="Evaluation", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        # Copy output.metrics to a dict
        metrics = dict(output.metrics) if getattr(output, "metrics", None) else {}
        
        if not self.enable_ast_eval:
            # Skip AST accuracy during training
            self.log(metrics)
            return metrics
        
        # Add AST-based evaluation only if fields exist
        from collections.abc import Mapping
        ok = len(eval_dataset) > 0 and isinstance(eval_dataset[0], Mapping) and \
             {'original_input','original_target'} <= set(eval_dataset[0].keys())
        
        # Run AST eval only every AST_EVAL_EVERY-th evaluation
        step_blocks = max(1, getattr(self.args, "eval_steps", 1))
        do_ast = ((self.state.global_step // step_blocks) % AST_EVAL_EVERY == 0)
        
        if ok and do_ast:
            print(f"\n=== Running AST Evaluation (step {self.state.global_step}) ===")
            ast_accuracy = self._compute_ast_accuracy(eval_dataset)
            metrics[f"{metric_key_prefix}_ast_accuracy"] = ast_accuracy
            print(f"=== AST Evaluation Complete: {ast_accuracy:.3%} ===\n")
            self.log({"ast_eval_samples": self._last_ast_n})
        
        self.log(metrics)
        
        return metrics
    
    def _compute_ast_accuracy(self, eval_dataset):
        """Compute AST-based accuracy."""
        import numpy as np
        
        # sample size: use eval dataset size directly, bounded
        n = max(AST_EVAL_MIN, min(AST_EVAL_MAX, len(eval_dataset)))

        rng = np.random.RandomState(0)
        idxs = rng.choice(len(eval_dataset), size=n, replace=False)

        correct = 0
        with torch.inference_mode():
            for i in idxs:
                original_input  = eval_dataset[i]['original_input']
                original_target = eval_dataset[i]['original_target']
                prompt = ensure_ans(original_input)
                pred = generate_prediction(self.model, self.tokenizer, prompt, max_new_tokens=AST_EVAL_MAX_NEW)
                exp = original_target.split('\n')[0].strip().rstrip(':')
                got = pred.split('\n')[0].strip().rstrip(':')
                if is_correct_ast(exp, got):
                    correct += 1

        self._last_ast_n = int(n)  # for logging
        accuracy = correct / float(n)
        print(f"AST Accuracy: {correct}/{n} = {accuracy*100:.2f}%")
        return accuracy

def finetune_model(
    tokenizer_path: str,
    pretrained_path: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    max_length: int = 512,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    early_stopping: int = 3,
    disable_ast_eval: bool = True,
    eval_subset_ratio: float = 0.05,
    eval_subset_cap: int = 1000,
    eval_token_acc: bool = False,
    ast_eval_after: int = 0,
):
    """Fine-tune model for if-condition prediction."""
    global MAX_LEN
    MAX_LEN = max_length
    print("=== Fine-tuning for If-Condition Prediction ===")
    
    # Setup tokenizer and model
    tokenizer, model = setup_tokenizer_and_model(tokenizer_path, pretrained_path)
    
    # Create datasets
    print("\n=== Creating Datasets ===")
    train_dataset = create_finetune_dataset(train_file, tokenizer)
    val_dataset = create_finetune_dataset(val_file, tokenizer)
    
    # Keep a full copy for post-train probe, then create the eval subset
    val_full = val_dataset
    if eval_subset_ratio < 1.0 or (eval_subset_cap and eval_subset_cap > 0):
        import random
        random.seed(42)
        n = int(len(val_dataset) * eval_subset_ratio)
        if eval_subset_cap:
            n = min(n, eval_subset_cap)
        n = max(1, n)  # ensure at least 1
        idx = random.sample(range(len(val_dataset)), n)
        val_dataset = val_dataset.select(idx)
    print(f"Eval subset: {len(val_dataset)} / {len(val_full)}")
    
    # Create data collator
    data_collator = DefaultDataCollator()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,   # was 2
        per_device_eval_batch_size=batch_size * 2,    # 2x batch size for faster eval
        gradient_accumulation_steps=8,  # Reduce accumulation for more frequent updates
        eval_accumulation_steps=4,  # Reduce accumulation steps
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        disable_tqdm=False,            
        logging_steps=50,              
        logging_first_step=True,       
        eval_strategy="steps",
        eval_steps=eval_steps,               
        save_strategy="steps",         
        save_steps=save_steps,
        report_to="none",              
        dataloader_pin_memory=False,   
        save_total_limit=3,  # Keep more checkpoints
        load_best_model_at_end=True,  # Load best model
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,  # Important: don't drop our custom fields
        dataloader_num_workers=0,  # Use single-worker dataloader to avoid hanging
        max_grad_norm=1.0,  # Gradient clipping
        warmup_ratio=0.1,  # 10% warmup
        prediction_loss_only=not eval_token_acc,  # key: no logits unless we need token acc
    )
    
    # Create trainer
    # Enable gradient checkpointing to reduce activation memory
    try:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    except Exception:
        pass

    # Choose trainer class
    use_ast_trainer = (not disable_ast_eval)
    if use_ast_trainer:
        print("WARNING: AST eval during training is ENABLED (slow/heavy).")
    else:
        print("AST eval during training is DISABLED (default). Using standard Trainer.")

    # compute_metrics only if we need token accuracy
    cm = compute_metrics if eval_token_acc else None

    if use_ast_trainer:
        trainer = ASTEvaluatorTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=cm,
            tokenizer=tokenizer,
            enable_ast_eval=True,
        )
    else:
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=cm,
            tokenizer=tokenizer,
        )
    
    # Train
    print("\n=== Starting Training ===")
    trainer.train()

    # --- Sanity check: one-batch token acc ---
    try:
        eval_loader = trainer.get_eval_dataloader()
        batch = next(iter(eval_loader))
        model.eval()
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device)
            )
        logits = out.logits.detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()

        pred = logits.argmax(-1)[:, :-1]
        lab  = labels[:, 1:]
        mask = (lab != -100)
        denom = mask.sum()
        one_batch_acc = float(((pred == lab) & mask).sum() / denom) if denom > 0 else 0.0
        print(f"[sanity] one-batch token_acc = {one_batch_acc:.4f} "
              f"(supervised tokens={int(denom)})")
    except Exception as e:
        print(f"[sanity] failed: {e}")
    # --- end sanity check ---
    
    # Save model
    print(f"\n=== Saving Model to {output_dir} ===")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("=== Fine-tuning Complete ===")
    
    # Post-train AST probe on a small subset if requested
    if ast_eval_after and ast_eval_after > 0:
        import random
        random.seed(42)
        k = min(ast_eval_after, len(val_full))
        probe_idx = random.sample(range(len(val_full)), k)
        correct = 0
        for i, j in enumerate(probe_idx, 1):
            original_input = val_full[j]['original_input']
            original_target = val_full[j]['original_target']
            prompt = ensure_ans(original_input)
            pred = generate_prediction(model, tokenizer, prompt)
            exp = original_target.split('\n')[0].strip().rstrip(':')
            got = pred.split('\n')[0].strip().rstrip(':')
            if is_correct_ast(exp, got):
                correct += 1
            if i % 50 == 0:
                print(f"[AST probe] {i}/{k}: running acc={correct/i:.3f}")
        final_acc = correct / k if k > 0 else 0.0
        print(f"\n=== Post-train AST probe on {k} samples: acc={final_acc:.3%} ===")
    
    # Test prediction
    print("\n=== Testing Prediction ===")
    model.eval()
    test_example = train_dataset[0]
    test_input = "def test(x):\n    if <IFMASK>:\n        return x"
    test_input = ensure_ans(test_input)
    prediction = generate_prediction(model, tokenizer, test_input)
    print(f"Test input: {test_input}")
    print(f"Generated prediction: {prediction}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune model for if-condition prediction")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer")
    parser.add_argument("--pretrained", required=True, help="Path to pre-trained model")
    parser.add_argument("--train", required=True, help="Path to training JSONL")
    parser.add_argument("--val", required=True, help="Path to validation JSONL")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Eval steps")
    parser.add_argument("--early-stopping", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--disable-ast-eval", action="store_true", default=True,
                        help="If set (default), skip AST-based evaluation during training/eval.")
    parser.add_argument("--eval-subset-ratio", type=float, default=0.05,
                        help="Fraction of val set used each eval step (default 0.05).")
    parser.add_argument("--eval-subset-cap", type=int, default=1000,
                        help="Cap for number of val examples per eval (default 1000).")
    parser.add_argument("--eval-token-acc", action="store_true", default=False,
                        help="If set, compute token-level accuracy during eval (uses logits; more RAM).")
    parser.add_argument("--ast-eval-after", type=int, default=0,
                        help="If >0, run AST evaluation on a random subset of this size after training.")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)
    
    if args.pretrained != "gpt2" and not os.path.exists(args.pretrained):
        print(f"Error: Pre-trained model not found: {args.pretrained}")
        sys.exit(1)
    
    if not os.path.exists(args.train):
        print(f"Error: Training data not found: {args.train}")
        sys.exit(1)
    
    if not os.path.exists(args.val):
        print(f"Error: Validation data not found: {args.val}")
        sys.exit(1)
    
    # Run fine-tuning
    finetune_model(
        tokenizer_path=args.tokenizer,
        pretrained_path=args.pretrained,
        train_file=args.train,
        val_file=args.val,
        output_dir=args.out_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        early_stopping=args.early_stopping,
        disable_ast_eval=args.disable_ast_eval,
        eval_subset_ratio=args.eval_subset_ratio,
        eval_subset_cap=args.eval_subset_cap,
        eval_token_acc=args.eval_token_acc,
        ast_eval_after=args.ast_eval_after,
    )

if __name__ == "__main__":
    main()