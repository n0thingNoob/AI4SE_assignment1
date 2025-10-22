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

def encode_if_example(prefix_ctx: str, target_if: str, tokenizer) -> Dict[str, List[int]]:
    inp = prefix_ctx.rstrip() + " <ANS> "
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
    
    # Validate labels
    sample_batch = [dataset[0], dataset[1]]
    valid_labels = sum(1 for item in sample_batch for label in item["labels"] if label != -100)
    print(f"Valid label tokens in sample: {valid_labels}")
    assert valid_labels > 0, "Labels are all -100; check encode_if_example logic"
    
    return dataset


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'\s+', ' ', text.strip())

def ensure_ans(prompt: str) -> str:
    p = prompt.rstrip()
    return p if p.endswith(" <ANS>") else (p + " <ANS> ")

@torch.no_grad()
def generate_prediction(model, tokenizer, input_text: str, max_new_tokens: int = 64) -> str:
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
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    input_length = inputs["input_ids"].shape[1]
    gen_tokens = generated[0][input_length:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    
    # Handle empty or invalid generation
    if not text or not text.strip():
        return "True"  # Fallback prediction
    
    lines = text.splitlines()
    if not lines:
        return "True"  # Fallback prediction
    
    line = lines[0].strip()
    if not line:
        return "True"  # Fallback prediction
        
    return line.rstrip(':')

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for evaluation using AST comparison."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    # Calculate accuracy on non-ignored tokens (fallback)
    mask = labels != -100
    if mask.sum() > 0:
        token_accuracy = (predictions[mask] == labels[mask]).mean()
    else:
        token_accuracy = 0.0
    
    # For now, return token accuracy as the main metric
    # TODO: Implement proper AST-based evaluation in a custom trainer
    return {
        "accuracy": float(token_accuracy),
        "token_accuracy": float(token_accuracy)
    }

class ASTEvaluatorTrainer(Trainer):
    """Custom trainer with AST-based evaluation."""
    
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.tokenizer = tokenizer
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to use AST-based comparison."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Run evaluation
        output = self.evaluation_loop(eval_dataloader, description="Evaluation", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        # Copy output.metrics to a dict
        metrics = dict(output.metrics) if getattr(output, "metrics", None) else {}
        
        # Add AST-based evaluation only if fields exist
        from collections.abc import Mapping
        ok = len(eval_dataset) > 0 and isinstance(eval_dataset[0], Mapping) and \
             {'original_input','original_target'} <= set(eval_dataset[0].keys())
        if ok:
            ast_accuracy = self._compute_ast_accuracy(eval_dataset)
            metrics[f"{metric_key_prefix}_ast_accuracy"] = ast_accuracy
        
        self.log(metrics)
        
        return metrics
    
    def _compute_ast_accuracy(self, eval_dataset):
        """Compute AST-based accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        print(f"\n=== Computing AST Accuracy on {len(eval_dataset)} samples ===")
        
        with torch.no_grad():
            for i in range(len(eval_dataset)):
                # Get original text
                original_input = eval_dataset[i]['original_input']
                original_target = eval_dataset[i]['original_target']
                
                # Generate prediction with consistent prompt format
                prompt = ensure_ans(original_input)
                prediction = generate_prediction(self.model, self.tokenizer, prompt)
                
                # Extract condition expressions
                expected_condition = original_target.split('\n')[0].strip().rstrip(':')
                predicted_condition = prediction.split('\n')[0].strip().rstrip(':')
                
                # Check correctness using AST comparison
                if is_correct_ast(expected_condition, predicted_condition):
                    correct += 1
                total += 1
                
                # Print progress every 100 samples
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1}/{len(eval_dataset)} samples, current accuracy: {correct}/{total} = {correct/total*100:.2f}%")
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"AST Accuracy: {correct}/{total} = {accuracy*100:.2f}%")
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
    )
    
    # Create trainer
    # Enable gradient checkpointing to reduce activation memory
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    trainer = ASTEvaluatorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # Add tokenizer for AST evaluation
    )
    
    # Train
    print("\n=== Starting Training ===")
    trainer.train()
    
    # Save model
    print(f"\n=== Saving Model to {output_dir} ===")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("=== Fine-tuning Complete ===")
    
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
    )

if __name__ == "__main__":
    main()