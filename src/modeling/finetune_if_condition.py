#!/usr/bin/env python3
"""
Fine-tune pre-trained model for if-condition prediction.

Takes the pre-trained CLM model and fine-tunes it to predict
masked if-conditions given the function context.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.modeling.data_collators import DataCollatorForIfConditionPrediction


PROMPT_TEMPLATE = "Task: Predict the masked if condition.\n\nFunction:\n{function}\n\n<answer>\n"


def load_jsonl_dataset(jsonl_path):
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_example(example):
    """
    Format a fine-tuning example into prompt + target format.
    
    Args:
        example: Dictionary with 'input' (masked function) and 'target' (condition)
        
    Returns:
        Dictionary with 'prompt', 'target', and 'full_text'
    """
    prompt = PROMPT_TEMPLATE.format(function=example['input'])
    target = example['target']
    full_text = prompt + target
    
    return {
        'prompt': prompt,
        'target': target,
        'full_text': full_text,
        'meta': example.get('meta', {}),
    }


def prepare_dataset(jsonl_path, tokenizer, max_length=1024):
    """
    Load and tokenize fine-tuning dataset.
    
    Args:
        jsonl_path: Path to JSONL file
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        HuggingFace Dataset
    """
    print(f"Loading dataset from: {jsonl_path}")
    
    # Load raw data
    raw_data = load_jsonl_dataset(jsonl_path)
    print(f"Loaded {len(raw_data)} examples.")
    
    # Format examples
    formatted_data = [format_example(ex) for ex in raw_data]
    
    # Tokenize
    def tokenize_function(examples):
        # Tokenize full text (prompt + target)
        tokenized = tokenizer(
            examples['full_text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return tokenized
    
    # Create HF dataset
    dataset = Dataset.from_list(formatted_data)
    
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['prompt', 'target', 'full_text', 'meta'],
        desc="Tokenizing",
    )
    
    print(f"Dataset ready: {len(tokenized)} examples")
    
    return tokenized


def finetune_model(
    tokenizer_path,
    pretrained_model_path,
    train_jsonl,
    val_jsonl,
    output_dir,
    max_length=1024,
    epochs=5,
    batch_size=4,
    learning_rate=2e-5,
    warmup_steps=100,
    save_steps=500,
    eval_steps=500,
    early_stopping_patience=3,
):
    """
    Fine-tune model for if-condition prediction.
    
    Args:
        tokenizer_path: Path to custom tokenizer
        pretrained_model_path: Path to pre-trained model
        train_jsonl: Path to training JSONL
        val_jsonl: Path to validation JSONL
        output_dir: Directory to save fine-tuned model
        max_length: Maximum sequence length
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        early_stopping_patience: Early stopping patience
    """
    # Load tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure special tokens are set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # Check for special tokens
    answer_token_id = tokenizer.convert_tokens_to_ids('<answer>')
    ifmask_token_id = tokenizer.convert_tokens_to_ids('<IFMASK>')
    
    print(f"Special token IDs:")
    print(f"  <answer>: {answer_token_id}")
    print(f"  <IFMASK>: {ifmask_token_id}")
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model from: {pretrained_model_path}")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,}")
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_jsonl, tokenizer, max_length)
    val_dataset = prepare_dataset(val_jsonl, tokenizer, max_length)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForIfConditionPrediction(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy='steps',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to=['tensorboard'],
        seed=42,
    )
    
    # Create trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting fine-tuning...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*60}")
    print("Fine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune model for if-condition prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python finetune_if_condition.py \\
        --tokenizer artifacts/tokenizer \\
        --pretrained artifacts/pretrain_gpt2 \\
        --train data/finetune_train.jsonl \\
        --val data/finetune_val.jsonl \\
        --out-dir artifacts/ifrec_finetuned \\
        --epochs 5 --batch-size 4
        """
    )
    
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer')
    parser.add_argument('--pretrained', required=True, help='Path to pre-trained model')
    parser.add_argument('--train', required=True, help='Path to training JSONL')
    parser.add_argument('--val', required=True, help='Path to validation JSONL')
    parser.add_argument('--out-dir', default='artifacts/ifrec_finetuned', help='Output directory')
    parser.add_argument('--max-length', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--save-steps', type=int, default=500, help='Save every N steps')
    parser.add_argument('--eval-steps', type=int, default=500, help='Eval every N steps')
    parser.add_argument('--early-stopping', type=int, default=3, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)
    
    if not os.path.exists(args.pretrained):
        print(f"Error: Pre-trained model not found: {args.pretrained}")
        sys.exit(1)
    
    if not os.path.exists(args.train):
        print(f"Error: Training data not found: {args.train}")
        sys.exit(1)
    
    if not os.path.exists(args.val):
        print(f"Error: Validation data not found: {args.val}")
        sys.exit(1)
    
    # Fine-tune
    finetune_model(
        args.tokenizer,
        args.pretrained,
        args.train,
        args.val,
        args.out_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping,
    )


if __name__ == '__main__':
    main()
