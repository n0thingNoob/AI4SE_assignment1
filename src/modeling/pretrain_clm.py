#!/usr/bin/env python3
"""
Fixed pre-training script for GPT-2 CLM.
Fixes: proper tokenizer setup, data collator, and training configuration.
"""

import json
import os
import sys
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    default_data_collator
)
from datasets import Dataset
import numpy as np

# Constants
MAX_LEN = 512
SPECIAL_TOKENS = ["<IFMASK>", "<ANS>", "<CODE>", "<TASK=IF_COND>"]

def load_corpus(corpus_path: str) -> list:
    """Load pre-training corpus."""
    print(f"Loading corpus from: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    print(f"Loaded {len(lines)} lines from corpus")
    return lines

def setup_tokenizer_and_model(tokenizer_path: str):
    """Setup tokenizer and model for pre-training."""
    print("=== Setting up Tokenizer and Model for Pre-training ===")
    
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
    
    # Load pre-trained GPT-2 model
    print("Loading pre-trained GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    
    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Set pad token in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.config.vocab_size}")
    
    return tokenizer, model

def create_pretrain_dataset(corpus: list, tokenizer, max_samples: int = None, block_size: int = 512) -> Dataset:
    """Create pre-training dataset with proper block grouping."""
    print("Creating pre-training dataset...")
    
    # Limit samples if specified
    if max_samples and len(corpus) > max_samples:
        corpus = corpus[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    raw = Dataset.from_dict({"text": corpus})

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=False, padding=False)

    tokenized = raw.map(tok_fn, batched=True, remove_columns=["text"], desc="Tokenizing")

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
                  for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    grouped = tokenized.map(group_texts, batched=True, desc="Grouping into blocks")
    print(f"Pre-training dataset ready: {len(grouped)} examples")
    return grouped

def pretrain_model(
    tokenizer_path: str,
    corpus_path: str,
    output_dir: str,
    block_size: int = 512,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 1000,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    max_samples: int = None,
):
    """Pre-train GPT-2 model on code corpus."""
    print("=== Pre-training GPT-2 on Code Corpus ===")
    
    # Setup tokenizer and model
    tokenizer, model = setup_tokenizer_and_model(tokenizer_path)
    
    # Load corpus
    corpus = load_corpus(corpus_path)
    
    # Create dataset
    print("\n=== Creating Pre-training Dataset ===")
    train_dataset = create_pretrain_dataset(corpus, tokenizer, max_samples)
    
    # Use default data collator for grouped blocks
    data_collator = default_data_collator
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="no",  # Disable evaluation during pre-training
        save_steps=save_steps,
        save_total_limit=3,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # Disable evaluation during pre-training
        data_collator=data_collator,
    )
    
    # Train
    print("\n=== Starting Pre-training ===")
    trainer.train()
    
    # Save model
    print(f"\n=== Saving Pre-trained Model to {output_dir} ===")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("=== Pre-training Complete ===")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-train GPT-2 on code corpus")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer")
    parser.add_argument("--corpus", required=True, help="Path to corpus file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--block-size", type=int, default=512, help="Block size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--eval-steps", type=int, default=1000, help="Eval steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)
    
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file not found: {args.corpus}")
        sys.exit(1)
    
    # Run pre-training
    pretrain_model(
        tokenizer_path=args.tokenizer,
        corpus_path=args.corpus,
        output_dir=args.out_dir,
        block_size=args.block_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_samples=args.max_samples,
    )

if __name__ == "__main__":
    main()