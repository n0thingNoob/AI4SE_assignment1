#!/usr/bin/env python3
"""
Pre-train a GPT-2 style decoder-only Transformer from scratch using CLM.

Trains a causal language model from randomly initialized weights
(no pre-trained model used) using our custom tokenizer.
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


def load_tokenizer(tokenizer_path):
    """Load custom tokenizer."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer


def create_model_from_scratch(vocab_size, config_kwargs=None):
    """
    Create GPT-2 model with random initialization.
    
    Args:
        vocab_size: Vocabulary size for embeddings
        config_kwargs: Optional dictionary of config overrides
        
    Returns:
        GPT2LMHeadModel with random weights
    """
    print("\nCreating GPT-2 model from scratch (random initialization)...")
    
    # Default config for small model
    default_config = {
        'vocab_size': vocab_size,
        'n_positions': 1024,       # Maximum sequence length
        'n_embd': 768,             # Embedding dimension
        'n_layer': 12,             # Number of transformer layers
        'n_head': 12,              # Number of attention heads
        'n_inner': 3072,           # FFN inner dimension
        'activation_function': 'gelu_new',
        'resid_pdrop': 0.1,
        'embd_pdrop': 0.1,
        'attn_pdrop': 0.1,
        'layer_norm_epsilon': 1e-5,
        'initializer_range': 0.02,
        'bos_token_id': 1,
        'eos_token_id': 2,
    }
    
    # Override with user config
    if config_kwargs:
        default_config.update(config_kwargs)
    
    config = GPT2Config(**default_config)
    
    print(f"Model config:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Attention heads: {config.n_head}")
    print(f"  FFN dim: {config.n_inner}")
    print(f"  Max sequence length: {config.n_positions}")
    
    # Create model with random weights
    model = GPT2LMHeadModel(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    
    return model


def prepare_dataset(corpus_path, tokenizer, block_size=512, max_samples=None):
    """
    Load and tokenize corpus for CLM training.
    
    Args:
        corpus_path: Path to text corpus
        tokenizer: Tokenizer instance
        block_size: Maximum sequence length
        max_samples: Maximum number of samples (for smoke tests)
        
    Returns:
        Tokenized dataset
    """
    print(f"\nLoading corpus from: {corpus_path}")
    
    # Load as text dataset
    dataset = load_dataset('text', data_files=corpus_path, split='train')
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} examples.")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=block_size,
            padding=False,
        )
    
    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    # Group into blocks
    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        
        # Drop remainder
        total_length = (total_length // block_size) * block_size
        
        # Split into chunks
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        
        return result
    
    print(f"Grouping texts into blocks of {block_size} tokens...")
    grouped = tokenized.map(
        group_texts,
        batched=True,
        desc="Grouping",
    )
    
    print(f"Final dataset size: {len(grouped)} blocks")
    
    return grouped


def pretrain_model(
    tokenizer_path,
    corpus_path,
    output_dir,
    block_size=512,
    epochs=3,
    batch_size=8,
    learning_rate=5e-4,
    warmup_steps=500,
    save_steps=1000,
    eval_steps=500,
    max_samples=None,
    model_config=None,
):
    """
    Pre-train GPT-2 model from scratch.
    
    Args:
        tokenizer_path: Path to custom tokenizer
        corpus_path: Path to pre-training corpus
        output_dir: Directory to save model checkpoints
        block_size: Maximum sequence length
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Warmup steps for learning rate scheduler
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        max_samples: Maximum samples for smoke tests
        model_config: Optional model config overrides
    """
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Create model from scratch
    model = create_model_from_scratch(len(tokenizer), model_config)
    
    # Prepare dataset
    train_dataset = prepare_dataset(corpus_path, tokenizer, block_size, max_samples)
    
    # Split for validation (use 5% for validation)
    split = train_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples:  {len(eval_dataset)}")
    
    # Data collator for CLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # CLM, not MLM
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
        eval_strategy='steps',  # Changed from evaluation_strategy
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to=['tensorboard'],
        seed=42,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting pre-training...")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*60}")
    print("Pre-training complete!")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train GPT-2 from scratch using CLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python pretrain_clm.py \\
        --tokenizer artifacts/tokenizer \\
        --corpus data/pretrain_corpus.txt \\
        --out-dir artifacts/pretrain_gpt2 \\
        --epochs 3 --batch-size 8
        """
    )
    
    parser.add_argument('--tokenizer', required=True, help='Path to custom tokenizer')
    parser.add_argument('--corpus', required=True, help='Path to pre-training corpus')
    parser.add_argument('--out-dir', default='artifacts/pretrain_gpt2', help='Output directory')
    parser.add_argument('--block-size', type=int, default=512, help='Max sequence length')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--save-steps', type=int, default=1000, help='Save every N steps')
    parser.add_argument('--eval-steps', type=int, default=500, help='Eval every N steps')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples (smoke test)')
    parser.add_argument('--n-layer', type=int, default=12, help='Number of layers')
    parser.add_argument('--n-embd', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--n-head', type=int, default=12, help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)
    
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus not found: {args.corpus}")
        sys.exit(1)
    
    # Model config overrides
    model_config = {
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'n_inner': args.n_embd * 4,
    }
    
    # Pre-train
    pretrain_model(
        args.tokenizer,
        args.corpus,
        args.out_dir,
        block_size=args.block_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        max_samples=args.max_samples,
        model_config=model_config,
    )


if __name__ == '__main__':
    main()
