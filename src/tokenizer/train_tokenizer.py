#!/usr/bin/env python3
"""
Train a Byte-level BPE tokenizer from scratch on Python code corpus.

This script trains a tokenizer from scratch (no pre-trained tokenizer used)
with special tokens required for our task: <pad>, <s>, </s>, <unk>, <mask>, <IFMASK>, <answer>.
"""

import argparse
import os
import sys
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def train_tokenizer(
    corpus_path,
    output_dir,
    vocab_size=52000,
    min_frequency=2,
):
    """
    Train a Byte-level BPE tokenizer from scratch.
    
    Args:
        corpus_path: Path to text corpus file
        output_dir: Directory to save tokenizer
        vocab_size: Vocabulary size
        min_frequency: Minimum frequency for tokens
        
    Returns:
        Path to saved tokenizer
    """
    print(f"Training tokenizer from scratch on: {corpus_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Output directory: {output_dir}")
    
    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Define special tokens
    special_tokens = [
        "<pad>",      # Padding token
        "<s>",        # Beginning of sequence
        "</s>",       # End of sequence
        "<unk>",      # Unknown token
        "<mask>",     # General mask token
        "<IFMASK>",   # Masked if-condition placeholder
        "<answer>",   # Answer delimiter in prompts
    ]
    
    print(f"\nSpecial tokens: {special_tokens}")
    
    # Train tokenizer
    print("\nTraining tokenizer (this may take a while)...")
    tokenizer.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    
    print(f"\nTokenizer saved to: {output_dir}")
    
    # Convert to HuggingFace format
    print("\nConverting to HuggingFace PreTrainedTokenizerFast...")
    
    # Load as PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(output_dir, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<IFMASK>", "<answer>"],
    )
    
    # Save in HuggingFace format
    fast_tokenizer.save_pretrained(output_dir)
    
    print(f"HuggingFace tokenizer saved to: {output_dir}")
    
    # Test tokenizer
    print("\n" + "="*60)
    print("Testing tokenizer:")
    print("="*60)
    
    test_code = "def foo(x):\n    if x > 0:\n        return True"
    encoded = fast_tokenizer.encode(test_code)
    decoded = fast_tokenizer.decode(encoded)
    
    print(f"Original: {test_code}")
    print(f"Encoded:  {encoded[:20]}... ({len(encoded)} tokens)")
    print(f"Decoded:  {decoded}")
    
    # Test special tokens
    test_masked = "def foo(x):\n    if <IFMASK>:\n        return True"
    encoded_masked = fast_tokenizer.encode(test_masked)
    print(f"\nWith <IFMASK>: {test_masked}")
    print(f"Encoded: {encoded_masked[:20]}... ({len(encoded_masked)} tokens)")
    
    # Check special token IDs
    print(f"\nSpecial token IDs:")
    print(f"  <pad>:    {fast_tokenizer.pad_token_id}")
    print(f"  <s>:      {fast_tokenizer.bos_token_id}")
    print(f"  </s>:     {fast_tokenizer.eos_token_id}")
    print(f"  <unk>:    {fast_tokenizer.unk_token_id}")
    print(f"  <mask>:   {fast_tokenizer.mask_token_id}")
    print(f"  <IFMASK>: {fast_tokenizer.convert_tokens_to_ids('<IFMASK>')}")
    print(f"  <answer>: {fast_tokenizer.convert_tokens_to_ids('<answer>')}")
    
    print(f"\nVocabulary size: {len(fast_tokenizer)}")
    print("="*60)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train a Byte-level BPE tokenizer from scratch on Python code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python train_tokenizer.py \\
        --corpus data/pretrain_corpus.txt \\
        --out-dir artifacts/tokenizer \\
        --vocab-size 52000
        """
    )
    
    parser.add_argument(
        '--corpus',
        type=str,
        required=True,
        help='Path to pre-training corpus text file'
    )
    
    parser.add_argument(
        '--out-dir',
        type=str,
        default='artifacts/tokenizer',
        help='Output directory for tokenizer (default: artifacts/tokenizer)'
    )
    
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=52000,
        help='Vocabulary size (default: 52000)'
    )
    
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='Minimum frequency for tokens (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.corpus):
        print(f"Error: Corpus file not found: {args.corpus}")
        sys.exit(1)
    
    # Train tokenizer
    output_dir = train_tokenizer(
        args.corpus,
        args.out_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    
    print(f"\n{'='*60}")
    print("Tokenizer training complete!")
    print(f"Load with: AutoTokenizer.from_pretrained('{output_dir}')")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
