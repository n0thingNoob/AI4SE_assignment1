#!/usr/bin/env python3
"""
Train a GPT-2–aligned byte-level BPE tokenizer on our own corpus.

Key points:
- Byte-level BPE with add_prefix_space=True (like GPT-2).
- No UNK token (byte-level covers all bytes). We'll still set eos and later use eos as pad in the model.
- Include task tokens (<IFMASK>, <ANS>, <CODE>, </CODE>, <TASK=IF_COND>) at TRAIN TIME.
- Save as HuggingFace PreTrainedTokenizerFast.
"""

import argparse
import os
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

def train_tokenizer(corpus_path, output_dir, vocab_size=50257, min_frequency=2):
    # GPT-2 aligns at byte-level with prefix space
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True, lowercase=False)

    # Special tokens:
    # GPT-2 has only an EOS token (</s> is not used); we use the literal  as eos for compatibility.
    # We also include our task tokens so they are single ids from day 1.
    special_tokens = [
        "",                # EOS (GPT-2 style)
        "<IFMASK>",
        "<ANS>",
        "<CODE>",
        "</CODE>",
        "<TASK=IF_COND>",
    ]

    print(f"Training tokenizer on: {corpus_path}")
    print(f"Vocab size: {vocab_size}; min_frequency: {min_frequency}")
    print(f"Special tokens: {special_tokens}")

    tokenizer.train(
        files=[corpus_path],
        vocab_size=vocab_size,               # GPT-2 default is 50257; feel free to use 52k~50k too
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    tokenizer.save_model(output_dir)  # writes vocab.json + merges.txt

    # Convert to HF fast tokenizer (GPT-2–like settings)
    fast = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(output_dir, "tokenizer.json"),
        eos_token="",         # GPT-2 eos
        bos_token=None,       # GPT-2 has no BOS
        unk_token=None,       # byte-level => no UNK
        pad_token=None,       # we will set pad==eos at MODEL time
        additional_special_tokens=[
            "<IFMASK>", "<ANS>", "<CODE>", "</CODE>", "<TASK=IF_COND>"
        ],
        model_max_length=1024  # adjust if you train with larger context
    )
    fast.save_pretrained(output_dir)

    print("\nTokenizer saved. Sanity check:")
    print(f"  vocab size: {len(fast)}")
    print(f"  eos id: {fast.eos_token_id}")
    print(f"  <IFMASK> id: {fast.convert_tokens_to_ids('<IFMASK>')}")
    print(f"  <ANS> id: {fast.convert_tokens_to_ids('<ANS>')}")
    print(f"  <CODE> id: {fast.convert_tokens_to_ids('<CODE>')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    if not os.path.exists(args.corpus):
        raise FileNotFoundError(args.corpus)

    train_tokenizer(
        corpus_path=args.corpus,
        output_dir=args.out_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

if __name__ == "__main__":
    main()