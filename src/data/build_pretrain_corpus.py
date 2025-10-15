#!/usr/bin/env python3
"""
Build pre-training corpus from extracted functions.

Reads functions.jsonl and outputs a plain text file with one function per line,
suitable for training a tokenizer and pre-training a language model.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm


def build_pretrain_corpus(functions_jsonl, output_txt, min_lines=5):
    """
    Convert functions JSONL to plain text corpus.
    
    Args:
        functions_jsonl: Path to functions.jsonl file
        output_txt: Path to output text file
        min_lines: Minimum lines to include (additional filter)
        
    Returns:
        Number of functions written
    """
    print(f"Reading functions from: {functions_jsonl}")
    
    functions = []
    with open(functions_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                func = json.loads(line)
                functions.append(func)
    
    print(f"Loaded {len(functions)} functions.")
    
    # Filter by minimum lines
    functions = [f for f in functions if f['n_lines'] >= min_lines]
    print(f"After min-lines filter ({min_lines}): {len(functions)} functions.")
    
    # Write to output file
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        for func in tqdm(functions, desc="Writing corpus"):
            # Normalize newlines and write one function per line
            code = func['code'].replace('\r\n', '\n').replace('\r', '\n')
            # Replace actual newlines with a special marker to keep one function per line
            # Actually, let's just write the code as-is with actual newlines
            # The tokenizer training can handle multi-line text
            f.write(code)
            f.write('\n\n')  # Double newline to separate functions
    
    print(f"Wrote {len(functions)} functions to: {output_txt}")
    return len(functions)


def main():
    parser = argparse.ArgumentParser(
        description="Build pre-training corpus from extracted functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python build_pretrain_corpus.py --functions data/functions.jsonl --out data/pretrain_corpus.txt
        """
    )
    
    parser.add_argument(
        '--functions',
        type=str,
        required=True,
        help='Path to functions.jsonl file'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='data/pretrain_corpus.txt',
        help='Output text file path (default: data/pretrain_corpus.txt)'
    )
    
    parser.add_argument(
        '--min-lines',
        type=int,
        default=5,
        help='Minimum lines per function (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.functions):
        print(f"Error: Functions file not found: {args.functions}")
        sys.exit(1)
    
    # Build corpus
    count = build_pretrain_corpus(
        args.functions,
        args.out,
        min_lines=args.min_lines
    )
    
    if count == 0:
        print("\nError: No functions written to corpus.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Corpus ready for tokenizer training and pre-training!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
