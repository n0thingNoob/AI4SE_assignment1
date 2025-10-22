#!/usr/bin/env python3
"""
Build fine-tuning dataset for if-condition prediction.

Reads functions.jsonl, selects functions with if statements,
masks exactly one if-condition per function, and creates train/val/test splits.

Output format (JSONL):
{
  "input": "function code with one if replaced by <IFMASK>",
  "target": "the original condition expression",
  "meta": {"file": "...", "function": "...", "original": "..."}
}
"""

import argparse
import json
import os
import sys
import random
import ast
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.modeling.utils import find_if_conditions_in_func, mask_one_if_condition


def create_masked_example(func_data, max_per_file=None):
    """
    Create masked if-condition examples from a function.
    
    Args:
        func_data: Function metadata dictionary
        max_per_file: Not used here (applied at file level)
        
    Returns:
        List of example dictionaries (one per if-condition in the function)
    """
    code = func_data['code']
    
    # Find all if conditions
    conditions = find_if_conditions_in_func(code)
    
    if not conditions:
        return []
    
    examples = []
    
    # Create one example for each if-condition
    for idx in range(len(conditions)):
        masked_code, condition_text = mask_one_if_condition(code, idx)
        
        if masked_code and condition_text:
            example = {
                'input': masked_code,
                'target': condition_text,
                'meta': {
                    'file': func_data['file'],
                    'function': func_data['function_name'],
                    'lineno': func_data['lineno'],
                    'if_index': idx,
                    'total_ifs': len(conditions),
                    'original': code,
                }
            }
            examples.append(example)
    
    return examples


def build_finetune_dataset(
    functions_jsonl,
    output_prefix,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    max_per_file=None,
    max_examples=None
):
    """
    Build fine-tuning dataset with train/val/test splits.
    
    Args:
        functions_jsonl: Path to functions.jsonl
        output_prefix: Prefix for output files (e.g., 'data/finetune')
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
        max_per_file: Maximum examples per source file (for balancing)
        max_examples: Maximum total examples to generate (for smoke tests)
        
    Returns:
        Dictionary with counts for train/val/test
    """
    random.seed(seed)
    
    print(f"Reading functions from: {functions_jsonl}")
    
    # Load functions
    functions = []
    with open(functions_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                func = json.loads(line)
                functions.append(func)
    
    print(f"Loaded {len(functions)} functions.")
    
    # Filter to only functions with if statements
    functions_with_if = [f for f in functions if f['has_if'] and f['num_ifs'] > 0]
    print(f"Functions with if statements: {len(functions_with_if)}")
    
    if not functions_with_if:
        print("Error: No functions with if statements found!")
        return None
    
    # Group by file for balanced sampling
    by_file = {}
    for func in functions_with_if:
        file_path = func['file']
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(func)
    
    print(f"Functions span {len(by_file)} files.")
    
    # Apply max_per_file limit
    if max_per_file:
        print(f"Applying max {max_per_file} functions per file...")
        for file_path in by_file:
            if len(by_file[file_path]) > max_per_file:
                by_file[file_path] = random.sample(by_file[file_path], max_per_file)
        
        # Flatten back
        functions_with_if = []
        for funcs in by_file.values():
            functions_with_if.extend(funcs)
        
        print(f"After per-file limit: {len(functions_with_if)} functions.")
    
    # Generate masked examples
    all_examples = []
    
    for func in tqdm(functions_with_if, desc="Creating masked examples"):
        examples = create_masked_example(func)
        all_examples.extend(examples)
        
        # Early exit for smoke tests
        if max_examples and len(all_examples) >= max_examples:
            break
    
    if max_examples:
        all_examples = all_examples[:max_examples]
    
    print(f"Generated {len(all_examples)} masked examples.")
    
    if not all_examples:
        print("Error: No examples generated!")
        return None
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Split into train/val/test
    n_total = len(all_examples)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    test_examples = all_examples[:n_test]
    val_examples = all_examples[n_test:n_test + n_val]
    train_examples = all_examples[n_test + n_val:]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_examples)}")
    print(f"  Val:   {len(val_examples)}")
    print(f"  Test:  {len(test_examples)}")
    
    # Save splits
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    def save_split(examples, suffix):
        path = f"{output_prefix}_{suffix}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        print(f"Saved {len(examples)} examples to: {path}")
    
    save_split(train_examples, 'train')
    save_split(val_examples, 'val')
    save_split(test_examples, 'test')
    
    return {
        'train': len(train_examples),
        'val': len(val_examples),
        'test': len(test_examples),
        'total': n_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build fine-tuning dataset for if-condition prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python build_finetune_dataset.py \\
        --functions data/functions.jsonl \\
        --out-prefix data/finetune \\
        --val 0.1 --test 0.1 --seed 42
        """
    )
    
    parser.add_argument(
        '--functions',
        type=str,
        required=True,
        help='Path to functions.jsonl file'
    )
    
    parser.add_argument(
        '--out-prefix',
        type=str,
        default='data/finetune',
        help='Output prefix for train/val/test files (default: data/finetune)'
    )
    
    parser.add_argument(
        '--val',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--max-per-file',
        type=int,
        default=None,
        help='Maximum examples per source file (for balancing, optional)'
    )
    
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum total examples to generate (for smoke tests, optional)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.functions):
        print(f"Error: Functions file not found: {args.functions}")
        sys.exit(1)
    
    if args.val < 0 or args.val > 0.5:
        print("Error: Validation ratio must be between 0 and 0.5")
        sys.exit(1)
    
    if args.test < 0 or args.test > 0.5:
        print("Error: Test ratio must be between 0 and 0.5")
        sys.exit(1)
    
    if args.val + args.test >= 1.0:
        print("Error: Val + test ratios must be < 1.0")
        sys.exit(1)
    
    # Build dataset
    result = build_finetune_dataset(
        args.functions,
        args.out_prefix,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        max_per_file=args.max_per_file,
        max_examples=args.max_examples
    )
    
    if not result:
        print("\nError: Failed to build dataset.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Fine-tuning dataset ready!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
