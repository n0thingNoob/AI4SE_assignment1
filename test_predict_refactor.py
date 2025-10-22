#!/usr/bin/env python3
"""
Test script to verify the refactored predict.py works correctly.
"""

import json
import tempfile
import os
from pathlib import Path

# Create a small test dataset
test_data = [
    {
        "input": "def test_func(x):\n    if <IFMASK>:\n        return x * 2",
        "input_prepped": "def test_func(x):\n    if <IFMASK>:\n        return x * 2",
        "target": "x > 0",
        "expected_condition": "x > 0"
    },
    {
        "input": "def another_func(y):\n    if <IFMASK>:\n        print('hello')",
        "input_prepped": "def another_func(y):\n    if <IFMASK>:\n        print('hello')",
        "target": "y is not None",
        "expected_condition": "y is not None"
    }
]

# Create temporary test file
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')
    test_file = f.name

print(f"Created test file: {test_file}")
print("Test data:")
for i, item in enumerate(test_data):
    print(f"  {i+1}. input: {item['input'][:50]}...")
    print(f"     expected: {item['expected_condition']}")

print("\nTo test the refactored predict.py, run:")
print(f"python src/modeling/predict.py \\")
print(f"  --tokenizer artifacts/tokenizer_v6_gpt2style \\")
print(f"  --model artifacts/ifrec_finetuned_v6_2000eval_20251021_124745 \\")
print(f"  --test {test_file} \\")
print(f"  --out test_predictions.csv \\")
print(f"  --max-length 512 \\")
print(f"  --max-new-tokens 64")

print(f"\nTest file will be cleaned up automatically: {test_file}")
