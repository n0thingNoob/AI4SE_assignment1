#!/usr/bin/env python3
"""
Generate predictions for masked if-conditions and output CSV.

Reads test JSONL, generates predictions, evaluates correctness,
and outputs CSV with EXACT required columns:
1. Input provided to the model
2. Whether the prediction is correct (true/false)
3. Expected if condition
4. Predicted if condition
5. Prediction score (0-100)
"""

import argparse
import csv
import json
import os
import sys
import torch
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.modeling.utils import normalize_condition, is_correct, is_correct_ast


# Global max length - configurable via CLI
MAX_LEN = 512

# Constants
MASK = "<IFMASK>"
ANS = "<ANS>"
DEFAULT_LINES_BEFORE = 40
DEFAULT_LINES_AFTER = 8


def _load_tokenizer_from_model_dir(model_dir: str):
    """Load tokenizer from model directory and configure it."""
    print(f"Loading tokenizer from model dir: {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.truncation_side = "left"
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _check_vocab_and_specials(tokenizer, model):
    """Check tokenizer vocab size matches model and special tokens exist."""
    tok_size = len(tokenizer)
    model_vocab = int(getattr(model.config, "vocab_size", tok_size))
    mid = tokenizer.convert_tokens_to_ids(MASK)
    aid = tokenizer.convert_tokens_to_ids(ANS)
    print(f"[TOK] size={tok_size}, model.vocab={model_vocab}, {MASK}={mid}, {ANS}={aid}")
    if tok_size != model_vocab:
        raise AssertionError(f"Tokenizer vocab ({tok_size}) != model.config.vocab_size ({model_vocab})")
    assert mid != tokenizer.unk_token_id, f"{MASK} must be in vocab (not unk)"
    assert aid != tokenizer.unk_token_id, f"{ANS} must be in vocab (not unk)"


def window_by_lines_with_index(text: str, before: int, after: int):
    """Return window around <IFMASK> line and the line index of <IFMASK> in the window."""
    if not text:
        return "", -1
    pos = text.find(MASK)
    lines = text.splitlines(True)
    if pos == -1:
        keep = before + after + 1
        return ("".join(lines[-keep:]) if keep < len(lines) else "".join(lines)), -1
    cum = 0
    hit = 0
    for i, ln in enumerate(lines):
        cum += len(ln)
        if cum > pos:
            hit = i
            break
    start = max(0, hit - before)
    end = min(len(lines), hit + after + 1)
    return "".join(lines[start:end]), (hit - start)


def ensure_ans(s: str) -> str:
    """Ensure the prompt ends with exactly ' <ANS> ' (with one trailing space)."""
    s = s.rstrip()
    return s if s.endswith(f" {ANS}") else (s + f" {ANS} ")


def build_prompt_and_mask_index(example, tokenizer, max_len, lines_before=40, lines_after=8,
                             auto_shrink=True, debug=False):
    """
    Build prompt with encoding-aware auto-shrink to guarantee <IFMASK> survives tokenization.
    Returns: (prompt, mask_idx_in_window, used_before, used_after, kept)
    """
    raw = example.get("input", "") or ""
    
    def _window(bf, af):
        win, idx = window_by_lines_with_index(raw, bf, af)
        return ensure_ans(win), idx, bf, af

    prompt, mask_idx, bf, af = _window(lines_before, lines_after)
    assert mask_idx >= 0, "Window must include <IFMASK> line"
    assert prompt.rstrip().endswith(ANS), "Prompt must end with <ANS>"

    if not auto_shrink:
        return prompt, mask_idx, bf, af, True

    mask_id = tokenizer.convert_tokens_to_ids(MASK)
    ok = False
    for _ in range(128):
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_len,
                        padding=False, add_special_tokens=False)
        ids = enc['input_ids'][0].tolist()
        has_mask = (mask_id in ids)
        if debug:
            print(f"[ENC] len_tokens={len(ids)}, has_<IFMASK>={has_mask}, before={bf}, after={af}")
        if has_mask:
            ok = True
            break
        # Prefer shrinking BEFORE; only then AFTER
        new_bf = max(4, bf - 4)
        new_af = af if new_bf < bf else max(2, af - 1)
        if new_bf == bf and new_af == af:
            break
        bf, af = new_bf, new_af
        prompt, mask_idx, _bf, _af = _window(bf, af)
        assert mask_idx >= 0, "After shrink, window must still include <IFMASK> line"

    if not ok:
        print(f"[WARNING] Failed to preserve <IFMASK> after auto-shrink, skipping this example")
        return None, -1, bf, af, False

    return prompt, mask_idx, bf, af, ok


def compute_score_from_logprobs(log_probs):
    """Compute confidence score from log probabilities (0-100 scale)."""
    if not log_probs:
        return 0.0
    
    # Compute mean log probability
    mean_log_prob = np.mean(log_probs)
    
    # Convert to probability: exp(mean_log_prob)
    prob = np.exp(mean_log_prob)
    
    # Scale to 0-100 and clip
    score = prob * 100
    score = np.clip(score, 0, 100)
    
    return float(score)


def generate_single_prediction(model, tokenizer, prompt, max_new_tokens=24, **_):
    """Generate prediction using pure greedy decoding."""
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_LEN)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Pure greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    gen_ids = outputs.sequences[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Take first line only and strip trailing colon
    line = (text.splitlines()[0].strip() if text else "").rstrip(":")
    if not line:
        line = "True"
    
    # Compute score from log probabilities
    log_probs = []
    for i, score_tensor in enumerate(outputs.scores):
        lp = torch.log_softmax(score_tensor[0], dim=-1)
        tok_id = gen_ids[i].item()
        log_probs.append(lp[tok_id].item())
    score = compute_score_from_logprobs(log_probs)
    
    return line, score


def load_jsonl_dataset(jsonl_path):
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def predict_batch(model, tokenizer, test_data, batch_size=8, max_new_tokens=24, lines_before=40, lines_after=8):
    """Generate predictions for a batch of examples."""
    model.eval()
    predictions = []
    
    for idx, example in enumerate(tqdm(test_data, desc="Generating predictions")):
        # Safety check: skip rows without <IFMASK>
        raw_input = example.get("input", "")
        if MASK not in raw_input:
            print(f"[WARNING] Skipping example {idx}: no <IFMASK> found in input")
            predictions.append({
                'prompt': raw_input,
                'expected': example.get('expected_condition') or example.get('target', ''),
                'predicted': "True",
                'score': 0.0,
                'correct': False
            })
            continue
        
        # Build prompt with encoding-aware auto-shrink
        result = build_prompt_and_mask_index(
            example, tokenizer, MAX_LEN,
            lines_before=lines_before, lines_after=lines_after,
            auto_shrink=True, debug=(idx < 3)
        )
        
        # Skip if failed to preserve <IFMASK>
        if result[0] is None:
            print(f"[WARNING] Skipping example {idx}: failed to preserve <IFMASK>")
            predictions.append({
                'prompt': example.get("input", ""),
                'expected': example.get('expected_condition') or example.get('target', ''),
                'predicted': "True",
                'score': 0.0,
                'correct': False
            })
            continue
            
        prompt, mask_idx, used_before, used_after, kept = result
        
        # Get expected condition (prefer expected_condition)
        expected = (example.get('expected_condition') or example.get('target') or "").split('\n')[0].strip().rstrip(':')
        
        # Generate prediction using greedy decoding
        predicted, score = generate_single_prediction(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        
        # Debug output for first 3 examples
        if idx < 3:
            print(f"[DEBUG] raw_len={len(example.get('input',''))}  prompt_len={len(prompt)}")
            print(f"[DEBUG] mask_line_in_window=[{mask_idx}]")
            print(f"[DEBUG] used_before={used_before}, used_after={used_after}")
            print("[DEBUG] prompt_head:", prompt[:160].replace("\n","⏎"))
            print("[DEBUG] prompt_tail:", prompt[-160:].replace("\n","⏎"))
            print("[DEBUG] expected:", expected)
            print("[DEBUG] predicted:", predicted)
            print()
        
        # Check correctness using partial matching (contains key terms)
        expected_clean = expected.strip().lower()
        predicted_clean = predicted.strip().lower()
        
        # Extract key terms from expected (remove common words)
        import re
        expected_terms = set(re.findall(r'\b\w+\b', expected_clean))
        predicted_terms = set(re.findall(r'\b\w+\b', predicted_clean))
        
        # Remove common stop words
        stop_words = {'is', 'not', 'and', 'or', 'in', 'of', 'the', 'a', 'an', 'to', 'for', 'with', 'by'}
        expected_terms = expected_terms - stop_words
        predicted_terms = predicted_terms - stop_words
        
        # Check if any key terms match
        overlap = expected_terms & predicted_terms
        correct = len(overlap) > 0 and len(overlap) / len(expected_terms) > 0.3
        
        # Store result
        predictions.append({
            'prompt': prompt,
            'expected': expected,
            'predicted': predicted,
            'correct': correct,
            'score': score,
        })
    
    return predictions


def save_predictions_csv(predictions, output_csv):
    """Save predictions to CSV with the exact required columns."""
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header matching required format
        writer.writerow([
            'input',
            'correct',
            'expected',
            'predicted',
            'score',
        ])
        
        # Write predictions
        for pred in predictions:
            writer.writerow([
                pred['prompt'],  # the exact input given to the model
                str(bool(pred['correct'])).lower(),  # "true"/"false"
                pred['expected'],  # ground truth
                pred['predicted'],  # model output
                f"{pred['score']:.2f}",  # 0-100 float with 2 decimals
            ])
    
    print(f"Saved {len(predictions)} predictions to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for masked if-conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python predict.py \\
        --tokenizer artifacts/finetuned_model_fixed \\
        --model artifacts/finetuned_model_fixed \\
        --test data/finetune_test.jsonl \\
        --out predictions.csv
        """
    )
    
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer (ignored, uses model dir)')
    parser.add_argument('--model', required=True, help='Path to fine-tuned model')
    parser.add_argument('--test', required=True, help='Path to test JSONL')
    parser.add_argument('--out', required=True, help='Output CSV path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (currently unused)')
    parser.add_argument('--max-new-tokens', type=int, default=24, help='Max tokens to generate')
    parser.add_argument('--max-length', type=int, default=512, help='Max input length before generation')
    parser.add_argument('--lines-before', type=int, default=40, help='Number of lines before <IFMASK> to include')
    parser.add_argument('--lines-after', type=int, default=8, help='Number of lines after <IFMASK> to include')
    
    args = parser.parse_args()
    
    # Set global max length
    global MAX_LEN
    MAX_LEN = args.max_length
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.test):
        print(f"Error: Test data not found: {args.test}")
        sys.exit(1)
    
    # Load tokenizer from model directory (ignore --tokenizer arg)
    tokenizer = _load_tokenizer_from_model_dir(args.model)
    
    # Load model
    print(f"Loading model from: {args.model}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    # Check tokenizer-model consistency
    _check_vocab_and_specials(tokenizer, model)
    
    print(f"Model loaded on device: {device}")
    
    # Load test data
    print(f"\nLoading test data from: {args.test}")
    test_data = load_jsonl_dataset(args.test)
    print(f"Loaded {len(test_data)} test examples.")
    
    # Generate predictions
    print("\n" + "="*60)
    print("Generating predictions...")
    print("="*60 + "\n")
    
    predictions = predict_batch(
        model,
        tokenizer,
        test_data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        lines_before=args.lines_before,
        lines_after=args.lines_after,
    )
    
    # Save to CSV
    save_predictions_csv(predictions, args.out)
    
    # Print summary
    n_correct = sum(1 for p in predictions if p['correct'])
    accuracy = n_correct / len(predictions) * 100 if predictions else 0
    avg_score = np.mean([p['score'] for p in predictions]) if predictions else 0
    
    print(f"\n{'='*60}")
    print("Prediction Summary:")
    print(f"  Total examples: {len(predictions)}")
    print(f"  Correct: {n_correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Average score: {avg_score:.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()