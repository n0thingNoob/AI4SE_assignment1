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
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.modeling.utils import normalize_condition, is_correct, is_correct_ast


# Global max length - configurable via CLI
MAX_LEN = 512


def ensure_ans(s: str) -> str:
    """Ensure prompt ends with ' <ANS> ' like in training."""
    s = s.rstrip()
    return s if s.endswith(" <ANS>") else (s + " <ANS> ")


def build_prompt(example: dict) -> str:
    """Build prompt using same format as training."""
    # Prefer pre-windowed input from preprocessing
    src = example.get("input_prepped") or example["input"]
    return ensure_ans(src)


def load_jsonl_dataset(jsonl_path):
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data




def compute_score_from_logprobs(log_probs):
    """
    Compute confidence score from log probabilities.
    
    Maps average log probability to 0-100 scale.
    
    Args:
        log_probs: List of log probabilities for generated tokens
        
    Returns:
        Score in range 0-100
    """
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


def _best_parseable_prefix(text: str) -> str:
    """Find the best parseable prefix of the generated text."""
    from src.modeling.utils import is_correct_ast
    t = text.strip().rstrip(':')
    if not t:
        return "True"
    candidates = [t]
    for sp in [" and ", " or ", ";", ","]:
        if sp in t:
            candidates.append(t.split(sp)[0].strip())
    for cand in candidates:
        try:
            if is_correct_ast(cand, cand):
                return cand
        except Exception:
            pass
    return t or "True"




def generate_prediction_with_strategies(model, tokenizer, prompt):
    """
    Generate prediction using multiple strategies and return the best one.
    """
    strategies = [
        # Pure greedy strategy - deterministic, matches training behavior
        {"max_new_tokens": 32, "temperature": None, "do_sample": False, "top_p": None, "top_k": None, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3},
    ]
    
    best_prediction = None
    best_score = -1
    
    for strategy in strategies:
        try:
            prediction, score = generate_single_prediction(model, tokenizer, prompt, **strategy)
            if score > best_score:
                best_score = score
                best_prediction = prediction
        except Exception as e:
            continue
    
    return best_prediction or "True", best_score

def generate_single_prediction(model, tokenizer, prompt, max_new_tokens=32, temperature=None, do_sample=False, 
                             top_p=None, top_k=None, repetition_penalty=1.2, no_repeat_ngram_size=3):
    """
    Generate prediction for a single example.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (None for greedy)
        
    Returns:
        Tuple of (predicted_text, score)
    """
    # Tokenize with left truncation to keep tail (where <IFMASK> and <ANS> live)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_LEN)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    # Safety check: ensure input length + max_new_tokens doesn't exceed model's limit
    limit = getattr(model.config, "max_position_embeddings", 1024) or 1024
    if input_ids.shape[1] + max_new_tokens > limit:
        max_new_tokens = max(8, limit - input_ids.shape[1])
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            early_stopping=True,  # Stop at EOS
        )
    
    # Decode generated tokens (only the new tokens, not the prompt)
    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    predicted_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Cut at first colon or newline, then trim whitespace
    if ':' in predicted_text:
        predicted_text = predicted_text.split(':')[0].strip()
    elif '\n' in predicted_text:
        predicted_text = predicted_text.split('\n')[0].strip()
    else:
        predicted_text = predicted_text.strip()
    
    # Find best parseable prefix
    predicted_text = _best_parseable_prefix(predicted_text)
    
    # Compute score from token scores
    log_probs = []
    for i, score_tensor in enumerate(outputs.scores):
        lp = torch.log_softmax(score_tensor[0], dim=-1)
        log_probs.append(lp[generated_ids[i]].item())
    
    score = compute_score_from_logprobs(log_probs)
    
    return predicted_text, score


def predict_batch(model, tokenizer, test_data, batch_size=8, max_new_tokens=32, temperature=0.0, do_sample=False):
    """
    Generate predictions for a batch of examples.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_data: List of test examples
        batch_size: Batch size (currently processes one at a time for scoring)
        max_new_tokens: Max tokens to generate
        
    Returns:
        List of prediction dictionaries
    """
    model.eval()
    predictions = []
    
    for idx, example in enumerate(tqdm(test_data, desc="Generating predictions")):
        # Build prompt using same format as training
        prompt = build_prompt(example)
        
        # Prefer cleaned expected_condition, fallback to target
        expected = example.get('expected_condition') or example['target']
        
        # Generate prediction using multiple strategies
        predicted, score = generate_prediction_with_strategies(model, tokenizer, prompt)
        
        # Extract condition expressions for comparison (ignore code block format differences)
        # Normalize both sides to first line, no trailing colon
        expected_condition = expected.split('\n')[0].strip().rstrip(':')
        predicted_condition = (predicted.split('\n')[0].strip().rstrip(':')) or "True"
        
        # Debug output for first few examples
        if idx <= 3:
            prompt_tail = prompt[-160:].replace('\n', '\\n')
            print(f"[DEBUG] prompt_tail: {prompt_tail}")
            print(f"[DEBUG] expected: {expected_condition}")
            print(f"[DEBUG] predicted: {predicted_condition}")
            print()
        
        # Check correctness using AST comparison (more robust than string comparison)
        correct = is_correct_ast(expected_condition, predicted_condition)
        
        # Store result
        predictions.append({
            'prompt': prompt,
            'expected': expected,
            'predicted': predicted,
            'correct': correct,
            'score': score,
            'meta': example.get('meta', {}),
        })
    
    return predictions


def save_predictions_csv(predictions, output_csv):
    """
    Save predictions to CSV with the exact required columns.
    
    Columns:
    1. input - the exact input given to the model
    2. correct - whether the prediction is correct (true/false)
    3. expected - ground truth if condition
    4. predicted - model output if condition
    5. score - confidence score (0-100)
    """
    output_dir = os.path.dirname(output_csv)
    if output_dir:  # Only create directory if path has a directory component
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
        --tokenizer artifacts/tokenizer \\
        --model artifacts/ifrec_finetuned \\
        --test data/finetune_test.jsonl \\
        --out predictions.csv
        """
    )
    
    parser.add_argument('--tokenizer', required=True, help='Path to tokenizer')
    parser.add_argument('--model', required=True, help='Path to fine-tuned model')
    parser.add_argument('--test', required=True, help='Path to test JSONL')
    parser.add_argument('--out', required=True, help='Output CSV path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (currently unused)')
    parser.add_argument('--max-new-tokens', type=int, default=32, help='Max tokens to generate')
    parser.add_argument('--max-length', type=int, default=512, help='Max input length before generation')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--do-sample', action='store_true', help='Use sampling instead of greedy decoding')
    
    args = parser.parse_args()
    
    # Set global max length
    global MAX_LEN
    MAX_LEN = args.max_length
    
    # Validate inputs
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.test):
        print(f"Error: Test data not found: {args.test}")
        sys.exit(1)
    
    # Load our custom tokenizer and model
    print(f"Loading custom tokenizer from: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Set truncation and padding to left to keep tail (where <IFMASK> and <ANS> live)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Warn if special tokens are not in vocab
    for tok in ["<IFMASK>", "<ANS>"]:
        if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id:
            print(f"WARNING: special token {tok} not in vocab!")
    
    print(f"Loading model from: {args.model}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
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
        temperature=args.temperature,
        do_sample=args.do_sample,
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
