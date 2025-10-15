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
from src.modeling.utils import normalize_condition, is_correct


PROMPT_TEMPLATE = "Task: Predict the masked if condition.\n\nFunction:\n{function}\n\n<answer>\n"


def load_jsonl_dataset(jsonl_path):
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_prompt(masked_function):
    """Format the prompt for prediction."""
    return PROMPT_TEMPLATE.format(function=masked_function)


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


def generate_prediction(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7):
    """
    Generate prediction for a single example.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Tuple of (predicted_text, score)
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # Decode generated tokens (only the new tokens, not the prompt)
    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    predicted_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Compute score from token scores
    # outputs.scores is a tuple of tensors, one per generated token
    log_probs = []
    for i, score_tensor in enumerate(outputs.scores):
        # score_tensor shape: (batch_size, vocab_size)
        # Get log probabilities
        log_probs_tensor = torch.log_softmax(score_tensor[0], dim=-1)
        # Get log prob of the selected token
        selected_token_id = generated_ids[i]
        log_prob = log_probs_tensor[selected_token_id].item()
        log_probs.append(log_prob)
    
    score = compute_score_from_logprobs(log_probs)
    
    # Clean up predicted text (remove leading/trailing whitespace)
    predicted_text = predicted_text.strip()
    
    # If prediction contains newlines, take only the first line
    if '\n' in predicted_text:
        predicted_text = predicted_text.split('\n')[0].strip()
    
    return predicted_text, score


def predict_batch(model, tokenizer, test_data, batch_size=8, max_new_tokens=128):
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
    
    for example in tqdm(test_data, desc="Generating predictions"):
        # Format prompt
        prompt = format_prompt(example['input'])
        expected = example['target']
        
        # Generate prediction
        predicted, score = generate_prediction(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens
        )
        
        # Check correctness
        correct = is_correct(expected, predicted)
        
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
    Save predictions to CSV with EXACT required columns.
    
    Columns:
    1. Input provided to the model
    2. Whether the prediction is correct (true/false)
    3. Expected if condition
    4. Predicted if condition
    5. Prediction score (0-100)
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header with EXACT column names
        writer.writerow([
            'Input provided to the model',
            'Whether the prediction is correct (true/false)',
            'Expected if condition',
            'Predicted if condition',
            'Prediction score (0-100)',
        ])
        
        # Write predictions
        for pred in predictions:
            writer.writerow([
                pred['prompt'],
                'true' if pred['correct'] else 'false',
                pred['expected'],
                pred['predicted'],
                f"{pred['score']:.2f}",
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
    parser.add_argument('--max-new-tokens', type=int, default=128, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    
    args = parser.parse_args()
    
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
    
    # Load model and tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
