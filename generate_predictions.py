#!/usr/bin/env python3
"""
Fixed prediction script for if-condition task.
Generates predictions and outputs CSV in required format.
"""

import argparse
import csv
import json
import os
import sys
import torch
import numpy as np
import ast
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
MAX_LEN = 512
SPECIAL_TOKENS = ["<IFMASK>", "<ANS>"]

def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def setup_model_and_tokenizer(model_dir):
    """Setup model and tokenizer with proper special tokens."""
    print("Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Ensure special tokens are available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens if not already present
    existing_tokens = tokenizer.get_vocab()
    missing_tokens = [token for token in SPECIAL_TOKENS if token not in existing_tokens]
    if missing_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
        print(f"Added missing special tokens: {missing_tokens}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer, device

def normalize_text(text):
    """Normalize text for comparison."""
    return re.sub(r'\s+', ' ', text.strip())

def extract_if_conditions_from_code(code):
    """Extract if conditions from function code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    
    if_conditions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            try:
                if hasattr(ast, 'get_source_segment'):
                    condition_source = ast.get_source_segment(code, node.test)
                else:
                    condition_source = ast.unparse(node.test)
                
                if condition_source:
                    if_conditions.append(condition_source.strip())
            except:
                continue
    
    return if_conditions

def generate_prediction(model, tokenizer, input_text, max_new_tokens=50, temperature=0.7, do_sample=True):
    """Generate if-condition prediction."""
    device = next(model.parameters()).device
    
    # Encode input with proper settings
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=MAX_LEN,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=0.9 if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # Decode only new tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs.sequences[0][input_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up prediction
    prediction = prediction.strip()
    
    # Stop at common delimiters
    stop_phrases = ['\n', '\n\n', '<TASK=', '<CODE>', '<COND>', '<ANS>']
    for phrase in stop_phrases:
        if phrase in prediction:
            prediction = prediction.split(phrase)[0].strip()
    
    # Calculate score from log probabilities
    log_probs = []
    for i, score_tensor in enumerate(outputs.scores):
        log_prob = torch.log_softmax(score_tensor[0], dim=-1)
        generated_token_id = generated_tokens[i]
        token_log_prob = log_prob[generated_token_id].item()
        log_probs.append(token_log_prob)
    
    if log_probs:
        avg_log_prob = np.mean(log_probs)
        score = min(100, max(0, int(100 * np.exp(avg_log_prob))))
    else:
        score = 0
    
    return prediction, score

def process_provided_testset(provided_csv, model, tokenizer, output_csv):
    """Process provided testset and generate predictions."""
    print("Processing provided testset...")
    
    predictions = []
    
    with open(provided_csv, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 100 == 0:
                print(f"  Processed {i} examples...")
            
            # Check if this is the expected format
            if 'Input provided to the model' in row:
                # This is already a prediction file, skip processing
                print("  Provided CSV appears to be a prediction file, skipping...")
                break
            
            # Try to find code in different possible column names
            code = None
            for col in ['code', 'Code', 'CODE', 'function', 'Function']:
                if col in row and row[col]:
                    code = row[col]
                    break
            
            if not code:
                print(f"  No code found in row {i}, skipping...")
                continue
            
            # Extract if conditions from the code
            if_conditions = extract_if_conditions_from_code(code)
            
            if not if_conditions:
                # No if conditions found, skip
                continue
            
            # Use the first if condition
            condition = if_conditions[0]
            
            # Create masked code
            masked_code = code.replace(f'if {condition}:', 'if <IFMASK>:')
            if masked_code == code:  # Try without space
                masked_code = code.replace(f'if{condition}:', 'if <IFMASK>:')
            
            # Create input template
            input_text = f"""<TASK=IF_COND>
<CODE>
{masked_code}
<COND>
"""
            
            # Generate prediction
            predicted_condition, score = generate_prediction(
                model, tokenizer, input_text,
                max_new_tokens=50, temperature=0.7, do_sample=True
            )
            
            # Check if correct
            is_correct = normalize_text(predicted_condition) == normalize_text(condition)
            
            predictions.append({
                'Input provided to the model': input_text,
                'Whether the prediction is correct (true/false)': 'true' if is_correct else 'false',
                'Expected if condition': condition,
                'Predicted if condition': predicted_condition,
                'Prediction score (0-100)': score
            })
    
    # Save predictions
    print(f"Saving predictions to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'Input provided to the model',
            'Whether the prediction is correct (true/false)',
            'Expected if condition',
            'Predicted if condition',
            'Prediction score (0-100)'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    
    # Calculate accuracy
    correct = sum(1 for p in predictions if p['Whether the prediction is correct (true/false)'] == 'true')
    accuracy = correct / len(predictions) * 100 if predictions else 0
    
    print(f"\nResults:")
    print(f"  Total examples: {len(predictions)}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return predictions

def process_generated_testset(test_file, model, tokenizer, output_csv):
    """Process our generated testset and generate predictions."""
    print("Processing generated testset...")
    
    test_data = load_jsonl(test_file)
    predictions = []
    
    for i, item in enumerate(test_data):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(test_data)} examples...")
        
        input_text = item['input']
        expected_condition = item['target']
        
        # Generate prediction
        predicted_condition, score = generate_prediction(
            model, tokenizer, input_text,
            max_new_tokens=50, temperature=0.7, do_sample=True
        )
        
        # Check if correct
        is_correct = normalize_text(predicted_condition) == normalize_text(expected_condition)
        
        predictions.append({
            'Input provided to the model': input_text,
            'Whether the prediction is correct (true/false)': 'true' if is_correct else 'false',
            'Expected if condition': expected_condition,
            'Predicted if condition': predicted_condition,
            'Prediction score (0-100)': score
        })
    
    # Save predictions
    print(f"Saving predictions to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'Input provided to the model',
            'Whether the prediction is correct (true/false)',
            'Expected if condition',
            'Predicted if condition',
            'Prediction score (0-100)'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    
    # Calculate accuracy
    correct = sum(1 for p in predictions if p['Whether the prediction is correct (true/false)'] == 'true')
    accuracy = correct / len(predictions) * 100 if predictions else 0
    
    print(f"\nResults:")
    print(f"  Total examples: {len(predictions)}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Generate predictions for if-condition task")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model")
    parser.add_argument("--provided_csv", default="provided-testset.csv", help="Provided testset CSV")
    parser.add_argument("--generated_test", default="data/finetune_test.jsonl", help="Generated testset JSONL")
    parser.add_argument("--output_provided", default="provided-testset.csv", help="Output CSV for provided testset")
    parser.add_argument("--output_generated", default="generated-testset.csv", help="Output CSV for generated testset")
    
    args = parser.parse_args()
    
    # Setup model and tokenizer
    model, tokenizer, device = setup_model_and_tokenizer(args.model_dir)
    
    # Process provided testset
    if os.path.exists(args.provided_csv):
        process_provided_testset(args.provided_csv, model, tokenizer, args.output_provided)
    else:
        print(f"Provided testset not found: {args.provided_csv}")
    
    # Process generated testset
    if os.path.exists(args.generated_test):
        process_generated_testset(args.generated_test, model, tokenizer, args.output_generated)
    else:
        print(f"Generated testset not found: {args.generated_test}")

if __name__ == "__main__":
    main()