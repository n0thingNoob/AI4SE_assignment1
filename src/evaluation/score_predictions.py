#!/usr/bin/env python3
"""
Score predictions from CSV file.

Reads the prediction CSV and computes accuracy and other metrics.
"""

import argparse
import csv
import sys
import numpy as np


def load_predictions_csv(csv_path):
    """
    Load predictions from CSV in benchmark format.
    
    Expected columns (benchmark_if_only.csv format):
    1. id - sequential ID
    2. code - expected if condition (ground truth)
    3. code_tokens - empty list
    4. docstring - predicted if condition
    5. docstring_tokens - empty list
    """
    predictions = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = row['code'].strip()
            predicted = row['docstring'].strip()
            
            # Normalize for comparison
            expected_norm = normalize_condition(expected)
            predicted_norm = normalize_condition(predicted)
            correct = (expected_norm == predicted_norm)
            
            predictions.append({
                'id': row['id'],
                'expected': expected,
                'predicted': predicted,
                'correct': correct,
                'score': 100.0 if correct else 0.0,  # Simple binary score
            })
    
    return predictions


def normalize_condition(text):
    """Normalize if condition for comparison."""
    # Remove extra whitespace and newlines
    normalized = ' '.join(text.split())
    return normalized.strip()


def compute_metrics(predictions):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Dictionary of metrics
    """
    n_total = len(predictions)
    n_correct = sum(1 for p in predictions if p['correct'])
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0
    
    scores = [p['score'] for p in predictions]
    avg_score = np.mean(scores) if scores else 0
    std_score = np.std(scores) if scores else 0
    
    # Compute metrics separately for correct and incorrect predictions
    correct_scores = [p['score'] for p in predictions if p['correct']]
    incorrect_scores = [p['score'] for p in predictions if not p['correct']]
    
    avg_score_correct = np.mean(correct_scores) if correct_scores else 0
    avg_score_incorrect = np.mean(incorrect_scores) if incorrect_scores else 0
    
    return {
        'total': n_total,
        'correct': n_correct,
        'incorrect': n_total - n_correct,
        'accuracy': accuracy,
        'avg_score': avg_score,
        'std_score': std_score,
        'avg_score_correct': avg_score_correct,
        'avg_score_incorrect': avg_score_incorrect,
    }


def print_metrics(metrics):
    """Print metrics in a formatted way."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Total examples:              {metrics['total']}")
    print(f"Correct predictions:         {metrics['correct']}")
    print(f"Incorrect predictions:       {metrics['incorrect']}")
    print(f"Accuracy:                    {metrics['accuracy']:.2f}%")
    print("-"*60)
    print(f"Average prediction score:    {metrics['avg_score']:.2f}")
    print(f"Std dev prediction score:    {metrics['std_score']:.2f}")
    print(f"Avg score (correct):         {metrics['avg_score_correct']:.2f}")
    print(f"Avg score (incorrect):       {metrics['avg_score_incorrect']:.2f}")
    print("="*60)


def print_sample_predictions(predictions, n_samples=5):
    """Print a few sample predictions."""
    print("\n" + "="*60)
    print(f"SAMPLE PREDICTIONS (first {n_samples})")
    print("="*60)
    
    for i, pred in enumerate(predictions[:n_samples]):
        status = "✓" if pred['correct'] else "✗"
        print(f"\n{status} Example {i+1}:")
        print(f"  Expected:  {pred['expected']}")
        print(f"  Predicted: {pred['predicted']}")
        print(f"  Score:     {pred['score']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Score predictions from CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python score_predictions.py --csv predictions.csv
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of sample predictions to display (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    try:
        with open(args.csv, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Load predictions
    print(f"Loading predictions from: {args.csv}")
    predictions = load_predictions_csv(args.csv)
    print(f"Loaded {len(predictions)} predictions.")
    
    if not predictions:
        print("Error: No predictions found in CSV.")
        sys.exit(1)
    
    # Compute metrics
    metrics = compute_metrics(predictions)
    
    # Print results
    print_metrics(metrics)
    
    if args.samples > 0:
        print_sample_predictions(predictions, n_samples=args.samples)
    
    print()  # Final newline


if __name__ == '__main__':
    main()
