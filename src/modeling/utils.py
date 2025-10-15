#!/usr/bin/env python3
"""
Utilities for working with Python functions and if-conditions.

Provides AST-based functions for:
- Finding if-conditions in Python source code
- Masking a specific if-condition with <IFMASK> token
- Normalizing conditions for comparison
- Checking correctness of predictions
"""

import ast
import re
import astor


def find_if_conditions_in_func(source_code):
    """
    Find all if-condition expressions in Python source code.
    
    Args:
        source_code: Python source code as string
        
    Returns:
        List of tuples (start_offset, end_offset, condition_text, node)
        Returns empty list if parsing fails or no if statements found
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    
    conditions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Extract the test condition
            try:
                # Use astor to convert AST node back to source
                condition_text = astor.to_source(node.test).strip()
                
                # Get position info (line and column)
                # Note: AST doesn't give us character offsets directly
                # We'll store the condition text and node for masking
                conditions.append({
                    'text': condition_text,
                    'line': node.lineno,
                    'col': node.col_offset,
                    'node': node,
                })
            except Exception:
                # Skip if we can't extract the condition
                continue
    
    return conditions


def mask_one_if_condition(source_code, condition_index=0):
    """
    Replace one if-condition in source code with <IFMASK> token.
    
    Args:
        source_code: Python source code as string
        condition_index: Index of which if-condition to mask (default: 0 = first)
        
    Returns:
        Tuple of (masked_code, original_condition_text)
        Returns (None, None) if masking fails
    """
    conditions = find_if_conditions_in_func(source_code)
    
    if not conditions or condition_index >= len(conditions):
        return None, None
    
    target_condition = conditions[condition_index]
    condition_text = target_condition['text']
    
    # Parse the source code
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None, None
    
    # Find the corresponding If node and replace its test
    if_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
    
    if condition_index >= len(if_nodes):
        return None, None
    
    target_node = if_nodes[condition_index]
    
    # Replace the test with a Name node representing <IFMASK>
    # This is a bit hacky but works for our purposes
    target_node.test = ast.Name(id='<IFMASK>', ctx=ast.Load())
    
    # Convert back to source
    try:
        masked_code = astor.to_source(tree)
        return masked_code, condition_text
    except Exception:
        return None, None


def normalize_condition(condition):
    """
    Normalize a condition string for comparison.
    
    Applies:
    - Strip whitespace
    - Collapse multiple spaces to single space
    - Remove surrounding parentheses if they match
    
    Args:
        condition: Condition string
        
    Returns:
        Normalized condition string
    """
    if not condition:
        return ""
    
    # Strip and collapse whitespace
    condition = ' '.join(condition.split())
    
    # Remove surrounding parentheses if balanced
    while condition.startswith('(') and condition.endswith(')'):
        # Check if these parens match
        depth = 0
        mismatched = False
        for i, char in enumerate(condition):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            if depth == 0 and i < len(condition) - 1:
                mismatched = True
                break
        
        if not mismatched:
            condition = condition[1:-1].strip()
        else:
            break
    
    return condition


def is_correct(expected, predicted):
    """
    Check if predicted condition matches expected condition.
    
    Uses normalized comparison (whitespace and parentheses).
    
    Args:
        expected: Expected condition string
        predicted: Predicted condition string
        
    Returns:
        Boolean indicating exact match after normalization
    """
    expected_norm = normalize_condition(expected)
    predicted_norm = normalize_condition(predicted)
    return expected_norm == predicted_norm


def extract_function_with_if(source_code):
    """
    Check if a function contains at least one if statement.
    
    Args:
        source_code: Python source code as string
        
    Returns:
        Boolean indicating presence of if statements
    """
    conditions = find_if_conditions_in_func(source_code)
    return len(conditions) > 0


# Alternative implementation without astor (in case it's not available)
# This version uses regex-based replacement as a fallback

def mask_one_if_condition_regex(source_code, condition_index=0):
    """
    Fallback implementation using regex to mask if-conditions.
    
    This is less robust than AST-based approach but works as backup.
    
    Args:
        source_code: Python source code as string
        condition_index: Which if-condition to mask
        
    Returns:
        Tuple of (masked_code, original_condition)
    """
    # Pattern to find if statements
    # This is a simplified pattern and may not catch all cases
    pattern = r'(\s+if\s+)(.*?)(\s*:)'
    
    matches = list(re.finditer(pattern, source_code))
    
    if condition_index >= len(matches):
        return None, None
    
    match = matches[condition_index]
    condition = match.group(2).strip()
    
    # Replace with <IFMASK>
    masked = (
        source_code[:match.start(2)] +
        '<IFMASK>' +
        source_code[match.end(2):]
    )
    
    return masked, condition


# Use regex version as fallback
try:
    import astor
    # astor is available, use AST-based methods
except ImportError:
    print("Warning: astor not installed. Using regex-based fallback for masking.")
    print("Install with: pip install astor")
    mask_one_if_condition = mask_one_if_condition_regex
