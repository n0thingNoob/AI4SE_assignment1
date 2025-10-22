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

# Try optional dependency astor lazily to support Python environments without it
try:
    import astor  # type: ignore
    _ASTOR_AVAILABLE = True
except Exception:
    astor = None  # type: ignore
    _ASTOR_AVAILABLE = False


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
                # Use astor if available to convert AST node back to source
                if _ASTOR_AVAILABLE:
                    condition_text = astor.to_source(node.test).strip()  # type: ignore
                else:
                    # Fallback: simple reconstruction via ast.unparse if available (Py>=3.9), else repr
                    if hasattr(ast, 'unparse'):
                        condition_text = ast.unparse(node.test).strip()  # type: ignore
                    else:
                        # Very rough fallback: use slice of source later; here keep a placeholder
                        condition_text = ''
                
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
        if _ASTOR_AVAILABLE:
            masked_code = astor.to_source(tree)  # type: ignore
        elif hasattr(ast, 'unparse'):
            masked_code = ast.unparse(tree)  # type: ignore
        else:
            # Regex-based replacement as last resort
            return mask_one_if_condition_regex(source_code, condition_index)
        return masked_code, condition_text
    except Exception:
        return mask_one_if_condition_regex(source_code, condition_index)


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


def ast_equal(expr1: str, expr2: str) -> bool:
    """Check if two expressions are AST-equivalent."""
    try:
        tree1 = ast.parse(expr1, mode="eval")
        tree2 = ast.parse(expr2, mode="eval")
        return ast.dump(tree1, annotate_fields=False) == ast.dump(tree2, annotate_fields=False)
    except SyntaxError:
        return False


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


def is_correct_ast(expected, predicted):
    """
    Check if predicted condition matches expected condition using AST equivalence.
    
    Args:
        expected: Expected condition string
        predicted: Predicted condition string
        
    Returns:
        Boolean indicating AST equivalence
    """
    # Remove trailing colon if present
    expected_clean = expected.rstrip(':').strip()
    predicted_clean = predicted.rstrip(':').strip()
    
    return ast_equal(expected_clean, predicted_clean)


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


# Use regex masking as final fallback when astor and ast.unparse are unavailable
if not _ASTOR_AVAILABLE and not hasattr(ast, 'unparse'):
    print("Warning: astor not installed and ast.unparse unavailable. Using regex-based fallback for masking.")
    print("Install with: pip install astor or upgrade Python (>=3.9) for better accuracy.")
    mask_one_if_condition = mask_one_if_condition_regex  # type: ignore
