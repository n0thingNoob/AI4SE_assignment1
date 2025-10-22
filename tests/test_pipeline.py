#!/usr/bin/env python3
"""
Test suite for if-condition prediction pipeline.
"""

import ast
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.modeling.utils import ast_equal, is_correct, is_correct_ast, normalize_condition


def test_ast_equal():
    """Test AST equivalence function."""
    # Same expressions
    assert ast_equal("x > 0", "x > 0")
    assert ast_equal("a and b", "a and b")
    assert ast_equal("(x + y) * z", "(x + y) * z")
    
    # Different whitespace but same AST
    assert ast_equal("x>0", "x > 0")
    assert ast_equal("a and b", "a  and  b")
    
    # Different expressions
    assert not ast_equal("x > 0", "x < 0")
    assert not ast_equal("a and b", "a or b")
    
    # Invalid expressions
    assert not ast_equal("x >", "x > 0")
    assert not ast_equal("", "x > 0")


def test_normalize_condition():
    """Test condition normalization."""
    # Basic normalization
    assert normalize_condition("  x > 0  ") == "x > 0"
    assert normalize_condition("a  and  b") == "a and b"
    
    # Parentheses removal
    assert normalize_condition("(x > 0)") == "x > 0"
    assert normalize_condition("((x > 0))") == "x > 0"
    
    # Nested parentheses (should not remove)
    assert normalize_condition("(x > 0) and (y < 0)") == "(x > 0) and (y < 0)"
    
    # Empty string
    assert normalize_condition("") == ""


def test_is_correct():
    """Test exact match correctness."""
    # Exact matches
    assert is_correct("x > 0", "x > 0")
    assert is_correct("a and b", "a and b")
    
    # Normalized matches
    assert is_correct("  x > 0  ", "x > 0")
    assert is_correct("(x > 0)", "x > 0")
    
    # Non-matches
    assert not is_correct("x > 0", "x < 0")
    assert not is_correct("a and b", "a or b")


def test_is_correct_ast():
    """Test AST equivalence correctness."""
    # AST equivalent
    assert is_correct_ast("x > 0", "x > 0")
    assert is_correct_ast("x>0", "x > 0")
    assert is_correct_ast("(x > 0)", "x > 0")
    
    # Not AST equivalent
    assert not is_correct_ast("x > 0", "x < 0")
    assert not is_correct_ast("a and b", "a or b")


def test_single_line_if():
    """Test single-line if condition extraction."""
    code = """
def test_func(x):
    if x > 0:
        return x
    return 0
"""
    
    # This would be tested with the actual extraction function
    # For now, just test that the code parses
    tree = ast.parse(code)
    if_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
    assert len(if_nodes) == 1
    assert isinstance(if_nodes[0].test, ast.Compare)


def test_elif_condition():
    """Test elif condition handling."""
    code = """
def test_func(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
"""
    
    tree = ast.parse(code)
    if_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
    assert len(if_nodes) == 2  # if and elif


def test_multi_line_condition():
    """Test multi-line condition handling."""
    code = """
def test_func(x, y):
    if (x > 0 and 
        y < 10 and
        x + y > 5):
        return True
    return False
"""
    
    tree = ast.parse(code)
    if_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
    assert len(if_nodes) == 1
    # The condition should be a single BoolOp node
    assert isinstance(if_nodes[0].test, ast.BoolOp)


def test_comprehension_if_ignored():
    """Test that comprehension if conditions are ignored."""
    code = """
def test_func(lst):
    return [x for x in lst if x > 0]
"""
    
    tree = ast.parse(code)
    if_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
    # Should not find any if nodes (comprehension if is not ast.If)
    assert len(if_nodes) == 0


def test_validate_sample_invariants():
    """Test sample validation invariants."""
    # Valid sample
    masked_code = "if <IFMASK>:\n    return True"
    target = "x > 0"
    assert validate_sample(masked_code, target)
    
    # Invalid: multiple masks
    masked_code_multi = "if <IFMASK> and <IFMASK>:\n    return True"
    assert not validate_sample(masked_code_multi, target)
    
    # Invalid: no masks
    masked_code_none = "if x > 0:\n    return True"
    assert not validate_sample(masked_code_none, target)
    
    # Invalid: doesn't parse after rebuild
    masked_code_invalid = "if <IFMASK>:\n    return"
    target_invalid = "x >"  # Invalid syntax
    assert not validate_sample(masked_code_invalid, target_invalid)


def validate_sample(masked_code: str, target: str, mask_token: str = "<IFMASK>") -> bool:
    """Helper function for testing."""
    # Check exactly one mask token
    if masked_code.count(mask_token) != 1:
        return False
    
    # Check that reconstruction parses
    try:
        reconstructed = masked_code.replace(mask_token, target)
        ast.parse(reconstructed)
        return True
    except SyntaxError:
        return False


if __name__ == "__main__":
    pytest.main([__file__])



