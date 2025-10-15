#!/usr/bin/env python3
"""
Extract Python functions from cloned repositories using AST parsing.

Walks through Python files, parses them with ast, extracts function definitions,
and outputs metadata as JSONL including:
- File path, function name, line numbers
- Number of lines, presence of if statements, number of ifs
- Function source code and normalized hash for deduplication
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from tqdm import tqdm


# Directories to exclude (vendor, venv, tests, builds, etc.)
EXCLUDE_DIRS = {
    'venv', 'env', '.env', 'virtualenv', '.venv',
    'node_modules', '__pycache__', '.git', '.svn', '.hg',
    'vendor', 'vendors', 'third_party', 'external',
    'build', 'dist', '.tox', '.pytest_cache',
    'test', 'tests', 'testing', 'spec', 'specs',
    '.idea', '.vscode', '.vs',
    'site-packages', 'lib', 'lib64',
}


def should_skip_directory(dir_path):
    """Check if directory should be skipped based on name."""
    dir_name = os.path.basename(dir_path).lower()
    return dir_name in EXCLUDE_DIRS or dir_name.startswith('.')


def normalize_code(code):
    """
    Normalize code for deduplication by removing extra whitespace.
    
    Args:
        code: Source code string
        
    Returns:
        Normalized string suitable for hashing
    """
    # Remove comments
    lines = []
    for line in code.split('\n'):
        # Remove inline comments (simple heuristic)
        if '#' in line:
            # Keep if it's in a string (very basic check)
            if line.count('"') % 2 == 0 and line.count("'") % 2 == 0:
                line = line.split('#')[0]
        lines.append(line)
    
    code = '\n'.join(lines)
    
    # Collapse whitespace
    code = re.sub(r'[ \t]+', ' ', code)
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()


def compute_hash(code):
    """Compute SHA-256 hash of normalized code."""
    normalized = normalize_code(code)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def count_if_statements(node):
    """
    Count number of if statements in an AST node.
    
    Args:
        node: AST node (typically a FunctionDef)
        
    Returns:
        Count of If nodes in the subtree
    """
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.If):
            count += 1
    return count


def extract_function(func_node, source_lines, file_path, rel_path):
    """
    Extract metadata and source code for a single function.
    
    Args:
        func_node: ast.FunctionDef or ast.AsyncFunctionDef node
        source_lines: List of source code lines
        file_path: Absolute path to file
        rel_path: Relative path from repo root
        
    Returns:
        Dictionary with function metadata or None if extraction fails
    """
    try:
        # Get line range
        start_line = func_node.lineno
        end_line = func_node.end_lineno
        
        if end_line is None:
            # Fallback for older Python versions
            end_line = start_line + 10  # Approximate
        
        # Extract source code
        func_lines = source_lines[start_line - 1:end_line]
        func_code = ''.join(func_lines)
        
        # Count lines
        n_lines = end_line - start_line + 1
        
        # Count if statements
        num_ifs = count_if_statements(func_node)
        has_if = num_ifs > 0
        
        # Compute hash for deduplication
        code_hash = compute_hash(func_code)
        
        return {
            'file': rel_path,
            'file_abs': file_path,
            'function_name': func_node.name,
            'lineno': start_line,
            'end_lineno': end_line,
            'n_lines': n_lines,
            'has_if': has_if,
            'num_ifs': num_ifs,
            'code': func_code,
            'hash': code_hash,
        }
    
    except Exception as e:
        # Silently skip functions that can't be extracted
        return None


def extract_functions_from_file(file_path, repos_root):
    """
    Parse a Python file and extract all function definitions.
    
    Args:
        file_path: Absolute path to Python file
        repos_root: Root directory of repositories (for relative paths)
        
    Returns:
        List of function metadata dictionaries
    """
    try:
        # Read source code
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            # Skip files with syntax errors
            return []
        
        # Get source lines
        source_lines = source.splitlines(keepends=True)
        
        # Compute relative path
        rel_path = os.path.relpath(file_path, repos_root)
        
        # Extract all function definitions
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_data = extract_function(node, source_lines, file_path, rel_path)
                if func_data:
                    functions.append(func_data)
        
        return functions
    
    except Exception as e:
        # Skip files that can't be processed
        return []


def extract_all_functions(repos_root, min_lines=5, max_lines=400):
    """
    Walk through repository directory and extract all Python functions.
    
    Args:
        repos_root: Root directory containing cloned repositories
        min_lines: Minimum lines for a function to be included
        max_lines: Maximum lines for a function to be included
        
    Returns:
        List of function metadata dictionaries
    """
    # Find all Python files
    python_files = []
    
    print(f"Scanning for Python files in: {repos_root}")
    for root, dirs, files in os.walk(repos_root):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not should_skip_directory(os.path.join(root, d))]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files.")
    
    # Extract functions from all files
    all_functions = []
    
    with tqdm(total=len(python_files), desc="Extracting functions") as pbar:
        for file_path in python_files:
            functions = extract_functions_from_file(file_path, repos_root)
            all_functions.extend(functions)
            pbar.update(1)
    
    print(f"Extracted {len(all_functions)} raw functions.")
    
    # Filter by line count
    filtered_functions = [
        f for f in all_functions
        if min_lines <= f['n_lines'] <= max_lines
    ]
    
    print(f"After filtering ({min_lines}-{max_lines} lines): {len(filtered_functions)} functions.")
    
    # Deduplicate by hash
    seen_hashes = set()
    deduplicated_functions = []
    
    for func in filtered_functions:
        if func['hash'] not in seen_hashes:
            seen_hashes.add(func['hash'])
            deduplicated_functions.append(func)
    
    print(f"After deduplication: {len(deduplicated_functions)} unique functions.")
    
    return deduplicated_functions


def save_functions_jsonl(functions, output_path):
    """
    Save functions to JSONL file (one JSON object per line).
    
    Args:
        functions: List of function metadata dictionaries
        output_path: Path to output JSONL file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for func in functions:
            json.dump(func, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved {len(functions)} functions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Python functions from cloned repositories using AST.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python extract_functions.py --repos-root data/raw_repos --out data/functions.jsonl
        """
    )
    
    parser.add_argument(
        '--repos-root',
        type=str,
        required=True,
        help='Root directory containing cloned repositories'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='data/functions.jsonl',
        help='Output JSONL file path (default: data/functions.jsonl)'
    )
    
    parser.add_argument(
        '--min-lines',
        type=int,
        default=5,
        help='Minimum number of lines for a function (default: 5)'
    )
    
    parser.add_argument(
        '--max-lines',
        type=int,
        default=400,
        help='Maximum number of lines for a function (default: 400)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.repos_root):
        print(f"Error: Repository root not found: {args.repos_root}")
        sys.exit(1)
    
    # Extract functions
    functions = extract_all_functions(
        args.repos_root,
        min_lines=args.min_lines,
        max_lines=args.max_lines
    )
    
    if not functions:
        print("\nError: No functions extracted.")
        sys.exit(1)
    
    # Save to JSONL
    save_functions_jsonl(functions, args.out)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Extraction Statistics:")
    print(f"  Total functions: {len(functions)}")
    print(f"  Functions with if statements: {sum(1 for f in functions if f['has_if'])}")
    print(f"  Average lines per function: {sum(f['n_lines'] for f in functions) / len(functions):.1f}")
    print(f"  Average if statements per function: {sum(f['num_ifs'] for f in functions) / len(functions):.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
