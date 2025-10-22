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
from textwrap import dedent
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


def _compute_end_line_compat(node, default_start):
    """
    Compute an approximate end line for a function node compatible with Python 3.7.
    Tries end_lineno if available; otherwise walks child nodes to find max lineno.

    Args:
        node: AST function node
        default_start: Fallback start line number

    Returns:
        Estimated end line number (int)
    """
    end_line = getattr(node, 'end_lineno', None)
    if isinstance(end_line, int):
        return end_line

    max_line = default_start
    for child in ast.walk(node):
        # Prefer end_lineno if present on child (Py>=3.8), else lineno
        child_end = getattr(child, 'end_lineno', None)
        if isinstance(child_end, int):
            if child_end > max_line:
                max_line = child_end
            continue
        child_line = getattr(child, 'lineno', None)
        if isinstance(child_line, int) and child_line > max_line:
            max_line = child_line
    # Ensure at least some span
    return max(max_line, default_start)


def extract_function(func_node, source_lines, file_path, rel_path):
    """
    Extract metadata and source code for a single function (decorators included, dedented).
    Returns None if the snippet cannot be parsed standalone.
    """
    try:
        # 1) include decorators above the def
        deco_lines = [getattr(d, 'lineno', func_node.lineno) for d in getattr(func_node, 'decorator_list', [])]
        start_line = min([func_node.lineno] + deco_lines)

        # 2) robust end line
        end_line = _compute_end_line_compat(func_node, start_line)

        # 3) slice raw source (inclusive)
        raw = ''.join(source_lines[start_line - 1:end_line])

        # 4) pre-clean: tabs -> spaces; strip simple fenced-code markers
        raw = raw.replace('\t', '    ')
        raw = '\n'.join(ln for ln in raw.splitlines() if not ln.strip().startswith('```'))

        # 5) dedent to make methods/nested defs parse at module level
        code = dedent(raw).rstrip() + '\n'

        # 6) sanity parse: ensure snippet is re-parseable standalone
        try:
            ast.parse(code)
        except SyntaxError:
            return None

        # 7) stats from original node
        n_lines = end_line - start_line + 1
        num_ifs = count_if_statements(func_node)
        has_if = num_ifs > 0

        # 8) hash on DEDENTED code (stable for dedup)
        code_hash = compute_hash(code)

        return {
            'file': rel_path,
            'file_abs': file_path,
            'function_name': func_node.name,
            'lineno': start_line,
            'end_lineno': end_line,
            'n_lines': n_lines,   # keep original span count
            'has_if': has_if,
            'num_ifs': num_ifs,
            'code': code,         # dedented, parseable
            'hash': code_hash,
        }
    except Exception:
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
