#!/usr/bin/env python3
"""
Data preparation and auditing script for IF-condition evaluation.
Cleans and validates JSONL data without loading any models.
"""

import json
import argparse
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file, ignoring blank lines."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip blank lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
    return data


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    """Save data to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def normalize_tokens(text: str) -> str:
    """Replace <IF_MASK> with <IFMASK> in text."""
    return text.replace("<IF_MASK>", "<IFMASK>")


def extract_condition(target: str) -> str:
    """
    Extract expected condition from target.
    Handle both single-line and multi-line conditions.
    """
    lines = target.strip().split('\n')
    if not lines:
        return ""
    
    first_line = lines[0].strip()
    
    # If starts with 'if ', extract between 'if ' and first ':' on original line
    if first_line.startswith('if '):
        colon_pos = first_line.find(':', 3)
        if colon_pos != -1:
            cond = first_line[3:colon_pos].strip()
        else:
            cond = first_line[3:].strip()
    else:
        cond = first_line
    
    # Handle multi-line conditions
    if cond.endswith('(') or cond.endswith('and') or cond.endswith('or') or cond.endswith('not'):
        # This looks like a multi-line condition, try to reconstruct
        full_cond = []
        for line in lines:
            line = line.strip()
            if line.endswith(':'):
                # Found the end of condition
                full_cond.append(line.rstrip(':').strip())
                break
            elif line:
                full_cond.append(line)
        
        if full_cond:
            cond = ' '.join(full_cond)
    
    # Remove trailing colon and inline comments
    cond = cond.rstrip(':').strip()
    # Strip Python inline comments safely (only if not inside quotes - best-effort)
    if '#' in cond:
        parts = cond.split('#', 1)
        cond = parts[0].rstrip()
    
    # Try to fix common issues
    cond = _fix_common_syntax_issues(cond)
    
    return cond


def _fix_common_syntax_issues(cond: str) -> str:
    """Try to fix common syntax issues in conditions."""
    if not cond:
        return cond
    
    # Remove trailing colon (should have been handled earlier, but double-check)
    cond = cond.rstrip(':').strip()
    
    # Remove trailing operators that make the condition incomplete
    trailing_ops = [' and', ' or', ' not', ' in', ' is']
    for op in trailing_ops:
        if cond.endswith(op):
            cond = cond[:-len(op)].strip()
    
    # Also handle cases where the operator is at the very end without space
    trailing_ops_no_space = ['and', 'or', 'not', 'in', 'is']
    for op in trailing_ops_no_space:
        if cond.endswith(op) and len(cond) > len(op):
            # Check if it's really at the end (not part of a word)
            if cond[-len(op)-1] in [' ', '(', '[', '{', '=', '!', '<', '>']:
                cond = cond[:-len(op)].strip()
    
    # Handle incomplete set/dict literals
    if cond.endswith('{'):
        cond = cond[:-1].strip()
    elif cond.endswith('['):
        cond = cond[:-1].strip()
    
    # Fix quote issues more intelligently
    cond = _fix_quote_issues(cond)
    
    # Try to close unclosed parentheses (simple heuristic)
    open_parens = cond.count('(')
    close_parens = cond.count(')')
    if open_parens > close_parens:
        cond += ')' * (open_parens - close_parens)
    
    return cond


def _fix_quote_issues(cond: str) -> str:
    """Fix quote-related syntax issues."""
    if not cond:
        return cond
    
    # Handle triple quotes
    if '"""' in cond:
        # If there's an unterminated triple quote, try to close it
        if cond.count('"""') % 2 == 1:
            cond += '"""'
    
    # Handle complex quote patterns like '"' in line or "''"" in line
    # Try to balance quotes by adding missing closing quotes at the end
    try:
        # Test if the condition can be parsed as-is
        test_snippet = f'if {cond}:\n    pass'
        ast.parse(test_snippet, mode='exec')
        return cond  # Already valid
    except SyntaxError as e:
        if 'unterminated string literal' in str(e):
            # Try to fix by adding quotes at the end
            # Count unescaped quotes
            single_quotes = 0
            double_quotes = 0
            i = 0
            while i < len(cond):
                if cond[i] == "'" and (i == 0 or cond[i-1] != '\\'):
                    single_quotes += 1
                elif cond[i] == '"' and (i == 0 or cond[i-1] != '\\'):
                    double_quotes += 1
                i += 1
            
            # Add missing closing quotes
            if single_quotes % 2 == 1:
                cond += "'"
            if double_quotes % 2 == 1:
                cond += '"'
    
    return cond


def window_around_mask(text: str, mask: str = "<IFMASK>", left: int = 280, right: int = 280) -> str:
    """Extract character window around mask token."""
    mask_pos = text.find(mask)
    if mask_pos == -1:
        return text  # Mask not found, return original
    
    start = max(0, mask_pos - left)
    end = min(len(text), mask_pos + len(mask) + right)
    
    return text[start:end]


def append_ans(prompt: str) -> str:
    """Ensure prompt ends with ' <ANS> '."""
    if not prompt.strip().endswith(" <ANS>"):
        return prompt.rstrip() + " <ANS> "
    return prompt


def _count_operators(condition: str) -> Counter:
    """Count operators with priority to composite forms to avoid double counting.
    Order:
      1) composite: "not in", "is not"
      2) single: "in" (excluding not in), "is" (excluding is not), "not" (excluding not in/is not)
      3) symbols: ==, !=, <=, >=, <, >
    """
    text = condition.lower()
    counts = Counter()

    # Composite patterns (non-overlapping)
    patterns_composite = {
        "not in": re.compile(r"\bnot\s+in\b"),
        "is not": re.compile(r"\bis\s+not\b"),
    }
    # Remove matched segments to avoid recounting
    for name, pat in patterns_composite.items():
        matches = list(pat.finditer(text))
        counts[name] += len(matches)
    # Replace composites with spaces of same length to keep indices simple
    def _mask(match: re.Match) -> str:
        return " " * (match.end() - match.start())
    for pat in patterns_composite.values():
        text = pat.sub(_mask, text)

    # Single-word operators
    patterns_single = {
        "in": re.compile(r"\bin\b"),
        "is": re.compile(r"\bis\b"),
        "not": re.compile(r"\bnot\b"),
        "and": re.compile(r"\band\b"),
        "or": re.compile(r"\bor\b"),
    }
    for name, pat in patterns_single.items():
        counts[name] += len(list(pat.finditer(text)))

    # Symbol operators (order to avoid <= counted as < + =)
    symbols = {
        "<=": re.compile(r"<="),
        ">=": re.compile(r">="),
        "==": re.compile(r"=="),
        "!=": re.compile(r"!="),
        "<": re.compile(r"<(?!=)"),
        ">": re.compile(r">(?!=)"),
    }
    for name, pat in symbols.items():
        counts[name] += len(list(pat.finditer(text)))

    return counts


def collect_stats(rows: List[Dict[str, Any]], token_lengths: Optional[List[int]] = None, max_len: Optional[int] = None, docstring_mask_hits: int = 0, function_unparseable_after_fill: int = 0) -> Dict[str, Any]:
    """Collect comprehensive statistics about the dataset."""
    total = len(rows)
    if total == 0:
        return {"total": 0}
    
    # Basic counts
    has_ifmask = sum(1 for row in rows if row.get('had_if_mask', False))
    unique_mask = sum(1 for row in rows if row.get('had_if_mask', False) and 
                     row.get('input', '').count('<IFMASK>') == 1)
    
    # Issue tracking
    issue_counts = Counter()
    for row in rows:
        for issue in row.get('issues', []):
            issue_counts[issue] += 1
    
    # Length statistics
    input_lengths = [len(row.get('input', '')) for row in rows]
    prepped_lengths = [len(row.get('input_prepped', '')) for row in rows]
    condition_lengths = [len(row.get('expected_condition', '')) for row in rows]
    
    # Operator counts (composite-first to avoid double counting)
    operator_counts = Counter()
    for row in rows:
        condition = row.get('expected_condition', '')
        operator_counts.update(_count_operators(condition))
    
    # Condition length histogram (bins: 0-10, 11-20, 21-50, 51-100, 100+)
    def get_length_bin(length: int) -> str:
        if length <= 10:
            return "0-10"
        elif length <= 20:
            return "11-20"
        elif length <= 50:
            return "21-50"
        elif length <= 100:
            return "51-100"
        else:
            return "100+"
    
    length_histogram = Counter(get_length_bin(length) for length in condition_lengths)

    # AST parse-ability of conditions
    parseable = 0
    for cond in (row.get('expected_condition', '') for row in rows):
        snippet = f"if {cond}:\n    pass"
        try:
            ast.parse(snippet, mode='exec')
            parseable += 1
        except Exception:
            pass

    # Token length distribution (if provided)
    token_stats = None
    if token_lengths is not None and len(token_lengths) == len(rows):
        sorted_lens = sorted(token_lengths)
        def pct(p: float) -> int:
            if not sorted_lens:
                return 0
            idx = min(len(sorted_lens) - 1, max(0, int(round(p * (len(sorted_lens) - 1)))))
            return int(sorted_lens[idx])
        token_stats = {
            "avg": sum(sorted_lens) / len(sorted_lens) if sorted_lens else 0,
            "min": sorted_lens[0] if sorted_lens else 0,
            "max": sorted_lens[-1] if sorted_lens else 0,
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max_len": max_len,
        }
    
    result = {
        "total": total,
        "has_ifmask": has_ifmask,
        "has_ifmask_pct": (has_ifmask / total) * 100 if total > 0 else 0,
        "unique_mask": unique_mask,
        "unique_mask_pct": (unique_mask / total) * 100 if total > 0 else 0,
        "issue_counts": dict(issue_counts),
        "lengths": {
            "input_avg": sum(input_lengths) / len(input_lengths) if input_lengths else 0,
            "input_min": min(input_lengths) if input_lengths else 0,
            "input_max": max(input_lengths) if input_lengths else 0,
            "prepped_avg": sum(prepped_lengths) / len(prepped_lengths) if prepped_lengths else 0,
            "prepped_min": min(prepped_lengths) if prepped_lengths else 0,
            "prepped_max": max(prepped_lengths) if prepped_lengths else 0,
            "condition_avg": sum(condition_lengths) / len(condition_lengths) if condition_lengths else 0,
            "condition_min": min(condition_lengths) if condition_lengths else 0,
            "condition_max": max(condition_lengths) if condition_lengths else 0,
        },
        "condition_length_histogram": dict(length_histogram),
        "operator_counts": dict(operator_counts),
        "condition_parseable": parseable,
        "condition_parseable_pct": (parseable / total) * 100 if total > 0 else 0,
    }
    if token_stats is not None:
        result["token_lengths"] = token_stats
    
    # Add hard filter counters
    result["docstring_mask_hits"] = docstring_mask_hits
    result["function_unparseable_after_fill"] = function_unparseable_after_fill
    
    return result


def is_mask_in_docstring(code: str, mask_pos: int) -> bool:
    """
    Check if the <IFMASK> position lies inside triple-quoted strings.
    Uses a simple 3-line neighborhood heuristic.
    """
    lines = code.splitlines(keepends=True)
    char_pos = 0
    current_line = 0
    
    # Find which line contains the mask position
    for i, line in enumerate(lines):
        if char_pos + len(line) > mask_pos:
            current_line = i
            break
        char_pos += len(line)
    
    # Check 3-line neighborhood for triple quotes
    start_line = max(0, current_line - 1)
    end_line = min(len(lines), current_line + 2)
    
    neighborhood = ''.join(lines[start_line:end_line])
    
    # Look for unmatched triple quotes before the mask position
    triple_single = neighborhood.count("'''")
    triple_double = neighborhood.count('"""')
    
    # Simple heuristic: if odd number of triple quotes, might be inside docstring
    return (triple_single % 2 == 1) or (triple_double % 2 == 1)


def is_function_parseable_after_fill(input_text: str, expected_condition: str) -> bool:
    """
    Check if the function is parseable after replacing <IFMASK> with expected_condition.
    """
    try:
        code_filled = input_text.replace('<IFMASK>', expected_condition)
        ast.parse(code_filled)
        return True
    except Exception:
        return False


def process_example(example: Dict[str, Any], left: int, right: int, append_ans_flag: bool, no_window: bool = False) -> Dict[str, Any]:
    """Process a single example and return cleaned data."""
    input_text = example.get('input', '')
    target = example.get('target', '')
    
    # Normalize special tokens
    input_normalized = normalize_tokens(input_text)
    target_normalized = normalize_tokens(target)
    
    # Check for <IFMASK> presence and uniqueness
    mask_count = input_normalized.count('<IFMASK>')
    had_if_mask = mask_count == 1
    issues = []
    
    if mask_count == 0:
        issues.append("missing_ifmask")
    elif mask_count > 1:
        issues.append("multiple_ifmask")
    
    # Extract expected condition
    expected_condition = extract_condition(target_normalized)
    # Sanitize: normalize unicode quotes and weird spaces
    trans_table = {
        ord('\u2018'): "'", ord('\u2019'): "'",
        ord('\u201c'): '"', ord('\u201d'): '"',
        ord('\u00a0'): ' ',  # non-breaking space
        ord('\t'): ' ',
    }
    expected_condition = expected_condition.translate(trans_table)
    # Remove trailing backslash line continuation
    if expected_condition.endswith('\\'):
        expected_condition = expected_condition[:-1].rstrip()
    # Collapse multiple spaces
    expected_condition = re.sub(r'\s+', ' ', expected_condition).strip()
    if not expected_condition.strip():
        issues.append("empty_condition")
    
    # Hard filter 1: Docstring guard - reject if <IFMASK> is inside triple-quoted strings
    if had_if_mask:
        mask_pos = input_normalized.find('<IFMASK>')
        if is_mask_in_docstring(input_normalized, mask_pos):
            issues.append("docstring_mask_hit")
    
    # Hard filter 2: Full-code parse check - ensure function is parseable after fill
    if had_if_mask and expected_condition.strip():
        if not is_function_parseable_after_fill(input_normalized, expected_condition):
            issues.append("function_unparseable_after_fill")
    
    # Prepare input with window around mask
    if had_if_mask and not no_window:
        input_prepped = window_around_mask(input_normalized, left=left, right=right)
    else:
        input_prepped = input_normalized
    
    # Append answer cue if requested
    if append_ans_flag:
        input_prepped = append_ans(input_prepped)
    
    return {
        "input": input_text,  # Keep original
        "input_prepped": input_prepped,
        "target": target,  # Keep original
        "expected_condition": expected_condition,
        "had_if_mask": had_if_mask,
        "issues": issues
    }


def _maybe_load_tokenizer(tokenizer_dir: Optional[str]):
    """Load a fast tokenizer from a directory if provided (no transformers import)."""
    if not tokenizer_dir:
        return None
    try:
        # Use tokenizers library directly to avoid transformers/torch deps
        from tokenizers import Tokenizer
        from tokenizers.processors import TemplateProcessing  # noqa: F401 (ensure package available)
        tok_path = Path(tokenizer_dir) / "tokenizer.json"
        if tok_path.exists():
            return Tokenizer.from_file(str(tok_path))
    except Exception as e:
        print(f"Warning: failed to load tokenizer from {tokenizer_dir}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare and audit IF-condition evaluation data")
    parser.add_argument("--in", dest="input_file", required=True, help="Input JSONL file")
    parser.add_argument("--out", dest="output_file", required=True, help="Output JSONL file")
    parser.add_argument("--summary", dest="summary_file", required=True, help="Summary JSON file")
    parser.add_argument("--left", type=int, default=280, help="Left window size around mask")
    parser.add_argument("--right", type=int, default=280, help="Right window size around mask")
    parser.add_argument("--no-window", action="store_true", help="Disable window cropping")
    parser.add_argument("--append-ans", action="store_true", help="Append <ANS> to input")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer directory (for token length stats)")
    parser.add_argument("--max-len", type=int, default=None, help="Model max sequence length for token length health check (e.g., 384/512)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} examples")
    
    # Process examples
    print("Processing examples...")
    processed_data = []
    dropped_count = 0
    docstring_mask_hits = 0
    function_unparseable_after_fill = 0
    
    for i, example in enumerate(data):
        processed = process_example(example, args.left, args.right, args.append_ans, args.no_window)
        
        # Apply hard filters - drop rows with critical issues
        if "docstring_mask_hit" in processed["issues"]:
            docstring_mask_hits += 1
            dropped_count += 1
            continue
        if "function_unparseable_after_fill" in processed["issues"]:
            function_unparseable_after_fill += 1
            dropped_count += 1
            continue
            
        processed_data.append(processed)
    
    print(f"Dropped {dropped_count} examples due to hard filters:")
    print(f"  - Docstring mask hits: {docstring_mask_hits}")
    print(f"  - Function unparseable after fill: {function_unparseable_after_fill}")
    
    # Optional token length statistics using tokenizers
    token_lengths: Optional[List[int]] = None
    if args.tokenizer:
        fast_tok = _maybe_load_tokenizer(args.tokenizer)
        if fast_tok is not None:
            token_lengths = []
            for row in processed_data:
                prompt = row.get('input_prepped', '')
                if args.append_ans and not prompt.strip().endswith("<ANS>"):
                    prompt = prompt.rstrip() + " <ANS> "
                try:
                    encoding = fast_tok.encode(prompt)
                    token_lengths.append(len(encoding.ids))
                except Exception:
                    token_lengths.append(0)

    # Collect statistics
    print("Collecting statistics...")
    stats = collect_stats(processed_data, token_lengths=token_lengths, max_len=args.max_len, 
                         docstring_mask_hits=docstring_mask_hits, 
                         function_unparseable_after_fill=function_unparseable_after_fill)

    # Print top problematic conditions (unparseable) for quick inspection
    unparseable_indices = []
    for idx, row in enumerate(processed_data):
        cond = row.get('expected_condition', '')
        snippet = f"if {cond}:\n    pass"
        try:
            ast.parse(snippet, mode='exec')
        except Exception:
            unparseable_indices.append((idx, cond))
    if unparseable_indices:
        print("\nUnparseable conditions (first 20):")
        for i, (idx, cond) in enumerate(unparseable_indices[:20]):
            print(f"  [{idx}] {cond}")
    
    # Save processed data
    print(f"Saving processed data to {args.output_file}...")
    save_jsonl(processed_data, args.output_file)
    
    # Save summary
    print(f"Saving summary to {args.summary_file}...")
    with open(args.summary_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print console report
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    print(f"Total examples: {stats['total']}")
    print(f"With <IFMASK>: {stats['has_ifmask']} ({stats['has_ifmask_pct']:.1f}%)")
    print(f"Unique <IFMASK>: {stats['unique_mask']} ({stats['unique_mask_pct']:.1f}%)")
    
    if stats['issue_counts']:
        print(f"\nIssues found:")
        for issue, count in sorted(stats['issue_counts'].items()):
            print(f"  {issue}: {count}")
    
    print(f"\nLength statistics:")
    print(f"  Input: avg={stats['lengths']['input_avg']:.1f}, "
          f"min={stats['lengths']['input_min']}, max={stats['lengths']['input_max']}")
    print(f"  Prepped: avg={stats['lengths']['prepped_avg']:.1f}, "
          f"min={stats['lengths']['prepped_min']}, max={stats['lengths']['prepped_max']}")
    print(f"  Condition: avg={stats['lengths']['condition_avg']:.1f}, "
          f"min={stats['lengths']['condition_min']}, max={stats['lengths']['condition_max']}")
    
    print(f"\nCondition length distribution:")
    for bin_name, count in sorted(stats['condition_length_histogram'].items()):
        print(f"  {bin_name}: {count}")
    
    print(f"\nOperator counts:")
    for op, count in sorted(stats['operator_counts'].items()):
        print(f"  {op}: {count}")
    
    print(f"\nParseable conditions: {stats['condition_parseable']}/{stats['total']} ({stats['condition_parseable_pct']:.2f}%)")
    
    # Show hard filter counters
    print(f"\nHard filter results:")
    print(f"  Docstring mask hits: {stats.get('docstring_mask_hits', 0)}")
    print(f"  Function unparseable after fill: {stats.get('function_unparseable_after_fill', 0)}")
    
    # Show top 5 examples with issues
    examples_with_issues = [(i, row) for i, row in enumerate(processed_data) if row['issues']]
    if examples_with_issues:
        print(f"\nTop 5 examples with issues:")
        for i, (idx, row) in enumerate(examples_with_issues[:5]):
            print(f"  {idx}: {row['issues']}")
    
    # Check for critical issues
    invalid_mask_ratio = (stats['total'] - stats['unique_mask']) / stats['total'] if stats['total'] > 0 else 0
    if invalid_mask_ratio > 0.01:  # More than 1% lack valid single <IFMASK>
        print(f"\nERROR: {invalid_mask_ratio*100:.1f}% of rows lack valid single <IFMASK> (threshold: 1%)")
        return 1

    # Warn if token lengths nearing model max
    if stats.get('token_lengths') and stats['token_lengths'].get('max_len'):
        tl = stats['token_lengths']
        max_len = tl.get('max_len') or 0
        p95 = tl.get('p95') or 0
        p99 = tl.get('p99') or 0
        if max_len and (p95 > max_len or p99 > max_len):
            print(f"\nWARNING: token length high percentiles exceed max_len (p95={p95}, p99={p99}, max_len={max_len}). Consider reducing window (e.g., 220/220).")
    
    print(f"\nData preparation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
