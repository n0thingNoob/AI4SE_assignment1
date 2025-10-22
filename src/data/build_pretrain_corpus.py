#!/usr/bin/env python3
"""
Build pre-training corpus from extracted functions (upgraded).

- Reads functions.jsonl (expects fields: code, n_lines)
- Filters by min lines and AST-parseable
- Deduplicates snippets (whitespace-insensitive)
- Optionally trims overlong snippets
- Light task-style augmentation (mask first if or append <ANS> <cond>) with --augment-ratio
- Wraps each snippet in <CODE> ... </CODE> blocks separated by a blank line
- Emits a JSON report with quality stats

Suitable for tokenizer training and LM pre-training (with block grouping in the trainer).
"""

import argparse
import json
import os
import re
import sys
import ast
import random
import hashlib
from collections import Counter
from statistics import mean
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm


# Replace old IF_REGEX with this:
IF_REGEX = re.compile(
    r"(?ms)^\s*if\s+(.+?):\s*(?:#.*)?$"
)
OPS = [
    r'\bnot\s+in\b', r'\bis\s+not\b', r'\band\b', r'\bor\b', r'\bnot\b', r'\bin\b', r'\bis\b',
    r'==', r'!=', r'<=', r'>=', r'<', r'>'
]

def is_parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def fingerprint(code: str) -> str:
    """Whitespace-insensitive fingerprint for deduplication."""
    key = re.sub(r'\s+', '', code)
    return hashlib.md5(key.encode('utf-8')).hexdigest()

def trim_long(code: str, max_chars: int) -> str:
    if max_chars and len(code) > max_chars:
        return code[:max_chars]
    return code

def wrap_block(code: str) -> str:
    return f"<CODE>\n{code}\n</CODE>\n"

def find_first_if_condition(code: str) -> Optional[Tuple[str, Tuple[int, int]]]:
    m = IF_REGEX.search(code)
    if not m:
        return None
    cond = m.group(1).strip()
    return cond, m.span()

def augment_mask_if(code: str) -> Optional[str]:
    m = IF_REGEX.search(code)
    if not m:
        return None
    start, end = m.span()
    indent = re.match(r'^\s*', code[start:end]).group(0)
    return code[:start] + f"{indent}if <IFMASK>:\n" + code[end:]

def augment_append_ans(code: str, cond: str) -> str:
    # Ensure trailing newline
    if not code.endswith('\n'):
        code += '\n'
    return code + f"<ANS> {cond}\n"

def operator_counts(code: str) -> Dict[str, int]:
    c = {}
    for pattern in OPS:
        c[pattern] = len(re.findall(pattern, code))
    return c

def build_pretrain_corpus(functions_jsonl: str,
                          output_txt: str,
                          min_lines: int = 5,
                          report_json: str = "",
                          augment_ratio: float = 0.08,
                          max_chars: int = 4000,
                          shuffle: bool = False,
                          max_samples: int = 0,
                          seed: int = 0) -> Dict[str, Any]:
    random.seed(seed)

    print(f"Reading functions from: {functions_jsonl}")
    items = []
    with open(functions_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            code = obj.get('code', '')
            n_lines = obj.get('n_lines', code.count('\n') + 1)
            items.append((code, n_lines))

    total = len(items)
    print(f"Loaded {total} functions.")

    # min lines
    items = [(c, n) for (c, n) in items if n >= min_lines]
    dropped_short = total - len(items)
    print(f"After min-lines filter ({min_lines}): {len(items)} kept, {dropped_short} dropped-short.")

    kept = []
    dropped_syntax = 0
    for code, _ in items:
        code = code.replace('\r\n', '\n').replace('\r', '\n').rstrip() + '\n'
        code = trim_long(code, max_chars)
        if not is_parseable(code):
            dropped_syntax += 1
            continue
        kept.append(code)

    print(f"AST-parseable: {len(kept)} / {len(items)}; dropped_syntax={dropped_syntax}")

    # dedup
    seen = set()
    deduped = []
    dup_count = 0
    for code in kept:
        fp = fingerprint(code)
        if fp in seen:
            dup_count += 1
            continue
        seen.add(fp)
        deduped.append(code)

    print(f"Deduplicated: {len(deduped)} kept, dup_count={dup_count}")

    if shuffle:
        random.shuffle(deduped)

    if max_samples and len(deduped) > max_samples:
        deduped = deduped[:max_samples]
        print(f"Truncated to max-samples={max_samples}")

    # After you have `deduped` (list of code strings), build an eligible index list:
    eligible = [i for i, code in enumerate(deduped) if IF_REGEX.search(code)]
    n_aug = int(len(eligible) * max(0.0, min(1.0, augment_ratio)))
    aug_idx = set(random.sample(eligible, n_aug)) if n_aug > 0 else set()

    augmented = 0
    if_hit = 0
    op_counter = Counter()
    out_lines = []

    for i, code in enumerate(tqdm(deduped, desc="Writing corpus")):
        # stats
        oc = operator_counts(code)
        op_counter.update(oc)

        info = find_first_if_condition(code)
        if info:
            if_hit += 1
        # only augment if i in aug_idx (eligible) and info is not None
        if i in aug_idx and info:
            cond, _ = info
            if random.random() < 0.5:
                masked = augment_mask_if(code)
                if masked is not None:
                    code = masked
                    augmented += 1
            else:
                code = augment_append_ans(code, cond)
                augmented += 1

        out_lines.append(wrap_block(code))

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        # Separate blocks with a blank line (wrap_block already includes trailing newline)
        for blk in out_lines:
            f.write(blk + "\n")

    kept_final = len(out_lines)
    ast_ok_rate = (len(kept) / max(1, len(items)))
    avg_lines = mean([(c.count('\n') + 1) for c in deduped]) if deduped else 0.0
    avg_chars = mean([len(c) for c in deduped]) if deduped else 0.0
    if_hit_rate = if_hit / max(1, kept_final)

    report: Dict[str, Any] = {
        "total": total,
        "after_min_lines": len(items),
        "kept": kept_final,
        "dropped_short": dropped_short,
        "dropped_syntax": dropped_syntax,
        "dup_count": dup_count,
        "augmented": augmented,
        "augment_ratio_target": augment_ratio,
        "ast_ok_rate": ast_ok_rate,
        "avg_lines": avg_lines,
        "avg_chars": avg_chars,
        "if_hit_rate": if_hit_rate,
        "operator_counts": dict(op_counter),
        "notes": []
    }

    if ast_ok_rate < 0.95:
        report["notes"].append("AST OK rate < 0.95 — check indentation and snippet boundaries.")
    if if_hit_rate < 0.20:
        report["notes"].append("Few 'if' statements detected — consider sampling more condition-rich code.")
    if augmented == 0 and augment_ratio > 0.0:
        report["notes"].append("Augmentation requested but none applied — regex might not find 'if <cond>:'.")

    if report_json:
        os.makedirs(os.path.dirname(report_json), exist_ok=True)
        with open(report_json, 'w', encoding='utf-8') as rf:
            json.dump(report, rf, indent=2, ensure_ascii=False)

    print(f"\nWrote {kept_final} blocks to: {output_txt}")
    if report_json:
        print(f"Report written to: {report_json}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Build pre-training corpus from extracted functions (upgraded).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python build_pretrain_corpus.py \\
    --functions data/functions.jsonl \\
    --out corpus/pretrain_corpus.txt \\
    --report corpus/report.json \\
    --min-lines 5 --augment-ratio 0.08 --max-chars 4000 --shuffle --seed 0
        """
    )
    parser.add_argument('--functions', type=str, required=True, help='Path to functions.jsonl')
    parser.add_argument('--out', type=str, default='data/pretrain_corpus.txt', help='Output text file')
    parser.add_argument('--report', type=str, default='data/pretrain_report.json', help='JSON report path')
    parser.add_argument('--min-lines', type=int, default=5, help='Minimum lines per function')
    parser.add_argument('--augment-ratio', type=float, default=0.08, help='Ratio of snippets to augment (0–1)')
    parser.add_argument('--max-chars', type=int, default=4000, help='Truncate very long snippets to this many chars')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle after dedup')
    parser.add_argument('--max-samples', type=int, default=0, help='Limit number of snippets (0 = no limit)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    if not os.path.exists(args.functions):
        print(f"Error: Functions file not found: {args.functions}")
        sys.exit(1)

    report = build_pretrain_corpus(
        functions_jsonl=args.functions,
        output_txt=args.out,
        min_lines=args.min_lines,
        report_json=args.report,
        augment_ratio=args.augment_ratio,
        max_chars=args.max_chars,
        shuffle=args.shuffle,
        max_samples=args.max_samples,
        seed=args.seed
    )

    if report.get("kept", 0) == 0:
        print("\nError: No snippets written to corpus.")
        sys.exit(1)

    print("\n" + "="*60)
    print("Corpus ready for tokenizer training and pre-training!")
    print("="*60)


if __name__ == '__main__':
    main()