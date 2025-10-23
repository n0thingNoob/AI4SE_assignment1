#!/usr/bin/env python3
"""
CSV to JSONL converter for if-condition prediction tasks.

This script converts CSV files containing Python code to JSONL format
suitable for if-condition prediction tasks. It extracts if conditions
from Python code and creates training samples where conditions are
replaced with <IFMASK> tokens.
"""

import argparse
import ast
import csv
import json
import sys
from typing import List, Tuple, Optional, Dict, Any, NamedTuple
from io import StringIO


class IfCondition(NamedTuple):
    """Represents an extracted if condition with position information."""
    condition_text: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


class IfExtractor(ast.NodeVisitor):
    """AST visitor to extract if conditions from Python code."""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.if_conditions: List[IfCondition] = []
        self.lines = source_code.splitlines()
    
    def visit_If(self, node: ast.If) -> None:
        """Visit an if statement and extract its condition."""
        condition_text = self._extract_condition_text(node)
        if condition_text:
            self.if_conditions.append(IfCondition(
                condition_text=condition_text,
                start_line=node.test.lineno,
                start_col=node.test.col_offset,
                end_line=getattr(node.test, 'end_lineno', node.test.lineno),
                end_col=getattr(node.test, 'end_col_offset', node.test.col_offset + len(condition_text))
            ))
        self.generic_visit(node)
    
    def _extract_condition_text(self, node: ast.If) -> Optional[str]:
        """Extract the source text of an if condition."""
        test_node = node.test
        
        # Method 1: Use end position fields (Python 3.8+)
        if hasattr(test_node, 'end_lineno') and hasattr(test_node, 'end_col_offset'):
            try:
                start_line = test_node.lineno - 1  # Convert to 0-based
                start_col = test_node.col_offset
                end_line = test_node.end_lineno - 1  # Convert to 0-based
                end_col = test_node.end_col_offset
                
                if start_line == end_line:
                    # Single line condition
                    line = self.lines[start_line]
                    return line[start_col:end_col]
                else:
                    # Multi-line condition
                    lines = []
                    for i in range(start_line, end_line + 1):
                        if i < len(self.lines):
                            if i == start_line:
                                lines.append(self.lines[i][start_col:])
                            elif i == end_line:
                                lines.append(self.lines[i][:end_col])
                            else:
                                lines.append(self.lines[i])
                    return '\n'.join(lines)
            except (IndexError, AttributeError):
                pass
        
        # Method 2: Use ast.get_source_segment (Python 3.8+)
        try:
            if hasattr(ast, 'get_source_segment'):
                return ast.get_source_segment(self.source_code, test_node)
        except (AttributeError, TypeError):
            pass
        
        # Method 3: Fallback for single-line if statements
        try:
            if test_node.lineno == test_node.end_lineno if hasattr(test_node, 'end_lineno') else True:
                line = self.lines[test_node.lineno - 1]
                if_start = line.find('if ', test_node.col_offset)
                if if_start != -1:
                    colon_pos = line.find(':', if_start)
                    if colon_pos != -1:
                        return line[if_start + 3:colon_pos].strip()
        except (IndexError, AttributeError):
            pass
        
        return None


def extract_if_conditions(code: str) -> List[IfCondition]:
    """
    Extract all if conditions from Python code.
    
    Args:
        code: Python source code as string
        
    Returns:
        List of IfCondition objects sorted by position
    """
    try:
        tree = ast.parse(code)
        extractor = IfExtractor(code)
        extractor.visit(tree)
        
        # Sort by position (line, column)
        return sorted(extractor.if_conditions, key=lambda x: (x.start_line, x.start_col))
    except SyntaxError:
        return []


def create_sample(code: str, condition: IfCondition, sample_id: str, if_index: int) -> Dict[str, Any]:
    """
    Create a training sample by replacing if condition with <IFMASK>.
    
    Args:
        code: Original Python code
        condition: The if condition to replace
        sample_id: Unique identifier for the sample
        if_index: Index of this if statement in the code
        
    Returns:
        Dictionary representing the training sample
    """
    lines = code.splitlines()
    
    # Replace the condition with <IFMASK>
    modified_lines = []
    
    if condition.start_line == condition.end_line:
        # Single line condition
        for i, line in enumerate(lines):
            if i == condition.start_line - 1:  # Convert to 0-based
                # Replace the condition part directly using position information
                before_condition = line[:condition.start_col]
                after_condition = line[condition.end_col:]
                modified_line = before_condition + '<IFMASK>' + after_condition
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)
    else:
        # Multi-line condition
        for i, line in enumerate(lines):
            if i == condition.start_line - 1:  # Convert to 0-based
                # First line: replace from condition start to end of line
                if_start = line.find('if ', condition.start_col)
                if if_start != -1:
                    before_if = line[:if_start + 3]
                    after_condition = line[condition.end_col:]
                    modified_line = before_if + '<IFMASK>' + after_condition
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)
            elif condition.start_line < i + 1 < condition.end_line:
                # Middle lines: replace entire line with empty line (preserving indentation)
                indent = len(line) - len(line.lstrip())
                modified_lines.append(' ' * indent)
            elif i == condition.end_line - 1:  # Convert to 0-based
                # Last line: replace from start to condition end
                before_condition = line[:condition.start_col]
                after_condition = line[condition.end_col:]
                modified_line = before_condition + '<IFMASK>' + after_condition
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)
    
    # Remove trailing colon from expected condition
    expected_condition = condition.condition_text.rstrip(':').strip()
    
    return {
        "input": '\n'.join(modified_lines),
        "expected_condition": expected_condition,
        "meta": {
            "id": sample_id,
            "if_index": if_index,
            "pos": {
                "lineno": condition.start_line,
                "col": condition.start_col,
                "end_lineno": condition.end_line,
                "end_col": condition.end_col
            }
        }
    }


def process_csv_to_jsonl(
    input_path: str,
    output_path: str,
    mode: str = "first",
    id_column: str = "id",
    code_column: str = "code",
    skip_on_parse_error: bool = True
) -> None:
    """
    Process CSV file and convert to JSONL format.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output JSONL file
        mode: "first" to extract only first if, "all" to extract all ifs
        id_column: Name of the ID column
        code_column: Name of the code column
        skip_on_parse_error: Whether to skip rows with parse errors
    """
    rows_in = 0
    samples_out = 0
    skipped = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            
            # Check required columns
            if code_column not in reader.fieldnames:
                print(f"Error: Required column '{code_column}' not found in CSV", file=sys.stderr)
                print(f"Available columns: {', '.join(reader.fieldnames)}", file=sys.stderr)
                sys.exit(2)
            
            for row in reader:
                rows_in += 1
                
                try:
                    code = row[code_column]
                    sample_id = row.get(id_column, str(rows_in))
                    
                    # Extract if conditions
                    conditions = extract_if_conditions(code)
                    
                    if not conditions:
                        skipped += 1
                        continue
                    
                    # Process conditions based on mode
                    if mode == "first":
                        conditions = [conditions[0]]
                    elif mode == "all":
                        pass  # Use all conditions
                    else:
                        print(f"Warning: Unknown mode '{mode}', using 'first'", file=sys.stderr)
                        conditions = [conditions[0]]
                    
                    # Create samples
                    for i, condition in enumerate(conditions):
                        sample = create_sample(code, condition, sample_id, i)
                        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        samples_out += 1
                        
                except Exception as e:
                    if skip_on_parse_error:
                        skipped += 1
                        if skipped <= 10:  # Only print first 10 errors to avoid spam
                            print(f"Warning: Skipping row {rows_in}: {str(e)}", file=sys.stderr)
                    else:
                        raise
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    print(f"[DONE] rows_in={rows_in}, samples_out={samples_out}, skipped={skipped}")
    print(f"[OUT] {output_path}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert CSV with Python code to JSONL for if-condition prediction tasks"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file path"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file path"
    )
    
    parser.add_argument(
        "--mode",
        choices=["first", "all"],
        default="first",
        help="Extraction mode: 'first' for only first if, 'all' for all ifs (default: first)"
    )
    
    parser.add_argument(
        "--id-column",
        default="id",
        help="Name of the ID column (default: id)"
    )
    
    parser.add_argument(
        "--code-column",
        default="code",
        help="Name of the code column (default: code)"
    )
    
    parser.add_argument(
        "--skip-on-parse-error",
        action="store_true",
        default=True,
        help="Skip rows with parse errors (default: True)"
    )
    
    args = parser.parse_args()
    
    process_csv_to_jsonl(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        id_column=args.id_column,
        code_column=args.code_column,
        skip_on_parse_error=args.skip_on_parse_error
    )


if __name__ == "__main__":
    main()
