#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "tqdm"]
# ///
"""Compute 18 structural text features and enrich a training CSV.

Ports the feature extraction logic from src/features.rs to Python.
Each feature is a standalone ``str -> float`` function with exact parity
to the Rust implementation, including edge cases and Unicode handling.

Usage:
    uv run featurize.py --input data/golden_raw.csv --output data/golden_featurized.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (match Rust features.rs)
# ---------------------------------------------------------------------------

SAMPLE_SIZE: int = 10_000
"""Maximum chars to sample from input text."""

UNIQUENESS_LINES: int = 500
"""Maximum lines to consider for line uniqueness."""

# Punctuation excluded from symbol_ratio (matches features.rs exactly):
# space, newline, tab, carriage-return, . , ; : ! ? - ' "
COMMON_PUNCTUATION: frozenset[str] = frozenset(' \n\t\r.,;:!?-\'"')


# ---------------------------------------------------------------------------
# Feature functions (18 total, matching features.rs)
# ---------------------------------------------------------------------------


def line_length_cv(text: str) -> float:
    """Coefficient of variation of line lengths: std_dev / mean."""
    lines = text.splitlines()
    if len(lines) < 2:
        return 0.0

    lengths = [len(line.encode("utf-8")) for line in lines]
    n = len(lengths)
    mean = sum(lengths) / n

    if mean < 1.0:
        return 0.0

    variance = sum((l - mean) ** 2 for l in lengths) / n
    std_dev = math.sqrt(variance)
    return std_dev / mean


def char_entropy(text: str) -> float:
    """Shannon entropy of character distribution (bits per character)."""
    if not text:
        return 0.0

    freq: Counter[str] = Counter(text)
    total = len(text)

    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0.0:
            entropy -= p * math.log2(p)

    return entropy


def leading_whitespace_ratio(text: str) -> float:
    """Fraction of lines starting with >2 columns of whitespace (tabs=4)."""
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    count = 0
    for line in lines:
        leading = 0
        for ch in line:
            if ch == '\t':
                leading += 4
            elif ch == ' ' or ch.isspace():
                leading += 1
            else:
                break
        if leading > 2:
            count += 1

    return count / n_lines


def tab_density(text: str) -> float:
    """Tab characters as a fraction of total characters."""
    total_chars = max(len(text), 1)
    tabs = text.count('\t')
    return tabs / total_chars


def sentence_punctuation_rate(text: str) -> float:
    """Sentence-ending punctuation (. ! ?) per word.

    Only counts punctuation preceded by an alphabetic character and followed
    by whitespace or end of text.  Avoids counting e.g. ``3.14``.
    """
    chars = list(text)
    word_count = max(len(text.split()), 1)
    count = 0

    for i, ch in enumerate(chars):
        if ch in '.!?':
            # Must be followed by whitespace or end of text
            if i + 1 >= len(chars) or chars[i + 1].isspace():
                # Previous char should be alphabetic
                if i > 0 and chars[i - 1].isalpha():
                    count += 1

    return count / word_count


def paragraph_break_rate(text: str) -> float:
    """Double-newline (paragraph break) frequency per line."""
    lines = text.splitlines()
    n_lines = max(len(lines), 1)
    breaks = text.count("\n\n")
    return breaks / n_lines


def alpha_ratio(text: str) -> float:
    """Fraction of characters that are alphanumeric or space."""
    total_chars = max(len(text), 1)
    alpha = sum(1 for ch in text if ch.isalnum() or ch == ' ')
    return alpha / total_chars


def line_uniqueness(text: str) -> float:
    """Ratio of unique lines to total lines within the first 500 lines."""
    lines = text.splitlines()
    sample = lines[:UNIQUENESS_LINES]
    sample_count = max(len(sample), 1)
    unique = len(set(sample))
    return unique / sample_count


def short_line_ratio(text: str) -> float:
    """Fraction of lines with 1-14 trimmed characters."""
    lines = text.splitlines()
    n_lines = max(len(lines), 1)
    short = sum(1 for line in lines if 1 <= len(line.strip()) <= 14)
    return short / n_lines


def symbol_ratio(text: str) -> float:
    """Fraction of non-alphanumeric chars, excluding common punctuation.

    Common punctuation excluded: space, newline, tab, CR, . , ; : ! ? - ' "
    Matches features.rs exactly.
    """
    total_chars = max(len(text), 1)
    symbols = sum(
        1 for ch in text if not ch.isalnum() and ch not in COMMON_PUNCTUATION
    )
    return symbols / total_chars


def delimiter_consistency(text: str) -> float:
    """Consistency of delimiter counts across lines.

    For each candidate delimiter (, | \\t ;), finds the mode count per line
    and returns the best (highest) fraction of lines matching that mode.
    Returns 0.0 if fewer than 3 lines.
    """
    lines = text.splitlines()
    if len(lines) < 3:
        return 0.0

    delimiters = [',', '|', '\t', ';']
    best = 0.0

    for delim in delimiters:
        counts = [line.count(delim) for line in lines]

        freq: Counter[int] = Counter(counts)
        if not freq:
            continue

        # Find mode frequency (most common count's frequency)
        mode_count, mode_freq = freq.most_common(1)[0]

        # Skip if mode is 0 occurrences (delimiter not present)
        if mode_count == 0:
            continue

        consistency = mode_freq / len(lines)
        if consistency > best:
            best = consistency

    return best


def json_brace_depth(text: str) -> float:
    """Fraction of JSON brace/bracket characters ({, }, [, ]) in text."""
    total_chars = max(len(text), 1)
    braces = frozenset('{}[]')
    count = sum(1 for ch in text if ch in braces)
    return count / total_chars


def key_value_ratio(text: str) -> float:
    """Fraction of lines with key-value patterns (key: value or key=value)."""
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    count = 0
    for line in lines:
        trimmed = line.strip()
        pos = trimmed.find(": ")
        if pos != -1:
            if pos > 0 and any(ch.isalnum() for ch in trimmed[:pos]):
                count += 1
                continue

        pos = trimmed.find("=")
        if pos != -1:
            if (
                pos > 0
                and pos + 1 < len(trimmed)
                and any(ch.isalnum() for ch in trimmed[:pos])
            ):
                count += 1

    return count / n_lines


def xml_tag_ratio(text: str) -> float:
    """Fraction of lines containing XML/HTML tags (<tag or </tag)."""
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    count = 0
    for line in lines:
        chars = list(line)
        found = False
        for i, ch in enumerate(chars):
            if ch == '<' and i + 1 < len(chars):
                nxt = chars[i + 1]
                if nxt.isalpha():
                    found = True
                    break
                if nxt == '/' and i + 2 < len(chars) and chars[i + 2].isalpha():
                    found = True
                    break
        if found:
            count += 1

    return count / n_lines


def log_line_ratio(text: str) -> float:
    """Fraction of lines starting with timestamp-like patterns.

    Detects: ``\\d{4}-\\d{2}-\\d{2}``, ``\\d{2}:\\d{2}:\\d{2}``,
    or ``[\\d{4}`` (bracket timestamps).
    """
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    count = 0
    for line in lines:
        trimmed = line.strip()
        chars = list(trimmed)
        n = len(chars)

        # Pattern: \d{4}-\d{2}-\d{2}
        if (
            n >= 10
            and chars[0].isascii() and chars[0].isdigit()
            and chars[1].isascii() and chars[1].isdigit()
            and chars[2].isascii() and chars[2].isdigit()
            and chars[3].isascii() and chars[3].isdigit()
            and chars[4] == '-'
            and chars[5].isascii() and chars[5].isdigit()
            and chars[6].isascii() and chars[6].isdigit()
            and chars[7] == '-'
            and chars[8].isascii() and chars[8].isdigit()
            and chars[9].isascii() and chars[9].isdigit()
        ):
            count += 1
            continue

        # Pattern: \d{2}:\d{2}:\d{2}
        if (
            n >= 8
            and chars[0].isascii() and chars[0].isdigit()
            and chars[1].isascii() and chars[1].isdigit()
            and chars[2] == ':'
            and chars[3].isascii() and chars[3].isdigit()
            and chars[4].isascii() and chars[4].isdigit()
            and chars[5] == ':'
            and chars[6].isascii() and chars[6].isdigit()
            and chars[7].isascii() and chars[7].isdigit()
        ):
            count += 1
            continue

        # Pattern: [\d{4} (bracket timestamp)
        if (
            n >= 5
            and chars[0] == '['
            and chars[1].isascii() and chars[1].isdigit()
            and chars[2].isascii() and chars[2].isdigit()
            and chars[3].isascii() and chars[3].isdigit()
            and chars[4].isascii() and chars[4].isdigit()
        ):
            count += 1
            continue

    return count / n_lines


def comment_ratio(text: str) -> float:
    """Fraction of lines starting with comment markers (# // /* -- %)."""
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    count = 0
    for line in lines:
        trimmed = line.strip()
        if (
            trimmed.startswith('#')
            or trimmed.startswith('//')
            or trimmed.startswith('/*')
            or trimmed.startswith('--')
            or trimmed.startswith('%')
        ):
            count += 1

    return count / n_lines


def numeric_field_ratio(text: str) -> float:
    """Fraction of whitespace-delimited tokens that parse as numbers.

    Strips commas before parsing (e.g. ``1,000`` -> ``1000``).
    """
    tokens = text.split()
    if not tokens:
        return 0.0

    numeric = 0
    for token in tokens:
        stripped = token.replace(',', '')
        try:
            float(stripped)
            numeric += 1
        except ValueError:
            pass

    return numeric / len(tokens)


def repetitive_structure_score(text: str) -> float:
    """Fraction of lines sharing the most common 'shape' (first 20).

    Shape = (number of whitespace tokens, tuple of delimiter presence).
    Returns 0.0 if fewer than 3 lines.
    """
    lines = text.splitlines()
    if len(lines) < 3:
        return 0.0

    sample_size = min(len(lines), 20)
    sample = lines[:sample_size]

    delimiters = [',', '|', '\t', ';']

    shapes: list[tuple[int, tuple[bool, ...]]] = []
    for line in sample:
        token_count = len(line.split())
        delim_present = tuple(delim in line for delim in delimiters)
        shapes.append((token_count, delim_present))

    freq: Counter[tuple[int, tuple[bool, ...]]] = Counter(shapes)
    max_freq = freq.most_common(1)[0][1] if freq else 0

    return max_freq / sample_size


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FEATURES: dict[str, Callable[[str], float]] = {
    "line_length_cv": line_length_cv,
    "char_entropy": char_entropy,
    "leading_whitespace_ratio": leading_whitespace_ratio,
    "tab_density": tab_density,
    "sentence_punctuation_rate": sentence_punctuation_rate,
    "paragraph_break_rate": paragraph_break_rate,
    "alpha_ratio": alpha_ratio,
    "line_uniqueness": line_uniqueness,
    "short_line_ratio": short_line_ratio,
    "symbol_ratio": symbol_ratio,
    "delimiter_consistency": delimiter_consistency,
    "json_brace_depth": json_brace_depth,
    "key_value_ratio": key_value_ratio,
    "xml_tag_ratio": xml_tag_ratio,
    "log_line_ratio": log_line_ratio,
    "comment_ratio": comment_ratio,
    "numeric_field_ratio": numeric_field_ratio,
    "repetitive_structure_score": repetitive_structure_score,
}


def extract_all(text: str) -> dict[str, float]:
    """Extract all 18 features from *text*, sampling the first 10k chars."""
    if not text:
        return {name: 0.0 for name in FEATURES}

    sample = text[:SAMPLE_SIZE]
    return {name: fn(sample) for name, fn in FEATURES.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 18 structural text features and enrich a CSV."
    )
    parser.add_argument(
        "--input",
        default="data/golden_raw.csv",
        help="Input CSV path (default: data/golden_raw.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/golden_featurized.csv",
        help="Output CSV path (default: data/golden_featurized.csv)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pl.read_csv(str(input_path))

    texts = df.get_column("text").to_list()

    # Compute features for each row with progress bar
    rows: list[dict[str, float]] = []
    for text in tqdm(texts, desc="Featurizing"):
        rows.append(extract_all(text if text is not None else ""))

    features_df = pl.DataFrame(rows, schema={name: pl.Float64 for name in FEATURES})
    result = pl.concat([df, features_df], how="horizontal")
    result.write_csv(str(output_path))

    print(f"Wrote {len(result)} rows x {len(result.columns)} columns to {output_path}")


if __name__ == "__main__":
    main()
