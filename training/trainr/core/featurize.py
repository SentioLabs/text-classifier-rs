#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "tqdm"]
# ///
"""Compute 40 structural text features and enrich a training dataset.

Ports the feature extraction logic from src/features.rs to Python.
Each feature is a standalone ``str -> float`` function with exact parity
to the Rust implementation, including edge cases and Unicode handling.

Usage:
    uv run featurize.py --input data/curated/train/golden_raw.parquet --output data/curated/train/golden_featurized.parquet
"""

from __future__ import annotations

import argparse
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

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

# Typographic symbols excluded from symbol_ratio (matches features.rs):
# en dash, em dash, bullet, degree, copyright, registered, trademark,
# horizontal ellipsis, multiplication sign, division sign
TYPOGRAPHIC_SYMBOLS: frozenset[str] = frozenset('–—•°©®™…×÷')

# Unicode decorative ranges excluded from symbol_ratio (matches features.rs):
# Arrows (U+2190-21FF), Box Drawing + Block Elements (U+2500-259F),
# Geometric Shapes + Misc Symbols + Dingbats (U+25A0-27BF)
UNICODE_DECORATIVE_RANGES: list[tuple[int, int]] = [
    (0x2190, 0x21FF),
    (0x2500, 0x259F),
    (0x25A0, 0x27BF),
]


def _is_unicode_decorative(ch: str) -> bool:
    """Check if a character falls in a Unicode decorative range."""
    cp = ord(ch)
    for lo, hi in UNICODE_DECORATIVE_RANGES:
        if lo <= cp <= hi:
            return True
    return False


_WORDLIST: frozenset[str] | None = None


def _load_wordlist() -> frozenset[str]:
    global _WORDLIST
    if _WORDLIST is None:
        path = Path(__file__).parent.parent.parent / "data" / "manual" / "wordlist.txt"
        _WORDLIST = frozenset(path.read_text().splitlines())
    return _WORDLIST


# ---------------------------------------------------------------------------
# Feature functions (28 total — 24 structural + 4 content-level)
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

    Excludes common punctuation (space, newline, tab, CR, . , ; : ! ? - ' "),
    Unicode decorative ranges (arrows, box drawing, geometric shapes, etc.),
    and typographic symbols (dashes, bullets, copyright, etc.).
    Matches features.rs exactly.
    """
    total_chars = max(len(text), 1)
    symbols = sum(
        1
        for ch in text
        if not ch.isalnum()
        and ch not in COMMON_PUNCTUATION
        and not _is_unicode_decorative(ch)
        and ch not in TYPOGRAPHIC_SYMBOLS
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


def hyphenated_line_break_ratio(text: str) -> float:
    """Fraction of line transitions that look like hyphenated line wraps."""
    lines = text.splitlines()
    n_transitions = max(len(lines) - 1, 1)

    count = 0
    for i in range(len(lines) - 1):
        current = lines[i].rstrip()
        if not current.endswith("-"):
            continue

        nxt = lines[i + 1].lstrip()
        if nxt and nxt[0].isalpha() and nxt[0].islower():
            count += 1

    return count / n_transitions


def short_repeated_line_ratio(text: str) -> float:
    """Fraction of short lines that are repeated (header/footer signal)."""
    short_lines = [line.strip() for line in text.splitlines() if 1 <= len(line.strip()) <= 40]
    if not short_lines:
        return 0.0

    freq: Counter[str] = Counter(short_lines)
    repeated_instances = sum(count for count in freq.values() if count > 1)
    return repeated_instances / len(short_lines)


_PAGE_NUMBER_ONLY_RE = re.compile(r"^\d{1,4}$")
_PAGE_N_OF_M_RE = re.compile(r"(?i)^page\s+\d{1,4}(\s+of\s+\d{1,4})?$")
_PAGE_FRACTION_RE = re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$")


def page_number_density(text: str) -> float:
    """Fraction of lines that resemble standalone page number markers."""
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    count = 0
    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            continue
        if _PAGE_NUMBER_ONLY_RE.fullmatch(trimmed):
            count += 1
            continue
        if _PAGE_N_OF_M_RE.fullmatch(trimmed):
            count += 1
            continue
        if _PAGE_FRACTION_RE.fullmatch(trimmed):
            count += 1

    return count / n_lines


_LABEL_VALUE_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9 _/().]{0,30}\s*[:\-]\s+\S"
)


def label_value_line_ratio(text: str) -> float:
    """Fraction of lines with extraction-style label/value formatting."""
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    count = 0
    for line in lines:
        if _LABEL_VALUE_RE.match(line.strip()):
            count += 1

    return count / n_lines


_SPACED_COLUMNS_RE = re.compile(r"\S+\s{2,}\S+")


def table_fragment_score(text: str) -> float:
    """Fraction of non-empty lines that look table-like."""
    non_empty = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not non_empty:
        return 0.0

    score = 0
    for line in non_empty:
        delimiter_hits = line.count(",") + line.count("|") + line.count("\t") + line.count(";")
        if delimiter_hits >= 2 or _SPACED_COLUMNS_RE.search(line):
            score += 1

    return score / len(non_empty)


def uppercase_header_ratio(text: str) -> float:
    """Fraction of lines that look like uppercase section headers."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    count = 0
    for line in lines:
        if len(line) > 80:
            continue
        alpha = [ch for ch in line if ch.isalpha()]
        if len(alpha) < 3:
            continue
        upper_alpha = sum(1 for ch in alpha if ch.isupper())
        if upper_alpha / len(alpha) >= 0.8 and line[-1] not in ".!?":
            count += 1

    return count / len(lines)


def dictionary_word_ratio(text: str) -> float:
    """Fraction of whitespace-delimited tokens found in a dictionary wordlist.

    Tokens are lowercased and stripped of leading/trailing punctuation before
    lookup.  Returns 0.0 for empty text or no valid tokens.
    """
    if not text:
        return 0.0

    wordlist = _load_wordlist()
    tokens = text.split()
    valid: list[str] = []
    for token in tokens:
        stripped = token.lower().strip(string.punctuation)
        if stripped:
            valid.append(stripped)

    if not valid:
        return 0.0

    found = sum(1 for w in valid if w in wordlist)
    return found / len(valid)


def encoding_error_ratio(text: str) -> float:
    """Fraction of characters that are encoding errors or mojibake sequences.

    Counts U+FFFD replacement characters and common mojibake byte sequences
    that result from UTF-8 text misinterpreted as Latin-1.
    """
    if not text:
        return 0.0

    fffd_count = text.count('\ufffd')

    mojibake_sequences = [
        'Ã©', 'Ã¨', 'Ã¼', 'Ã¶', 'Ã¤', 'Â°', 'Â©',
        '\u00e2\u0080\u0093',  # â€"
        '\u00e2\u0080\u0099',  # â€™
        '\u00e2\u0080\u009c',  # â€œ
        '\u00e2\u0080\u009d',  # â€\x9d
    ]
    mojibake_count = sum(text.count(seq) for seq in mojibake_sequences)

    return (fffd_count + mojibake_count) / max(len(text), 1)


def repeated_ngram_ratio(text: str) -> float:
    """Fraction of unique 3-gram types that appear more than once.

    Splits text into words, forms 3-grams, and returns the ratio of repeated
    3-gram types to total unique 3-gram types.  Returns 0.0 if fewer than 3
    words.
    """
    words = text.split()
    if len(words) < 3:
        return 0.0

    ngrams: list[tuple[str, ...]] = []
    for i in range(len(words) - 2):
        ngrams.append((words[i], words[i + 1], words[i + 2]))

    freq: Counter[tuple[str, ...]] = Counter(ngrams)
    total_unique = len(freq)
    if total_unique == 0:
        return 0.0

    repeated = sum(1 for count in freq.values() if count > 1)
    return repeated / total_unique


def sentence_coherence_score(text: str) -> float:
    """Fraction of non-empty lines that look like proper sentences.

    A line is considered a proper sentence if it starts with an uppercase
    letter and ends with sentence-ending punctuation (. ! ?).
    Empty lines are ignored.  Returns 0.0 for empty text.
    """
    if not text:
        return 0.0

    lines = text.splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]

    if not non_empty:
        return 0.0

    proper = 0
    for line in non_empty:
        if line[0].isupper() and line[-1] in '.!?':
            proper += 1

    return proper / len(non_empty)


# ---------------------------------------------------------------------------
# New features (10 additional — 29-38)
# ---------------------------------------------------------------------------

# Programming operators (multi-char only — single chars overlap with prose).
_CODE_OPERATORS: list[str] = [
    "==", "!=", ">=", "<=", "&&", "||", "=>", "->",
    "+=", "-=", "*=", "/=", "**", "<<", ">>", "::",
]

_MARKUP_HEADING_RE = re.compile(
    r"^#{1,6}\s+\S"  # Markdown: # Heading
)
_RST_UNDERLINE_CHARS = frozenset("=-*~^\"'`")

_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*•]\s+\S|\d+[.)]\s+\S)"
)

_INLINE_MARKUP_RE = re.compile(
    r"\*\*[^*]+\*\*"    # **bold**
    r"|`[^`]+`"          # `code`
    r"|\*[^*\s][^*]*\*"  # *italic*
    r"|\[[^\]]+\]\([^)]+\)"  # [link](url)
)


def avg_words_per_line(text: str) -> float:
    """Average whitespace-delimited tokens per non-empty line.

    Prose: ~10-80 words/line (long paragraphs).
    Code: ~3-8 words/line (short statements).
    Structured: ~2-10 words/line (key-value pairs).
    """
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return 0.0
    total_words = sum(len(line.split()) for line in non_empty)
    return total_words / len(non_empty)


def operator_density(text: str) -> float:
    """Programming operators per 1000 characters.

    Counts multi-character operators (==, !=, >=, &&, etc.) that are
    strong code signals. Single-char operators like + and - overlap
    with prose and are excluded.
    """
    total_chars = max(len(text), 1)
    count = sum(text.count(op) for op in _CODE_OPERATORS)
    return count / total_chars * 1000


def inline_markup_count(text: str) -> float:
    """Count of inline markup patterns per 1000 characters.

    Detects **bold**, `code`, *italic*, and [link](url) patterns that
    signal Markdown/RST prose rather than code.
    """
    total_chars = max(len(text), 1)
    matches = _INLINE_MARKUP_RE.findall(text)
    return len(matches) / total_chars * 1000


def indentation_consistency(text: str) -> float:
    """Whether indentation follows a regular pattern (0.0-1.0).

    Code has consistent indentation (multiples of 2 or 4 spaces).
    Prose has no indentation. Structured data has shallow/irregular indentation.
    Returns the fraction of indented lines whose indent level is a multiple
    of the detected base indent (2, 3, or 4 spaces).
    """
    lines = text.splitlines()
    indent_levels: list[int] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or line == stripped:
            continue
        indent = len(line) - len(stripped)
        # Convert tabs to 4 spaces
        indent = line[:len(line) - len(stripped)].replace('\t', '    ')
        indent_levels.append(len(indent))

    if len(indent_levels) < 3:
        return 0.0

    # Try base indents of 2, 3, 4
    best_consistency = 0.0
    for base in [2, 3, 4]:
        consistent = sum(1 for i in indent_levels if i % base == 0)
        ratio = consistent / len(indent_levels)
        if ratio > best_consistency:
            best_consistency = ratio

    return best_consistency


def markup_heading_ratio(text: str) -> float:
    """Fraction of non-empty lines that are Markdown/RST section headings.

    Detects:
    - Markdown headings: lines starting with 1-6 # followed by space
    - RST underlines: lines consisting entirely of =, -, *, ~, ^, etc.
    """
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return 0.0

    count = 0
    for line in non_empty:
        stripped = line.strip()
        # Markdown heading
        if _MARKUP_HEADING_RE.match(stripped):
            count += 1
            continue
        # RST underline (all same char, length >= 3)
        if len(stripped) >= 3 and all(c in _RST_UNDERLINE_CHARS for c in stripped):
            count += 1

    return count / len(non_empty)


def code_fence_density(text: str) -> float:
    """Fraction of lines inside triple-backtick fenced code blocks.

    Prose documents *contain* code fences (the fenced lines are a minority).
    Actual source code does not use fences.
    """
    lines = text.splitlines()
    if not lines:
        return 0.0

    inside = False
    fenced_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            inside = not inside
            continue
        if inside:
            fenced_lines += 1

    return fenced_lines / len(lines)


def prose_paragraph_ratio(text: str) -> float:
    """Fraction of lines in multi-sentence paragraph blocks.

    A paragraph block is 3+ consecutive non-empty lines of >40 characters
    without structural delimiters (|, tab, {, }).
    Prose has these; code and structured data do not.
    """
    lines = text.splitlines()
    if not lines:
        return 0.0

    structural_chars = frozenset('|{}\t')
    para_lines = 0
    streak = 0

    for line in lines:
        stripped = line.strip()
        is_paragraph_line = (
            len(stripped) > 40
            and not any(c in stripped for c in structural_chars)
            and not stripped.startswith('#')
        )
        if is_paragraph_line:
            streak += 1
        else:
            if streak >= 3:
                para_lines += streak
            streak = 0

    if streak >= 3:
        para_lines += streak

    return para_lines / len(lines)


def semicolon_line_ending_ratio(text: str) -> float:
    """Fraction of non-empty lines ending with semicolons.

    Strong code signal (C, Java, JavaScript, CSS, SQL).
    Prose and structured data almost never end lines with semicolons.
    """
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return 0.0

    count = sum(1 for line in non_empty if line.rstrip().endswith(';'))
    return count / len(non_empty)


def list_item_ratio(text: str) -> float:
    """Fraction of non-empty lines that are list items.

    Detects: - item, * item, bullet item, 1. item, a) item.
    Prose and Markdown have lists; code and structured data typically do not.
    """
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return 0.0

    count = sum(1 for line in non_empty if _LIST_ITEM_RE.match(line))
    return count / len(non_empty)


def parenthesis_density(text: str) -> float:
    """Parentheses per 1000 characters.

    Code has high parenthesis density (function calls, conditionals).
    Prose has moderate. Structured data has low.
    """
    total_chars = max(len(text), 1)
    count = text.count('(') + text.count(')')
    return count / total_chars * 1000


def section_header_ratio(text: str) -> float:
    """Fraction of non-empty lines that are INI-style section headers: [section.name]."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    pattern = re.compile(r'^\[[\w\s.\-]+\]$')
    count = sum(1 for l in lines if pattern.match(l.strip()))
    return count / len(lines)


def json_lines_ratio(text: str) -> float:
    """Fraction of non-empty lines that look like JSON lines ({...})."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    count = sum(1 for l in lines if l.startswith('{') and l.endswith('}'))
    return count / len(lines)


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
    "hyphenated_line_break_ratio": hyphenated_line_break_ratio,
    "short_repeated_line_ratio": short_repeated_line_ratio,
    "page_number_density": page_number_density,
    "label_value_line_ratio": label_value_line_ratio,
    "table_fragment_score": table_fragment_score,
    "uppercase_header_ratio": uppercase_header_ratio,
    "dictionary_word_ratio": dictionary_word_ratio,
    "encoding_error_ratio": encoding_error_ratio,
    "repeated_ngram_ratio": repeated_ngram_ratio,
    "sentence_coherence_score": sentence_coherence_score,
    # New features (v2)
    "avg_words_per_line": avg_words_per_line,
    "operator_density": operator_density,
    "inline_markup_count": inline_markup_count,
    "indentation_consistency": indentation_consistency,
    "markup_heading_ratio": markup_heading_ratio,
    "code_fence_density": code_fence_density,
    "prose_paragraph_ratio": prose_paragraph_ratio,
    "semicolon_line_ending_ratio": semicolon_line_ending_ratio,
    "list_item_ratio": list_item_ratio,
    "parenthesis_density": parenthesis_density,
    "section_header_ratio": section_header_ratio,
    "json_lines_ratio": json_lines_ratio,
}


def _normalize_escaped_newlines(text: str) -> str:
    """Replace literal backslash-n sequences with real newlines.

    Text arriving from JSON, CSV, or API payloads sometimes contains
    literal ``\\n`` instead of actual newline characters, which breaks
    all line-based feature extraction.
    """
    return text.replace("\\n", "\n")


def extract_all(text: str) -> dict[str, float]:
    """Extract all features from *text*, sampling the first 10k chars."""
    if not text:
        return {name: 0.0 for name in FEATURES}

    sample = _normalize_escaped_newlines(text[:SAMPLE_SIZE])
    return {name: fn(sample) for name, fn in FEATURES.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 40 structural text features and enrich a dataset."
    )
    parser.add_argument(
        "--input",
        default="data/curated/train/golden_raw.parquet",
        help="Input parquet path (default: data/curated/train/golden_raw.parquet)",
    )
    parser.add_argument(
        "--output",
        default="data/curated/train/golden_featurized.parquet",
        help="Output parquet path (default: data/curated/train/golden_featurized.parquet)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import polars as pl
    from tqdm import tqdm

    args = parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pl.read_parquet(str(input_path))

    texts = df.get_column("text").to_list()

    # Compute features for each row with progress bar
    rows: list[dict[str, float]] = []
    for text in tqdm(texts, desc="Featurizing"):
        rows.append(extract_all(text if text is not None else ""))

    features_df = pl.DataFrame(rows, schema={name: pl.Float32 for name in FEATURES})
    result = pl.concat([df, features_df], how="horizontal")
    result.write_parquet(str(output_path))

    print(f"Wrote {len(result)} rows x {len(result.columns)} columns to {output_path}")



if __name__ == "__main__":
    main()
