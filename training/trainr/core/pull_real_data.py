"""Pull real data from HuggingFace's The Stack for code/structured sub_types.

Streams code and structured samples from bigcode/the-stack (v1), filters by
size and format validators, and appends to golden_train.parquet with null
feature columns.

Usage:
    cd training && uv run python3 -m trainr.core.pull_real_data --phase code
    cd training && uv run python3 -m trainr.core.pull_real_data --phase structured
"""

import argparse
import csv
import io
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import polars as pl
from tqdm import tqdm

try:
    import datasets
except ImportError:
    datasets = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Feature columns (38 total) — must match existing parquet schema
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "line_length_cv",
    "char_entropy",
    "leading_whitespace_ratio",
    "tab_density",
    "sentence_punctuation_rate",
    "paragraph_break_rate",
    "alpha_ratio",
    "line_uniqueness",
    "short_line_ratio",
    "symbol_ratio",
    "delimiter_consistency",
    "json_brace_depth",
    "key_value_ratio",
    "xml_tag_ratio",
    "log_line_ratio",
    "comment_ratio",
    "numeric_field_ratio",
    "repetitive_structure_score",
    "hyphenated_line_break_ratio",
    "short_repeated_line_ratio",
    "page_number_density",
    "label_value_line_ratio",
    "table_fragment_score",
    "uppercase_header_ratio",
    "dictionary_word_ratio",
    "encoding_error_ratio",
    "repeated_ngram_ratio",
    "sentence_coherence_score",
    "avg_words_per_line",
    "operator_density",
    "inline_markup_count",
    "indentation_consistency",
    "markup_heading_ratio",
    "code_fence_density",
    "prose_paragraph_ratio",
    "semicolon_line_ending_ratio",
    "list_item_ratio",
    "parenthesis_density",
]

# ---------------------------------------------------------------------------
# Extension -> sub_type mapping
# ---------------------------------------------------------------------------

_EXT_TO_SUB_TYPE: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "sql": "sql",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
    "css": "css",
    "scss": "css",
    "html": "html",
    "htm": "html",
    "xml": "xml",
    "xsl": "xml",
    "xslt": "xml",
    "xsd": "xml",
    "svg": "xml",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "mk": "makefile",
    # config
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "ini": "ini",
    "cfg": "ini",
    # structured
    "jsonl": "jsonl",
    "ndjson": "jsonl",
    "json": "json",
    "csv": "csv",
    "tsv": "tsv",
    "properties": "key_value",
    "env": "key_value",
    "log": "log_lines",
    # prose
    "md": "markdown",
    "markdown": "markdown",
    "rst": "rst",
    "tex": "latex",
    "latex": "latex",
}


def sub_type_for_extension(ext: str) -> str | None:
    """Map a file extension (without dot) or filename to a sub_type.

    Returns None if the extension is not recognised.
    """
    return _EXT_TO_SUB_TYPE.get(ext.lower())


# ---------------------------------------------------------------------------
# Sub_type -> category mapping
# ---------------------------------------------------------------------------

SUB_TYPE_CATEGORY: dict[str, str] = {
    # code
    "python": "code",
    "javascript": "code",
    "typescript": "code",
    "rust": "code",
    "go": "code",
    "java": "code",
    "sql": "code",
    "shell": "code",
    "css": "code",
    "html": "code",
    "xml": "code",
    "dockerfile": "code",
    "makefile": "code",
    "unknown": "code",
    # config (category is "code" per types.rs — yaml/toml/ini are under Code)
    "yaml": "code",
    "toml": "code",
    "ini": "code",
    # structured
    "csv": "structured",
    "tsv": "structured",
    "jsonl": "structured",
    "json": "structured",
    "pipe_table": "structured",
    "fixed_width": "structured",
    "key_value": "structured",
    "log_lines": "structured",
    # prose
    "markdown": "prose",
    "rst": "prose",
    "latex": "prose",
}

# ---------------------------------------------------------------------------
# Phase -> sub_type groups
# ---------------------------------------------------------------------------

PHASE_SUB_TYPES: dict[str, list[str]] = {
    "code": [
        "python",
        "javascript",
        "typescript",
        "rust",
        "go",
        "java",
        "sql",
        "shell",
        "css",
        "html",
        "xml",
        "dockerfile",
        "makefile",
        "unknown",
    ],
    "config": ["yaml", "toml", "ini"],
    "structured": [
        "csv",
        "tsv",
        "jsonl",
        "json",
        "pipe_table",
        "fixed_width",
        "key_value",
        "log_lines",
    ],
    "prose": ["markdown", "rst", "latex"],
}

# ---------------------------------------------------------------------------
# Target counts per sub_type
# ---------------------------------------------------------------------------

TARGET_COUNTS: dict[str, int] = {
    "python": 3000,
    "javascript": 3000,
    "typescript": 3000,
    "rust": 3000,
    "go": 3000,
    "java": 3000,
    "sql": 2500,
    "shell": 2500,
    "css": 2500,
    "html": 2500,
    "xml": 2500,
    "dockerfile": 2500,
    "makefile": 2500,
    "unknown": 2500,
    # config
    "yaml": 1000,
    "toml": 1000,
    "ini": 500,
    # structured
    "jsonl": 1500,
    "json": 1500,
    "csv": 1000,
    "tsv": 1500,
    "pipe_table": 1000,
    "fixed_width": 1000,
    "key_value": 1000,
    "log_lines": 1000,
    # prose
    "markdown": 2500,
    "rst": 1500,
    "latex": 2000,
}

# ---------------------------------------------------------------------------
# Stack v1 data_dir mapping (sub_type -> HF data_dir path)
# ---------------------------------------------------------------------------

STACK_DATA_DIR: dict[str, str] = {
    "python": "data/python",
    "javascript": "data/javascript",
    "typescript": "data/typescript",
    "rust": "data/rust",
    "go": "data/go",
    "java": "data/java",
    "sql": "data/sql",
    "shell": "data/shell",
    "css": "data/css",
    "html": "data/html",
    "xml": "data/xml",
    "dockerfile": "data/dockerfile",
    "makefile": "data/makefile",
    "unknown": "data/c",  # use C as proxy for "unknown" code
    # config
    "yaml": "data/yaml",
    "toml": "data/toml",
    "ini": "data/ini",
    # structured (types with direct Stack sources)
    "json": "data/json",
    "csv": "data/csv",
    "tsv": "data/tsv",
    "jsonl": "data/json",  # filter JSONL from JSON stream
    # prose
    "markdown": "data/markdown",
    "rst": "data/restructuredtext",
    "latex": "data/tex",
}

# ---------------------------------------------------------------------------
# Format validators for structured sub_types
# ---------------------------------------------------------------------------


def validate_csv(text: str) -> bool:
    """Return True if *text* looks like valid CSV with consistent columns."""
    try:
        reader = csv.reader(io.StringIO(text))
        rows = [r for r in reader if r]
        if len(rows) < 2:
            return False
        col_count = len(rows[0])
        return col_count >= 2 and all(len(r) == col_count for r in rows[:20])
    except csv.Error:
        return False


def validate_tsv(text: str) -> bool:
    """Return True if *text* looks like valid TSV with consistent columns."""
    try:
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        rows = [r for r in reader if r]
        if len(rows) < 2:
            return False
        col_count = len(rows[0])
        return col_count >= 2 and all(len(r) == col_count for r in rows[:20])
    except csv.Error:
        return False


def validate_jsonl(text: str) -> bool:
    """Return True if *text* contains at least 2 valid JSON lines."""
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    try:
        for line in lines[:20]:
            json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def validate_json(text: str) -> bool:
    """Return True if *text* is valid JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def validate_pipe_table(text: str) -> bool:
    """Return True if *text* looks like a pipe-delimited table."""
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    pipe_lines = [line for line in lines if "|" in line]
    if len(pipe_lines) < len(lines) * 0.7:
        return False
    col_counts = [line.count("|") for line in pipe_lines[:20]]
    return len(set(col_counts)) <= 2


def validate_fixed_width(text: str) -> bool:
    """Return True if *text* has aligned multi-space gaps across lines.

    Checks that there is at least one character position that falls inside a
    multi-space gap on every sampled line, which indicates column alignment.
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        return False
    # For each line, collect the set of char positions that are inside a
    # multi-space gap (2+ consecutive spaces).
    gap_char_sets: list[set[int]] = []
    for line in lines[:20]:
        positions: set[int] = set()
        for m in re.finditer(r"  +", line):
            for pos in range(m.start(), m.end()):
                positions.add(pos)
        gap_char_sets.append(positions)
    if not gap_char_sets:
        return False
    common = gap_char_sets[0]
    for gp in gap_char_sets[1:]:
        common &= gp
    return len(common) >= 1


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "csv": validate_csv,
    "tsv": validate_tsv,
    "jsonl": validate_jsonl,
    "json": validate_json,
    "pipe_table": validate_pipe_table,
    "fixed_width": validate_fixed_width,
}


# ---------------------------------------------------------------------------
# Size filtering
# ---------------------------------------------------------------------------

_MIN_BYTES = 50
_MAX_BYTES = 50_000


def passes_size_filter(text: str) -> bool:
    """Return True if text is between 50 bytes and 50KB (inclusive)."""
    n = len(text.encode("utf-8", errors="replace"))
    return _MIN_BYTES <= n <= _MAX_BYTES


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def build_row(text: str, sub_type: str, category: str = "code") -> dict:
    """Build a single row dict matching the golden_train.parquet schema.

    Sets source and model to ``real/the-stack-v2`` and all 38 feature
    columns to None.  The *category* defaults to ``"code"`` but should
    be set to ``"prose"`` for prose sub_types.
    """
    row: dict = {
        "text": text,
        "category": category,
        "sub_type": sub_type,
        "source": "real/the-stack-v2",
        "model": "real/the-stack-v2",
    }
    for col in FEATURE_COLUMNS:
        row[col] = None
    return row


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


_FALLBACK_DATA_DIRS: dict[str, list[str]] = {
    "pipe_table": ["data/markdown", "data/restructuredtext"],
    "fixed_width": ["data/text", "data/csv"],
    "key_value": ["data/ini", "data/yaml"],
    "log_lines": ["data/text", "data/shell"],
    "tsv": ["data/csv"],
}


def _stream_sub_type(
    sub_type: str,
    target: int,
    seed: int = 42,
) -> list[dict]:
    """Stream samples from The Stack for a single sub_type.

    For sub_types with a direct ``STACK_DATA_DIR`` entry the stream is read
    from that directory.  For rare structured types without a direct source
    (pipe_table, fixed_width, key_value, log_lines) we fall back to broader
    data dirs and rely on the format validator to filter.
    """
    if datasets is None:
        raise ImportError("The 'datasets' package is required: pip install datasets")

    validator = VALIDATORS.get(sub_type)

    # Determine which Stack data_dirs to stream from.
    # Start with primary data_dir, then append fallbacks.
    data_dirs: list[str] = []
    if sub_type in STACK_DATA_DIR:
        data_dirs.append(STACK_DATA_DIR[sub_type])
    if sub_type in _FALLBACK_DATA_DIRS:
        for fb in _FALLBACK_DATA_DIRS[sub_type]:
            if fb not in data_dirs:
                data_dirs.append(fb)
    if not data_dirs:
        print(
            f"  WARNING: no Stack data_dir or fallback for {sub_type}, skipping",
            file=sys.stderr,
        )
        return []

    rows: list[dict] = []
    pbar = tqdm(total=target, desc=sub_type, unit="rows")

    for data_dir in data_dirs:
        if len(rows) >= target:
            break

        try:
            ds = datasets.load_dataset(
                "bigcode/the-stack",
                data_dir=data_dir,
                streaming=True,
                split="train",
                token=True,
            )
        except Exception as exc:
            print(
                f"  WARNING: failed to load {data_dir} for {sub_type}: {exc}",
                file=sys.stderr,
            )
            continue

        skipped = 0
        max_skips = target * 50  # stop after too many misses

        for item in ds.shuffle(seed=seed, buffer_size=5000):
            content = item.get("content", "")
            if not content or not passes_size_filter(content):
                skipped += 1
                if skipped >= max_skips:
                    break
                continue

            # Apply format validator if one exists
            if validator and not validator(content):
                skipped += 1
                if skipped >= max_skips:
                    break
                continue

            category = SUB_TYPE_CATEGORY.get(sub_type, "code")
            rows.append(
                build_row(text=content, sub_type=sub_type, category=category)
            )
            pbar.update(1)

            if len(rows) >= target:
                break

        if len(rows) < target and len(data_dirs) == 1:
            print(
                f"  WARNING: only found {len(rows)}/{target} for {sub_type} "
                f"from {data_dir}",
                file=sys.stderr,
            )

    pbar.close()

    if len(rows) < 500 and target >= 500:
        print(
            f"  WARNING: {sub_type} has only {len(rows)} rows (minimum 500 "
            f"recommended)",
            file=sys.stderr,
        )

    return rows


# ---------------------------------------------------------------------------
# Parquet append
# ---------------------------------------------------------------------------

_DEFAULT_PARQUET = "data/curated/train/golden_train.parquet"


def append_to_parquet(
    new_rows: list[dict],
    parquet_path: str = _DEFAULT_PARQUET,
) -> int:
    """Append new rows to the existing golden_train.parquet.

    Reads the existing parquet, creates a new DataFrame from new_rows with
    matching schema, vstacks them, and writes back.

    Returns the number of rows appended.
    """
    path = Path(parquet_path)
    existing = pl.read_parquet(path)

    new_df = pl.DataFrame(new_rows, schema=existing.schema)
    combined = pl.concat([existing, new_df], how="vertical")
    combined.write_parquet(path)

    return len(new_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Pull real code samples from The Stack on HuggingFace"
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_PARQUET,
        help=f"Output parquet path (default: {_DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--phase",
        choices=["code", "config", "prose", "structured", "all"],
        default="all",
        help="Which sub_type phase to pull (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without downloading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Determine which sub_types to pull
    if args.phase == "all":
        sub_types = []
        for phase_list in PHASE_SUB_TYPES.values():
            sub_types.extend(phase_list)
    else:
        if args.phase not in PHASE_SUB_TYPES:
            print(
                f"Phase '{args.phase}' has no sub_types configured. "
                f"Available: {', '.join(sorted(PHASE_SUB_TYPES.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)
        sub_types = PHASE_SUB_TYPES[args.phase]

    # Compute plan
    plan: list[tuple[str, int]] = []
    for st in sub_types:
        target = TARGET_COUNTS.get(st, 0)
        if target > 0:
            plan.append((st, target))

    total_target = sum(t for _, t in plan)

    print(f"Phase: {args.phase}")
    print(f"Sub-types: {len(plan)}")
    print(f"Total target: {total_target:,}")
    print(f"Output: {args.output}")
    print()

    for st, target in plan:
        print(f"  {st:15s} → {target:,} samples")

    if args.dry_run:
        print("\n[dry-run] Exiting without downloading.")
        return

    # Pull data
    all_rows: list[dict] = []
    for st, target in plan:
        print(f"\nPulling {st} ({target:,} target)...")
        rows = _stream_sub_type(st, target, seed=args.seed)
        print(f"  Collected {len(rows):,} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows collected. Exiting.")
        return

    # Append to parquet
    print(f"\nAppending {len(all_rows):,} rows to {args.output}...")
    appended = append_to_parquet(all_rows, parquet_path=args.output)
    print(f"Done. Appended {appended:,} rows.")


if __name__ == "__main__":
    main()
