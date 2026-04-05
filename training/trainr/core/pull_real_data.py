"""Pull real data from HuggingFace's The Stack for code sub_types.

Streams code samples from bigcode/the-stack (v1), filters by size,
and appends to golden_train.parquet with null feature columns.

Usage:
    cd training && uv run python3 -m trainr.core.pull_real_data --phase code
"""

import argparse
import sys
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
    # prose
    "markdown": "data/markdown",
    "rst": "data/restructuredtext",
    "latex": "data/tex",
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


def _stream_sub_type(
    sub_type: str,
    target: int,
    seed: int = 42,
) -> list[dict]:
    """Stream samples from The Stack for a single sub_type."""
    if datasets is None:
        raise ImportError("The 'datasets' package is required: pip install datasets")

    data_dir = STACK_DATA_DIR[sub_type]
    ds = datasets.load_dataset(
        "bigcode/the-stack",
        data_dir=data_dir,
        streaming=True,
        split="train",
        token=True,
    )

    rows: list[dict] = []
    pbar = tqdm(total=target, desc=sub_type, unit="rows")

    for item in ds.shuffle(seed=seed, buffer_size=5000):
        content = item.get("content", "")
        if not content or not passes_size_filter(content):
            continue

        category = SUB_TYPE_CATEGORY.get(sub_type, "code")
        rows.append(build_row(text=content, sub_type=sub_type, category=category))
        pbar.update(1)

        if len(rows) >= target:
            break

    pbar.close()
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
