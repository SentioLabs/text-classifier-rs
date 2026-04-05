"""Heuristic audit for label quality validation.

Runs format-specific heuristics on a labelled dataset to estimate mislabel
rates per sub_type.  Each heuristic returns True when the text plausibly
matches the claimed sub_type label.

Phase 3a of the validation checkpoint.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Callable

import polars as pl

from trainr.core.pull_real_data import (
    validate_csv,
    validate_fixed_width,
    validate_json,
    validate_jsonl,
    validate_pipe_table,
    validate_tsv,
)

# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def _check_python(text: str) -> bool:
    return any(kw in text for kw in ("def ", "import ", "class ", "print("))


def _check_javascript(text: str) -> bool:
    return any(kw in text for kw in ("function ", "const ", "let ", "=>", "require("))


def _check_rust(text: str) -> bool:
    return any(kw in text for kw in ("fn ", "let ", "struct ", "impl ", "use "))


def _check_go(text: str) -> bool:
    return any(kw in text for kw in ("func ", "package ", "import ("))


def _check_java(text: str) -> bool:
    return any(kw in text for kw in ("class ", "public ", "private ", "import "))


def _check_sql(text: str) -> bool:
    upper = text.upper()
    return any(kw in upper for kw in ("SELECT", "INSERT", "CREATE", "ALTER"))


def _check_shell(text: str) -> bool:
    return text.startswith("#!/") or any(
        kw in text for kw in ("if [", "then", "echo ", "fi")
    )


def _check_html(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ("<html", "<div", "<body", "<head"))


def _check_xml(text: str) -> bool:
    if "<?xml" in text:
        return True
    return bool(re.search(r"<[a-z]+>.*</[a-z]+>", text, re.DOTALL))


def _check_css(text: str) -> bool:
    return "{" in text and any(
        kw in text for kw in ("color:", "margin:", "display:", "font-")
    )


def _check_yaml(text: str) -> bool:
    # At least one line with key: value pattern (colon followed by space)
    return bool(re.search(r"^\w[\w\s]*:\s", text, re.MULTILINE))


def _check_toml(text: str) -> bool:
    return bool(re.search(r"^\[.+\]", text, re.MULTILINE)) and bool(
        re.search(r"^\w[\w\s]*=\s", text, re.MULTILINE)
    )


def _check_ini(text: str) -> bool:
    return bool(re.search(r"^\[.+\]", text, re.MULTILINE)) and bool(
        re.search(r"^\w[\w\s]*\s*=\s*", text, re.MULTILINE)
    )


def _check_dockerfile(text: str) -> bool:
    return any(kw in text for kw in ("FROM ", "RUN ", "CMD "))


def _check_makefile(text: str) -> bool:
    # A line ending with ':' followed by a tab-indented line
    return bool(re.search(r"^[^\t\n]+:\s*$\n\t", text, re.MULTILINE))


def _check_markdown(text: str) -> bool:
    return any(
        (
            bool(re.search(r"^#+ ", text, re.MULTILINE)),
            "**" in text,
            "```" in text,
            bool(re.search(r"\[.+?\]\(.+?\)", text)),
        )
    )


def _check_rst(text: str) -> bool:
    return any(
        (
            ".. " in text,
            bool(re.search(r":[a-z]+:", text)),
            bool(re.search(r"^[=\-]{3,}$", text, re.MULTILINE)),
        )
    )


def _check_latex(text: str) -> bool:
    return any(kw in text for kw in ("\\begin{", "\\end{", "\\section", "\\documentclass"))


def _always_true(text: str) -> bool:
    return True


def _check_key_value(text: str) -> bool:
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return False
    kv_pattern = re.compile(r"^\w[\w\s]*\s*[=:]\s*\S")
    matches = sum(1 for line in lines if kv_pattern.search(line))
    return matches / len(lines) >= 0.3


def _check_log_lines(text: str) -> bool:
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return False
    # Timestamp-like: starts with digits followed by colon, dash, or T
    ts_pattern = re.compile(r"^\d+[\-:T]")
    matches = sum(1 for line in lines if ts_pattern.search(line))
    return matches / len(lines) >= 0.3


# ---------------------------------------------------------------------------
# HEURISTICS registry
# ---------------------------------------------------------------------------

HEURISTICS: dict[str, Callable[[str], bool]] = {
    # Structured — reuse validators from pull_real_data
    "csv": validate_csv,
    "tsv": validate_tsv,
    "jsonl": validate_jsonl,
    "json": validate_json,
    "pipe_table": validate_pipe_table,
    "fixed_width": validate_fixed_width,
    # Code
    "python": _check_python,
    "javascript": _check_javascript,
    "typescript": _check_javascript,  # same keywords
    "rust": _check_rust,
    "go": _check_go,
    "java": _check_java,
    "sql": _check_sql,
    "shell": _check_shell,
    "html": _check_html,
    "xml": _check_xml,
    "css": _check_css,
    # Config / markup
    "yaml": _check_yaml,
    "toml": _check_toml,
    "ini": _check_ini,
    "dockerfile": _check_dockerfile,
    "makefile": _check_makefile,
    "markdown": _check_markdown,
    "rst": _check_rst,
    "latex": _check_latex,
    # Always-pass
    "plain": _always_true,
    "unknown": _always_true,
    # Pattern-based
    "key_value": _check_key_value,
    "log_lines": _check_log_lines,
}


# ---------------------------------------------------------------------------
# Audit runner
# ---------------------------------------------------------------------------


def run_audit(input_path: str, output_path: str) -> pl.DataFrame:
    """Read a labelled parquet, apply heuristics, write audit output.

    Adds a ``heuristic_pass`` boolean column.  Returns the result DataFrame.
    Prints a per-sub_type summary table to stdout.
    """
    df = pl.read_parquet(input_path)

    def _apply(row_text: str, row_sub_type: str) -> bool:
        heuristic = HEURISTICS.get(row_sub_type, _always_true)
        return heuristic(row_text)

    results = [
        _apply(text, sub_type)
        for text, sub_type in zip(
            df["text"].to_list(), df["sub_type"].to_list()
        )
    ]

    df = df.with_columns(pl.Series("heuristic_pass", results))
    df.write_parquet(output_path)

    # Print summary
    summary = (
        df.group_by("sub_type")
        .agg(
            pl.len().alias("total"),
            pl.col("heuristic_pass").sum().alias("pass_count"),
        )
        .with_columns(
            (pl.col("pass_count") / pl.col("total") * 100.0).alias("pass_pct")
        )
        .sort("pass_pct")
    )
    print("\n=== Heuristic Audit Summary ===")
    print(f"{'sub_type':<20} {'total':>6} {'pass':>6} {'pass%':>7}")
    print("-" * 42)
    for row in summary.iter_rows(named=True):
        print(
            f"{row['sub_type']:<20} {row['total']:>6} {row['pass_count']:>6} {row['pass_pct']:>6.1f}%"
        )

    total_rows = df.shape[0]
    total_pass = df["heuristic_pass"].sum()
    print("-" * 42)
    print(
        f"{'TOTAL':<20} {total_rows:>6} {total_pass:>6} {total_pass / total_rows * 100:.1f}%"
    )
    print()

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run heuristic audit on labelled training data."
    )
    parser.add_argument(
        "--input",
        default="data/curated/train/golden_train.parquet",
        help="Input parquet path (default: data/curated/train/golden_train.parquet)",
    )
    parser.add_argument(
        "--output",
        default="data/curated/train/golden_train_audit.parquet",
        help="Output parquet path (default: data/curated/train/golden_train_audit.parquet)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_audit(args.input, args.output)


if __name__ == "__main__":
    main()
