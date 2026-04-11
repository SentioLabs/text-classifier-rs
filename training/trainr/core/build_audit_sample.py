"""Build the 5k audit input for semantic detection label validation.

Combines a stratified sample of the training corpus with targeted
injection of known-positive candidates for the 3 new semantic labels
(log_content, stack_trace, diff_patch). Injected rows are tagged with
an `audit_source` column so the downstream audit can:

1. Compute inter-annotator agreement on the `stratified` rows only
   (injected positives would inflate the stat).
2. Compute recall per new label on the `inject_*` rows.

Usage:
    uv run python -m trainr.core.build_audit_sample \\
        --input data/curated/train/golden_train.parquet \\
        --output data/audit/iter16_5k_input.parquet \\
        --stratified-n 5000 \\
        --injection-per-label 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from trainr.core.annotate_detections import stratified_sample

# Regex patterns for targeted injection. Each pattern targets a label's
# high-precision surface features so that matches are very likely to be
# true positives for manual review.
INJECTION_PATTERNS: dict[str, list[str]] = {
    "stack_trace": [
        # Python: Traceback header + "File " adjacency within ~200 chars.
        # [\s\S]{0,200} is the polars-compatible equivalent of (?s:.{0,200}) —
        # "any char including newlines", up to a bounded count.
        r'Traceback \(most recent call last\):[\s\S]{0,200}File "',
        # Java: "Exception in thread" header + "at <class>.<method>(File.java:N)"
        # frame adjacency within ~400 chars.
        r"Exception in thread[\s\S]{0,400}\s+at [\w.$]+\(.*\.java:\d+\)",
        # Go is unchanged — "goroutine N [" header is already specific.
        r"goroutine \d+ \[",
        # Rust: (?m)^ anchors to line start, which excludes the test directive
        # "// error-pattern:thread 'main' panicked at" (not at line start) while
        # still matching real runtime panics (printed at line start).
        r"(?m)^thread '[^']+' panicked at",
        # .NET is unchanged — the "in <file>.cs:line N" suffix is specific.
        r"^\s+at \w+\.\w+\.\w+\(\) in .*\.cs:line \d+",
    ],
    "diff_patch": [
        r"(?m)^@@ -\d+(,\d+)? \+\d+(,\d+)? @@",
        r"(?m)^diff --git a/",
    ],
    "log_content": [
        # Severity token + timestamp-shaped substring in same row.
        # Left broad — this is a generous superset; annotators judge.
        r"\b(INFO|WARN|ERROR|DEBUG|TRACE|FATAL)\b.*\d{4}-\d{2}-\d{2}",
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}.*\b(INFO|WARN|ERROR|DEBUG|TRACE|FATAL)\b",
        r"\[\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}",  # apache access log timestamp
    ],
}


def find_injection_candidates(
    corpus: pl.DataFrame,
    label: str,
    patterns: list[str],
    n: int,
    seed: int,
) -> pl.DataFrame:
    """Return up to `n` rows from `corpus` whose `text` matches any pattern.

    Matches are case-insensitive only if the regex uses inline flags.
    Uses polars' string.contains (regex) with OR composition across
    patterns. Samples uniformly from the match set.
    """
    # Build an OR regex
    combined = "|".join(f"(?:{p})" for p in patterns)
    matched = corpus.filter(pl.col("text").str.contains(combined))
    if len(matched) == 0:
        print(
            f"  WARNING: zero corpus matches for {label!r} injection "
            f"regexes — check patterns or corpus content.",
            file=sys.stderr,
        )
        return matched.with_columns(pl.lit(f"inject_{label}").alias("audit_source"))
    if len(matched) > n:
        matched = matched.sample(n=n, seed=seed)
    return matched.with_columns(pl.lit(f"inject_{label}").alias("audit_source"))


def build_audit_sample(
    input_path: str,
    output_path: str,
    stratified_n: int = 5000,
    injection_per_label: int = 50,
    seed: int = 42,
) -> pl.DataFrame:
    """Build the audit input parquet.

    Args:
        input_path: Path to the corpus parquet (golden_train.parquet).
        output_path: Where to write the audit input parquet.
        stratified_n: Size of the stratified sample.
        injection_per_label: How many known-positive candidates to inject
            per semantic label (log_content, stack_trace, diff_patch).
        seed: RNG seed for sampling reproducibility.

    Returns:
        The final DataFrame with an `audit_source` column distinguishing
        'stratified' rows from 'inject_<label>' rows.
    """
    corpus = pl.read_parquet(input_path)
    print(
        f"  Corpus loaded: {len(corpus)} rows.",
        file=sys.stderr,
    )
    if "sub_type" not in corpus.columns:
        raise ValueError(
            f"Corpus at {input_path} is missing required 'sub_type' column; "
            f"got columns: {corpus.columns}"
        )
    if "text" not in corpus.columns:
        raise ValueError(
            f"Corpus at {input_path} is missing required 'text' column; "
            f"got columns: {corpus.columns}"
        )

    # 1. Stratified sample tagged as 'stratified'
    stratified = stratified_sample(corpus, n=stratified_n, seed=seed)
    stratified = stratified.with_columns(
        pl.lit("stratified").alias("audit_source"),
    )
    print(
        f"  Stratified sample: {len(stratified)} rows.",
        file=sys.stderr,
    )

    # 2. Per-label targeted injection
    injected_parts: list[pl.DataFrame] = []
    for label, patterns in INJECTION_PATTERNS.items():
        part = find_injection_candidates(
            corpus=corpus,
            label=label,
            patterns=patterns,
            n=injection_per_label,
            seed=seed,
        )
        print(
            f"  Injected {label}: {len(part)} rows.",
            file=sys.stderr,
        )
        injected_parts.append(part)

    # 3. Concatenate; schema alignment via how='diagonal_relaxed' in case
    # injected parts have slightly different column orders
    all_parts = [stratified] + injected_parts
    combined = pl.concat(all_parts, how="diagonal_relaxed")

    # Ensure output dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)
    print(
        f"  Wrote {len(combined)} rows to {output_path}.",
        file=sys.stderr,
    )
    return combined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Corpus parquet path")
    parser.add_argument("--output", required=True, help="Audit input parquet path")
    parser.add_argument("--stratified-n", type=int, default=5000)
    parser.add_argument("--injection-per-label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    build_audit_sample(
        input_path=args.input,
        output_path=args.output,
        stratified_n=args.stratified_n,
        injection_per_label=args.injection_per_label,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
