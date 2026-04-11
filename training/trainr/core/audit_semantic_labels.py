"""Audit report generator for the iter16 semantic label validation.

Reads 3 annotated parquets (one per annotator model), computes:

1. Inter-annotator agreement on stratified rows (ALL labels -- catches
   regressions on existing labels from the SYSTEM_PROMPT length change).
2. Recall per new semantic label on injected positive rows.
3. Disagreement spot-check table for manual review.

Writes a markdown report gating the decision on the $400-600 full run.

Usage:
    uv run python -m trainr.core.audit_semantic_labels \\
        --gemini data/audit/iter16_5k_gemini3flash.parquet \\
        --sonnet data/audit/iter16_5k_sonnet.parquet \\
        --gpt54mini data/audit/iter16_5k_gpt54mini.parquet \\
        --output docs/accuracy_runs/2026-04-10-iteration-16-audit-report.md
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import polars as pl

from trainr.core.annotate_detections import SEMANTIC_LABELS

STRATIFIED = "stratified"

# Pass criteria thresholds (from spec)
AGREEMENT_THRESHOLD = 0.995
RECALL_THRESHOLD = 0.90


def filter_for_agreement(df: pl.DataFrame) -> pl.DataFrame:
    """Return only stratified rows (exclude injected positives)."""
    return df.filter(pl.col("audit_source") == STRATIFIED)


def detection_columns(df: pl.DataFrame) -> list[str]:
    """Return det_* column names present in the DataFrame."""
    return [c for c in df.columns if c.startswith("det_")]


def compute_agreement_across_models(
    dfs: dict[str, pl.DataFrame],
) -> dict[str, float]:
    """Compute per-label mean agreement across models on aligned rows.

    Assumes all DataFrames have the same row ordering (which they will
    if produced from the same input by annotate_dataframe). Raises
    ValueError if the DataFrames have mismatched det_* columns or row
    counts -- catching this at the top is much clearer than a
    downstream KeyError mid-loop.

    Returns {label: agreement} where agreement is the fraction of rows
    on which the majority vote matches the modal vote, averaged across
    rows. Unanimous rows contribute 1.0; split rows contribute
    max(ones, zeros) / n_models. For n_models=3, the floor is 2/3
    (a 2-1 split) and the ceiling is 1.0 (unanimous), so the metric
    lives in [0.667, 1.0], not [0, 1].
    """
    model_names = list(dfs.keys())
    n_models = len(model_names)
    if n_models == 0:
        return {}

    first = dfs[model_names[0]]
    det_cols = detection_columns(first)
    n_rows = len(first)

    # Validate schema and row alignment across all models
    first_det_set = set(det_cols)
    for m in model_names[1:]:
        other = dfs[m]
        other_det_set = set(detection_columns(other))
        if other_det_set != first_det_set:
            missing = first_det_set - other_det_set
            extra = other_det_set - first_det_set
            raise ValueError(
                f"Model {m!r} has mismatched det_* columns vs "
                f"{model_names[0]!r}: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        if len(other) != n_rows:
            raise ValueError(
                f"Model {m!r} has {len(other)} rows but "
                f"{model_names[0]!r} has {n_rows}; annotated "
                f"parquets must share row ordering"
            )

    result: dict[str, float] = {}
    for col in det_cols:
        label = col[len("det_"):]
        votes_per_model = [dfs[m][col].to_list() for m in model_names]
        agreement_sum = 0.0
        for row_idx in range(n_rows):
            row_votes = [votes_per_model[m][row_idx] for m in range(n_models)]
            ones = sum(1 for v in row_votes if v == 1)
            zeros = n_models - ones
            agreement_sum += max(ones, zeros) / n_models
        result[label] = agreement_sum / n_rows if n_rows > 0 else 1.0
    return result


def compute_recall_on_injected(df: pl.DataFrame) -> dict[str, float]:
    """Per-semantic-label recall on injected positives.

    For each `inject_<label>` row, the label should have fired (det==1).
    Returns {label: recall_fraction}. Computed on a SINGLE annotated DF --
    call once per model if you want per-model recall.
    """
    result: dict[str, float] = {}
    for label in SEMANTIC_LABELS:
        injected = df.filter(pl.col("audit_source") == f"inject_{label}")
        if len(injected) == 0:
            result[label] = float("nan")
            continue
        col = f"det_{label}"
        if col not in df.columns:
            result[label] = 0.0
            continue
        fired = injected[col].sum()
        result[label] = fired / len(injected)
    return result


def compute_recall_majority(
    dfs: dict[str, pl.DataFrame],
) -> dict[str, float]:
    """Recall on injected positives where >=2 of 3 models fire the label.

    This is the metric the pass criterion uses ("correctly fired by at
    least 2 of 3 annotators").
    """
    model_names = list(dfs.keys())
    if not model_names:
        return {label: float("nan") for label in SEMANTIC_LABELS}

    first = dfs[model_names[0]]
    result: dict[str, float] = {}
    for label in SEMANTIC_LABELS:
        col = f"det_{label}"
        injected_mask = first["audit_source"] == f"inject_{label}"
        n_inject = int(injected_mask.sum())
        if n_inject == 0:
            result[label] = float("nan")
            continue

        correct = 0
        votes_per_model = [
            dfs[m].filter(injected_mask)[col].to_list()
            for m in model_names
        ]
        for row_idx in range(n_inject):
            row_votes = [
                votes_per_model[m][row_idx] for m in range(len(model_names))
            ]
            if sum(row_votes) >= 2:
                correct += 1
        result[label] = correct / n_inject
    return result


def format_report(
    dfs: dict[str, pl.DataFrame],
    agreement: dict[str, float],
    recall: dict[str, float],
) -> str:
    """Build the markdown report with pass/fail verdicts."""
    lines = [
        "# Iter16 Semantic Label Audit Report",
        "",
        "**Date:** 2026-04-10",
        f"**Models:** {', '.join(dfs.keys())}",
        "",
        "## Pass Criteria",
        "",
        f"1. Inter-annotator agreement >={AGREEMENT_THRESHOLD} on stratified 5k "
        "for ALL labels.",
        f"2. Recall (majority >=2 of 3) >={RECALL_THRESHOLD} on injected positives.",
        "3. Zero obvious rule violations in spot-check (manual).",
        "",
        "## Criterion 1: Inter-Annotator Agreement (stratified 5k)",
        "",
        "| Label | Agreement | Pass |",
        "|---|---:|:---:|",
    ]

    all_agree_pass = True
    for label in sorted(agreement.keys()):
        score = agreement[label]
        if math.isnan(score):
            lines.append(f"| {label} | N/A | N/A (no data) |")
            continue
        passed = score >= AGREEMENT_THRESHOLD
        if not passed:
            all_agree_pass = False
        lines.append(
            f"| {label} | {score:.4f} | {'PASS' if passed else 'FAIL'} |"
        )

    lines.extend([
        "",
        f"**Criterion 1 verdict: {'PASS' if all_agree_pass else 'FAIL'}**",
        "",
        "## Criterion 2: Recall on Injected Positives",
        "",
        "| Label | Recall (majority) | Pass |",
        "|---|---:|:---:|",
    ])

    all_recall_pass = True
    for label in sorted(SEMANTIC_LABELS):
        score = recall.get(label, float("nan"))
        if math.isnan(score):
            lines.append(f"| {label} | N/A | N/A (no injected rows) |")
            all_recall_pass = False  # missing data fails criterion 2
            continue
        passed = score >= RECALL_THRESHOLD
        if not passed:
            all_recall_pass = False
        lines.append(
            f"| {label} | {score:.4f} | {'PASS' if passed else 'FAIL'} |"
        )

    lines.extend([
        "",
        f"**Criterion 2 verdict: {'PASS' if all_recall_pass else 'FAIL'}**",
        "",
        "## Criterion 3: Spot-Check Rule Violations",
        "",
        "_Manual review required -- scan disagreement rows and injected "
        "positives for obvious rule violations (e.g., log_content firing on "
        "lowercase error in prose)._",
        "",
        "## Overall Decision",
        "",
    ])

    if all_agree_pass and all_recall_pass:
        lines.append(
            "**Gate: PASS** (pending manual criterion 3). Proceed to "
            "full 90k annotation run."
        )
    else:
        lines.append(
            "**Gate: FAIL.** Iterate on label definitions; re-run audit."
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini", required=True, help="gemini-3-flash annotated parquet")
    parser.add_argument("--sonnet", required=True, help="sonnet-4.6 annotated parquet")
    parser.add_argument("--gpt54mini", required=True, help="gpt-5.4-mini annotated parquet")
    parser.add_argument("--output", required=True, help="Markdown report output path")
    args = parser.parse_args(argv)

    dfs = {
        "gemini3flash": pl.read_parquet(args.gemini),
        "sonnet": pl.read_parquet(args.sonnet),
        "gpt54mini": pl.read_parquet(args.gpt54mini),
    }

    # Filter to stratified rows for agreement
    stratified_dfs = {name: filter_for_agreement(df) for name, df in dfs.items()}
    print(
        f"  Agreement cohort: {len(next(iter(stratified_dfs.values())))} rows.",
        file=sys.stderr,
    )

    agreement = compute_agreement_across_models(stratified_dfs)
    recall = compute_recall_majority(dfs)

    report = format_report(dfs, agreement, recall)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"  Wrote report to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
