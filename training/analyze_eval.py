#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Slice analysis for eval_onnx prediction records."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval_schema import CATEGORY_ORDER


def load_prediction_records(paths: list[str]) -> list[dict[str, Any]]:
    """Load one or more JSONL prediction files."""
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path) as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                records.append(json.loads(raw_line))
    return records


def ordered_categories(records: list[dict[str, Any]]) -> list[str]:
    """Return categories in stable order for analysis output."""
    seen = {
        record["expected_category"]
        for record in records
        if record.get("expected_category") is not None
    }
    seen.update(
        record["predicted_category"]
        for record in records
        if record.get("predicted_category") is not None
    )
    ordered = [category for category in CATEGORY_ORDER if category in seen]
    extras = sorted(seen.difference(ordered))
    return ordered + extras


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    categories: list[str],
) -> dict[str, Any]:
    """Compute accuracy, per-category precision/recall/F1, and confusion matrix."""
    n = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    overall_accuracy = correct / max(n, 1)

    cat_to_idx = {category: idx for idx, category in enumerate(categories)}
    n_cats = len(categories)
    confusion_matrix = [[0] * n_cats for _ in range(n_cats)]
    for truth, pred in zip(y_true, y_pred):
        truth_idx = cat_to_idx.get(truth)
        pred_idx = cat_to_idx.get(pred)
        if truth_idx is not None and pred_idx is not None:
            confusion_matrix[truth_idx][pred_idx] += 1

    per_category: dict[str, dict[str, float]] = {}
    for idx, category in enumerate(categories):
        tp = confusion_matrix[idx][idx]
        fn = sum(confusion_matrix[idx][col] for col in range(n_cats)) - tp
        fp = sum(confusion_matrix[row][idx] for row in range(n_cats)) - tp
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_category[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": support,
        }

    return {
        "overall_accuracy": overall_accuracy,
        "per_category": per_category,
        "confusion_matrix": confusion_matrix,
        "categories": categories,
        "total_samples": n,
    }


def compute_slice_accuracy(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float | int]]:
    """Compute accuracy by the given field, skipping missing values."""
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for record in records:
        value = record.get(field)
        if value is None:
            continue
        key = str(value)
        stats[key][0] += 1
        if record["expected_category"] == record["predicted_category"]:
            stats[key][1] += 1

    return {
        key: {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }
        for key, (total, correct) in sorted(stats.items())
    }


def build_top_confusions(
    records: list[dict[str, Any]],
    field: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the most common misclassifications grouped by field."""
    counts: Counter[tuple[str, str, str | None]] = Counter()

    for record in records:
        expected = record["expected_category"]
        predicted = record["predicted_category"]
        if expected == predicted:
            continue

        value = None if field is None else record.get(field)
        if field is not None and value is None:
            continue

        counts[(expected, predicted, None if value is None else str(value))] += 1

    results: list[dict[str, Any]] = []
    for (expected, predicted, value), count in counts.most_common(limit):
        entry: dict[str, Any] = {
            "expected_category": expected,
            "predicted_category": predicted,
            "count": count,
        }
        if field is not None:
            entry[field] = value
        results.append(entry)
    return results


def build_slice_report(
    records: list[dict[str, Any]],
    *,
    eval_file: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Build a machine-readable slice report."""
    y_true = [record["expected_category"] for record in records]
    y_pred = [record["predicted_category"] for record in records]
    metrics = compute_metrics(y_true, y_pred, ordered_categories(records))

    report: dict[str, Any] = {
        "overall_accuracy": metrics["overall_accuracy"],
        "total_samples": metrics["total_samples"],
        "per_category": metrics["per_category"],
        "confusion_matrix": metrics["confusion_matrix"],
        "categories": metrics["categories"],
        "per_sub_type": compute_slice_accuracy(records, "sub_type"),
        "per_boundary_pair": compute_slice_accuracy(records, "boundary_pair"),
        "top_confusions": {
            "by_category_pair": build_top_confusions(records, limit=limit),
            "by_sub_type": build_top_confusions(records, "sub_type", limit=limit),
            "by_content_domain": build_top_confusions(
                records, "content_domain", limit=limit
            ),
            "by_length_bucket": build_top_confusions(
                records, "length_bucket", limit=limit
            ),
            "by_model": build_top_confusions(records, "model", limit=limit),
        },
    }
    if eval_file is not None:
        report["eval_file"] = eval_file
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze eval_onnx prediction records into slice reports."
    )
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Path to prediction JSONL file (can be specified multiple times)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. Defaults to stdout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum entries per top-confusions group (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    records = load_prediction_records(args.predictions)
    report = build_slice_report(
        records,
        eval_file=Path(args.predictions[0]).name,
        limit=args.limit,
    )
    output = json.dumps(report, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
