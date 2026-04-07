"""Compare Python and Rust featurizer outputs for alignment verification.

Samples N texts from an eval JSONL, runs both featurizers, and reports
per-feature divergence. The Rust featurizer is invoked via the `classify
features` CLI command, the Python featurizer via `trainr.core.featurize`.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from trainr.core.featurize import extract_all


@dataclass
class FeatureStats:
    mean_abs_error: float
    max_abs_error: float
    mean_python: float
    mean_rust: float


@dataclass
class ComparisonReport:
    feature_names: list[str]
    per_feature: dict[str, FeatureStats]
    n_samples: int = 0

    @property
    def max_mean_abs_error(self) -> float:
        if not self.per_feature:
            return 0.0
        return max(s.mean_abs_error for s in self.per_feature.values())

    @classmethod
    def from_paired_features(
        cls,
        python_rows: list[dict[str, float]],
        rust_rows: list[dict[str, float]],
        feature_names: list[str],
    ) -> ComparisonReport:
        n = len(python_rows)
        per_feature: dict[str, FeatureStats] = {}
        for feat in feature_names:
            diffs = [abs(python_rows[i][feat] - rust_rows[i][feat]) for i in range(n)]
            py_vals = [python_rows[i][feat] for i in range(n)]
            rs_vals = [rust_rows[i][feat] for i in range(n)]
            per_feature[feat] = FeatureStats(
                mean_abs_error=sum(diffs) / max(n, 1),
                max_abs_error=max(diffs) if diffs else 0.0,
                mean_python=sum(py_vals) / max(n, 1),
                mean_rust=sum(rs_vals) / max(n, 1),
            )
        return cls(
            feature_names=feature_names,
            per_feature=per_feature,
            n_samples=n,
        )

    def format_table(self) -> str:
        lines = [f"Featurizer comparison ({self.n_samples} samples)"]
        lines.append(f"{'Feature':35s} {'MAE':>10s} {'MaxAE':>10s} {'PyMean':>10s} {'RsMean':>10s}")
        lines.append("-" * 77)
        for feat in self.feature_names:
            s = self.per_feature[feat]
            flag = " <<<" if s.mean_abs_error > 0.01 else ""
            lines.append(
                f"{feat:35s} {s.mean_abs_error:10.6f} {s.max_abs_error:10.6f} "
                f"{s.mean_python:10.6f} {s.mean_rust:10.6f}{flag}"
            )
        lines.append(f"\nMax mean abs error: {self.max_mean_abs_error:.6f}")
        threshold = 0.01
        if self.max_mean_abs_error > threshold:
            lines.append(f"FAIL: exceeds threshold {threshold}")
        else:
            lines.append(f"PASS: within threshold {threshold}")
        return "\n".join(lines)


def _extract_rust_features(
    jsonl_path: str | Path, classify_bin: str = "target/release/classify"
) -> list[dict[str, float]]:
    """Run Rust classify features on a JSONL file, return list of feature dicts."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        csv_path = tmp.name

    subprocess.run(
        [classify_bin, "features", str(jsonl_path), "--output", csv_path],
        check=True,
        capture_output=True,
    )
    rows: list[dict[str, float]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if k != "text"})
    Path(csv_path).unlink(missing_ok=True)
    return rows


def _extract_python_features(jsonl_path: str | Path) -> list[dict[str, float]]:
    """Run Python featurizer on a JSONL file, return list of feature dicts."""
    rows: list[dict[str, float]] = []
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get("text", "")
            rows.append(extract_all(text))
    return rows


def compare_features(
    jsonl_path: str | Path,
    n_samples: int = 500,
    classify_bin: str = "target/release/classify",
) -> ComparisonReport:
    """Compare Python and Rust featurizers on samples from a JSONL file."""
    samples: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if len(samples) > n_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, n_samples)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp:
        for s in samples:
            tmp.write(json.dumps(s) + "\n")
        tmp_path = tmp.name

    try:
        python_rows = _extract_python_features(tmp_path)
        rust_rows = _extract_rust_features(tmp_path, classify_bin)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    feature_names = list(python_rows[0].keys()) if python_rows else []

    return ComparisonReport.from_paired_features(
        python_rows, rust_rows, feature_names
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare Python and Rust featurizers")
    parser.add_argument("--eval", required=True, help="Eval JSONL file")
    parser.add_argument("--n-samples", type=int, default=500, help="Samples to compare")
    parser.add_argument(
        "--classify-bin",
        default="target/release/classify",
        help="Path to Rust classify binary",
    )
    args = parser.parse_args(argv)

    report = compare_features(args.eval, args.n_samples, args.classify_bin)
    print(report.format_table())


if __name__ == "__main__":
    main()
