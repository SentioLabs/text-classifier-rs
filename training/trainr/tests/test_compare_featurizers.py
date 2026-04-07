import json
import tempfile
from pathlib import Path

from trainr.core.compare_featurizers import compare_features, ComparisonReport


def test_comparison_report_structure():
    """A comparison report has per-feature stats and an overall summary."""
    python_features = {"alpha_ratio": 0.85, "symbol_ratio": 0.02}
    rust_features = {"alpha_ratio": 0.85, "symbol_ratio": 0.02}

    report = ComparisonReport.from_paired_features(
        [python_features], [rust_features], ["alpha_ratio", "symbol_ratio"]
    )

    assert report.feature_names == ["alpha_ratio", "symbol_ratio"]
    assert len(report.per_feature) == 2
    assert report.per_feature["alpha_ratio"].mean_abs_error == 0.0
    assert report.per_feature["alpha_ratio"].max_abs_error == 0.0
    assert report.max_mean_abs_error == 0.0


def test_comparison_report_detects_divergence():
    """Report correctly computes divergence stats when features differ."""
    python_features = {"alpha_ratio": 0.85, "symbol_ratio": 0.05}
    rust_features = {"alpha_ratio": 0.80, "symbol_ratio": 0.02}

    report = ComparisonReport.from_paired_features(
        [python_features], [rust_features], ["alpha_ratio", "symbol_ratio"]
    )

    assert abs(report.per_feature["alpha_ratio"].mean_abs_error - 0.05) < 1e-9
    assert abs(report.per_feature["symbol_ratio"].mean_abs_error - 0.03) < 1e-9
    assert abs(report.max_mean_abs_error - 0.05) < 1e-9
