"""Tests for detection head metrics in trainr.core.eval_onnx."""

import numpy as np
import pytest

from trainr.core.eval_onnx import (
    compute_detection_metrics,
    format_detection_report,
    predict_samples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSession:
    """Minimal stand-in for onnxruntime.InferenceSession."""

    def __init__(self, outputs: list[np.ndarray]):
        self._outputs = outputs

    def run(self, _output_names, _feed_dict):
        return self._outputs


def _make_config(*, with_detection: bool = False) -> dict:
    """Build a minimal model config dict."""
    config = {
        "category_map": {"prose": 0, "code": 1},
        "feature_names": ["f1", "f2"],
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
    }
    if with_detection:
        config["detection_map"] = {
            "markdown": 0,
            "python": 1,
            "yaml": 2,
        }
    return config


def _make_sample(**overrides) -> dict:
    """Create a minimal eval sample."""
    base = {"text": "hello world", "expected_category": "prose"}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# predict_samples — detection output extraction
# ---------------------------------------------------------------------------


class TestPredictSamplesDetection:
    """predict_samples should extract detection logits from the 3rd output."""

    def test_no_detection_outputs_returns_records_without_detection(self):
        """When model has only 2 outputs, records lack detection fields."""
        cat_logits = np.array([[2.0, 0.1]], dtype=np.float32)
        sub_logits = np.array([[0.5, 0.3]], dtype=np.float32)
        session = FakeSession([cat_logits, sub_logits])
        config = _make_config(with_detection=False)
        samples = [_make_sample()]

        results = predict_samples(
            session, config, samples,
            feature_extractor=lambda _text: {"f1": 0.0, "f2": 0.0},
        )

        assert len(results) == 1
        assert "detected_subtypes" not in results[0]
        assert "detection_scores" not in results[0]

    def test_detection_outputs_above_threshold(self):
        """When model has 3 outputs, records include detected_subtypes."""
        cat_logits = np.array([[2.0, 0.1]], dtype=np.float32)
        sub_logits = np.array([[0.5, 0.3]], dtype=np.float32)
        # Logits: sigmoid(2.0)=0.88, sigmoid(-3.0)=0.047, sigmoid(0.5)=0.62
        det_logits = np.array([[2.0, -3.0, 0.5]], dtype=np.float32)

        session = FakeSession([cat_logits, sub_logits, det_logits])
        config = _make_config(with_detection=True)
        samples = [_make_sample()]

        results = predict_samples(
            session, config, samples,
            feature_extractor=lambda _text: {"f1": 0.0, "f2": 0.0},
        )

        assert len(results) == 1
        rec = results[0]
        assert "detected_subtypes" in rec
        assert "detection_scores" in rec
        # markdown (sigmoid(2.0) ~ 0.88) above 0.5 threshold
        assert "markdown" in rec["detected_subtypes"]
        # python (sigmoid(-3.0) ~ 0.05) below 0.5
        assert "python" not in rec["detected_subtypes"]
        # yaml (sigmoid(0.5) ~ 0.62) above 0.5
        assert "yaml" in rec["detected_subtypes"]
        # Scores should be a dict with all labels
        assert set(rec["detection_scores"].keys()) == {"markdown", "python", "yaml"}

    def test_detection_outputs_without_detection_map_in_config(self):
        """When 3 outputs exist but no detection_map, skip detection."""
        cat_logits = np.array([[2.0, 0.1]], dtype=np.float32)
        sub_logits = np.array([[0.5, 0.3]], dtype=np.float32)
        det_logits = np.array([[2.0, -3.0, 0.5]], dtype=np.float32)

        session = FakeSession([cat_logits, sub_logits, det_logits])
        config = _make_config(with_detection=False)
        samples = [_make_sample()]

        results = predict_samples(
            session, config, samples,
            feature_extractor=lambda _text: {"f1": 0.0, "f2": 0.0},
        )

        assert "detected_subtypes" not in results[0]


# ---------------------------------------------------------------------------
# compute_detection_metrics
# ---------------------------------------------------------------------------


class TestComputeDetectionMetrics:
    """Tests for per-label P/R/F1, micro/macro F1, and hamming loss."""

    def test_perfect_predictions(self):
        """Perfect predictions yield P=R=F1=1.0 and hamming=0.0."""
        detection_map = {"markdown": 0, "python": 1}
        predictions = [
            {"detected_subtypes": ["markdown"], "detection_scores": {"markdown": 0.9, "python": 0.1}},
            {"detected_subtypes": ["python"], "detection_scores": {"markdown": 0.1, "python": 0.9}},
        ]
        samples = [
            {"det_markdown": 1, "det_python": 0},
            {"det_markdown": 0, "det_python": 1},
        ]

        metrics = compute_detection_metrics(predictions, samples, detection_map)

        assert metrics["per_label"]["markdown"]["precision"] == 1.0
        assert metrics["per_label"]["markdown"]["recall"] == 1.0
        assert metrics["per_label"]["markdown"]["f1"] == 1.0
        assert metrics["per_label"]["python"]["precision"] == 1.0
        assert metrics["per_label"]["python"]["f1"] == 1.0
        assert metrics["micro_f1"] == pytest.approx(1.0)
        assert metrics["macro_f1"] == pytest.approx(1.0)
        assert metrics["hamming_loss"] == pytest.approx(0.0)

    def test_all_wrong_predictions(self):
        """Completely wrong predictions yield P=R=F1=0.0 and hamming=1.0."""
        detection_map = {"markdown": 0, "python": 1}
        predictions = [
            {"detected_subtypes": ["python"], "detection_scores": {"markdown": 0.1, "python": 0.9}},
            {"detected_subtypes": ["markdown"], "detection_scores": {"markdown": 0.9, "python": 0.1}},
        ]
        samples = [
            {"det_markdown": 1, "det_python": 0},
            {"det_markdown": 0, "det_python": 1},
        ]

        metrics = compute_detection_metrics(predictions, samples, detection_map)

        assert metrics["per_label"]["markdown"]["precision"] == 0.0
        assert metrics["per_label"]["markdown"]["recall"] == 0.0
        assert metrics["per_label"]["markdown"]["f1"] == 0.0
        assert metrics["hamming_loss"] == pytest.approx(1.0)

    def test_mixed_predictions(self):
        """Partial correctness computes reasonable intermediate values."""
        detection_map = {"markdown": 0, "python": 1}
        # Sample 0: true=[md, py], pred=[md] => md TP, py FN
        # Sample 1: true=[md], pred=[md, py] => md TP, py FP
        predictions = [
            {"detected_subtypes": ["markdown"], "detection_scores": {"markdown": 0.9, "python": 0.1}},
            {"detected_subtypes": ["markdown", "python"], "detection_scores": {"markdown": 0.9, "python": 0.9}},
        ]
        samples = [
            {"det_markdown": 1, "det_python": 1},
            {"det_markdown": 1, "det_python": 0},
        ]

        metrics = compute_detection_metrics(predictions, samples, detection_map)

        # markdown: TP=2, FP=0, FN=0 => P=1.0, R=1.0, F1=1.0
        assert metrics["per_label"]["markdown"]["precision"] == pytest.approx(1.0)
        assert metrics["per_label"]["markdown"]["recall"] == pytest.approx(1.0)
        # python: TP=0, FP=1, FN=1 => P=0, R=0, F1=0
        assert metrics["per_label"]["python"]["precision"] == pytest.approx(0.0)
        assert metrics["per_label"]["python"]["recall"] == pytest.approx(0.0)

        # Hamming: 2 wrong out of 4 total label slots => 0.5
        assert metrics["hamming_loss"] == pytest.approx(0.5)

    def test_support_counts(self):
        """Per-label n (support) counts ground-truth positives."""
        detection_map = {"markdown": 0, "python": 1}
        predictions = [
            {"detected_subtypes": ["markdown"], "detection_scores": {"markdown": 0.9, "python": 0.1}},
            {"detected_subtypes": [], "detection_scores": {"markdown": 0.1, "python": 0.1}},
        ]
        samples = [
            {"det_markdown": 1, "det_python": 0},
            {"det_markdown": 1, "det_python": 0},
        ]

        metrics = compute_detection_metrics(predictions, samples, detection_map)

        assert metrics["per_label"]["markdown"]["n"] == 2
        assert metrics["per_label"]["python"]["n"] == 0

    def test_empty_inputs(self):
        """Empty inputs produce zero metrics without errors."""
        detection_map = {"markdown": 0}
        metrics = compute_detection_metrics([], [], detection_map)

        assert metrics["hamming_loss"] == pytest.approx(0.0)
        assert metrics["micro_f1"] == pytest.approx(0.0)
        assert metrics["macro_f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# format_detection_report
# ---------------------------------------------------------------------------


class TestFormatDetectionReport:
    """Tests for human-readable detection report formatting."""

    def test_contains_header(self):
        metrics = {
            "per_label": {
                "markdown": {"precision": 0.95, "recall": 0.92, "f1": 0.93, "n": 142},
            },
            "micro_f1": 0.89,
            "macro_f1": 0.86,
            "hamming_loss": 0.04,
        }
        report = format_detection_report(metrics)
        assert "Detection Metrics" in report

    def test_contains_label_row(self):
        metrics = {
            "per_label": {
                "markdown": {"precision": 0.95, "recall": 0.92, "f1": 0.93, "n": 142},
            },
            "micro_f1": 0.89,
            "macro_f1": 0.86,
            "hamming_loss": 0.04,
        }
        report = format_detection_report(metrics)
        assert "markdown" in report
        assert "0.95" in report
        assert "0.92" in report
        assert "142" in report

    def test_contains_summary_line(self):
        metrics = {
            "per_label": {
                "markdown": {"precision": 0.95, "recall": 0.92, "f1": 0.93, "n": 142},
            },
            "micro_f1": 0.89,
            "macro_f1": 0.86,
            "hamming_loss": 0.04,
        }
        report = format_detection_report(metrics)
        assert "Micro F1" in report
        assert "Macro F1" in report
        assert "Hamming" in report

    def test_column_headers_present(self):
        metrics = {
            "per_label": {
                "markdown": {"precision": 0.95, "recall": 0.92, "f1": 0.93, "n": 142},
            },
            "micro_f1": 0.89,
            "macro_f1": 0.86,
            "hamming_loss": 0.04,
        }
        report = format_detection_report(metrics)
        assert "Precision" in report
        assert "Recall" in report
        assert "F1" in report
        assert "N" in report
