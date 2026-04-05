"""Tests for eval_onnx.py — validates ONNX model evaluation pipeline."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the training module is importable
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_MAP = {"prose": 0, "code": 1, "structured": 2}
SUB_TYPE_MAP = {"plain": 0, "markdown": 1, "python": 2, "csv": 3}
FEATURE_NAMES = [
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
]


def _make_config(*, feature_names=None, feature_mean=None, feature_std=None,
                 category_map=None, sub_type_map=None):
    """Create a model config dict with sensible defaults."""
    names = feature_names or FEATURE_NAMES
    n = len(names)
    return {
        "feature_names": names,
        "feature_mean": feature_mean or [0.5] * n,
        "feature_std": feature_std or [0.25] * n,
        "category_map": category_map or CATEGORY_MAP,
        "sub_type_map": sub_type_map or SUB_TYPE_MAP,
    }


def _write_config(path, config=None):
    """Write a model config JSON file."""
    config = config or _make_config()
    with open(path, "w") as f:
        json.dump(config, f)
    return config


def _make_sample(*, text="Hello world.", expected_category="prose", sub_type=None):
    """Create an eval sample dict."""
    sample = {"text": text, "expected_category": expected_category}
    if sub_type is not None:
        sample["sub_type"] = sub_type
    return sample


def _write_jsonl(path, samples):
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Tests for load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_config(self):
        from eval_onnx import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = _make_config()
            json.dump(config, f)
            f.flush()
            result = load_config(f.name)

        assert result["feature_names"] == FEATURE_NAMES
        assert len(result["feature_mean"]) == 18
        assert len(result["feature_std"]) == 18
        assert result["category_map"] == CATEGORY_MAP

    def test_raises_on_missing_file(self):
        from eval_onnx import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.json")


# ---------------------------------------------------------------------------
# Tests for load_eval_samples
# ---------------------------------------------------------------------------


class TestLoadEvalSamples:
    def test_loads_samples_from_jsonl(self):
        from eval_onnx import load_eval_samples

        samples = [
            _make_sample(text="Hello.", expected_category="prose"),
            _make_sample(text="def foo():", expected_category="code"),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            f.flush()
            result = load_eval_samples([f.name])

        assert len(result) == 2
        assert result[0]["expected_category"] == "prose"
        assert result[1]["text"] == "def foo():"

    def test_loads_from_multiple_files(self):
        from eval_onnx import load_eval_samples

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f1:
            f1.write(json.dumps(_make_sample(text="A")) + "\n")
            f1.flush()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f2:
                f2.write(json.dumps(_make_sample(text="B")) + "\n")
                f2.flush()
                result = load_eval_samples([f1.name, f2.name])

        assert len(result) == 2

    def test_skips_empty_lines(self):
        from eval_onnx import load_eval_samples

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_sample()) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_sample(text="B")) + "\n")
            f.flush()
            result = load_eval_samples([f.name])

        assert len(result) == 2

    def test_loads_samples_from_parquet(self, tmp_path):
        import polars as pl
        from eval_onnx import load_eval_samples

        parquet_path = str(tmp_path / "test.parquet")
        df = pl.DataFrame({
            "text": ["Hello world.", "def foo():"],
            "category": ["prose", "code"],
            "sub_type": ["plain", "python"],
        })
        df.write_parquet(parquet_path)

        result = load_eval_samples([parquet_path])

        assert len(result) == 2
        assert result[0]["text"] == "Hello world."
        assert result[0]["category"] == "prose"
        assert result[0]["sub_type"] == "plain"
        assert result[1]["text"] == "def foo():"
        assert result[1]["category"] == "code"

    def test_parquet_samples_same_format_as_jsonl(self, tmp_path):
        import polars as pl
        from eval_onnx import load_eval_samples

        # Create a Parquet file
        parquet_path = str(tmp_path / "test.parquet")
        df = pl.DataFrame({
            "text": ["Hello world."],
            "category": ["prose"],
            "sub_type": ["plain"],
        })
        df.write_parquet(parquet_path)

        # Create a JSONL file with the same data
        jsonl_path = str(tmp_path / "test.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "Hello world.", "category": "prose", "sub_type": "plain"}) + "\n")

        parquet_result = load_eval_samples([parquet_path])
        jsonl_result = load_eval_samples([jsonl_path])

        assert parquet_result[0] == jsonl_result[0]

    def test_mixed_jsonl_and_parquet_input(self, tmp_path):
        import polars as pl
        from eval_onnx import load_eval_samples

        # Create JSONL file
        jsonl_path = str(tmp_path / "test.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "JSONL sample", "category": "prose", "sub_type": "plain"}) + "\n")

        # Create Parquet file
        parquet_path = str(tmp_path / "test.parquet")
        df = pl.DataFrame({
            "text": ["Parquet sample"],
            "category": ["code"],
            "sub_type": ["python"],
        })
        df.write_parquet(parquet_path)

        result = load_eval_samples([jsonl_path, parquet_path])

        assert len(result) == 2
        assert result[0]["text"] == "JSONL sample"
        assert result[0]["category"] == "prose"
        assert result[1]["text"] == "Parquet sample"
        assert result[1]["category"] == "code"


# ---------------------------------------------------------------------------
# Tests for normalize_features
# ---------------------------------------------------------------------------


class TestNormalizeFeatures:
    def test_zscore_normalization(self):
        from eval_onnx import normalize_features

        raw = {"f1": 1.0, "f2": 2.0}
        config = {
            "feature_names": ["f1", "f2"],
            "feature_mean": [0.5, 1.0],
            "feature_std": [0.5, 0.5],
        }
        result = normalize_features(raw, config)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 1], 2.0, atol=1e-6)

    def test_handles_zero_std(self):
        from eval_onnx import normalize_features

        raw = {"f1": 5.0}
        config = {
            "feature_names": ["f1"],
            "feature_mean": [5.0],
            "feature_std": [0.0],
        }
        result = normalize_features(raw, config)
        # Should not produce inf/nan; use 0.0 when std is 0
        assert np.isfinite(result[0, 0])
        assert result[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Tests for compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self):
        from eval_onnx import compute_metrics

        y_true = ["prose", "code", "prose", "code"]
        y_pred = ["prose", "code", "prose", "code"]
        categories = ["prose", "code"]
        metrics = compute_metrics(y_true, y_pred, categories)

        assert metrics["overall_accuracy"] == 1.0
        assert metrics["per_category"]["prose"]["precision"] == 1.0
        assert metrics["per_category"]["prose"]["recall"] == 1.0
        assert metrics["per_category"]["prose"]["f1"] == 1.0
        assert metrics["per_category"]["code"]["precision"] == 1.0

    def test_partial_predictions(self):
        from eval_onnx import compute_metrics

        # prose: 2 correct, 1 missed (predicted as code)
        # code: 1 correct, 0 missed
        y_true = ["prose", "prose", "prose", "code"]
        y_pred = ["prose", "prose", "code", "code"]
        categories = ["prose", "code"]
        metrics = compute_metrics(y_true, y_pred, categories)

        assert metrics["overall_accuracy"] == pytest.approx(0.75)
        assert metrics["per_category"]["prose"]["recall"] == pytest.approx(2 / 3, abs=1e-6)
        assert metrics["per_category"]["prose"]["precision"] == pytest.approx(1.0)
        assert metrics["per_category"]["code"]["precision"] == pytest.approx(1 / 2, abs=1e-6)
        assert metrics["per_category"]["code"]["recall"] == pytest.approx(1.0)

    def test_confusion_matrix(self):
        from eval_onnx import compute_metrics

        y_true = ["prose", "prose", "code"]
        y_pred = ["prose", "code", "code"]
        categories = ["prose", "code"]
        metrics = compute_metrics(y_true, y_pred, categories)

        cm = metrics["confusion_matrix"]
        # rows = true, columns = predicted
        assert cm == [[1, 1], [0, 1]]

    def test_handles_zero_support_category(self):
        from eval_onnx import compute_metrics

        y_true = ["prose", "prose"]
        y_pred = ["prose", "prose"]
        categories = ["prose", "code"]
        metrics = compute_metrics(y_true, y_pred, categories)

        # code has no support
        assert metrics["per_category"]["code"]["precision"] == 0.0
        assert metrics["per_category"]["code"]["recall"] == 0.0
        assert metrics["per_category"]["code"]["f1"] == 0.0
        assert metrics["per_category"]["code"]["n"] == 0


# ---------------------------------------------------------------------------
# Tests for format_report (human-readable)
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_contains_accuracy(self):
        from eval_onnx import format_report

        metrics = {
            "overall_accuracy": 0.938,
            "per_category": {
                "prose": {"precision": 0.95, "recall": 0.93, "f1": 0.94, "n": 100},
            },
            "confusion_matrix": [[100]],
            "categories": ["prose"],
        }
        output = format_report(metrics, "clear.jsonl")
        assert "0.938" in output
        assert "clear.jsonl" in output

    def test_contains_per_category_headers(self):
        from eval_onnx import format_report

        metrics = {
            "overall_accuracy": 0.9,
            "per_category": {
                "prose": {"precision": 0.95, "recall": 0.93, "f1": 0.94, "n": 100},
                "code": {"precision": 0.91, "recall": 0.89, "f1": 0.90, "n": 100},
            },
            "confusion_matrix": [[90, 10], [11, 89]],
            "categories": ["prose", "code"],
        }
        output = format_report(metrics, "test.jsonl")
        assert "Precision" in output
        assert "Recall" in output
        assert "F1" in output
        assert "prose" in output
        assert "code" in output

    def test_contains_confusion_matrix(self):
        from eval_onnx import format_report

        metrics = {
            "overall_accuracy": 0.9,
            "per_category": {
                "prose": {"precision": 0.95, "recall": 0.93, "f1": 0.94, "n": 100},
            },
            "confusion_matrix": [[95]],
            "categories": ["prose"],
        }
        output = format_report(metrics, "test.jsonl")
        assert "Confusion Matrix" in output


# ---------------------------------------------------------------------------
# Tests for format_json_report
# ---------------------------------------------------------------------------


class TestFormatJsonReport:
    def test_produces_valid_json(self):
        from eval_onnx import format_json_report

        metrics = {
            "overall_accuracy": 0.938,
            "per_category": {
                "prose": {"precision": 0.95, "recall": 0.93, "f1": 0.94, "n": 100},
            },
            "confusion_matrix": [[100]],
            "categories": ["prose"],
        }
        result = format_json_report(metrics, "test.jsonl")
        parsed = json.loads(result)
        assert parsed["eval_file"] == "test.jsonl"
        assert parsed["overall_accuracy"] == 0.938


# ---------------------------------------------------------------------------
# Tests for prediction records
# ---------------------------------------------------------------------------


class TestPredictionRecords:
    def test_predict_samples_preserves_provenance(self):
        from eval_onnx import predict_samples

        config = _make_config()
        sample = {
            "text": "col1,col2\n1,2",
            "expected_category": "structured",
            "sub_type": "csv",
            "boundary_pair": "code_structured",
            "content_domain": "finance",
            "length_bucket": "short",
            "model": "openai/gpt-5",
        }

        mock_session = MagicMock()
        cat_logits = np.array([[0.0, 0.0, 10.0]], dtype=np.float32)
        sub_logits = np.zeros((1, len(SUB_TYPE_MAP)), dtype=np.float32)
        mock_session.run.return_value = [cat_logits, sub_logits]

        predictions = predict_samples(
            mock_session,
            config,
            [sample],
            feature_extractor=lambda _text: {
                name: 0.0 for name in config["feature_names"]
            },
        )

        assert len(predictions) == 1
        assert predictions[0]["expected_category"] == "structured"
        assert predictions[0]["predicted_category"] == "structured"
        assert predictions[0]["sub_type"] == "csv"
        assert predictions[0]["boundary_pair"] == "code_structured"
        assert predictions[0]["content_domain"] == "finance"
        assert predictions[0]["length_bucket"] == "short"
        assert predictions[0]["model"] == "openai/gpt-5"

    def test_write_prediction_records_jsonl(self, tmp_path):
        from eval_onnx import write_prediction_records

        output_path = tmp_path / "predictions.jsonl"
        records = [
            {
                "text": "hello",
                "expected_category": "prose",
                "predicted_category": "prose",
                "sub_type": "plain",
                "boundary_pair": None,
            }
        ]

        write_prediction_records(output_path, records)

        loaded = json.loads(output_path.read_text().strip())
        assert loaded["predicted_category"] == "prose"


# ---------------------------------------------------------------------------
# Tests for parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        from eval_onnx import parse_args

        args = parse_args(["--eval", "test.jsonl"])
        assert args.model == "output/model.onnx"
        assert args.config == "output/model_config.json"
        assert args.eval == ["test.jsonl"]
        assert args.json is False

    def test_multiple_eval_files(self):
        from eval_onnx import parse_args

        args = parse_args(["--eval", "a.jsonl", "--eval", "b.jsonl"])
        assert args.eval == ["a.jsonl", "b.jsonl"]

    def test_json_flag(self):
        from eval_onnx import parse_args

        args = parse_args(["--eval", "test.jsonl", "--json"])
        assert args.json is True

    def test_predictions_output_dir_flag(self):
        from eval_onnx import parse_args

        args = parse_args(
            ["--eval", "test.jsonl", "--predictions-output-dir", "output"]
        )
        assert args.predictions_output_dir == "output"

    def test_custom_model_and_config(self):
        from eval_onnx import parse_args

        args = parse_args([
            "--model", "my_model.onnx",
            "--config", "my_config.json",
            "--eval", "test.jsonl",
        ])
        assert args.model == "my_model.onnx"
        assert args.config == "my_config.json"


# ---------------------------------------------------------------------------
# Tests for invert_category_map
# ---------------------------------------------------------------------------


class TestInvertCategoryMap:
    def test_inverts_map(self):
        from eval_onnx import invert_category_map

        cat_map = {"prose": 0, "code": 1, "structured": 2}
        inv = invert_category_map(cat_map)
        assert inv == {0: "prose", 1: "code", 2: "structured"}


# ---------------------------------------------------------------------------
# Tests for run_evaluation (integration with mocked ONNX)
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def test_end_to_end_with_mock_session(self):
        from eval_onnx import run_evaluation

        config = _make_config()

        samples = [
            _make_sample(text="This is a prose sentence.", expected_category="prose"),
            _make_sample(text="def foo(): pass", expected_category="code"),
            _make_sample(text="col1,col2\n1,2\n3,4", expected_category="structured"),
        ]

        # Mock ONNX session: always predict category index 0 (prose)
        mock_session = MagicMock()
        cat_logits = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)
        sub_logits = np.zeros((1, len(SUB_TYPE_MAP)), dtype=np.float32)
        mock_session.run.return_value = [cat_logits, sub_logits]

        metrics = run_evaluation(
            mock_session,
            config,
            samples,
            feature_extractor=lambda _text: {
                name: 0.0 for name in config["feature_names"]
            },
        )

        assert metrics["overall_accuracy"] == pytest.approx(1 / 3)
        assert metrics["per_category"]["prose"]["recall"] == 1.0
        assert mock_session.run.call_count == 3

    def test_perfect_mock_session(self):
        from eval_onnx import run_evaluation

        config = _make_config()

        samples = [
            _make_sample(text="Hello.", expected_category="prose"),
            _make_sample(text="x = 1", expected_category="code"),
        ]

        # Build a mock that returns the correct category for each sample
        def side_effect(output_names, inputs):
            features = inputs["features"]
            # We'll just alternate: first call prose, second call code
            return [
                np.array([[0.0] * 3], dtype=np.float32),
                np.zeros((1, len(SUB_TYPE_MAP)), dtype=np.float32),
            ]

        call_count = [0]

        def smart_side_effect(output_names, inputs):
            idx = call_count[0]
            call_count[0] += 1
            logits = np.zeros((1, 3), dtype=np.float32)
            if idx == 0:
                logits[0, 0] = 10.0  # prose
            else:
                logits[0, 1] = 10.0  # code
            sub_logits = np.zeros((1, len(SUB_TYPE_MAP)), dtype=np.float32)
            return [logits, sub_logits]

        mock_session = MagicMock()
        mock_session.run.side_effect = smart_side_effect

        metrics = run_evaluation(
            mock_session,
            config,
            samples,
            feature_extractor=lambda _text: {
                name: 0.0 for name in config["feature_names"]
            },
        )
        assert metrics["overall_accuracy"] == 1.0
