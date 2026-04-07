"""Tests for trainr.core.train — TextClassifier training pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn as nn

from trainr.core.train import (
    CATEGORY_MAP,
    FEATURE_COLUMNS,
    NUM_CATEGORIES,
    TextClassifier,
    export_onnx,
    load_and_prepare_data,
    parse_args,
    resolve_feature_columns,
    save_config,
    train_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parquet(
    path: Path,
    n_rows: int = 50,
    include_det: bool = False,
    seed: int = 42,
) -> Path:
    """Create a minimal Parquet file with required columns for training."""
    rng = np.random.default_rng(seed)
    categories = ["prose", "code", "structured"]
    sub_types = ["plain", "markdown", "python"]

    data: dict = {
        col: rng.standard_normal(n_rows).astype(np.float32).tolist()
        for col in FEATURE_COLUMNS
    }
    data["category"] = [categories[i % len(categories)] for i in range(n_rows)]
    data["sub_type"] = [sub_types[i % len(sub_types)] for i in range(n_rows)]

    if include_det:
        # Add det_* columns for a few detection labels
        for label in ["plain", "markdown", "python"]:
            data[f"det_{label}"] = [int(rng.integers(0, 2)) for _ in range(n_rows)]

    df = pl.DataFrame(data)
    df.write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# TextClassifier model tests
# ---------------------------------------------------------------------------


class TestTextClassifierInit:
    """Test TextClassifier.__init__ with and without detection head."""

    def test_default_no_detection_head(self):
        model = TextClassifier(n_features=10, n_categories=3, n_sub_types=5)
        assert model.detection_head is None

    def test_detection_head_created_when_n_detection_labels_positive(self):
        model = TextClassifier(
            n_features=10, n_categories=3, n_sub_types=5, n_detection_labels=7
        )
        assert model.detection_head is not None
        assert isinstance(model.detection_head, nn.Linear)
        assert model.detection_head.out_features == 7
        assert model.detection_head.in_features == 32

    def test_detection_head_none_when_zero(self):
        model = TextClassifier(
            n_features=10, n_categories=3, n_sub_types=5, n_detection_labels=0
        )
        assert model.detection_head is None


class TestTextClassifierForward:
    """Test TextClassifier.forward returns 3-tuple."""

    def test_forward_returns_3_tuple(self):
        model = TextClassifier(n_features=10, n_categories=3, n_sub_types=5)
        x = torch.randn(4, 10)
        result = model(x)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_forward_det_logits_none_without_detection_head(self):
        model = TextClassifier(n_features=10, n_categories=3, n_sub_types=5)
        x = torch.randn(4, 10)
        cat_logits, sub_logits, det_logits = model(x)
        assert cat_logits.shape == (4, 3)
        assert sub_logits.shape == (4, 5)
        assert det_logits is None

    def test_forward_det_logits_with_detection_head(self):
        model = TextClassifier(
            n_features=10, n_categories=3, n_sub_types=5, n_detection_labels=7
        )
        x = torch.randn(4, 10)
        cat_logits, sub_logits, det_logits = model(x)
        assert cat_logits.shape == (4, 3)
        assert sub_logits.shape == (4, 5)
        assert det_logits is not None
        assert det_logits.shape == (4, 7)


# ---------------------------------------------------------------------------
# load_and_prepare_data tests
# ---------------------------------------------------------------------------


class TestLoadAndPrepareDataDetection:
    """Test detection column handling in load_and_prepare_data."""

    def test_no_det_columns_no_detection_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_parquet(Path(tmpdir) / "train.parquet", include_det=False)
            data = load_and_prepare_data(path)
            assert "detection_map" not in data
            assert "y_det_train" not in data
            assert "y_det_val" not in data

    def test_det_columns_produce_detection_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_parquet(Path(tmpdir) / "train.parquet", include_det=True)
            data = load_and_prepare_data(path)
            assert "detection_map" in data
            assert "y_det_train" in data
            assert "y_det_val" in data

    def test_detection_map_has_correct_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_parquet(Path(tmpdir) / "train.parquet", include_det=True)
            data = load_and_prepare_data(path)
            detection_map = data["detection_map"]
            # We added det_plain, det_markdown, det_python
            assert "plain" in detection_map
            assert "markdown" in detection_map
            assert "python" in detection_map
            assert len(detection_map) == 3

    def test_y_det_is_float32_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_parquet(Path(tmpdir) / "train.parquet", n_rows=50, include_det=True)
            data = load_and_prepare_data(path)
            assert data["y_det_train"].dtype == np.float32
            assert data["y_det_val"].dtype == np.float32
            # Should have columns equal to number of detection labels
            assert data["y_det_train"].shape[1] == 3
            assert data["y_det_val"].shape[1] == 3


# ---------------------------------------------------------------------------
# train_model with detection tests
# ---------------------------------------------------------------------------


class TestTrainModelDetection:
    """Test train_model with detection head."""

    def test_train_with_detection_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_parquet(Path(tmpdir) / "train.parquet", n_rows=60, include_det=True)
            data = load_and_prepare_data(path)
            n_sub_types = len(data["sub_type_map"])
            model, metrics = train_model(
                data,
                n_sub_types=n_sub_types,
                epochs=2,
                batch_size=16,
                patience=5,
                device=torch.device("cpu"),
                detection_weight=0.3,
            )
            assert model.detection_head is not None
            assert model.detection_head.out_features == 3

    def test_train_without_detection_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_parquet(Path(tmpdir) / "train.parquet", n_rows=60, include_det=False)
            data = load_and_prepare_data(path)
            n_sub_types = len(data["sub_type_map"])
            model, metrics = train_model(
                data,
                n_sub_types=n_sub_types,
                epochs=2,
                batch_size=16,
                patience=5,
                device=torch.device("cpu"),
            )
            assert model.detection_head is None

    def test_detection_metrics_present_when_detection_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_parquet(Path(tmpdir) / "train.parquet", n_rows=60, include_det=True)
            data = load_and_prepare_data(path)
            n_sub_types = len(data["sub_type_map"])
            _, metrics = train_model(
                data,
                n_sub_types=n_sub_types,
                epochs=2,
                batch_size=16,
                patience=5,
                device=torch.device("cpu"),
                detection_weight=0.3,
            )
            assert "val_detection_f1" in metrics


# ---------------------------------------------------------------------------
# export_onnx tests
# ---------------------------------------------------------------------------


class TestExportOnnxDetection:
    """Test ONNX export with and without detection head."""

    def test_export_with_detection_head(self):
        model = TextClassifier(
            n_features=10, n_categories=3, n_sub_types=5, n_detection_labels=7
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            export_onnx(model, path, n_features=10)
            assert path.exists()

            import onnxruntime as ort

            session = ort.InferenceSession(str(path))
            output_names = [o.name for o in session.get_outputs()]
            assert "detection_logits" in output_names
            assert len(output_names) == 3

    def test_export_without_detection_head(self):
        model = TextClassifier(n_features=10, n_categories=3, n_sub_types=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            export_onnx(model, path, n_features=10)
            assert path.exists()

            import onnxruntime as ort

            session = ort.InferenceSession(str(path))
            output_names = [o.name for o in session.get_outputs()]
            assert "detection_logits" not in output_names
            assert len(output_names) == 2


# ---------------------------------------------------------------------------
# save_config tests
# ---------------------------------------------------------------------------


class TestSaveConfigDetection:
    """Test save_config with detection_map."""

    def test_config_includes_detection_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            detection_map = {"plain": 0, "markdown": 1, "python": 2}
            save_config(
                path,
                feature_names=["f1", "f2"],
                feature_mean=np.array([0.0, 0.0]),
                feature_std=np.array([1.0, 1.0]),
                sub_type_map={"plain": 0},
                detection_map=detection_map,
            )
            config = json.loads(path.read_text())
            assert "detection_map" in config
            assert config["detection_map"] == detection_map

    def test_config_no_detection_map_when_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            save_config(
                path,
                feature_names=["f1", "f2"],
                feature_mean=np.array([0.0, 0.0]),
                feature_std=np.array([1.0, 1.0]),
                sub_type_map={"plain": 0},
            )
            config = json.loads(path.read_text())
            assert "detection_map" not in config


# ---------------------------------------------------------------------------
# parse_args tests
# ---------------------------------------------------------------------------


class TestParseArgsDetection:
    """Test --detection-weight CLI argument."""

    def test_detection_weight_default(self):
        args = parse_args(["--data", "foo.parquet", "--output", "out/"])
        assert args.detection_weight == 0.3

    def test_detection_weight_custom(self):
        args = parse_args(
            ["--data", "foo.parquet", "--output", "out/", "--detection-weight", "0.5"]
        )
        assert args.detection_weight == 0.5
