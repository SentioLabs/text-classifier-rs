"""Tests for training/train.py — validates data loading, model, training, and export."""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the training module is importable
sys.path.insert(0, str(Path(__file__).parent))

from train import (
    CATEGORY_MAP,
    FEATURE_COLUMNS,
    TextClassifier,
    load_and_prepare_data,
    parse_args,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DUMMY_SUB_TYPES = [
    "narrative",
    "python",
    "csv_tsv",
    "ocr_text",
    "boilerplate",
    "dialogue",
    "javascript",
    "json_data",
    "log_output",
    "whitespace",
]


def _make_dummy_csv(path: Path, n_rows: int = 200) -> None:
    """Write a small but valid training CSV with known structure."""
    import random

    random.seed(42)
    categories = list(CATEGORY_MAP.keys())
    rows_per_cat = n_rows // len(categories)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = list(FEATURE_COLUMNS) + ["line_count", "category", "sub_type"]
        writer.writerow(header)
        for cat in categories:
            for _ in range(rows_per_cat):
                features = [round(random.uniform(0.0, 1.0), 4) for _ in FEATURE_COLUMNS]
                line_count = random.randint(1, 500)
                sub_type = random.choice(DUMMY_SUB_TYPES)
                writer.writerow(features + [line_count, cat, sub_type])


@pytest.fixture
def dummy_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "combined.csv"
    _make_dummy_csv(csv_path)
    return csv_path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "models"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_feature_columns_count(self):
        assert len(FEATURE_COLUMNS) == 18

    def test_category_map_entries(self):
        assert CATEGORY_MAP == {
            "prose": 0,
            "code": 1,
            "structured": 2,
            "artifact": 3,
            "skip": 4,
        }


class TestModel:
    def test_forward_shapes(self):
        import torch

        model = TextClassifier(n_features=18, n_categories=5, n_sub_types=10)
        x = torch.randn(4, 18)
        cat_logits, sub_logits = model(x)
        assert cat_logits.shape == (4, 5)
        assert sub_logits.shape == (4, 10)

    def test_shared_layers_exist(self):
        model = TextClassifier(n_features=18, n_categories=5, n_sub_types=10)
        assert hasattr(model, "shared")
        assert hasattr(model, "category_head")
        assert hasattr(model, "sub_type_head")

    def test_three_shared_linear_layers(self):
        """Model should have 3 linear layers: 18->128, 128->64, 64->32."""
        import torch.nn as nn

        model = TextClassifier(n_features=18, n_categories=5, n_sub_types=10)
        linear_layers = [m for m in model.shared if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3, f"Expected 3 Linear layers, got {len(linear_layers)}"
        assert linear_layers[0].in_features == 18
        assert linear_layers[0].out_features == 128
        assert linear_layers[1].in_features == 128
        assert linear_layers[1].out_features == 64
        assert linear_layers[2].in_features == 64
        assert linear_layers[2].out_features == 32

    def test_dropout_rate_is_0_3(self):
        """Dropout layers should use p=0.3."""
        import torch.nn as nn

        model = TextClassifier(n_features=18, n_categories=5, n_sub_types=10)
        dropout_layers = [m for m in model.shared if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) >= 1, "No Dropout layers found"
        for d in dropout_layers:
            assert d.p == pytest.approx(0.3), f"Expected dropout p=0.3, got {d.p}"


class TestTrainingDefaults:
    def test_default_epochs_200(self):
        """Default epochs should be 200."""
        args = parse_args(["--data", "dummy.csv", "--output", "out"])
        assert args.epochs == 200

    def test_default_patience_15(self):
        """Default patience should be 15."""
        args = parse_args(["--data", "dummy.csv", "--output", "out"])
        assert args.patience == 15

    def test_lr_scheduler_used_in_training(self, dummy_csv):
        """Training should use ReduceLROnPlateau scheduler."""
        import unittest.mock as mock

        data = load_and_prepare_data(dummy_csv, val_fraction=0.2, seed=42)
        n_sub_types = len(data["sub_type_map"])

        with mock.patch("torch.optim.lr_scheduler.ReduceLROnPlateau") as mock_sched_cls:
            mock_scheduler = mock.MagicMock()
            mock_sched_cls.return_value = mock_scheduler

            train_model(data, n_sub_types=n_sub_types, epochs=3, patience=5)

            # Scheduler should be created with correct params
            mock_sched_cls.assert_called_once()
            _, kwargs = mock_sched_cls.call_args
            assert kwargs.get("mode") == "min"
            assert kwargs.get("factor") == 0.5
            assert kwargs.get("patience") == 5

            # scheduler.step should be called once per epoch
            assert mock_scheduler.step.call_count == 3


class TestDataLoading:
    def test_load_returns_expected_keys(self, dummy_csv):
        result = load_and_prepare_data(dummy_csv, val_fraction=0.2, seed=42)
        expected_keys = {
            "X_train",
            "X_val",
            "y_cat_train",
            "y_cat_val",
            "y_sub_train",
            "y_sub_val",
            "feature_mean",
            "feature_std",
            "sub_type_map",
        }
        assert set(result.keys()) == expected_keys

    def test_features_are_standardized(self, dummy_csv):
        import numpy as np

        result = load_and_prepare_data(dummy_csv, val_fraction=0.2, seed=42)
        X_train = result["X_train"]
        # After z-score, training set should have ~0 mean and ~1 std
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)
        assert np.allclose(means, 0.0, atol=0.1)
        assert np.allclose(stds, 1.0, atol=0.2)

    def test_train_val_split_proportions(self, dummy_csv):
        result = load_and_prepare_data(dummy_csv, val_fraction=0.2, seed=42)
        n_train = len(result["X_train"])
        n_val = len(result["X_val"])
        total = n_train + n_val
        assert 0.15 <= n_val / total <= 0.25

    def test_sub_type_map_covers_data(self, dummy_csv):
        result = load_and_prepare_data(dummy_csv, val_fraction=0.2, seed=42)
        sub_map = result["sub_type_map"]
        # All sub_type labels should be non-negative integers
        assert all(isinstance(v, int) and v >= 0 for v in sub_map.values())
        assert len(sub_map) == len(DUMMY_SUB_TYPES)


# ---------------------------------------------------------------------------
# Integration / end-to-end test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_script_produces_output_files(self, dummy_csv, output_dir):
        """Run the training script as a subprocess and check outputs."""
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "train.py"),
                "--data",
                str(dummy_csv),
                "--output",
                str(output_dir),
                "--epochs",
                "3",
                "--batch-size",
                "32",
                "--patience",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Script failed:\nstderr={result.stderr}\nstdout={result.stdout}"

        # Check all expected output files exist
        assert (output_dir / "model.onnx").exists(), "model.onnx not found"
        assert (output_dir / "model_config.json").exists(), "model_config.json not found"
        assert (output_dir / "metrics.json").exists(), "metrics.json not found"

    def test_model_config_structure(self, dummy_csv, output_dir):
        """Verify model_config.json has the expected keys and structure."""
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "train.py"),
                "--data",
                str(dummy_csv),
                "--output",
                str(output_dir),
                "--epochs",
                "2",
                "--batch-size",
                "32",
                "--patience",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        config = json.loads((output_dir / "model_config.json").read_text())
        assert "feature_names" in config
        assert "feature_mean" in config
        assert "feature_std" in config
        assert "category_map" in config
        assert "sub_type_map" in config
        assert len(config["feature_names"]) == 18
        assert len(config["feature_mean"]) == 18
        assert len(config["feature_std"]) == 18

    def test_metrics_json_structure(self, dummy_csv, output_dir):
        """Verify metrics.json has expected fields."""
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "train.py"),
                "--data",
                str(dummy_csv),
                "--output",
                str(output_dir),
                "--epochs",
                "2",
                "--batch-size",
                "32",
                "--patience",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        metrics = json.loads((output_dir / "metrics.json").read_text())
        assert "best_val_loss" in metrics
        assert "best_epoch" in metrics
        assert "total_epochs" in metrics
        assert "val_category_accuracy" in metrics
        assert "val_sub_type_accuracy" in metrics

    def test_onnx_model_loadable(self, dummy_csv, output_dir):
        """Verify the exported ONNX model can be loaded and has correct I/O."""
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "train.py"),
                "--data",
                str(dummy_csv),
                "--output",
                str(output_dir),
                "--epochs",
                "2",
                "--batch-size",
                "32",
                "--patience",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        import onnx

        model = onnx.load(str(output_dir / "model.onnx"))
        onnx.checker.check_model(model)

        input_names = [inp.name for inp in model.graph.input]
        output_names = [out.name for out in model.graph.output]
        assert "features" in input_names
        assert "category_logits" in output_names
        assert "sub_type_logits" in output_names
