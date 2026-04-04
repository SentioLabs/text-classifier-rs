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
    NUM_CATEGORIES,
    TextClassifier,
    load_and_prepare_data,
    parse_args,
    resolve_device,
    resolve_feature_columns,
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


def _make_source_csv(path: Path, source_sizes: dict[str, int], seed: int = 123) -> None:
    """Write training CSV with explicit source-group sizes."""
    import random

    random.seed(seed)
    categories = list(CATEGORY_MAP.keys())
    header = list(FEATURE_COLUMNS) + ["line_count", "category", "sub_type", "source"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for source, size in source_sizes.items():
            for i in range(size):
                features = [round(random.uniform(0.0, 1.0), 4) for _ in FEATURE_COLUMNS]
                line_count = random.randint(1, 500)
                category = categories[i % len(categories)]
                sub_type = DUMMY_SUB_TYPES[i % len(DUMMY_SUB_TYPES)]
                writer.writerow(features + [line_count, category, sub_type, source])


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
        assert len(FEATURE_COLUMNS) == 28

    def test_feature_columns_order(self):
        expected = [
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
            "hyphenated_line_break_ratio",
            "short_repeated_line_ratio",
            "page_number_density",
            "label_value_line_ratio",
            "table_fragment_score",
            "uppercase_header_ratio",
            "dictionary_word_ratio",
            "encoding_error_ratio",
            "repeated_ngram_ratio",
            "sentence_coherence_score",
        ]
        assert list(FEATURE_COLUMNS) == expected

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

        model = TextClassifier(n_features=len(FEATURE_COLUMNS), n_categories=5, n_sub_types=10)
        x = torch.randn(4, len(FEATURE_COLUMNS))
        cat_logits, sub_logits = model(x)
        assert cat_logits.shape == (4, 5)
        assert sub_logits.shape == (4, 10)

    def test_shared_layers_exist(self):
        model = TextClassifier(n_features=len(FEATURE_COLUMNS), n_categories=5, n_sub_types=10)
        assert hasattr(model, "shared")
        assert hasattr(model, "category_head")
        assert hasattr(model, "sub_type_head")

    def test_three_shared_linear_layers(self):
        """Model should have 3 linear layers: 18->128, 128->64, 64->32."""
        import torch.nn as nn

        model = TextClassifier(n_features=len(FEATURE_COLUMNS), n_categories=5, n_sub_types=10)
        linear_layers = [m for m in model.shared if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3, f"Expected 3 Linear layers, got {len(linear_layers)}"
        assert linear_layers[0].in_features == len(FEATURE_COLUMNS)
        assert linear_layers[0].out_features == 128
        assert linear_layers[1].in_features == 128
        assert linear_layers[1].out_features == 64
        assert linear_layers[2].in_features == 64
        assert linear_layers[2].out_features == 32

    def test_dropout_rate_is_0_3(self):
        """Dropout layers should use p=0.3."""
        import torch.nn as nn

        model = TextClassifier(n_features=len(FEATURE_COLUMNS), n_categories=5, n_sub_types=10)
        dropout_layers = [m for m in model.shared if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) >= 1, "No Dropout layers found"
        for d in dropout_layers:
            assert d.p == pytest.approx(0.3), f"Expected dropout p=0.3, got {d.p}"


class TestDeviceFlag:
    def test_device_argument_accepted_by_parser(self):
        """The --device argument should be accepted by the parser."""
        args = parse_args(["--data", "dummy.csv", "--output", "out", "--device", "cpu"])
        assert args.device == "cpu"

    def test_device_default_is_auto(self):
        """Default value for --device should be 'auto'."""
        args = parse_args(["--data", "dummy.csv", "--output", "out"])
        assert args.device == "auto"

    def test_device_resolution_cpu(self):
        """When --device cpu is passed, resolve_device returns torch.device('cpu')."""
        import torch

        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_device_resolution_auto_without_cuda(self):
        """When --device auto and CUDA unavailable, resolve to CPU."""
        from unittest import mock
        import torch

        with mock.patch("torch.cuda.is_available", return_value=False):
            device = resolve_device("auto")
        assert device == torch.device("cpu")

    def test_device_resolution_auto_with_cuda(self):
        """When --device auto and CUDA available, resolve to CUDA."""
        from unittest import mock
        import torch

        with mock.patch("torch.cuda.is_available", return_value=True):
            device = resolve_device("auto")
        assert device == torch.device("cuda")

    def test_device_resolution_specific_cuda_index(self):
        """When --device cuda:0 is passed, resolve to torch.device('cuda:0')."""
        import torch

        device = resolve_device("cuda:0")
        assert device == torch.device("cuda:0")

    def test_train_model_accepts_device_argument(self, dummy_csv):
        """train_model should accept and use a device parameter."""
        import torch

        data = load_and_prepare_data(dummy_csv, val_fraction=0.2, seed=42)
        n_sub_types = len(data["sub_type_map"])
        model, metrics = train_model(
            data, n_sub_types=n_sub_types, epochs=1, patience=2, device=torch.device("cpu")
        )
        assert model is not None
        assert "best_val_loss" in metrics


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

    def test_new_slice_aware_flags_default_off(self):
        args = parse_args(["--data", "dummy.csv", "--output", "out"])
        assert args.group_val_by_source is False
        assert args.balance_artifact_subtypes is False

    def test_new_slice_aware_flags_can_be_enabled(self):
        args = parse_args(
            [
                "--data",
                "dummy.csv",
                "--output",
                "out",
                "--group-val-by-source",
                "--balance-artifact-subtypes",
            ]
        )
        assert args.group_val_by_source is True
        assert args.balance_artifact_subtypes is True


class TestFeatureAblation:
    def test_drop_features_cli_allows_comma_separated_values(self):
        args = parse_args(
            [
                "--data",
                "dummy.csv",
                "--output",
                "out",
                "--drop-features",
                "tab_density,page_number_density",
            ]
        )
        assert args.drop_features == ["tab_density,page_number_density"]

    def test_drop_features_cli_allows_repeated_flags(self):
        args = parse_args(
            [
                "--data",
                "dummy.csv",
                "--output",
                "out",
                "--drop-features",
                "tab_density",
                "--drop-features",
                "page_number_density",
            ]
        )
        assert args.drop_features == ["tab_density", "page_number_density"]

    def test_resolve_feature_columns_removes_dropped_features(self):
        active = resolve_feature_columns(["tab_density,page_number_density"])
        assert "tab_density" not in active
        assert "page_number_density" not in active
        assert len(active) == len(FEATURE_COLUMNS) - 2

    def test_resolve_feature_columns_raises_for_unknown_feature(self):
        with pytest.raises(ValueError):
            resolve_feature_columns(["does_not_exist"])

    def test_resolve_feature_columns_raises_when_all_features_dropped(self):
        with pytest.raises(ValueError, match="at least one feature must remain"):
            resolve_feature_columns(list(FEATURE_COLUMNS))


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
            "feature_names",
            "sub_type_map",
            "class_weights",
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

    def test_class_weights_length(self, dummy_csv):
        """class_weights array should have length NUM_CATEGORIES."""
        result = load_and_prepare_data(dummy_csv, val_fraction=0.2, seed=42)
        weights = result["class_weights"]
        assert len(weights) == NUM_CATEGORIES

    def test_class_weights_inversely_proportional(self, tmp_path):
        """Less frequent classes should receive higher weights."""
        import random

        random.seed(99)
        csv_path = tmp_path / "imbalanced.csv"
        counts_per_cat = {"prose": 100, "code": 10, "structured": 10, "artifact": 10, "skip": 10}

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = list(FEATURE_COLUMNS) + ["line_count", "category", "sub_type"]
            writer.writerow(header)
            for cat in CATEGORY_MAP:
                for _ in range(counts_per_cat[cat]):
                    features = [round(random.uniform(0.0, 1.0), 4) for _ in FEATURE_COLUMNS]
                    line_count = random.randint(1, 500)
                    sub_type = random.choice(DUMMY_SUB_TYPES)
                    writer.writerow(features + [line_count, cat, sub_type])

        result = load_and_prepare_data(csv_path, val_fraction=0.2, seed=42)
        weights = result["class_weights"]

        prose_weight = weights[CATEGORY_MAP["prose"]]
        code_weight = weights[CATEGORY_MAP["code"]]
        assert code_weight > prose_weight, (
            f"Minority class 'code' (weight={code_weight:.4f}) should have higher "
            f"weight than majority class 'prose' (weight={prose_weight:.4f})"
        )

    def test_class_weights_all_positive(self, dummy_csv):
        """All class weights should be positive."""
        result = load_and_prepare_data(dummy_csv, val_fraction=0.2, seed=42)
        weights = result["class_weights"]
        assert all(w > 0 for w in weights), f"All weights should be positive, got {weights}"

    def test_group_val_by_source_uses_disjoint_sources(self, tmp_path):
        import random

        random.seed(123)
        csv_path = tmp_path / "with_source.csv"
        header = list(FEATURE_COLUMNS) + ["line_count", "category", "sub_type", "source"]
        rows = []
        for category in CATEGORY_MAP:
            for source_idx in range(10):
                source = f"source-{source_idx}"
                features = [round(random.uniform(0.0, 1.0), 4) for _ in FEATURE_COLUMNS]
                rows.append(features + [source_idx + 1, category, "pdf_dump", source])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        result = load_and_prepare_data(csv_path, val_fraction=0.3, seed=42, group_val_by_source=True)
        train_sources = set(result["source_train"])
        val_sources = set(result["source_val"])
        assert train_sources
        assert val_sources
        assert train_sources.isdisjoint(val_sources)

    def test_group_val_by_source_single_source_falls_back_without_error(self, tmp_path):
        csv_path = tmp_path / "single_source.csv"
        _make_source_csv(csv_path, {"golden_train": 120})

        result = load_and_prepare_data(
            csv_path,
            val_fraction=0.2,
            seed=42,
            group_val_by_source=True,
        )
        total = len(result["X_train"]) + len(result["X_val"])
        val_ratio = len(result["X_val"]) / total

        assert len(result["X_train"]) > 0
        assert len(result["X_val"]) > 0
        assert 0.15 <= val_ratio <= 0.25
        assert set(result["source_train"]) == {"golden_train"}
        assert set(result["source_val"]) == {"golden_train"}

    def test_group_val_by_source_targets_row_fraction_for_uneven_groups(self, tmp_path):
        csv_path = tmp_path / "uneven_sources.csv"
        _make_source_csv(
            csv_path,
            {
                "source-large": 70,
                "source-target": 20,
                "source-small": 10,
            },
        )

        result = load_and_prepare_data(
            csv_path,
            val_fraction=0.2,
            seed=42,
            group_val_by_source=True,
        )
        total = len(result["X_train"]) + len(result["X_val"])
        val_ratio = len(result["X_val"]) / total

        assert 0.15 <= val_ratio <= 0.25
        assert set(result["source_train"]).isdisjoint(set(result["source_val"]))

    def test_group_val_by_source_falls_back_when_group_drift_is_extreme(self, tmp_path):
        csv_path = tmp_path / "extreme_uneven_sources.csv"
        _make_source_csv(
            csv_path,
            {
                "dominant-source": 96,
                "tiny-source-a": 2,
                "tiny-source-b": 2,
            },
        )

        result = load_and_prepare_data(
            csv_path,
            val_fraction=0.2,
            seed=42,
            group_val_by_source=True,
        )
        total = len(result["X_train"]) + len(result["X_val"])
        val_ratio = len(result["X_val"]) / total
        train_sources = set(result["source_train"])
        val_sources = set(result["source_val"])

        assert 0.15 <= val_ratio <= 0.25
        assert train_sources.intersection(val_sources)

    def test_group_val_by_source_falls_back_when_best_disjoint_ratio_is_too_far(self, tmp_path):
        csv_path = tmp_path / "coarse_groups.csv"
        _make_source_csv(
            csv_path,
            {
                "source-a": 45,
                "source-b": 45,
                "source-c": 10,
            },
        )

        result = load_and_prepare_data(
            csv_path,
            val_fraction=0.2,
            seed=42,
            group_val_by_source=True,
        )
        total = len(result["X_train"]) + len(result["X_val"])
        val_ratio = len(result["X_val"]) / total
        train_sources = set(result["source_train"])
        val_sources = set(result["source_val"])

        assert 0.15 <= val_ratio <= 0.25
        assert train_sources.intersection(val_sources)

    def test_balance_artifact_subtypes_adds_sampler_weights_for_rare_artifacts(self, tmp_path):
        import random

        random.seed(777)
        csv_path = tmp_path / "artifact_balance.csv"
        header = list(FEATURE_COLUMNS) + ["line_count", "category", "sub_type", "source"]

        rows = []
        for _ in range(80):
            features = [round(random.uniform(0.0, 1.0), 4) for _ in FEATURE_COLUMNS]
            rows.append(features + [1, "prose", "narrative", "book-src"])

        for _ in range(15):
            features = [round(random.uniform(0.0, 1.0), 4) for _ in FEATURE_COLUMNS]
            rows.append(features + [1, "artifact", "pdf_dump", "artifact-major"])

        for _ in range(5):
            features = [round(random.uniform(0.0, 1.0), 4) for _ in FEATURE_COLUMNS]
            rows.append(features + [1, "artifact", "ocr_text", "artifact-minor"])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        result = load_and_prepare_data(
            csv_path,
            val_fraction=0.2,
            seed=42,
            balance_artifact_subtypes=True,
        )
        sample_weights = result["sample_weights_train"]
        y_cat_train = result["y_cat_train"]
        y_sub_train = result["y_sub_train"]
        inverse_sub_map = {v: k for k, v in result["sub_type_map"].items()}

        artifact_weights_by_subtype: dict[str, list[float]] = {"pdf_dump": [], "ocr_text": []}
        for idx, cat_id in enumerate(y_cat_train):
            if cat_id != CATEGORY_MAP["artifact"]:
                continue
            sub_type = inverse_sub_map[int(y_sub_train[idx])]
            if sub_type in artifact_weights_by_subtype:
                artifact_weights_by_subtype[sub_type].append(float(sample_weights[idx]))

        assert artifact_weights_by_subtype["pdf_dump"]
        assert artifact_weights_by_subtype["ocr_text"]
        assert (
            sum(artifact_weights_by_subtype["ocr_text"]) / len(artifact_weights_by_subtype["ocr_text"])
            > sum(artifact_weights_by_subtype["pdf_dump"]) / len(artifact_weights_by_subtype["pdf_dump"])
        )

    def test_train_model_uses_weighted_sampler_when_balancing_enabled(self, tmp_path):
        import random
        import torch
        from unittest import mock

        random.seed(9)
        csv_path = tmp_path / "sampler.csv"
        header = list(FEATURE_COLUMNS) + ["line_count", "category", "sub_type"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for category in CATEGORY_MAP:
                for _ in range(40):
                    features = [round(random.uniform(0.0, 1.0), 4) for _ in FEATURE_COLUMNS]
                    sub_type = "pdf_dump" if category == "artifact" else "narrative"
                    writer.writerow(features + [2, category, sub_type])

        data = load_and_prepare_data(
            csv_path,
            val_fraction=0.2,
            seed=42,
            balance_artifact_subtypes=True,
        )

        with mock.patch(
            "torch.utils.data.WeightedRandomSampler",
            wraps=torch.utils.data.WeightedRandomSampler,
        ) as mock_sampler:
            train_model(
                data,
                n_sub_types=len(data["sub_type_map"]),
                epochs=1,
                batch_size=32,
                patience=2,
                balance_artifact_subtypes=True,
            )
            assert mock_sampler.called, "Expected WeightedRandomSampler to be used when balancing is enabled"


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
        assert config["feature_names"] == list(FEATURE_COLUMNS)
        assert len(config["feature_mean"]) == len(FEATURE_COLUMNS)
        assert len(config["feature_std"]) == len(FEATURE_COLUMNS)

    def test_model_config_uses_active_feature_names_after_drops(self, dummy_csv, output_dir):
        dropped = ["tab_density", "page_number_density", "table_fragment_score"]
        result = subprocess.run(
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
                "--drop-features",
                ",".join(dropped),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Script failed:\nstderr={result.stderr}\nstdout={result.stdout}"

        config = json.loads((output_dir / "model_config.json").read_text())
        expected_active = [f for f in FEATURE_COLUMNS if f not in dropped]
        assert config["feature_names"] == expected_active
        assert len(config["feature_mean"]) == len(expected_active)
        assert len(config["feature_std"]) == len(expected_active)

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
