"""Tests for the FAISS two-layer dedup pipeline."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure the training directory is on the import path.
sys.path.insert(0, os.path.dirname(__file__))

from dedup import dedup_pipeline, feature_dedup, semantic_dedup


# ---------------------------------------------------------------------------
# feature_dedup tests
# ---------------------------------------------------------------------------


class TestFeatureDedup:
    """Tests for feature-space deduplication."""

    def _make_df(self, rows: list[list[float]], cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=cols)

    def test_removes_exact_duplicates(self):
        cols = [f"f{i}" for i in range(3)]
        rows = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],  # exact duplicate
            [9.0, 8.0, 7.0],
        ]
        df = self._make_df(rows, cols)
        result = feature_dedup(df, cols, threshold=0.1)
        assert len(result) == 2
        # First occurrence kept, duplicate removed
        assert list(result.iloc[0]) == [1.0, 2.0, 3.0]
        assert list(result.iloc[1]) == [9.0, 8.0, 7.0]

    def test_removes_near_duplicates_within_threshold(self):
        cols = [f"f{i}" for i in range(3)]
        rows = [
            [1.0, 2.0, 3.0],
            [1.01, 2.01, 3.01],  # very close
            [9.0, 8.0, 7.0],
        ]
        df = self._make_df(rows, cols)
        result = feature_dedup(df, cols, threshold=0.1)
        assert len(result) == 2

    def test_keeps_distinct_samples(self):
        cols = [f"f{i}" for i in range(3)]
        rows = [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
        ]
        df = self._make_df(rows, cols)
        result = feature_dedup(df, cols, threshold=0.1)
        assert len(result) == 3

    def test_preserves_non_feature_columns(self):
        cols = ["f0", "f1"]
        df = pd.DataFrame(
            {"f0": [1.0, 1.0, 5.0], "f1": [2.0, 2.0, 6.0], "label": ["a", "b", "c"]}
        )
        result = feature_dedup(df, cols, threshold=0.1)
        assert "label" in result.columns
        assert len(result) == 2
        # First occurrence kept
        assert result.iloc[0]["label"] == "a"

    def test_single_row_unchanged(self):
        cols = ["f0"]
        df = pd.DataFrame({"f0": [1.0]})
        result = feature_dedup(df, cols, threshold=0.1)
        assert len(result) == 1

    def test_logs_removal_count(self, caplog):
        import logging

        cols = ["f0"]
        df = pd.DataFrame({"f0": [1.0, 1.0, 5.0]})
        with caplog.at_level(logging.INFO):
            feature_dedup(df, cols, threshold=0.1)
        assert "Feature dedup: removed 1 of 3 samples" in caplog.text


# ---------------------------------------------------------------------------
# semantic_dedup tests
# ---------------------------------------------------------------------------


class TestSemanticDedup:
    """Tests for semantic deduplication."""

    def test_removes_identical_texts(self):
        df = pd.DataFrame({"text": ["hello world", "hello world", "quantum physics"]})
        result = semantic_dedup(df, text_col="text", threshold=0.9)
        assert len(result) == 2

    def test_removes_near_identical_texts(self):
        df = pd.DataFrame(
            {
                "text": [
                    "The quick brown fox jumps over the lazy dog.",
                    "The quick brown fox jumped over the lazy dog.",
                    "Quantum mechanics is a fundamental theory in physics.",
                ]
            }
        )
        result = semantic_dedup(df, text_col="text", threshold=0.9)
        # The two fox sentences should be very similar
        assert len(result) <= 2

    def test_keeps_distinct_texts(self):
        df = pd.DataFrame(
            {
                "text": [
                    "The weather is sunny today.",
                    "import numpy as np; x = np.array([1,2,3])",
                    "SELECT * FROM users WHERE id = 42",
                ]
            }
        )
        result = semantic_dedup(df, text_col="text", threshold=0.9)
        assert len(result) == 3

    def test_preserves_non_text_columns(self):
        df = pd.DataFrame(
            {
                "text": ["hello world", "hello world", "something different"],
                "label": ["a", "b", "c"],
            }
        )
        result = semantic_dedup(df, text_col="text", threshold=0.9)
        assert "label" in result.columns
        assert result.iloc[0]["label"] == "a"

    def test_single_row_unchanged(self):
        df = pd.DataFrame({"text": ["hello world"]})
        result = semantic_dedup(df, text_col="text", threshold=0.9)
        assert len(result) == 1

    def test_logs_removal_count(self, caplog):
        import logging

        df = pd.DataFrame({"text": ["hello world", "hello world", "something else"]})
        with caplog.at_level(logging.INFO):
            semantic_dedup(df, text_col="text", threshold=0.9)
        assert "Semantic dedup: removed" in caplog.text


# ---------------------------------------------------------------------------
# dedup_pipeline tests
# ---------------------------------------------------------------------------


class TestDedupPipeline:
    """Tests for the end-to-end pipeline."""

    def test_pipeline_reads_and_writes_csv(self, tmp_path):
        input_csv = str(tmp_path / "input.csv")
        output_csv = str(tmp_path / "output.csv")

        df = pd.DataFrame(
            {
                "text": [
                    "hello world",
                    "hello world",
                    "completely different text about science",
                ],
                "f0": [1.0, 1.0, 9.0],
                "f1": [2.0, 2.0, 8.0],
            }
        )
        df.to_csv(input_csv, index=False)

        dedup_pipeline(
            input_csv=input_csv,
            output_csv=output_csv,
            feature_threshold=0.1,
            semantic_threshold=0.9,
        )

        result = pd.read_csv(output_csv)
        # At least the exact feature duplicate should be removed
        assert len(result) < len(df)
        assert os.path.exists(output_csv)

    def test_pipeline_prints_summary(self, tmp_path, capsys):
        input_csv = str(tmp_path / "input.csv")
        output_csv = str(tmp_path / "output.csv")

        df = pd.DataFrame(
            {
                "text": ["a", "a", "b"],
                "f0": [1.0, 1.0, 9.0],
                "f1": [2.0, 2.0, 8.0],
            }
        )
        df.to_csv(input_csv, index=False)

        dedup_pipeline(
            input_csv=input_csv,
            output_csv=output_csv,
            feature_threshold=0.1,
            semantic_threshold=0.9,
        )

        captured = capsys.readouterr()
        assert "Original count:" in captured.out
        assert "Final count:" in captured.out


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the argparse CLI entry point."""

    def test_cli_runs(self, tmp_path):
        input_csv = str(tmp_path / "input.csv")
        output_csv = str(tmp_path / "output.csv")

        df = pd.DataFrame(
            {
                "text": ["hello", "hello", "world"],
                "f0": [1.0, 1.0, 9.0],
                "f1": [2.0, 2.0, 8.0],
            }
        )
        df.to_csv(input_csv, index=False)

        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "dedup.py"),
                "--input",
                input_csv,
                "--output",
                output_csv,
                "--feature-threshold",
                "0.1",
                "--semantic-threshold",
                "0.9",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert os.path.exists(output_csv)
