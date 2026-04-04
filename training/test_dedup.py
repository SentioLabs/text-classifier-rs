"""Tests for the FAISS two-layer dedup pipeline."""

import os
import sys
import tempfile

import numpy as np
import polars as pl
import pytest

# Ensure the training directory is on the import path.
sys.path.insert(0, os.path.dirname(__file__))

from dedup import dedup_pipeline, feature_dedup, semantic_dedup


# ---------------------------------------------------------------------------
# feature_dedup tests
# ---------------------------------------------------------------------------


class TestFeatureDedup:
    """Tests for feature-space deduplication."""

    def _make_df(self, rows: list[list[float]], cols: list[str]) -> pl.DataFrame:
        return pl.DataFrame(rows, schema=cols, orient="row")

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
        dicts = result.to_dicts()
        assert dicts[0] == {"f0": 1.0, "f1": 2.0, "f2": 3.0}
        assert dicts[1] == {"f0": 9.0, "f1": 8.0, "f2": 7.0}

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
        df = pl.DataFrame(
            {"f0": [1.0, 1.0, 5.0], "f1": [2.0, 2.0, 6.0], "label": ["a", "b", "c"]}
        )
        result = feature_dedup(df, cols, threshold=0.1)
        assert "label" in result.columns
        assert len(result) == 2
        # First occurrence kept
        assert result.get_column("label").to_list()[0] == "a"

    def test_single_row_unchanged(self):
        cols = ["f0"]
        df = pl.DataFrame({"f0": [1.0]})
        result = feature_dedup(df, cols, threshold=0.1)
        assert len(result) == 1

    def test_logs_removal_count(self, caplog):
        import logging

        cols = ["f0"]
        df = pl.DataFrame({"f0": [1.0, 1.0, 5.0]})
        with caplog.at_level(logging.INFO):
            feature_dedup(df, cols, threshold=0.1)
        assert "Feature dedup: removed 1 of 3 samples" in caplog.text


# ---------------------------------------------------------------------------
# semantic_dedup tests
# ---------------------------------------------------------------------------


class TestSemanticDedup:
    """Tests for semantic deduplication."""

    def test_removes_identical_texts(self):
        df = pl.DataFrame({"text": ["hello world", "hello world", "quantum physics"]})
        result = semantic_dedup(df, text_col="text", threshold=0.9)
        assert len(result) == 2

    def test_removes_near_identical_texts(self):
        df = pl.DataFrame(
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
        df = pl.DataFrame(
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
        df = pl.DataFrame(
            {
                "text": ["hello world", "hello world", "something different"],
                "label": ["a", "b", "c"],
            }
        )
        result = semantic_dedup(df, text_col="text", threshold=0.9)
        assert "label" in result.columns
        assert result.get_column("label").to_list()[0] == "a"

    def test_single_row_unchanged(self):
        df = pl.DataFrame({"text": ["hello world"]})
        result = semantic_dedup(df, text_col="text", threshold=0.9)
        assert len(result) == 1

    def test_logs_removal_count(self, caplog):
        import logging

        df = pl.DataFrame({"text": ["hello world", "hello world", "something else"]})
        with caplog.at_level(logging.INFO):
            semantic_dedup(df, text_col="text", threshold=0.9)
        assert "Semantic dedup: removed" in caplog.text


# ---------------------------------------------------------------------------
# dedup_pipeline tests
# ---------------------------------------------------------------------------


class TestDedupPipeline:
    """Tests for the end-to-end pipeline."""

    def test_pipeline_reads_and_writes_parquet(self, tmp_path):
        input_path = str(tmp_path / "input.parquet")
        output_path = str(tmp_path / "output.parquet")

        df = pl.DataFrame(
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
        df.write_parquet(input_path)

        dedup_pipeline(
            input_csv=input_path,
            output_csv=output_path,
            feature_threshold=0.1,
            semantic_threshold=0.9,
        )

        result = pl.read_parquet(output_path)
        # At least the exact feature duplicate should be removed
        assert len(result) < len(df)
        assert os.path.exists(output_path)

    def test_pipeline_prints_summary(self, tmp_path, capsys):
        input_path = str(tmp_path / "input.parquet")
        output_path = str(tmp_path / "output.parquet")

        df = pl.DataFrame(
            {
                "text": ["a", "a", "b"],
                "f0": [1.0, 1.0, 9.0],
                "f1": [2.0, 2.0, 8.0],
            }
        )
        df.write_parquet(input_path)

        dedup_pipeline(
            input_csv=input_path,
            output_csv=output_path,
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
        input_path = str(tmp_path / "input.parquet")
        output_path = str(tmp_path / "output.parquet")

        df = pl.DataFrame(
            {
                "text": ["hello", "hello", "world"],
                "f0": [1.0, 1.0, 9.0],
                "f1": [2.0, 2.0, 8.0],
            }
        )
        df.write_parquet(input_path)

        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "dedup.py"),
                "--input",
                input_path,
                "--output",
                output_path,
                "--feature-threshold",
                "0.1",
                "--semantic-threshold",
                "0.9",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert os.path.exists(output_path)
