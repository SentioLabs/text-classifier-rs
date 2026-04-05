"""Tests for trainr.core.drop_synthetic — synthetic data removal pipeline."""

import tempfile
from pathlib import Path

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parquet(path: Path, n_real: int = 20, n_synthetic: int = 10) -> Path:
    """Create a minimal Parquet file with real and synthetic rows."""
    rows = []
    for i in range(n_real):
        rows.append({
            "text": f"real text {i}",
            "category": "prose",
            "sub_type": "article" if i % 2 == 0 else "essay",
            "source": f"real/source_{i % 3}",
        })
    for i in range(n_synthetic):
        rows.append({
            "text": f"synthetic text {i}",
            "category": "code",
            "sub_type": "python" if i % 2 == 0 else "javascript",
            "source": "unknown",
        })
    df = pl.DataFrame(rows)
    df.write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Tests for filter_synthetic
# ---------------------------------------------------------------------------


class TestFilterSynthetic:
    """Tests for the filter_synthetic function."""

    def test_removes_unknown_source_rows(self, tmp_path: Path) -> None:
        input_path = _make_parquet(tmp_path / "input.parquet")
        from trainr.core.drop_synthetic import filter_synthetic

        filtered, dropped = filter_synthetic(input_path)
        assert filtered.filter(pl.col("source") == "unknown").height == 0

    def test_keeps_all_real_rows(self, tmp_path: Path) -> None:
        input_path = _make_parquet(tmp_path / "input.parquet", n_real=20, n_synthetic=10)
        from trainr.core.drop_synthetic import filter_synthetic

        filtered, dropped = filter_synthetic(input_path)
        assert filtered.height == 20

    def test_dropped_contains_only_unknown(self, tmp_path: Path) -> None:
        input_path = _make_parquet(tmp_path / "input.parquet", n_real=20, n_synthetic=10)
        from trainr.core.drop_synthetic import filter_synthetic

        filtered, dropped = filter_synthetic(input_path)
        assert dropped.height == 10
        assert (dropped["source"] == "unknown").all()

    def test_total_rows_preserved(self, tmp_path: Path) -> None:
        input_path = _make_parquet(tmp_path / "input.parquet", n_real=25, n_synthetic=15)
        from trainr.core.drop_synthetic import filter_synthetic

        filtered, dropped = filter_synthetic(input_path)
        assert filtered.height + dropped.height == 40

    def test_no_synthetic_rows(self, tmp_path: Path) -> None:
        input_path = _make_parquet(tmp_path / "input.parquet", n_real=10, n_synthetic=0)
        from trainr.core.drop_synthetic import filter_synthetic

        filtered, dropped = filter_synthetic(input_path)
        assert filtered.height == 10
        assert dropped.height == 0

    def test_all_synthetic_rows(self, tmp_path: Path) -> None:
        input_path = _make_parquet(tmp_path / "input.parquet", n_real=0, n_synthetic=10)
        from trainr.core.drop_synthetic import filter_synthetic

        filtered, dropped = filter_synthetic(input_path)
        assert filtered.height == 0
        assert dropped.height == 10

    def test_case_sensitive_match(self, tmp_path: Path) -> None:
        """Ensure 'Unknown' (capital U) is NOT dropped — only exact 'unknown'."""
        df = pl.DataFrame([
            {"text": "a", "category": "prose", "sub_type": "x", "source": "Unknown"},
            {"text": "b", "category": "prose", "sub_type": "x", "source": "unknown"},
            {"text": "c", "category": "prose", "sub_type": "x", "source": "UNKNOWN"},
        ])
        path = tmp_path / "input.parquet"
        df.write_parquet(path)
        from trainr.core.drop_synthetic import filter_synthetic

        filtered, dropped = filter_synthetic(path)
        assert filtered.height == 2
        assert dropped.height == 1


# ---------------------------------------------------------------------------
# Tests for run_drop_synthetic (full pipeline)
# ---------------------------------------------------------------------------


class TestRunDropSynthetic:
    """Tests for the full pipeline that writes files."""

    def test_overwrites_input_and_creates_archive(self, tmp_path: Path) -> None:
        input_path = _make_parquet(tmp_path / "input.parquet", n_real=20, n_synthetic=10)
        archive_dir = tmp_path / "archive"
        from trainr.core.drop_synthetic import run_drop_synthetic

        run_drop_synthetic(input_path=input_path, archive_dir=archive_dir)

        # Input should be overwritten with filtered data
        result = pl.read_parquet(input_path)
        assert result.height == 20
        assert result.filter(pl.col("source") == "unknown").height == 0

        # Archive should contain dropped rows
        archive_path = archive_dir / "synthetic_dropped.parquet"
        assert archive_path.exists()
        archive = pl.read_parquet(archive_path)
        assert archive.height == 10
        assert (archive["source"] == "unknown").all()

    def test_creates_archive_dir_if_missing(self, tmp_path: Path) -> None:
        input_path = _make_parquet(tmp_path / "input.parquet")
        archive_dir = tmp_path / "nested" / "archive"
        assert not archive_dir.exists()
        from trainr.core.drop_synthetic import run_drop_synthetic

        run_drop_synthetic(input_path=input_path, archive_dir=archive_dir)
        assert (archive_dir / "synthetic_dropped.parquet").exists()
