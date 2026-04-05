"""Tests for trainr.core.drop_synthetic — synthetic data removal pipeline."""

from pathlib import Path

import polars as pl

from trainr.core.drop_synthetic import filter_synthetic, run_drop_synthetic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_real: int = 20, n_synthetic: int = 10) -> pl.DataFrame:
    """Create a minimal DataFrame with real and synthetic rows."""
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
    return pl.DataFrame(rows)


def _make_parquet(path: Path, n_real: int = 20, n_synthetic: int = 10) -> Path:
    """Create a minimal Parquet file with real and synthetic rows."""
    df = _make_df(n_real, n_synthetic)
    df.write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Tests for filter_synthetic
# ---------------------------------------------------------------------------


class TestFilterSynthetic:
    """Tests for the filter_synthetic function."""

    def test_removes_unknown_source_rows(self) -> None:
        df = _make_df()
        filtered, dropped = filter_synthetic(df)
        assert filtered.filter(pl.col("source") == "unknown").height == 0

    def test_keeps_all_real_rows(self) -> None:
        df = _make_df(n_real=20, n_synthetic=10)
        filtered, dropped = filter_synthetic(df)
        assert filtered.height == 20

    def test_dropped_contains_only_unknown(self) -> None:
        df = _make_df(n_real=20, n_synthetic=10)
        filtered, dropped = filter_synthetic(df)
        assert dropped.height == 10
        assert (dropped["source"] == "unknown").all()

    def test_total_rows_preserved(self) -> None:
        df = _make_df(n_real=25, n_synthetic=15)
        filtered, dropped = filter_synthetic(df)
        assert filtered.height + dropped.height == 40

    def test_no_synthetic_rows(self) -> None:
        df = _make_df(n_real=10, n_synthetic=0)
        filtered, dropped = filter_synthetic(df)
        assert filtered.height == 10
        assert dropped.height == 0

    def test_all_synthetic_rows(self) -> None:
        df = _make_df(n_real=0, n_synthetic=10)
        filtered, dropped = filter_synthetic(df)
        assert filtered.height == 0
        assert dropped.height == 10

    def test_case_sensitive_match(self) -> None:
        """Ensure 'Unknown' (capital U) is NOT dropped — only exact 'unknown'."""
        df = pl.DataFrame([
            {"text": "a", "category": "prose", "sub_type": "x", "source": "Unknown"},
            {"text": "b", "category": "prose", "sub_type": "x", "source": "unknown"},
            {"text": "c", "category": "prose", "sub_type": "x", "source": "UNKNOWN"},
        ])
        filtered, dropped = filter_synthetic(df)
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

        run_drop_synthetic(input_path=input_path, archive_dir=archive_dir)
        assert (archive_dir / "synthetic_dropped.parquet").exists()

    def test_atomic_write_preserves_input_on_no_crash(self, tmp_path: Path) -> None:
        """Verify input is overwritten atomically via temp file rename."""
        input_path = _make_parquet(tmp_path / "input.parquet", n_real=15, n_synthetic=5)
        archive_dir = tmp_path / "archive"

        run_drop_synthetic(input_path=input_path, archive_dir=archive_dir)

        # After successful run, no temp file should remain
        tmp_files = list(tmp_path.glob("*.tmp.parquet"))
        assert len(tmp_files) == 0

        result = pl.read_parquet(input_path)
        assert result.height == 15

    def test_empty_archive_skips_file(self, tmp_path: Path) -> None:
        """When no rows are dropped, archive file should not be created."""
        input_path = _make_parquet(tmp_path / "input.parquet", n_real=10, n_synthetic=0)
        archive_dir = tmp_path / "archive"

        run_drop_synthetic(input_path=input_path, archive_dir=archive_dir)

        archive_path = archive_dir / "synthetic_dropped.parquet"
        assert not archive_path.exists()
