"""Smoke tests for trainr.core.audit_semantic_labels."""

import re
from pathlib import Path

import polars as pl
import pytest

from trainr.core.audit_semantic_labels import compute_prevalence_per_label


def _make_fake_annotated(
    audit_source_values: list[str],
    det_log_content: list[int],
    det_stack_trace: list[int],
    det_diff_patch: list[int],
) -> pl.DataFrame:
    """Build a minimal annotated-parquet fixture."""
    n = len(audit_source_values)
    return pl.DataFrame({
        "text": [f"row {i}" for i in range(n)],
        "sub_type": ["plain"] * n,
        "audit_source": audit_source_values,
        "det_log_content": det_log_content,
        "det_stack_trace": det_stack_trace,
        "det_diff_patch": det_diff_patch,
    })


def test_compute_recall_on_injected_positives_all_correct():
    from trainr.core.audit_semantic_labels import compute_recall_on_injected

    df = _make_fake_annotated(
        audit_source_values=[
            "inject_stack_trace", "inject_stack_trace",
            "inject_diff_patch",
            "inject_log_content", "inject_log_content",
            "stratified",
        ],
        det_log_content=[0, 0, 0, 1, 1, 0],
        det_stack_trace=[1, 1, 0, 0, 0, 0],
        det_diff_patch=[0, 0, 1, 0, 0, 0],
    )
    recall = compute_recall_on_injected(df)
    assert recall["stack_trace"] == 1.0
    assert recall["diff_patch"] == 1.0
    assert recall["log_content"] == 1.0


def test_compute_recall_on_injected_positives_partial():
    from trainr.core.audit_semantic_labels import compute_recall_on_injected

    df = _make_fake_annotated(
        audit_source_values=[
            "inject_stack_trace", "inject_stack_trace", "inject_stack_trace",
            "inject_stack_trace", "inject_stack_trace",
        ],
        det_log_content=[0, 0, 0, 0, 0],
        det_stack_trace=[1, 1, 1, 1, 0],  # 4/5 fired
        det_diff_patch=[0, 0, 0, 0, 0],
    )
    recall = compute_recall_on_injected(df)
    assert recall["stack_trace"] == 0.8


def test_agreement_excludes_injected_rows():
    """Inter-annotator agreement must be computed on stratified rows only."""
    from trainr.core.audit_semantic_labels import filter_for_agreement

    df = _make_fake_annotated(
        audit_source_values=[
            "stratified", "stratified", "inject_stack_trace",
        ],
        det_log_content=[0, 0, 1],
        det_stack_trace=[0, 0, 1],
        det_diff_patch=[0, 0, 0],
    )
    filtered = filter_for_agreement(df)
    assert len(filtered) == 2
    assert all(s == "stratified" for s in filtered["audit_source"].to_list())


class TestComputePrevalencePerLabel:
    def test_majority_fire_rate_single_model(self):
        # Single model: prevalence is simply mean fire rate.
        df = pl.DataFrame({
            "audit_source": ["stratified"] * 10,
            "det_python": [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "det_markdown": [0] * 10,
        })
        result = compute_prevalence_per_label({"only": df})
        assert result["python"] == 0.3
        assert result["markdown"] == 0.0

    def test_majority_of_three_fire(self):
        # 3 models: prevalence = fraction of rows where majority (>=2 of 3) fires.
        base = pl.DataFrame({
            "audit_source": ["stratified"] * 4,
        })
        df1 = base.with_columns(pl.Series("det_python", [1, 1, 0, 0]))
        df2 = base.with_columns(pl.Series("det_python", [1, 0, 1, 0]))
        df3 = base.with_columns(pl.Series("det_python", [0, 1, 1, 0]))
        # Row-by-row: 2+, 2+, 2+, 0 → majority fires on rows 0, 1, 2 → prev = 3/4
        result = compute_prevalence_per_label({"a": df1, "b": df2, "c": df3})
        assert result["python"] == 0.75

    def test_zero_rows_returns_zero(self):
        df = pl.DataFrame({
            "audit_source": pl.Series([], dtype=pl.Utf8),
            "det_python": pl.Series([], dtype=pl.Int64),
        })
        result = compute_prevalence_per_label({"only": df})
        assert result["python"] == 0.0

    def test_filters_out_injected_rows(self):
        # Only stratified rows count. Injected rows that fire must be excluded.
        df = pl.DataFrame({
            "audit_source": ["stratified", "stratified", "inject_det_python"],
            "det_python": [0, 0, 1],  # only the injected row fires
        })
        result = compute_prevalence_per_label({"only": df})
        assert result["python"] == 0.0

    def test_skips_label_missing_from_some_models(self):
        # det_markdown is in df_a only. Function must skip it rather than crash.
        df_a = pl.DataFrame({
            "audit_source": ["stratified"] * 2,
            "det_python": [1, 0],
            "det_markdown": [0, 0],
        })
        df_b = pl.DataFrame({
            "audit_source": ["stratified"] * 2,
            "det_python": [1, 1],
            # intentionally missing det_markdown
        })
        result = compute_prevalence_per_label({"a": df_a, "b": df_b})
        assert "python" in result
        assert "markdown" not in result, (
            "Labels missing from one frame must be skipped, not included"
        )


class TestLoadAnnotatorParquets:
    def test_expected_slug_set_constant(self):
        from trainr.core.audit_semantic_labels import EXPECTED_MODEL_SLUGS
        assert EXPECTED_MODEL_SLUGS == frozenset({"gemini3flash", "sonnet", "gpt54mini"})

    def test_loads_three_parquets_by_slug(self, tmp_path):
        from trainr.core.audit_semantic_labels import load_annotator_parquets

        def _make(path: Path, val: int):
            pl.DataFrame({
                "audit_source": ["stratified"],
                "det_python": [val],
            }).write_parquet(path)

        _make(tmp_path / "iter17_ab_iter15_gemini3flash.parquet", 1)
        _make(tmp_path / "iter17_ab_iter15_sonnet.parquet", 0)
        _make(tmp_path / "iter17_ab_iter15_gpt54mini.parquet", 1)

        paths = sorted(tmp_path.glob("iter17_ab_iter15_*.parquet"))
        result = load_annotator_parquets(paths)

        assert set(result.keys()) == {"gemini3flash", "sonnet", "gpt54mini"}
        assert result["gemini3flash"]["det_python"][0] == 1
        assert result["sonnet"]["det_python"][0] == 0

    def test_rejects_wrong_count(self, tmp_path):
        from trainr.core.audit_semantic_labels import load_annotator_parquets

        pl.DataFrame({"audit_source": ["stratified"]}).write_parquet(
            tmp_path / "iter17_ab_iter15_gemini3flash.parquet"
        )
        paths = [tmp_path / "iter17_ab_iter15_gemini3flash.parquet"]
        with pytest.raises(ValueError, match="expected 3 parquets"):
            load_annotator_parquets(paths)

    def test_rejects_duplicate_slug(self, tmp_path):
        from trainr.core.audit_semantic_labels import load_annotator_parquets

        def _make(path: Path):
            pl.DataFrame({"audit_source": ["stratified"]}).write_parquet(path)

        _make(tmp_path / "iter17_ab_iter15_gemini3flash.parquet")
        _make(tmp_path / "iter17_ab_iter15_sonnet.parquet")
        _make(tmp_path / "iter17_ab_iter15_sonnet_copy.parquet")
        # The "copy" one won't match the regex expected set and will be rejected.
        paths = sorted(tmp_path.glob("iter17_ab_iter15_*.parquet"))
        with pytest.raises(ValueError):
            load_annotator_parquets(paths)

    def test_rejects_unknown_slug(self, tmp_path):
        from trainr.core.audit_semantic_labels import load_annotator_parquets

        def _make(path: Path):
            pl.DataFrame({"audit_source": ["stratified"]}).write_parquet(path)

        _make(tmp_path / "iter17_ab_iter15_gemini3flash.parquet")
        _make(tmp_path / "iter17_ab_iter15_sonnet.parquet")
        _make(tmp_path / "iter17_ab_iter15_claude35.parquet")

        paths = sorted(tmp_path.glob("iter17_ab_iter15_*.parquet"))
        with pytest.raises(ValueError, match="unexpected slugs"):
            load_annotator_parquets(paths)

