"""Smoke tests for trainr.core.audit_semantic_labels."""

import polars as pl

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

