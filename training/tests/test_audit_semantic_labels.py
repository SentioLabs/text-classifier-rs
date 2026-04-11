"""Smoke tests for trainr.core.audit_semantic_labels."""

import tempfile
from pathlib import Path

import polars as pl


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
