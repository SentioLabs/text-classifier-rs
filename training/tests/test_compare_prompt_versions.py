"""Unit tests for compare_prompt_versions.py.

Fixture pattern: construct small polars DataFrames in-memory and call the
public API directly. No filesystem I/O except where explicitly testing
filesystem-interacting functions (glob handling, parquet reading).
"""

from __future__ import annotations

import polars as pl
import pytest

from trainr.core.compare_prompt_versions import (
    DeltaReport,
    LabelVerdict,
    compare_prompt_versions,
)


def _make_input_frame(n_strat: int = 10, n_inject: int = 2) -> pl.DataFrame:
    """Minimal input parquet fixture matching the real iter16_5k_input schema."""
    rows = [
        {"text": f"row-{i}", "sub_type": "python", "audit_source": "stratified"}
        for i in range(n_strat)
    ] + [
        {"text": f"inj-{i}", "sub_type": "python", "audit_source": "inject_det_python"}
        for i in range(n_inject)
    ]
    return pl.DataFrame(rows)


def _make_annotator_frame(
    input_frame: pl.DataFrame,
    det_columns: dict[str, list[int]],
) -> pl.DataFrame:
    """Clone the input frame and append det_* columns with given values."""
    result = input_frame.clone()
    for col, values in det_columns.items():
        result = result.with_columns(pl.Series(col, values))
    return result


class TestHappyPath:
    def test_all_shared_all_pass(self):
        """Baseline: 3 shared labels, zero delta, identical noise floor, all PASS."""
        input_frame = _make_input_frame(n_strat=10, n_inject=0)
        # Every model fires det_python on rows 0-2 (30% prevalence).
        votes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        before_frames = {
            "gemini3flash": _make_annotator_frame(input_frame, {"det_python": votes}),
            "sonnet": _make_annotator_frame(input_frame, {"det_python": votes}),
            "gpt54mini": _make_annotator_frame(input_frame, {"det_python": votes}),
        }
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames = {k: v.clone() for k, v in before_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

        assert isinstance(report, DeltaReport)
        assert "python" in report.shared_labels
        py = report.labels["python"]
        assert py.verdict == LabelVerdict.PASS
        assert py.delta_agreement == pytest.approx(0.0)
        assert py.iter15_prevalence == pytest.approx(0.3)
        assert py.iter16_prevalence == pytest.approx(0.3)
        assert py.prevalence_ratio == pytest.approx(1.0)
        assert py.noise_floor == pytest.approx(0.0)
