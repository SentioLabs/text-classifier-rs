"""Unit tests for compare_prompt_versions.py.

Fixture pattern: construct small polars DataFrames in-memory and call the
public API directly. No filesystem I/O except where explicitly testing
filesystem-interacting functions (glob handling, parquet reading).
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from trainr.core.compare_prompt_versions import (
    AGREEMENT_DELTA_THRESHOLD,
    DeltaReport,
    LabelCategory,
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


class TestColumnCategorization:
    def _make_frames(self, det_columns: list[str]) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        frames = {}
        for slug in ("gemini3flash", "sonnet", "gpt54mini"):
            frames[slug] = _make_annotator_frame(
                input_frame,
                {col: [0] * 5 for col in det_columns},
            )
        return frames, input_frame

    def test_iter15_only_label_categorized(self):
        before_frames, input_frame = self._make_frames(["det_python", "det_log_lines"])
        after_frames, _ = self._make_frames(["det_python"])
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

        assert "log_lines" in report.iter15_only_labels
        assert "log_lines" in report.labels
        assert report.labels["log_lines"].category == LabelCategory.ITER15_ONLY
        assert report.labels["log_lines"].verdict == LabelVerdict.NO_VERDICT
        assert report.labels["log_lines"].iter16_agreement is None
        assert report.labels["log_lines"].delta_agreement is None
        assert "python" in report.shared_labels

    def test_iter16_only_label_categorized(self):
        before_frames, input_frame = self._make_frames(["det_python"])
        after_frames, _ = self._make_frames(["det_python", "det_log_content"])
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

        assert "log_content" in report.iter16_only_labels
        assert report.labels["log_content"].category == LabelCategory.ITER16_ONLY
        assert report.labels["log_content"].verdict == LabelVerdict.NO_VERDICT
        assert report.labels["log_content"].iter15_agreement is None

    def test_mixed_asymmetry(self):
        before_frames, input_frame = self._make_frames(["det_python", "det_log_lines"])
        after_frames, _ = self._make_frames(["det_python", "det_log_content"])
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

        assert report.shared_labels == ["python"]
        assert report.iter15_only_labels == ["log_lines"]
        assert report.iter16_only_labels == ["log_content"]

    def test_after_and_noise_floor_column_mismatch_raises(self):
        before_frames, input_frame = self._make_frames(["det_python"])
        after_frames, _ = self._make_frames(["det_python", "det_log_content"])
        # noise_floor missing det_log_content — this must fail loud.
        noise_frames, _ = self._make_frames(["det_python"])

        with pytest.raises(ValueError, match="det_log_content"):
            compare_prompt_versions(
                before_frames=before_frames,
                after_frames=after_frames,
                noise_floor_frames=noise_frames,
                input_frame=input_frame,
            )


class TestVerdictLogic:
    """Verdict logic per the spec's hard/soft gate rules."""

    def _frames_with_votes(
        self,
        before_votes: list[list[int]],
        after_votes: list[list[int]],
        noise_votes: list[list[int]] | None = None,
    ) -> tuple[dict, dict, dict, pl.DataFrame]:
        """Build 3-model frame sets from per-model vote lists.

        Each *_votes arg is a list of 3 lists (one per model), each of
        length n_rows. Returns (before_frames, after_frames, noise_frames,
        input_frame).
        """
        n_rows = len(before_votes[0])
        input_frame = _make_input_frame(n_strat=n_rows, n_inject=0)
        noise_votes_actual = noise_votes if noise_votes is not None else after_votes

        def _build(vote_lists):
            return {
                slug: _make_annotator_frame(input_frame, {"det_python": votes})
                for slug, votes in zip(
                    ("gemini3flash", "sonnet", "gpt54mini"),
                    vote_lists,
                )
            }

        return _build(before_votes), _build(after_votes), _build(noise_votes_actual), input_frame

    def test_pass_when_delta_zero(self):
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[1, 1, 0, 0]] * 3,  # unanimous 0.5 prev, agr=1.0
            after_votes=[[1, 1, 0, 0]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        assert report.labels["python"].verdict == LabelVerdict.PASS

    def test_fail_agreement_hard_gate(self):
        # iter15 unanimous on all 10 rows (agr=1.0); iter16 has 1 split (2-1)
        # on row 9, giving iter16_agr = (9 + 2/3) / 10 ≈ 0.9666. Δ ≈ -0.0333
        # which exceeds the 0.005 threshold → FAIL-agreement.
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            after_votes=[
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # row 9 dissent
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            # noise == after → noise_floor = 0
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        assert abs(row.delta_agreement) > AGREEMENT_DELTA_THRESHOLD
        assert row.verdict == LabelVerdict.FAIL_AGREEMENT

    def test_warn_prevalence_low_ratio(self):
        # iter15 fires 3/4 rows, iter16 fires 1/4 → ratio = 0.33 → WARN.
        # Agreement is full 1.0 on both sides so no FAIL — pure WARN test.
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[1, 1, 1, 0]] * 3,
            after_votes=[[1, 0, 0, 0]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        assert row.iter15_prevalence == 0.75
        assert row.iter16_prevalence == 0.25
        assert row.prevalence_ratio == pytest.approx(0.333, rel=0.01)
        # Note: Δagr=0 so this should be WARN_PREVALENCE, NOT FAIL_AND_WARN.
        assert row.verdict == LabelVerdict.WARN_PREVALENCE

    def test_warn_prevalence_high_ratio(self):
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[1, 0, 0, 0]] * 3,
            after_votes=[[1, 1, 1, 0]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        assert row.prevalence_ratio == pytest.approx(3.0)
        assert row.verdict == LabelVerdict.WARN_PREVALENCE

    def test_fail_and_warn_co_occur(self):
        # Construct a case where Δagr > 0.005 AND prev_ratio > 2.0 both hold.
        # iter15: 1/10 rows fire unanimously (agr=1.0, prev=0.1)
        # iter16: 3/10 rows with 2-1 split on one of them
        #   → prev = 3/10 = 0.3, ratio = 3.0 (out of [0.5, 2.0], WARN)
        #   → agr = (9 + 2/3)/10 ≈ 0.9666, Δ ≈ -0.0333 (FAIL)
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            after_votes=[
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],  # 4 fires with 1 dissent on row 9
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            ],
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        # Agreement: iter15=1.0 (all unanimous); iter16 has 1 row with 2-1 split.
        assert abs(row.delta_agreement) > AGREEMENT_DELTA_THRESHOLD
        # Prevalence ratio: iter15=1/10=0.1; iter16 majority fires on rows
        # 0,1,2 (the 2-1 row on row 9 does NOT meet majority). ratio=0.3/0.1=3.0
        assert row.prevalence_ratio == pytest.approx(3.0)
        assert row.verdict == LabelVerdict.FAIL_AND_WARN

    def test_zero_prevalence_both_sides_no_warn(self):
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[0, 0, 0, 0]] * 3,
            after_votes=[[0, 0, 0, 0]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        assert row.prevalence_ratio == 1.0
        assert row.verdict == LabelVerdict.PASS

    def test_zero_to_nonzero_warn(self):
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[0, 0, 0, 0]] * 3,
            after_votes=[[1, 1, 1, 1]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        assert math.isinf(row.prevalence_ratio)
        assert row.verdict == LabelVerdict.WARN_PREVALENCE


def test_agreement_delta_threshold_constant():
    """Pin the threshold so future edits don't silently move it."""
    from trainr.core.compare_prompt_versions import AGREEMENT_DELTA_THRESHOLD
    assert AGREEMENT_DELTA_THRESHOLD == 0.005


def test_prevalence_ratio_bounds_constants():
    from trainr.core.compare_prompt_versions import (
        PREVALENCE_RATIO_HIGH,
        PREVALENCE_RATIO_LOW,
    )
    assert PREVALENCE_RATIO_LOW == 0.5
    assert PREVALENCE_RATIO_HIGH == 2.0
