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


class TestRowAlignmentFingerprint:
    def test_passing_fingerprint_does_not_raise(self):
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        before_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": [0] * 5})
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames = {k: v.clone() for k, v in before_frames.items()}

        # Should succeed without raising.
        compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

    def test_text_mutation_raises_with_row_index(self):
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        before_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": [0] * 5})
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        # Corrupt one model's parquet by mutating a text row.
        bad = before_frames["gemini3flash"].with_columns(
            pl.Series("text", ["row-0", "row-1", "CORRUPTED", "row-3", "row-4"])
        )
        before_frames["gemini3flash"] = bad
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames = {k: v.clone() for k, v in before_frames.items()}

        with pytest.raises(ValueError) as excinfo:
            compare_prompt_versions(
                before_frames=before_frames,
                after_frames=after_frames,
                noise_floor_frames=noise_frames,
                input_frame=input_frame,
            )
        msg = str(excinfo.value)
        # Error must name the row index and the diverging column.
        assert "row 2" in msg or "index 2" in msg, f"row index missing from error: {msg!r}"
        assert "text" in msg, f"column name missing from error: {msg!r}"
        assert "gemini3flash" in msg, f"source slug missing from error: {msg!r}"

    def test_row_count_mismatch_raises(self):
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        smaller = input_frame.head(3)
        before_frames = {
            "gemini3flash": _make_annotator_frame(smaller, {"det_python": [0, 0, 0]}),
            "sonnet": _make_annotator_frame(input_frame, {"det_python": [0] * 5}),
            "gpt54mini": _make_annotator_frame(input_frame, {"det_python": [0] * 5}),
        }
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames = {k: v.clone() for k, v in before_frames.items()}

        with pytest.raises(ValueError, match="row count"):
            compare_prompt_versions(
                before_frames=before_frames,
                after_frames=after_frames,
                noise_floor_frames=noise_frames,
                input_frame=input_frame,
            )

    def test_missing_non_det_column_raises(self):
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        # Drop sub_type from one annotation frame to simulate a column-schema
        # mismatch on a non-det column. The fingerprint check should catch it.
        before_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": [0] * 5})
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        before_frames["sonnet"] = before_frames["sonnet"].drop("sub_type")
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        # Restore sub_type on after/noise so the mismatch is only on before.
        after_frames["sonnet"] = _make_annotator_frame(input_frame, {"det_python": [0] * 5})
        noise_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames["sonnet"] = _make_annotator_frame(input_frame, {"det_python": [0] * 5})

        with pytest.raises(ValueError, match="sub_type|non-det"):
            compare_prompt_versions(
                before_frames=before_frames,
                after_frames=after_frames,
                noise_floor_frames=noise_frames,
                input_frame=input_frame,
            )


class TestFormatReport:
    def test_report_contains_all_required_sections(self):
        from trainr.core.compare_prompt_versions import format_delta_report

        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        before_frames = {
            slug: _make_annotator_frame(
                input_frame, {"det_python": [1, 0, 0, 0, 0], "det_log_lines": [0] * 5}
            )
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        after_frames = {
            slug: _make_annotator_frame(
                input_frame,
                {"det_python": [1, 0, 0, 0, 0], "det_log_content": [0] * 5},
            )
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        text = format_delta_report(report)

        assert "# iter17 A/B Regression Audit Report" in text
        assert "## Gate verdict" in text
        assert "## Shared labels" in text
        assert "## iter15-only labels" in text
        assert "## iter16-only labels" in text
        assert "## Noise floor table" in text
        # Shared label row present
        assert "python" in text
        # iter15-only label present in its section
        assert "log_lines" in text
        # iter16-only label present in its section
        assert "log_content" in text
        # No FAIL rows in this fixture → PASS gate verdict
        assert "**PASS**" in text

    def test_report_summary_shows_fail_count(self):
        from trainr.core.compare_prompt_versions import format_delta_report

        input_frame = _make_input_frame(n_strat=10, n_inject=0)
        # Construct a FAIL-agreement row: iter15 unanimous, iter16 has 2-1 split.
        before_votes = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] * 3
        after_votes = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        before_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": v})
            for slug, v in zip(("gemini3flash", "sonnet", "gpt54mini"), before_votes)
        }
        after_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": v})
            for slug, v in zip(("gemini3flash", "sonnet", "gpt54mini"), after_votes)
        }
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        text = format_delta_report(report)
        assert "**FAIL**" in text
        # Summary line should mention the FAIL count
        assert "1 FAIL-agreement" in text or "FAIL-agreement: 1" in text or "1 FAIL" in text

    def test_report_handles_inf_prevalence_ratio(self):
        """A label with iter15_prev=0 and iter16_prev>0 should format cleanly."""
        from trainr.core.compare_prompt_versions import format_delta_report

        input_frame = _make_input_frame(n_strat=4, n_inject=0)
        # iter15: 0 fires; iter16: all 4 fire → ratio = inf
        before_votes = [[0, 0, 0, 0]] * 3
        after_votes = [[1, 1, 1, 1]] * 3
        before_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": v})
            for slug, v in zip(("gemini3flash", "sonnet", "gpt54mini"), before_votes)
        }
        after_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": v})
            for slug, v in zip(("gemini3flash", "sonnet", "gpt54mini"), after_votes)
        }
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        text = format_delta_report(report)
        # Should not crash, should not say "nan". "inf" is OK as a formatted value.
        assert "nan" not in text.lower() or "inf" in text
        assert "python" in text


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


class TestMainCLI:
    def test_main_end_to_end_from_parquets(self, tmp_path):
        """Run main() against a small set of real parquet files on disk."""
        from trainr.core.compare_prompt_versions import main as compare_main

        # Build a tiny input parquet and 9 annotation parquets.
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        input_path = tmp_path / "input.parquet"
        input_frame.write_parquet(input_path)

        def _write_side(side: str):
            for slug in ("gemini3flash", "sonnet", "gpt54mini"):
                df = _make_annotator_frame(input_frame, {"det_python": [0] * 5})
                df.write_parquet(tmp_path / f"iter17_ab_{side}_{slug}.parquet")

        _write_side("iter15")
        _write_side("iter16a")
        _write_side("iter16b")

        output_report = tmp_path / "report.md"
        compare_main([
            "--before", str(tmp_path / "iter17_ab_iter15_*.parquet"),
            "--after", str(tmp_path / "iter17_ab_iter16a_*.parquet"),
            "--noise-floor", str(tmp_path / "iter17_ab_iter16b_*.parquet"),
            "--input", str(input_path),
            "--output", str(output_report),
        ])

        assert output_report.exists()
        content = output_report.read_text()
        assert "# iter17 A/B Regression Audit Report" in content
        assert "**PASS**" in content

    def test_main_exits_2_on_fail(self, tmp_path):
        """main() should sys.exit(2) if any FAIL-agreement row exists."""
        import sys

        from trainr.core.compare_prompt_versions import main as compare_main

        input_frame = _make_input_frame(n_strat=10, n_inject=0)
        input_path = tmp_path / "input.parquet"
        input_frame.write_parquet(input_path)

        # iter15 unanimous all-1, iter16 has a 2-1 split on row 9 → FAIL.
        before_votes = [[1] * 10] * 3
        after_votes = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1] * 10,
            [1] * 10,
        ]

        def _write_side(side: str, votes_list):
            for slug, votes in zip(("gemini3flash", "sonnet", "gpt54mini"), votes_list):
                df = _make_annotator_frame(input_frame, {"det_python": votes})
                df.write_parquet(tmp_path / f"iter17_ab_{side}_{slug}.parquet")

        _write_side("iter15", before_votes)
        _write_side("iter16a", after_votes)
        _write_side("iter16b", after_votes)

        output_report = tmp_path / "report.md"
        with pytest.raises(SystemExit) as excinfo:
            compare_main([
                "--before", str(tmp_path / "iter17_ab_iter15_*.parquet"),
                "--after", str(tmp_path / "iter17_ab_iter16a_*.parquet"),
                "--noise-floor", str(tmp_path / "iter17_ab_iter16b_*.parquet"),
                "--input", str(input_path),
                "--output", str(output_report),
            ])
        assert excinfo.value.code == 2
        # Report should still be written before the exit.
        assert output_report.exists()
        assert "**FAIL**" in output_report.read_text()

    def test_main_raises_on_empty_glob(self, tmp_path):
        from trainr.core.compare_prompt_versions import main as compare_main

        # No parquets exist at these paths.
        input_path = tmp_path / "input.parquet"
        _make_input_frame(n_strat=5, n_inject=0).write_parquet(input_path)
        output_report = tmp_path / "report.md"

        with pytest.raises(ValueError, match="matched zero files"):
            compare_main([
                "--before", str(tmp_path / "nonexistent_*.parquet"),
                "--after", str(tmp_path / "also_missing_*.parquet"),
                "--noise-floor", str(tmp_path / "also_gone_*.parquet"),
                "--input", str(input_path),
                "--output", str(output_report),
            ])
