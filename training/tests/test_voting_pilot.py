"""Tests for trainr.core.voting_pilot — Tier 1 voting pilot for escalation measurement."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Routing table tests
# ---------------------------------------------------------------------------


class TestTier1Routing:
    def test_routing_table_defined(self):
        from trainr.core.voting_pilot import TIER1_ROUTING

        assert isinstance(TIER1_ROUTING, dict)
        assert len(TIER1_ROUTING) > 0

    def test_routing_covers_all_detection_labels(self):
        from trainr.core.annotate_detections import DETECTION_LABELS
        from trainr.core.voting_pilot import TIER1_ROUTING

        # Every sub_type that appears in DETECTION_LABELS should have a routing entry
        # (unknown is extra, not in DETECTION_LABELS)
        for label in DETECTION_LABELS:
            assert label in TIER1_ROUTING, f"Missing routing for sub_type: {label}"

    def test_routing_entries_are_tuples_of_model_and_backend(self):
        from trainr.core.voting_pilot import TIER1_ROUTING

        valid_backends = {"openrouter", "anthropic"}
        for sub_type, (model_id, backend) in TIER1_ROUTING.items():
            assert isinstance(model_id, str), f"model_id for {sub_type} not a string"
            assert isinstance(backend, str), f"backend for {sub_type} not a string"
            assert backend in valid_backends, (
                f"backend for {sub_type} is '{backend}', expected one of {valid_backends}"
            )

    def test_unknown_sub_type_has_routing(self):
        from trainr.core.voting_pilot import TIER1_ROUTING

        assert "unknown" in TIER1_ROUTING


# ---------------------------------------------------------------------------
# Stratified sample tests
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def _make_df(self, sub_type_counts: dict[str, int]) -> pl.DataFrame:
        """Helper: build a DataFrame with given sub_type distributions."""
        rows = []
        for st, count in sub_type_counts.items():
            for i in range(count):
                rows.append({"text": f"text_{st}_{i}", "sub_type": st})
        return pl.DataFrame(rows)

    def test_returns_requested_sample_size(self):
        from trainr.core.voting_pilot import stratified_sample

        df = self._make_df({"python": 200, "rust": 200, "go": 200})
        result = stratified_sample(df, n=100, seed=42)
        assert result.height == 100

    def test_all_sub_types_represented(self):
        from trainr.core.voting_pilot import stratified_sample

        df = self._make_df({"python": 500, "rust": 500, "go": 10, "css": 5})
        result = stratified_sample(df, n=100, seed=42)
        result_types = set(result["sub_type"].to_list())
        assert result_types == {"python", "rust", "go", "css"}

    def test_small_groups_fully_included(self):
        from trainr.core.voting_pilot import stratified_sample

        # Group "rare" has only 3 rows; it should get all 3
        df = self._make_df({"common": 500, "rare": 3})
        result = stratified_sample(df, n=100, seed=42)
        rare_count = result.filter(pl.col("sub_type") == "rare").height
        assert rare_count == 3

    def test_sample_does_not_exceed_dataframe_size(self):
        from trainr.core.voting_pilot import stratified_sample

        df = self._make_df({"python": 10, "rust": 10})
        result = stratified_sample(df, n=5000, seed=42)
        assert result.height <= df.height

    def test_seed_determinism(self):
        from trainr.core.voting_pilot import stratified_sample

        df = self._make_df({"python": 200, "rust": 200})
        r1 = stratified_sample(df, n=50, seed=42).sort("text")
        r2 = stratified_sample(df, n=50, seed=42).sort("text")
        assert r1.equals(r2)

    def test_single_group(self):
        from trainr.core.voting_pilot import stratified_sample

        df = self._make_df({"python": 200})
        result = stratified_sample(df, n=50, seed=42)
        assert result.height == 50
        assert set(result["sub_type"].to_list()) == {"python"}


# ---------------------------------------------------------------------------
# Agreement check logic tests
# ---------------------------------------------------------------------------


class TestAgreementCheck:
    def test_agrees_when_detection_is_one(self):
        from trainr.core.voting_pilot import check_agreement

        detections = {"det_python": 1, "det_rust": 0, "det_go": 0}
        assert check_agreement("python", detections) is True

    def test_disagrees_when_detection_is_zero(self):
        from trainr.core.voting_pilot import check_agreement

        detections = {"det_python": 0, "det_rust": 1, "det_go": 0}
        assert check_agreement("python", detections) is False

    def test_disagrees_when_key_missing(self):
        from trainr.core.voting_pilot import check_agreement

        detections = {"det_rust": 1}
        assert check_agreement("python", detections) is False

    def test_unknown_sub_type(self):
        from trainr.core.voting_pilot import check_agreement

        detections = {"det_unknown": 1}
        assert check_agreement("unknown", detections) is True


# ---------------------------------------------------------------------------
# Dry-run / CLI tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_parser_defaults(self):
        from trainr.core.voting_pilot import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.input == "data/curated/train/golden_train.parquet"
        assert args.output == "data/curated/train/voting_pilot_results.parquet"
        assert args.sample_size == 5000
        assert args.seed == 42
        assert args.concurrency == 20
        assert args.dry_run is False

    def test_parser_custom_args(self):
        from trainr.core.voting_pilot import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--input", "my_input.parquet",
            "--output", "my_output.parquet",
            "--sample-size", "1000",
            "--seed", "99",
            "--concurrency", "10",
            "--dry-run",
        ])
        assert args.input == "my_input.parquet"
        assert args.output == "my_output.parquet"
        assert args.sample_size == 1000
        assert args.seed == 99
        assert args.concurrency == 10
        assert args.dry_run is True


class TestDryRun:
    def test_dry_run_prints_plan(self, capsys):
        from trainr.core.voting_pilot import run_dry_run, TIER1_ROUTING

        df = pl.DataFrame({
            "text": [f"text_{i}" for i in range(20)],
            "sub_type": ["python"] * 10 + ["rust"] * 10,
        })

        run_dry_run(df)
        captured = capsys.readouterr()
        # Should mention sub_type distribution and model assignments
        assert "python" in captured.out
        assert "rust" in captured.out


# ---------------------------------------------------------------------------
# Summary stats tests
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_summary_per_sub_type(self):
        from trainr.core.voting_pilot import compute_summary

        results_df = pl.DataFrame({
            "sub_type": ["python", "python", "python", "rust", "rust"],
            "tier1_agrees": [True, True, False, True, False],
        })
        summary = compute_summary(results_df)
        assert isinstance(summary, pl.DataFrame)
        # Should have per-sub_type rows
        python_row = summary.filter(pl.col("sub_type") == "python")
        assert python_row["agreement_rate"][0] == pytest.approx(2.0 / 3.0, abs=0.01)
        rust_row = summary.filter(pl.col("sub_type") == "rust")
        assert rust_row["agreement_rate"][0] == pytest.approx(0.5, abs=0.01)

    def test_summary_includes_escalation_rate(self):
        from trainr.core.voting_pilot import compute_summary

        results_df = pl.DataFrame({
            "sub_type": ["python", "python", "rust", "rust"],
            "tier1_agrees": [True, False, True, False],
        })
        summary = compute_summary(results_df)
        # Each sub_type has 50% disagreement = 50% escalation
        for row in summary.iter_rows(named=True):
            assert "escalation_rate" in row
            assert row["escalation_rate"] == pytest.approx(0.5, abs=0.01)
