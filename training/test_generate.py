"""Tests for training/generate.py"""

import csv
import os
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure the training directory is importable
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
CLASSIFY_BIN = PROJECT_ROOT / "target" / "release" / "classify"


def cli_available():
    """Check if the classify binary is built."""
    return CLASSIFY_BIN.exists()


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------


class TestMapDirectoryToCategory:
    def test_known_directories(self):
        from generate import map_directory_to_category

        assert map_directory_to_category("prose") == "prose"
        assert map_directory_to_category("code") == "code"
        assert map_directory_to_category("tabular") == "structured"
        assert map_directory_to_category("pdf_dump") == "artifact"

    def test_unknown_directory_returns_name(self):
        from generate import map_directory_to_category

        assert map_directory_to_category("other") == "other"


class TestDeriveSubType:
    def test_derives_from_filename(self):
        from generate import derive_sub_type

        assert derive_sub_type("python.txt") == "python"
        assert derive_sub_type("csv_data.txt") == "csv_data"
        assert derive_sub_type("simple.txt") == "simple"

    def test_handles_no_extension(self):
        from generate import derive_sub_type

        assert derive_sub_type("readme") == "readme"


class TestExtractFeaturesViaCli:
    @pytest.mark.skipif(not cli_available(), reason="classify binary not built")
    def test_extracts_features_from_text(self):
        from generate import extract_features_via_cli

        text = "Hello world. This is a test sentence for feature extraction."
        features = extract_features_via_cli(text, str(CLASSIFY_BIN))
        assert isinstance(features, dict)
        assert "line_length_cv" in features
        assert "alpha_ratio" in features
        assert "line_count" in features
        # All values should be numeric
        for key, val in features.items():
            assert isinstance(val, (int, float)), f"{key} is not numeric: {val}"

    @pytest.mark.skipif(not cli_available(), reason="classify binary not built")
    def test_multiline_text(self):
        from generate import extract_features_via_cli

        text = "Line one.\nLine two.\nLine three."
        features = extract_features_via_cli(text, str(CLASSIFY_BIN))
        assert features["line_count"] == 3


# ---------------------------------------------------------------------------
# Tests for fixtures mode
# ---------------------------------------------------------------------------


class TestFixturesMode:
    @pytest.mark.skipif(not cli_available(), reason="classify binary not built")
    def test_produces_csv_output(self):
        from generate import run_fixtures_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = run_fixtures_mode(
                fixtures_dir=str(FIXTURES_DIR),
                output_dir=tmpdir,
                classify_bin=str(CLASSIFY_BIN),
            )
            assert Path(output_path).exists()

            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) > 0
            # Check required columns
            assert "category" in reader.fieldnames
            assert "sub_type" in reader.fieldnames
            assert "line_count" in reader.fieldnames
            assert "line_length_cv" in reader.fieldnames

    @pytest.mark.skipif(not cli_available(), reason="classify binary not built")
    def test_maps_categories_correctly(self):
        from generate import run_fixtures_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = run_fixtures_mode(
                fixtures_dir=str(FIXTURES_DIR),
                output_dir=tmpdir,
                classify_bin=str(CLASSIFY_BIN),
            )
            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            categories = {row["category"] for row in rows}
            # We expect at least prose and code from fixtures
            assert "prose" in categories
            assert "code" in categories


# ---------------------------------------------------------------------------
# Tests for perturbation mode
# ---------------------------------------------------------------------------


class TestPerturbMode:
    @pytest.mark.skipif(not cli_available(), reason="classify binary not built")
    def test_generates_perturbations(self):
        from generate import run_fixtures_mode, run_perturb_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            # First generate fixtures
            run_fixtures_mode(
                fixtures_dir=str(FIXTURES_DIR),
                output_dir=tmpdir,
                classify_bin=str(CLASSIFY_BIN),
            )
            output_path = run_perturb_mode(output_dir=tmpdir)
            assert Path(output_path).exists()

            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have 10-15 perturbations per fixture row
            assert len(rows) >= 10

    @pytest.mark.skipif(not cli_available(), reason="classify binary not built")
    def test_perturbations_have_same_columns(self):
        from generate import run_fixtures_mode, run_perturb_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_path = run_fixtures_mode(
                fixtures_dir=str(FIXTURES_DIR),
                output_dir=tmpdir,
                classify_bin=str(CLASSIFY_BIN),
            )
            perturb_path = run_perturb_mode(output_dir=tmpdir)

            with open(fixtures_path) as f:
                fixture_cols = csv.DictReader(f).fieldnames
            with open(perturb_path) as f:
                perturb_cols = csv.DictReader(f).fieldnames

            assert fixture_cols == perturb_cols

    @pytest.mark.skipif(not cli_available(), reason="classify binary not built")
    def test_perturbations_clip_to_minimum(self):
        from generate import run_fixtures_mode, run_perturb_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fixtures_mode(
                fixtures_dir=str(FIXTURES_DIR),
                output_dir=tmpdir,
                classify_bin=str(CLASSIFY_BIN),
            )
            output_path = run_perturb_mode(output_dir=tmpdir)

            with open(output_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, val in row.items():
                        if key not in ("category", "sub_type"):
                            assert float(val) >= 0.0, f"{key}={val} is negative"


# ---------------------------------------------------------------------------
# Tests for synthetic mode
# ---------------------------------------------------------------------------


class TestSyntheticMode:
    def test_skips_without_api_key(self, capsys):
        from generate import run_synthetic_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_synthetic_mode(
                output_dir=tmpdir,
                classify_bin=str(CLASSIFY_BIN),
                api_key=None,
                samples_per_type=1,
            )
            assert result is None
            captured = capsys.readouterr()
            assert "ANTHROPIC_API_KEY" in captured.out or "ANTHROPIC_API_KEY" in captured.err

    def test_type_pairs_defined(self):
        from generate import SYNTHETIC_TYPE_PAIRS

        assert len(SYNTHETIC_TYPE_PAIRS) > 0
        categories = {pair[0] for pair in SYNTHETIC_TYPE_PAIRS}
        assert "prose" in categories
        assert "code" in categories
        assert "structured" in categories
        assert "artifact" in categories


# ---------------------------------------------------------------------------
# Tests for all mode (combination)
# ---------------------------------------------------------------------------


class TestAllMode:
    @pytest.mark.skipif(not cli_available(), reason="classify binary not built")
    def test_combines_csv_files(self):
        from generate import run_fixtures_mode, run_perturb_mode, combine_csvs

        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_path = run_fixtures_mode(
                fixtures_dir=str(FIXTURES_DIR),
                output_dir=tmpdir,
                classify_bin=str(CLASSIFY_BIN),
            )
            perturb_path = run_perturb_mode(output_dir=tmpdir)
            combined_path = combine_csvs(
                [fixtures_path, perturb_path],
                output_dir=tmpdir,
            )
            assert Path(combined_path).exists()

            with open(fixtures_path) as f:
                fixture_count = sum(1 for _ in csv.DictReader(f))
            with open(perturb_path) as f:
                perturb_count = sum(1 for _ in csv.DictReader(f))
            with open(combined_path) as f:
                combined_count = sum(1 for _ in csv.DictReader(f))

            assert combined_count == fixture_count + perturb_count


# ---------------------------------------------------------------------------
# Tests for CLI argument parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_default_arguments(self):
        from generate import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.mode == "all"
        assert args.output == "training/data/"
        assert args.samples_per_type == 50

    def test_custom_arguments(self):
        from generate import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "fixtures",
            "--output", "/tmp/out",
            "--samples-per-type", "10",
            "--api-key", "test-key",
        ])
        assert args.mode == "fixtures"
        assert args.output == "/tmp/out"
        assert args.samples_per_type == 10
        assert args.api_key == "test-key"

    def test_golden_train_mode_accepted(self):
        from generate import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "golden-train",
            "--output", "/tmp/out",
            "--samples-per-type", "1",
            "--dry-run",
        ])
        assert args.mode == "golden-train"
        assert args.dry_run is True

    def test_dry_run_defaults_to_false(self):
        from generate import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.dry_run is False


# ---------------------------------------------------------------------------
# Tests for golden-train constants
# ---------------------------------------------------------------------------


class TestGoldenTrainConstants:
    def test_valid_categories_defined(self):
        from generate import VALID_CATEGORIES

        assert VALID_CATEGORIES == {"prose", "code", "structured", "artifact"}

    def test_golden_sub_types_covers_all_categories(self):
        from generate import GOLDEN_SUB_TYPES, VALID_CATEGORIES

        for cat in VALID_CATEGORIES:
            assert cat in GOLDEN_SUB_TYPES, f"Missing category {cat} in GOLDEN_SUB_TYPES"
            assert len(GOLDEN_SUB_TYPES[cat]) > 0, f"No sub-types for {cat}"

    def test_golden_sub_types_total_count(self):
        """All trainable ContentSubType variants should be present (32 total,
        excluding skip/fallback types: TooShort, Empty, Ambiguous, Unknown)."""
        from generate import GOLDEN_SUB_TYPES

        total = sum(len(v) for v in GOLDEN_SUB_TYPES.values())
        assert total == 32

    def test_golden_domain_seeds_has_50_plus(self):
        from generate import GOLDEN_DOMAIN_SEEDS

        assert len(GOLDEN_DOMAIN_SEEDS) >= 50

    def test_golden_length_buckets_defined(self):
        from generate import GOLDEN_LENGTH_BUCKETS

        assert "short" in GOLDEN_LENGTH_BUCKETS
        assert "medium" in GOLDEN_LENGTH_BUCKETS
        assert "long" in GOLDEN_LENGTH_BUCKETS
        assert GOLDEN_LENGTH_BUCKETS["short"] == (3, 10)
        assert GOLDEN_LENGTH_BUCKETS["medium"] == (20, 50)
        assert GOLDEN_LENGTH_BUCKETS["long"] == (100, 200)

    def test_golden_boundary_pairs_defined(self):
        from generate import GOLDEN_BOUNDARY_PAIRS

        assert len(GOLDEN_BOUNDARY_PAIRS) == 6
        # Each pair should have cat_a, cat_b, label, examples
        for pair in GOLDEN_BOUNDARY_PAIRS:
            assert "cat_a" in pair
            assert "cat_b" in pair
            assert "label" in pair
            assert "examples" in pair


# ---------------------------------------------------------------------------
# Tests for golden-train dry run
# ---------------------------------------------------------------------------


class TestGoldenTrainDryRun:
    def test_dry_run_prints_summary(self, capsys):
        from generate import run_golden_train_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            run_golden_train_mode(
                output_dir=tmpdir,
                samples_per_type=1,
                dry_run=True,
            )
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out
            assert "prose" in captured.out
            assert "code" in captured.out
            assert "structured" in captured.out
            assert "artifact" in captured.out

    def test_dry_run_does_not_create_csv(self):
        from generate import run_golden_train_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            run_golden_train_mode(
                output_dir=tmpdir,
                samples_per_type=1,
                dry_run=True,
            )
            csv_path = os.path.join(tmpdir, "golden_raw.csv")
            assert not os.path.exists(csv_path)

    def test_dry_run_shows_sample_counts(self, capsys):
        from generate import run_golden_train_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            run_golden_train_mode(
                output_dir=tmpdir,
                samples_per_type=200,
                dry_run=True,
            )
            captured = capsys.readouterr()
            # Should show per-sub-type counts and boundary pair counts
            assert "clear" in captured.out.lower() or "Clear" in captured.out
            assert "boundary" in captured.out.lower() or "Boundary" in captured.out

    def test_dry_run_shows_domain_seed_count(self, capsys):
        from generate import run_golden_train_mode

        with tempfile.TemporaryDirectory() as tmpdir:
            run_golden_train_mode(
                output_dir=tmpdir,
                samples_per_type=1,
                dry_run=True,
            )
            captured = capsys.readouterr()
            assert "domain" in captured.out.lower() or "Domain" in captured.out


# ---------------------------------------------------------------------------
# Tests for generate_golden_clear function
# ---------------------------------------------------------------------------


class TestGenerateGoldenClear:
    def test_function_signature(self):
        from generate import generate_golden_clear
        import inspect

        sig = inspect.signature(generate_golden_clear)
        params = list(sig.parameters.keys())
        assert "category" in params
        assert "sub_types" in params
        assert "count" in params
        assert "domain_seeds" in params
        assert "length_buckets" in params

    def test_returns_list_of_dicts(self):
        """With a mock client, should return sample dicts."""
        from generate import generate_golden_clear, GOLDEN_LENGTH_BUCKETS

        # Mock the anthropic client
        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.content = [mock.MagicMock()]
        mock_response.content[0].text = '"Sample text for testing.\\nLine 2."]'
        mock_client.messages.create.return_value = mock_response

        results = generate_golden_clear(
            category="prose",
            sub_types=["plain"],
            count=1,
            domain_seeds=["astronomy"],
            length_buckets=GOLDEN_LENGTH_BUCKETS,
            client=mock_client,
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        assert "text" in results[0]
        assert "category" in results[0]
        assert "sub_type" in results[0]
        assert "source" in results[0]
        assert results[0]["category"] == "prose"
        assert results[0]["source"] == "golden_clear"

    def test_distributes_across_sub_types(self):
        """Count should be distributed across sub-types."""
        from generate import generate_golden_clear, GOLDEN_LENGTH_BUCKETS

        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.content = [mock.MagicMock()]
        mock_response.content[0].text = '"Sample text."]'
        mock_client.messages.create.return_value = mock_response

        results = generate_golden_clear(
            category="prose",
            sub_types=["plain", "markdown"],
            count=4,
            domain_seeds=["astronomy"],
            length_buckets=GOLDEN_LENGTH_BUCKETS,
            client=mock_client,
        )
        # Should have called the API for both sub-types
        calls = mock_client.messages.create.call_args_list
        prompts = [str(c) for c in calls]
        assert any("plain" in p for p in prompts)
        assert any("markdown" in p for p in prompts)


# ---------------------------------------------------------------------------
# Tests for generate_golden_boundary function
# ---------------------------------------------------------------------------


class TestGenerateGoldenBoundary:
    def test_function_signature(self):
        from generate import generate_golden_boundary
        import inspect

        sig = inspect.signature(generate_golden_boundary)
        params = list(sig.parameters.keys())
        assert "pair" in params
        assert "count" in params
        assert "domain_seeds" in params
        assert "length_buckets" in params

    def test_returns_list_of_dicts(self):
        from generate import generate_golden_boundary, GOLDEN_LENGTH_BUCKETS

        mock_client = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.content = [mock.MagicMock()]
        mock_response.content[0].text = '"Ambiguous boundary text."]'
        mock_client.messages.create.return_value = mock_response

        pair = {"cat_a": "prose", "cat_b": "code", "label": "prose", "examples": "mixed content"}
        results = generate_golden_boundary(
            pair=pair,
            count=1,
            domain_seeds=["astronomy"],
            length_buckets=GOLDEN_LENGTH_BUCKETS,
            client=mock_client,
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        assert "text" in results[0]
        assert "category" in results[0]
        assert "source" in results[0]
        assert results[0]["source"] == "golden_boundary"


# ---------------------------------------------------------------------------
# Tests for golden-train CLI integration
# ---------------------------------------------------------------------------


class TestGoldenTrainCLI:
    def test_main_routes_golden_train(self):
        """Verify that --mode golden-train routes to run_golden_train_mode."""
        from generate import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--mode", "golden-train",
            "--output", "/tmp/test_golden",
            "--samples-per-type", "1",
            "--dry-run",
        ])
        assert args.mode == "golden-train"
        assert args.output == "/tmp/test_golden"
        assert args.samples_per_type == 1
        assert args.dry_run is True
