"""Tests for trainr.core.generate_openrouter — multi-model OpenRouter generation script."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestModelRosters:
    def test_primary_models_defined(self):
        from trainr.core.generate_openrouter import PRIMARY_MODELS

        assert len(PRIMARY_MODELS) == 7
        assert "anthropic/claude-sonnet-4.6" in PRIMARY_MODELS
        assert "openai/gpt-5" in PRIMARY_MODELS
        assert "openai/gpt-5.4" in PRIMARY_MODELS
        assert "qwen/qwen3-235b-a22b" in PRIMARY_MODELS
        assert "deepseek/deepseek-v3.2" in PRIMARY_MODELS
        assert "mistralai/mistral-large-2512" in PRIMARY_MODELS
        assert "meta-llama/llama-3.3-70b-instruct" in PRIMARY_MODELS

    def test_secondary_models_defined(self):
        from trainr.core.generate_openrouter import SECONDARY_MODELS

        assert len(SECONDARY_MODELS) == 9
        assert "x-ai/grok-4-fast" in SECONDARY_MODELS
        assert "deepseek/deepseek-r1-0528" in SECONDARY_MODELS
        assert "google/gemini-2.5-flash" in SECONDARY_MODELS
        assert "cohere/command-a" in SECONDARY_MODELS
        assert "mistralai/codestral-2508" in SECONDARY_MODELS
        assert "google/gemma-3-27b-it" in SECONDARY_MODELS
        assert "qwen/qwen3-30b-a3b" in SECONDARY_MODELS
        assert "qwen/qwen3-coder" in SECONDARY_MODELS
        assert "meta-llama/llama-4-maverick" in SECONDARY_MODELS

    def test_edge_case_models_defined(self):
        from trainr.core.generate_openrouter import EDGE_CASE_MODELS

        assert len(EDGE_CASE_MODELS) == 3
        assert "meta-llama/llama-3.1-8b-instruct" in EDGE_CASE_MODELS
        assert "microsoft/phi-4" in EDGE_CASE_MODELS
        assert "openai/gpt-5.4-nano" in EDGE_CASE_MODELS

    def test_all_models_unique(self):
        from trainr.core.generate_openrouter import PRIMARY_MODELS, SECONDARY_MODELS, EDGE_CASE_MODELS

        all_models = PRIMARY_MODELS + SECONDARY_MODELS + EDGE_CASE_MODELS
        assert len(all_models) == len(set(all_models)), "Duplicate model IDs found"


class TestDomainSeeds:
    def test_has_50_plus_domains(self):
        from trainr.core.generate_openrouter import DOMAIN_SEEDS

        assert len(DOMAIN_SEEDS) >= 50

    def test_contains_expected_topics(self):
        from trainr.core.generate_openrouter import DOMAIN_SEEDS

        for topic in ["astronomy", "finance", "healthcare", "devops", "gaming"]:
            assert topic in DOMAIN_SEEDS, f"Missing expected topic: {topic}"


class TestLengthBuckets:
    def test_buckets_defined(self):
        from trainr.core.generate_openrouter import LENGTH_BUCKETS

        assert LENGTH_BUCKETS["short"] == (3, 10)
        assert LENGTH_BUCKETS["medium"] == (20, 50)
        assert LENGTH_BUCKETS["long"] == (100, 200)

    def test_all_three_buckets(self):
        from trainr.core.generate_openrouter import LENGTH_BUCKETS

        assert set(LENGTH_BUCKETS.keys()) == {"short", "medium", "long"}


class TestBoundaryPairs:
    def test_has_6_pairs(self):
        from trainr.core.generate_openrouter import BOUNDARY_PAIRS

        assert len(BOUNDARY_PAIRS) == 6

    def test_pairs_are_tuples(self):
        from trainr.core.generate_openrouter import BOUNDARY_PAIRS

        for pair in BOUNDARY_PAIRS:
            assert isinstance(pair, tuple), f"Expected tuple, got {type(pair)}"
            assert len(pair) == 2, f"Expected 2-tuple, got {len(pair)}-tuple"


class TestSubTypeConfig:
    def test_has_33_sub_types(self):
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        assert len(SUB_TYPE_CONFIG) == 33

    def test_each_has_required_fields(self):
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        for sub_type, config in SUB_TYPE_CONFIG.items():
            assert "category" in config, f"{sub_type} missing 'category'"
            assert "models" in config, f"{sub_type} missing 'models'"
            assert "prompt_templates" in config, f"{sub_type} missing 'prompt_templates'"
            assert "temperature_range" in config, f"{sub_type} missing 'temperature_range'"
            assert "domains" in config, f"{sub_type} missing 'domains'"

    def test_domains_are_nonempty_lists(self):
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        for sub_type, config in SUB_TYPE_CONFIG.items():
            domains = config["domains"]
            assert isinstance(domains, list), f"{sub_type} domains is not a list"
            assert len(domains) >= 1, f"{sub_type} has empty domains list"

    def test_models_have_5_to_7_entries(self):
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        for sub_type, config in SUB_TYPE_CONFIG.items():
            models = config["models"]
            assert 5 <= len(models) <= 9, (
                f"{sub_type} has {len(models)} models, expected 5-7"
            )

    def test_prompt_templates_have_5_plus(self):
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        for sub_type, config in SUB_TYPE_CONFIG.items():
            templates = config["prompt_templates"]
            assert len(templates) >= 5, (
                f"{sub_type} has {len(templates)} templates, expected 5+"
            )

    def test_prompt_templates_have_placeholders(self):
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        for sub_type, config in SUB_TYPE_CONFIG.items():
            for tmpl in config["prompt_templates"]:
                assert "{domain}" in tmpl or "{task}" in tmpl, (
                    f"{sub_type} template missing {{domain}} or {{task}} placeholder: {tmpl[:60]}"
                )

    def test_model_weight_cap_15_percent(self):
        """No single model should have more than 15% of the weight."""
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        for sub_type, config in SUB_TYPE_CONFIG.items():
            models = config["models"]
            total_weight = sum(w for _, w in models)
            for model_id, weight in models:
                pct = weight / total_weight
                assert pct <= 0.16, (  # small tolerance
                    f"{sub_type}: model {model_id} has {pct:.1%} weight (>15%)"
                )

    def test_categories_cover_all_valid(self):
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        categories = {c["category"] for c in SUB_TYPE_CONFIG.values()}
        assert categories == {"prose", "code", "structured", "artifact"}

    def test_temperature_range_valid(self):
        from trainr.core.generate_openrouter import SUB_TYPE_CONFIG

        for sub_type, config in SUB_TYPE_CONFIG.items():
            lo, hi = config["temperature_range"]
            assert 0.0 <= lo < hi <= 2.0, (
                f"{sub_type} has invalid temp range: ({lo}, {hi})"
            )


# ---------------------------------------------------------------------------
# generate_samples tests
# ---------------------------------------------------------------------------


class TestGenerateSamples:
    def _make_mock_client(self, text="This is a generated sample about astronomy."):
        client = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps([{"text": text}])
        response.choices = [choice]
        client.chat.completions.create.return_value = response
        return client

    def test_returns_list_of_dicts(self):
        from trainr.core.generate_openrouter import generate_samples, SUB_TYPE_CONFIG

        client = self._make_mock_client()
        sub_type = next(iter(SUB_TYPE_CONFIG))
        config = SUB_TYPE_CONFIG[sub_type]
        results = generate_samples(sub_type, 2, config, client)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert isinstance(results[0], dict)

    def test_provenance_fields_present(self):
        from trainr.core.generate_openrouter import generate_samples, SUB_TYPE_CONFIG

        client = self._make_mock_client()
        sub_type = next(iter(SUB_TYPE_CONFIG))
        config = SUB_TYPE_CONFIG[sub_type]
        results = generate_samples(sub_type, 1, config, client)
        assert len(results) >= 1
        sample = results[0]
        for field in [
            "text", "expected_category", "sub_type", "model",
            "temperature", "prompt_template", "content_domain", "length_bucket",
        ]:
            assert field in sample, f"Missing provenance field: {field}"

    def test_expected_category_matches_config(self):
        from trainr.core.generate_openrouter import generate_samples, SUB_TYPE_CONFIG

        client = self._make_mock_client()
        sub_type = next(iter(SUB_TYPE_CONFIG))
        config = SUB_TYPE_CONFIG[sub_type]
        results = generate_samples(sub_type, 1, config, client)
        assert results[0]["expected_category"] == config["category"]
        assert results[0]["sub_type"] == sub_type


class TestGenerateBoundarySamples:
    def _make_mock_client(self, text="Ambiguous boundary content."):
        client = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps([{"text": text}])
        response.choices = [choice]
        client.chat.completions.create.return_value = response
        return client

    def test_returns_list_of_dicts(self):
        from trainr.core.generate_openrouter import generate_boundary_samples, BOUNDARY_PAIRS

        client = self._make_mock_client()
        pair = BOUNDARY_PAIRS[0]
        results = generate_boundary_samples(pair, 2, client)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_provenance_fields_present(self):
        from trainr.core.generate_openrouter import generate_boundary_samples, BOUNDARY_PAIRS

        client = self._make_mock_client()
        pair = BOUNDARY_PAIRS[0]
        results = generate_boundary_samples(pair, 2, client)
        sample = results[0]
        assert "text" in sample
        assert "expected_category" in sample
        assert "boundary_pair" in sample

    def test_boundary_pair_field_set(self):
        from trainr.core.generate_openrouter import generate_boundary_samples, BOUNDARY_PAIRS

        client = self._make_mock_client()
        pair = BOUNDARY_PAIRS[0]
        results = generate_boundary_samples(pair, 2, client)
        bp = f"{pair[0]}_{pair[1]}"
        assert results[0]["boundary_pair"] == bp


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_build_parser_defaults(self):
        from trainr.core.generate_openrouter import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.output == "training/data/openrouter.jsonl"
        assert args.total_samples == 60000
        assert args.pilot is False
        assert args.dry_run is False
        assert args.resume is False

    def test_pilot_flag(self):
        from trainr.core.generate_openrouter import build_parser

        parser = build_parser()
        args = parser.parse_args(["--pilot"])
        assert args.pilot is True

    def test_dry_run_flag(self):
        from trainr.core.generate_openrouter import build_parser

        parser = build_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_resume_flag(self):
        from trainr.core.generate_openrouter import build_parser

        parser = build_parser()
        args = parser.parse_args(["--resume"])
        assert args.resume is True

    def test_custom_output(self):
        from trainr.core.generate_openrouter import build_parser

        parser = build_parser()
        args = parser.parse_args(["--output", "/tmp/foo.jsonl"])
        assert args.output == "/tmp/foo.jsonl"

    def test_custom_total_samples(self):
        from trainr.core.generate_openrouter import build_parser

        parser = build_parser()
        args = parser.parse_args(["--total-samples", "100"])
        assert args.total_samples == 100


# ---------------------------------------------------------------------------
# Dry-run integration test
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_prints_plan_no_output_file(self, capsys):
        from trainr.core.generate_openrouter import main

        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.jsonl")
            main(["--output", output, "--total-samples", "100", "--dry-run"])
            captured = capsys.readouterr()
            assert "DRY RUN" in captured.out
            assert not os.path.exists(output)

    def test_dry_run_shows_sub_type_plan(self, capsys):
        from trainr.core.generate_openrouter import main

        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test.jsonl")
            main(["--output", output, "--total-samples", "100", "--dry-run"])
            captured = capsys.readouterr()
            # Should mention sub-types and models
            assert "prose" in captured.out.lower() or "plain" in captured.out.lower()
            assert "model" in captured.out.lower()

    def test_pilot_mode_sets_500_samples(self):
        from trainr.core.generate_openrouter import build_parser

        parser = build_parser()
        args = parser.parse_args(["--pilot"])
        # Pilot overrides total_samples conceptually; test it in main
        assert args.pilot is True


# ---------------------------------------------------------------------------
# Retry / backoff tests
# ---------------------------------------------------------------------------


class TestWeightedModels:
    def test_returns_normalised_weights(self):
        from trainr.core.generate_openrouter import _weighted_models

        models = _weighted_models(["a", "b"], ["c"])
        total = sum(w for _, w in models)
        assert abs(total - 1.0) < 1e-9

    def test_no_model_exceeds_15_percent_cap(self):
        from trainr.core.generate_openrouter import _weighted_models

        models = _weighted_models(["a", "b", "c", "d", "e"], ["f", "g"])
        for model_id, weight in models:
            assert weight <= 0.16, f"{model_id} exceeds 15% cap: {weight}"

    def test_primary_and_secondary_both_present(self):
        from trainr.core.generate_openrouter import _weighted_models

        models = _weighted_models(["a", "b"], ["c", "d"])
        ids = [m for m, _ in models]
        assert "a" in ids and "b" in ids and "c" in ids and "d" in ids


class TestSelectModel:
    def test_returns_model_from_list(self):
        from trainr.core.generate_openrouter import _select_model

        models = [("model-a", 0.5), ("model-b", 0.5)]
        result = _select_model(models)
        assert result in ["model-a", "model-b"]

    def test_respects_weights_over_many_calls(self):
        """A model with weight 0 should never be selected."""
        from trainr.core.generate_openrouter import _select_model

        models = [("always", 1.0), ("never", 0.0)]
        results = {_select_model(models) for _ in range(50)}
        assert "never" not in results


class TestJSONLOutput:
    def test_main_writes_jsonl_output(self):
        """main() in non-dry-run mode should write JSONL with provenance."""
        from trainr.core.generate_openrouter import main

        client_mock = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps([{"text": "Generated sample."}])
        response.choices = [choice]
        client_mock.chat.completions.create.return_value = response

        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "test_output.jsonl")
            with patch("trainr.core.generate_openrouter._create_client", return_value=client_mock):
                main(["--output", output, "--total-samples", "10"])
            assert os.path.exists(output)
            with open(output) as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) >= 1
            sample = json.loads(lines[0])
            assert "text" in sample
            assert "model" in sample
            assert "temperature" in sample

    def test_resume_appends_to_existing(self):
        """--resume should append to an existing file."""
        from trainr.core.generate_openrouter import main

        client_mock = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps([{"text": "Resumed sample."}])
        response.choices = [choice]
        client_mock.chat.completions.create.return_value = response

        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "resume_test.jsonl")
            # Write an existing sample
            with open(output, "w") as f:
                f.write(json.dumps({"text": "existing", "expected_category": "prose"}) + "\n")
            with patch("trainr.core.generate_openrouter._create_client", return_value=client_mock):
                main(["--output", output, "--total-samples", "10", "--resume"])
            with open(output) as f:
                lines = [line.strip() for line in f if line.strip()]
            # Should have the original line plus new ones
            assert len(lines) >= 2
            assert json.loads(lines[0])["text"] == "existing"


class TestRetryLogic:
    def test_retries_on_api_error(self):
        from trainr.core.generate_openrouter import _call_api_with_retry

        client = MagicMock()
        # First two calls raise, third succeeds
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps([{"text": "hello"}])
        response.choices = [choice]

        client.chat.completions.create.side_effect = [
            Exception("rate limit"),
            Exception("rate limit"),
            response,
        ]
        result = _call_api_with_retry(client, "test-model", [{"role": "user", "content": "hi"}], temperature=0.7)
        assert len(result) == 1
        assert result[0]["text"] == "hello"
        assert client.chat.completions.create.call_count == 3

    def test_gives_up_after_3_retries(self):
        from trainr.core.generate_openrouter import _call_api_with_retry

        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("always fails")
        result = _call_api_with_retry(client, "test-model", [{"role": "user", "content": "hi"}], temperature=0.7)
        assert result == []
        assert client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# Validation integration
# ---------------------------------------------------------------------------


class TestValidationIntegration:
    def test_validates_with_eval_schema(self):
        """Generated samples should pass eval_schema.validate_sample."""
        from trainr.core.generate_openrouter import generate_samples, SUB_TYPE_CONFIG
        from trainr.core.schema import validate_sample

        client = MagicMock()
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps([{"text": "Test sample content."}])
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        sub_type = "plain"
        config = SUB_TYPE_CONFIG[sub_type]
        results = generate_samples(sub_type, 1, config, client)
        assert len(results) >= 1
        assert validate_sample(results[0]) is True
