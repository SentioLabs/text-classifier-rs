"""Tests for trainr.core.generate_eval — the GPT-5.4 eval set generator."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import trainr.core.generate_eval as generate_eval


# ── Shared contract tests ────────────────────────────────────────────

class TestValidateConstants:
    """Verify the shared constants are present and correct."""

    def test_valid_categories(self):
        assert generate_eval.VALID_CATEGORIES == {"prose", "code", "structured", "artifact"}

    def test_valid_boundary_pairs(self):
        expected = {
            "prose_code", "prose_structured", "prose_artifact",
            "code_structured", "code_artifact", "structured_artifact",
        }
        assert generate_eval.VALID_BOUNDARY_PAIRS == expected


class TestValidateSample:
    """validate_sample must accept well-formed dicts and reject bad ones."""

    def test_accepts_valid_clear_sample(self):
        sample = {
            "text": "This is a clear prose paragraph about astronomy.",
            "label": "prose",
            "kind": "clear",
        }
        assert generate_eval.validate_sample(sample) is True

    def test_rejects_missing_text(self):
        sample = {"label": "prose", "kind": "clear"}
        assert generate_eval.validate_sample(sample) is False

    def test_rejects_empty_text(self):
        sample = {"text": "", "label": "prose", "kind": "clear"}
        assert generate_eval.validate_sample(sample) is False

    def test_rejects_invalid_label_for_clear(self):
        sample = {"text": "some text", "label": "bogus", "kind": "clear"}
        assert generate_eval.validate_sample(sample) is False

    def test_rejects_missing_label(self):
        sample = {"text": "some text", "kind": "clear"}
        assert generate_eval.validate_sample(sample) is False

    def test_rejects_missing_kind(self):
        sample = {"text": "some text", "label": "prose"}
        assert generate_eval.validate_sample(sample) is False

    def test_accepts_valid_boundary_sample(self):
        sample = {
            "text": "apiVersion: v1\nkind: ConfigMap",
            "label": "code",
            "kind": "boundary",
            "boundary_pair": "code_structured",
        }
        assert generate_eval.validate_sample(sample) is True

    def test_rejects_boundary_without_pair(self):
        sample = {
            "text": "some text",
            "label": "code",
            "kind": "boundary",
        }
        assert generate_eval.validate_sample(sample) is False

    def test_rejects_boundary_with_invalid_pair(self):
        sample = {
            "text": "some text",
            "label": "code",
            "kind": "boundary",
            "boundary_pair": "code_bogus",
        }
        assert generate_eval.validate_sample(sample) is False

    def test_rejects_boundary_label_not_in_pair(self):
        sample = {
            "text": "some text",
            "label": "prose",
            "kind": "boundary",
            "boundary_pair": "code_structured",
        }
        assert generate_eval.validate_sample(sample) is False


# ── Domain seeds and length buckets ──────────────────────────────────

class TestDomainSeedsAndBuckets:
    """Domain seeds list and length buckets must exist with required properties."""

    def test_domain_seeds_has_at_least_50(self):
        assert len(generate_eval.DOMAIN_SEEDS) >= 50

    def test_length_buckets_defined(self):
        assert "short" in generate_eval.LENGTH_BUCKETS
        assert "medium" in generate_eval.LENGTH_BUCKETS
        assert "long" in generate_eval.LENGTH_BUCKETS

    def test_length_bucket_ranges(self):
        short = generate_eval.LENGTH_BUCKETS["short"]
        medium = generate_eval.LENGTH_BUCKETS["medium"]
        long_ = generate_eval.LENGTH_BUCKETS["long"]
        assert short == (3, 10)
        assert medium == (20, 50)
        assert long_ == (100, 200)


# ── generate_clear_samples ───────────────────────────────────────────

class TestGenerateClearSamples:
    """generate_clear_samples must call the API and return validated dicts."""

    @patch("trainr.core.generate_eval.openai")
    def test_returns_list_of_validated_samples(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client

        # Build a fake API response with 1 valid sample
        sample = {
            "text": "The Milky Way is a barred spiral galaxy.",
            "label": "prose",
            "kind": "clear",
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([sample])
        mock_client.chat.completions.create.return_value = mock_response

        results = generate_eval.generate_clear_samples(
            category="prose",
            count=1,
            domain_seeds=["astronomy"],
            client=mock_client,
            model="gpt-5.4",
        )
        assert len(results) >= 1
        assert results[0]["label"] == "prose"
        assert results[0]["kind"] == "clear"

    @patch("trainr.core.generate_eval.openai")
    def test_skips_invalid_samples(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client

        good = {"text": "Valid sample text.", "label": "code", "kind": "clear"}
        bad = {"text": "", "label": "code", "kind": "clear"}  # empty text -> invalid
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([good, bad])
        mock_client.chat.completions.create.return_value = mock_response

        results = generate_eval.generate_clear_samples(
            category="code",
            count=2,
            domain_seeds=["devops"],
            client=mock_client,
            model="gpt-5.4",
        )
        # Only the good sample should survive validation
        valid_count = sum(1 for r in results if r["text"] == "Valid sample text.")
        assert valid_count >= 1


# ── generate_boundary_samples ────────────────────────────────────────

class TestGenerateBoundarySamples:
    @patch("trainr.core.generate_eval.openai")
    def test_returns_validated_boundary_samples(self, mock_openai_mod):
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client

        sample = {
            "text": "apiVersion: v1\nkind: Service",
            "label": "code",
            "kind": "boundary",
            "boundary_pair": "code_structured",
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([sample])
        mock_client.chat.completions.create.return_value = mock_response

        results = generate_eval.generate_boundary_samples(
            pair="code_structured",
            count=1,
            domain_seeds=["devops"],
            client=mock_client,
            model="gpt-5.4",
        )
        assert len(results) >= 1
        assert results[0]["kind"] == "boundary"
        assert results[0]["boundary_pair"] == "code_structured"


# ── CLI / argparse ───────────────────────────────────────────────────

class TestCLIParsing:
    def test_default_arguments(self):
        args = generate_eval.parse_args(["--mode", "clear", "--output-dir", "/tmp/x"])
        assert args.mode == "clear"
        assert args.output_dir == "/tmp/x"
        assert args.samples_per_category == 1000
        assert args.samples_per_pair == 1000
        assert args.model == "gpt-5.4"
        assert args.dry_run is False

    def test_dry_run_flag(self):
        args = generate_eval.parse_args(["--mode", "all", "--output-dir", "/tmp/x", "--dry-run"])
        assert args.dry_run is True

    def test_custom_counts(self):
        args = generate_eval.parse_args([
            "--mode", "boundary",
            "--output-dir", "/tmp/x",
            "--samples-per-category", "500",
            "--samples-per-pair", "200",
        ])
        assert args.samples_per_category == 500
        assert args.samples_per_pair == 200

    def test_model_override(self):
        args = generate_eval.parse_args(["--mode", "clear", "--output-dir", "/tmp/x", "--model", "gpt-4o"])
        assert args.model == "gpt-4o"


# ── Dry-run integration ─────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_clear_does_not_call_api(self):
        """--dry-run should produce output summary without calling OpenAI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = generate_eval.parse_args([
                "--mode", "clear",
                "--output-dir", tmpdir,
                "--samples-per-category", "1",
                "--dry-run",
            ])
            # Should not raise and should not need OPENAI_API_KEY
            generate_eval.run(args)
            # No files should be written in dry-run
            assert not os.path.exists(os.path.join(tmpdir, "clear.jsonl"))

    def test_dry_run_boundary_does_not_call_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = generate_eval.parse_args([
                "--mode", "boundary",
                "--output-dir", tmpdir,
                "--samples-per-pair", "1",
                "--dry-run",
            ])
            generate_eval.run(args)
            assert not os.path.exists(os.path.join(tmpdir, "boundary.jsonl"))

    def test_dry_run_all_does_not_call_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = generate_eval.parse_args([
                "--mode", "all",
                "--output-dir", tmpdir,
                "--samples-per-category", "1",
                "--samples-per-pair", "1",
                "--dry-run",
            ])
            generate_eval.run(args)
            assert not os.path.exists(os.path.join(tmpdir, "clear.jsonl"))
            assert not os.path.exists(os.path.join(tmpdir, "boundary.jsonl"))


# ── File output ──────────────────────────────────────────────────────

class TestFileOutput:
    def test_writes_clear_jsonl(self):
        mock_client = MagicMock()

        def _fake_create(**kwargs):
            """Return a sample whose label matches whatever the prompt asked for."""
            prompt = kwargs.get("messages", [{}])[-1].get("content", "")
            # Extract category from the prompt text
            for cat in generate_eval.VALID_CATEGORIES:
                if f"'{cat}'" in prompt:
                    label = cat
                    break
            else:
                label = "prose"
            sample = {"text": f"Sample text for {label}.", "label": label, "kind": "clear"}
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = json.dumps([sample])
            return resp

        mock_client.chat.completions.create.side_effect = _fake_create

        with tempfile.TemporaryDirectory() as tmpdir:
            args = generate_eval.parse_args([
                "--mode", "clear",
                "--output-dir", tmpdir,
                "--samples-per-category", "1",
            ])
            with patch("trainr.core.generate_eval._create_client", return_value=mock_client):
                generate_eval.run(args)

            path = os.path.join(tmpdir, "clear.jsonl")
            assert os.path.exists(path)
            with open(path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            # Should have 1 sample per category = 4 total
            assert len(lines) == 4
            labels = {line["label"] for line in lines}
            assert labels == generate_eval.VALID_CATEGORIES
