"""Tests for training/validate_pilot.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the training directory is importable
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    *,
    text="Hello world.",
    expected_category="prose",
    sub_type="plain",
    model="claude-3-opus",
    temperature=0.7,
    prompt_template="clear_v1",
    length_bucket="short",
    content_domain="astronomy",
    reasoning_mode=False,
):
    """Create a sample dict with sensible defaults."""
    return {
        "text": text,
        "expected_category": expected_category,
        "sub_type": sub_type,
        "model": model,
        "temperature": temperature,
        "prompt_template": prompt_template,
        "length_bucket": length_bucket,
        "content_domain": content_domain,
        "reasoning_mode": reasoning_mode,
    }


def _write_jsonl(path, samples):
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Tests for load_samples
# ---------------------------------------------------------------------------


class TestLoadSamples:
    def test_reads_jsonl_file(self):
        from validate_pilot import load_samples

        samples = [_make_sample(), _make_sample(text="Second sample.")]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            path = f.name

        try:
            result = load_samples(path)
            assert len(result) == 2
            assert result[0]["text"] == "Hello world."
            assert result[1]["text"] == "Second sample."
        finally:
            os.unlink(path)

    def test_returns_list_of_dicts(self):
        from validate_pilot import load_samples

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_sample()) + "\n")
            path = f.name

        try:
            result = load_samples(path)
            assert isinstance(result, list)
            assert isinstance(result[0], dict)
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty_list(self):
        from validate_pilot import load_samples

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            result = load_samples(path)
            assert result == []
        finally:
            os.unlink(path)

    def test_skips_blank_lines(self):
        from validate_pilot import load_samples

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(_make_sample()) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_sample(text="After blank.")) + "\n")
            path = f.name

        try:
            result = load_samples(path)
            assert len(result) == 2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests for diversity_report
# ---------------------------------------------------------------------------


class TestDiversityReport:
    def _varied_samples(self):
        """Build a varied sample set for diversity testing."""
        models = ["claude-3-opus", "claude-3-sonnet", "gpt-4", "gpt-4o", "gemini-pro",
                   "llama-3", "mistral-large"]
        templates = ["clear_v1", "clear_v2", "boundary_v1", "boundary_v2", "edge_v1"]
        temps = [0.3, 0.5, 0.7, 0.9, 1.0]
        domains = ["astronomy", "finance", "healthcare", "devops", "gaming"]
        buckets = ["short", "medium", "long"]

        samples = []
        for i in range(30):
            samples.append(_make_sample(
                sub_type="plain",
                model=models[i % len(models)],
                temperature=temps[i % len(temps)],
                prompt_template=templates[i % len(templates)],
                length_bucket=buckets[i % len(buckets)],
                content_domain=domains[i % len(domains)],
            ))
        return samples

    def test_returns_dict_keyed_by_sub_type(self):
        from validate_pilot import diversity_report

        samples = [_make_sample(sub_type="plain"), _make_sample(sub_type="markdown")]
        report = diversity_report(samples)
        assert "plain" in report
        assert "markdown" in report

    def test_sample_count(self):
        from validate_pilot import diversity_report

        samples = [_make_sample(sub_type="plain")] * 5
        report = diversity_report(samples)
        assert report["plain"]["sample_count"] == 5

    def test_model_distribution(self):
        from validate_pilot import diversity_report

        samples = [
            _make_sample(model="claude-3-opus"),
            _make_sample(model="claude-3-opus"),
            _make_sample(model="gpt-4"),
        ]
        report = diversity_report(samples)
        assert report["plain"]["model_distribution"]["claude-3-opus"] == 2
        assert report["plain"]["model_distribution"]["gpt-4"] == 1

    def test_temperature_values(self):
        from validate_pilot import diversity_report

        samples = [
            _make_sample(temperature=0.3),
            _make_sample(temperature=0.7),
            _make_sample(temperature=0.3),
        ]
        report = diversity_report(samples)
        assert report["plain"]["temperature_values"] == {0.3, 0.7}

    def test_prompt_templates(self):
        from validate_pilot import diversity_report

        samples = [
            _make_sample(prompt_template="clear_v1"),
            _make_sample(prompt_template="boundary_v1"),
        ]
        report = diversity_report(samples)
        assert report["plain"]["prompt_templates"] == {"clear_v1", "boundary_v1"}

    def test_length_buckets(self):
        from validate_pilot import diversity_report

        samples = [
            _make_sample(length_bucket="short"),
            _make_sample(length_bucket="long"),
        ]
        report = diversity_report(samples)
        assert report["plain"]["length_buckets"] == {"short", "long"}

    def test_content_domains(self):
        from validate_pilot import diversity_report

        samples = [
            _make_sample(content_domain="astronomy"),
            _make_sample(content_domain="finance"),
        ]
        report = diversity_report(samples)
        assert report["plain"]["content_domains"] == {"astronomy", "finance"}


# ---------------------------------------------------------------------------
# Tests for check_diversity_checklist
# ---------------------------------------------------------------------------


class TestCheckDiversityChecklist:
    def _passing_stats(self, *, reasoning_sub_type=False):
        """Return stats that should pass all 7 checks."""
        stats = {
            "sample_count": 100,
            "model_distribution": {
                "claude-3-opus": 10,
                "claude-3-sonnet": 10,
                "gpt-4": 10,
                "gpt-4o": 10,
                "gemini-pro": 10,
                "llama-3": 10,
                "mistral-large": 10,
            },
            "temperature_values": {0.3, 0.5, 0.7},
            "prompt_templates": {"clear_v1", "clear_v2", "boundary_v1", "edge_v1"},
            "length_buckets": {"short", "medium", "long"},
            "content_domains": {"astronomy", "finance", "healthcare"},
        }
        if reasoning_sub_type:
            stats["has_reasoning_mode"] = True
        return stats

    def test_all_pass(self):
        from validate_pilot import check_diversity_checklist

        results = check_diversity_checklist(self._passing_stats())
        for name, passed, detail in results:
            if name == "reasoning_mode_included":
                continue  # Only required for prose sub-types
            assert passed, f"Check {name!r} failed: {detail}"

    def test_returns_list_of_tuples(self):
        from validate_pilot import check_diversity_checklist

        results = check_diversity_checklist(self._passing_stats())
        assert isinstance(results, list)
        assert len(results) == 7
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 3

    def test_fewer_than_5_models_fails(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats()
        stats["model_distribution"] = {"claude-3-opus": 50, "gpt-4": 50}
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["at_least_5_models"] is False

    def test_single_model_exceeds_15_percent_fails(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats()
        # 50 out of 100 = 50% for one model
        stats["model_distribution"] = {
            "claude-3-opus": 50,
            "gpt-4": 10,
            "gemini-pro": 10,
            "llama-3": 10,
            "mistral-large": 10,
        }
        stats["sample_count"] = 90
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["no_model_exceeds_15_pct"] is False

    def test_fewer_than_3_temperatures_fails(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats()
        stats["temperature_values"] = {0.7, 0.9}
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["at_least_3_temperatures"] is False

    def test_fewer_than_4_templates_fails(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats()
        stats["prompt_templates"] = {"clear_v1", "clear_v2"}
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["at_least_4_templates"] is False

    def test_missing_length_bucket_fails(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats()
        stats["length_buckets"] = {"short", "medium"}
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["all_length_buckets_present"] is False

    def test_fewer_than_3_domains_fails(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats()
        stats["content_domains"] = {"astronomy", "finance"}
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["at_least_3_domains"] is False

    def test_reasoning_mode_check_passes_when_present(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats(reasoning_sub_type=True)
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["reasoning_mode_included"] is True

    def test_reasoning_mode_check_fails_when_missing(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats()
        stats["has_reasoning_mode"] = False
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["reasoning_mode_included"] is False

    def test_model_at_exactly_15_percent_passes(self):
        from validate_pilot import check_diversity_checklist

        stats = self._passing_stats()
        # 15 out of 100 = exactly 15%
        stats["model_distribution"] = {
            "claude-3-opus": 15,
            "claude-3-sonnet": 15,
            "gpt-4": 15,
            "gpt-4o": 15,
            "gemini-pro": 15,
            "llama-3": 15,
            "mistral-large": 10,
        }
        results = check_diversity_checklist(stats)
        check = {name: passed for name, passed, _ in results}
        assert check["no_model_exceeds_15_pct"] is True


# ---------------------------------------------------------------------------
# Tests for run_classifier_audit
# ---------------------------------------------------------------------------


class TestRunClassifierAudit:
    def test_returns_dict_when_binary_missing(self, capsys):
        from validate_pilot import run_classifier_audit

        samples = [_make_sample()]
        result = run_classifier_audit(samples, classify_bin="/nonexistent/classify")
        assert result is None
        captured = capsys.readouterr()
        assert "warning" in captured.err.lower() or "not found" in captured.err.lower()


# ---------------------------------------------------------------------------
# Tests for print_audit_samples
# ---------------------------------------------------------------------------


class TestPrintAuditSamples:
    def test_prints_samples(self, capsys):
        from validate_pilot import print_audit_samples

        samples = []
        for st in ["plain", "markdown", "python", "csv"]:
            for i in range(5):
                samples.append(_make_sample(sub_type=st, text=f"Text for {st} {i}"))

        print_audit_samples(samples, n=8)
        captured = capsys.readouterr()
        # Should print some sample text
        assert "Text for" in captured.out

    def test_respects_n_parameter(self, capsys):
        from validate_pilot import print_audit_samples

        samples = [_make_sample(sub_type="plain", text=f"Item {i}") for i in range(20)]
        print_audit_samples(samples, n=3)
        captured = capsys.readouterr()
        # Count the "--- Sample N ---" header lines to determine how many were printed
        header_lines = [l for l in captured.out.splitlines() if l.strip().startswith("--- Sample")]
        assert len(header_lines) <= 3


# ---------------------------------------------------------------------------
# Tests for CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_default_input_path(self):
        from validate_pilot import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.input == "data/source/pilot/pilot_samples.jsonl"

    def test_custom_input_path(self):
        from validate_pilot import build_parser

        parser = build_parser()
        args = parser.parse_args(["--input", "/tmp/test.jsonl"])
        assert args.input == "/tmp/test.jsonl"
