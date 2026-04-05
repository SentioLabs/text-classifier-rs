"""Tests for trainr.core.sample

Tests only programmatic generators and the emit_sample helper.
HF streaming functions are NOT tested (they require network access).
"""

import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# emit_sample
# ---------------------------------------------------------------------------


class TestEmitSample:
    def test_valid_sample(self):
        from trainr.core.sample import emit_sample

        result = emit_sample("x" * 100, "prose", "plain", "wikipedia")
        assert result is not None
        assert result["text"] == "x" * 100
        assert result["expected_category"] == "prose"
        assert result["sub_type"] == "plain"
        assert result["source"] == "real/wikipedia"
        assert result["model"] == "real/wikipedia"

    def test_too_short_returns_none(self):
        from trainr.core.sample import emit_sample

        assert emit_sample("short", "prose", "plain", "src") is None

    def test_exactly_50_chars(self):
        from trainr.core.sample import emit_sample

        result = emit_sample("x" * 50, "prose", "plain", "src")
        assert result is not None

    def test_too_long_returns_none(self):
        from trainr.core.sample import emit_sample

        assert emit_sample("x" * 10001, "prose", "plain", "src") is None

    def test_exactly_10000_chars(self):
        from trainr.core.sample import emit_sample

        result = emit_sample("x" * 10000, "prose", "plain", "src")
        assert result is not None

    def test_strips_whitespace(self):
        from trainr.core.sample import emit_sample

        text = "  " + "x" * 100 + "  "
        result = emit_sample(text, "prose", "plain", "src")
        assert result is not None
        assert result["text"] == "x" * 100

    def test_stripped_text_too_short(self):
        from trainr.core.sample import emit_sample

        # After stripping, only whitespace remains
        assert emit_sample("   ", "prose", "plain", "src") is None

    def test_output_is_json_serializable(self):
        from trainr.core.sample import emit_sample

        result = emit_sample("Hello world " * 10, "artifact", "pdf_dump", "finepdfs")
        assert result is not None
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# generate_skip_samples
# ---------------------------------------------------------------------------


class TestGenerateSkipSamples:
    def test_yields_dicts(self):
        from trainr.core.sample import generate_skip_samples

        samples = list(generate_skip_samples(n=10))
        assert len(samples) == 10
        for s in samples:
            assert isinstance(s, dict)

    def test_output_structure(self):
        from trainr.core.sample import generate_skip_samples

        for s in generate_skip_samples(n=5):
            assert s["expected_category"] == "artifact"
            assert s["sub_type"] == "skip"
            assert s["source"].startswith("real/")
            assert s["model"].startswith("real/")
            assert "text" in s

    def test_deterministic_with_seed(self):
        from trainr.core.sample import generate_skip_samples

        a = list(generate_skip_samples(n=20, seed=42))
        b = list(generate_skip_samples(n=20, seed=42))
        assert a == b

    def test_texts_are_short_fragments(self):
        """Skip samples should be empty/whitespace/single-word fragments."""
        from trainr.core.sample import generate_skip_samples

        for s in generate_skip_samples(n=50):
            # skip samples are intentionally short/empty - they bypass the
            # normal length filter since they represent garbage input
            assert len(s["text"]) < 50


# ---------------------------------------------------------------------------
# generate_ocr_garbage
# ---------------------------------------------------------------------------


class TestGenerateOcrGarbage:
    def test_yields_requested_count(self):
        from trainr.core.sample import generate_ocr_garbage

        samples = list(generate_ocr_garbage(n=20))
        assert len(samples) == 20

    def test_output_structure(self):
        from trainr.core.sample import generate_ocr_garbage

        for s in generate_ocr_garbage(n=5):
            assert s["expected_category"] == "artifact"
            assert s["sub_type"] == "ocr_garbage"
            assert s["source"].startswith("real/")
            assert "text" in s
            assert len(s["text"]) >= 50

    def test_deterministic_with_seed(self):
        from trainr.core.sample import generate_ocr_garbage

        a = list(generate_ocr_garbage(n=10, seed=99))
        b = list(generate_ocr_garbage(n=10, seed=99))
        assert a == b


# ---------------------------------------------------------------------------
# generate_hybrid_document_samples
# ---------------------------------------------------------------------------


class TestGenerateHybridDocumentSamples:
    def test_yields_requested_count(self):
        from trainr.core.sample import generate_hybrid_document_samples

        samples = list(generate_hybrid_document_samples(n=10))
        assert len(samples) == 10

    def test_output_structure(self):
        from trainr.core.sample import generate_hybrid_document_samples

        for s in generate_hybrid_document_samples(n=5):
            assert s["expected_category"] == "artifact"
            assert s["sub_type"] == "pdf_dump"
            assert s["source"] == "real/generated_hybrid_pdf_dump"
            assert len(s["text"]) >= 50

    def test_deterministic_with_seed(self):
        from trainr.core.sample import generate_hybrid_document_samples

        a = list(generate_hybrid_document_samples(n=10, seed=7))
        b = list(generate_hybrid_document_samples(n=10, seed=7))
        assert a == b


# ---------------------------------------------------------------------------
# generate_csv_samples
# ---------------------------------------------------------------------------


class TestGenerateCsvSamples:
    def test_yields_requested_count(self):
        from trainr.core.sample import generate_csv_samples

        samples = list(generate_csv_samples(n=20))
        assert len(samples) == 20

    def test_output_structure(self):
        from trainr.core.sample import generate_csv_samples

        for s in generate_csv_samples(n=10):
            assert s["expected_category"] == "structured"
            assert s["sub_type"] in ("csv", "tsv")
            assert s["source"].startswith("real/")
            assert len(s["text"]) >= 50

    def test_contains_delimiter(self):
        from trainr.core.sample import generate_csv_samples

        for s in generate_csv_samples(n=20):
            # Every CSV/TSV sample should have either commas or tabs
            assert "," in s["text"] or "\t" in s["text"]

    def test_has_multiple_lines(self):
        from trainr.core.sample import generate_csv_samples

        for s in generate_csv_samples(n=10):
            lines = s["text"].strip().split("\n")
            assert len(lines) >= 2, "CSV samples should have header + data rows"

    def test_deterministic_with_seed(self):
        from trainr.core.sample import generate_csv_samples

        a = list(generate_csv_samples(n=10, seed=7))
        b = list(generate_csv_samples(n=10, seed=7))
        assert a == b


# ---------------------------------------------------------------------------
# generate_log_samples
# ---------------------------------------------------------------------------


class TestGenerateLogSamples:
    def test_yields_requested_count(self):
        from trainr.core.sample import generate_log_samples

        samples = list(generate_log_samples(n=15))
        assert len(samples) == 15

    def test_output_structure(self):
        from trainr.core.sample import generate_log_samples

        for s in generate_log_samples(n=5):
            assert s["expected_category"] == "structured"
            assert s["sub_type"] == "log_lines"
            assert s["source"].startswith("real/")
            assert len(s["text"]) >= 50

    def test_has_multiple_lines(self):
        from trainr.core.sample import generate_log_samples

        for s in generate_log_samples(n=5):
            lines = s["text"].strip().split("\n")
            assert len(lines) >= 3, "Log samples should have multiple log lines"

    def test_deterministic_with_seed(self):
        from trainr.core.sample import generate_log_samples

        a = list(generate_log_samples(n=10, seed=12))
        b = list(generate_log_samples(n=10, seed=12))
        assert a == b


# ---------------------------------------------------------------------------
# generate_kv_samples
# ---------------------------------------------------------------------------


class TestGenerateKvSamples:
    def test_yields_requested_count(self):
        from trainr.core.sample import generate_kv_samples

        samples = list(generate_kv_samples(n=15))
        assert len(samples) == 15

    def test_output_structure(self):
        from trainr.core.sample import generate_kv_samples

        for s in generate_kv_samples(n=10):
            assert s["expected_category"] == "structured"
            assert s["sub_type"] in ("ini", "key_value", "xml")
            assert s["source"].startswith("real/")
            assert len(s["text"]) >= 50

    def test_deterministic_with_seed(self):
        from trainr.core.sample import generate_kv_samples

        a = list(generate_kv_samples(n=10, seed=55))
        b = list(generate_kv_samples(n=10, seed=55))
        assert a == b


# ---------------------------------------------------------------------------
# generate_prose_variants
# ---------------------------------------------------------------------------


class TestGenerateProseVariants:
    def test_yields_requested_count(self):
        from trainr.core.sample import generate_prose_variants

        samples = list(generate_prose_variants(n=15))
        assert len(samples) == 15

    def test_output_structure(self):
        from trainr.core.sample import generate_prose_variants

        for s in generate_prose_variants(n=10):
            assert s["expected_category"] == "prose"
            assert s["sub_type"] == "plain"
            assert s["source"].startswith("real/")
            assert len(s["text"]) >= 50

    def test_deterministic_with_seed(self):
        from trainr.core.sample import generate_prose_variants

        a = list(generate_prose_variants(n=10, seed=33))
        b = list(generate_prose_variants(n=10, seed=33))
        assert a == b


# ---------------------------------------------------------------------------
# Source planning helpers
# ---------------------------------------------------------------------------


class TestSourcePlanning:
    def test_summary_keeps_artifact_synthetic_below_real(self):
        from trainr.core.sample import SOURCE_PLANS, summarize_source_plans

        summary = summarize_source_plans(SOURCE_PLANS)
        assert summary["artifact_synthetic"] <= summary["artifact_real"]
        assert summary["hybrid_pdf_dump"] > 0
        assert summary["structured_real"] > 0

    def test_format_source_plan_mentions_quota_summary(self):
        from trainr.core.sample import SOURCE_PLANS, format_source_plan

        output = format_source_plan(SOURCE_PLANS)
        assert "Quota summary:" in output
        assert "hybrid_pdf_dump" in output
        assert "sample_finepdfs" in output

    def test_finepdfs_candidates_try_schema_tolerant_options(self):
        from trainr.core.sample import finepdfs_stream_candidates

        candidates = finepdfs_stream_candidates()
        assert candidates[0]["dataset_name"] == "HuggingFaceFW/finepdfs_100BT"
        assert candidates[0]["kwargs"]["trust_remote_code"] is True
        assert "data_files" in candidates[1]["kwargs"] or "data_files" in candidates[2]["kwargs"]

    def test_stack_config_candidates_cover_json_yaml_toml_ini(self):
        from trainr.core.sample import stack_dataset_candidates

        candidates = stack_dataset_candidates("configs")
        assert [c["sub_type"] for c in candidates] == ["json", "yaml", "toml", "ini"]
        assert all(c["dataset_name"] == "bigcode/the-stack-dedup" for c in candidates)


# ---------------------------------------------------------------------------
# Integration: all generators produce valid samples
# ---------------------------------------------------------------------------


class TestAllGeneratorsIntegration:
    """Verify every generator function produces structurally valid output."""

    REQUIRED_KEYS = {"text", "expected_category", "sub_type", "source", "model"}
    VALID_CATEGORIES = {"prose", "artifact", "structured"}

    @pytest.mark.parametrize(
        "gen_func,gen_kwargs",
        [
            ("generate_skip_samples", {"n": 5}),
            ("generate_ocr_garbage", {"n": 5}),
            ("generate_hybrid_document_samples", {"n": 5}),
            ("generate_csv_samples", {"n": 5}),
            ("generate_log_samples", {"n": 5}),
            ("generate_kv_samples", {"n": 5}),
            ("generate_prose_variants", {"n": 5}),
        ],
    )
    def test_valid_structure(self, gen_func, gen_kwargs):
        import trainr.core.sample as sample_real_data

        func = getattr(sample_real_data, gen_func)
        samples = list(func(**gen_kwargs))
        assert len(samples) > 0, f"{gen_func} produced no samples"
        for s in samples:
            assert self.REQUIRED_KEYS <= set(s.keys()), (
                f"{gen_func}: missing keys {self.REQUIRED_KEYS - set(s.keys())}"
            )
            assert s["expected_category"] in self.VALID_CATEGORIES, (
                f"{gen_func}: invalid category {s['expected_category']}"
            )
            assert s["source"].startswith("real/"), (
                f"{gen_func}: source should start with 'real/'"
            )
            # All samples should be JSON-serializable
            json.dumps(s)
