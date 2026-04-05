"""Tests for trainr.core.annotate_detections — LLM-based multi-label annotation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestDetectionLabels:
    def test_labels_defined(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        assert isinstance(DETECTION_LABELS, list)
        assert len(DETECTION_LABELS) == 29

    def test_expected_labels_present(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        expected = [
            "plain", "markdown", "rst", "latex",
            "python", "javascript", "typescript", "rust", "go", "java", "sql", "shell", "css",
            "yaml", "toml", "ini", "dockerfile", "makefile",
            "html", "xml", "sgml",
            "csv", "tsv", "pipe_table", "fixed_width",
            "json", "jsonl", "key_value", "log_lines",
        ]
        for label in expected:
            assert label in DETECTION_LABELS, f"Missing label: {label}"

    def test_no_duplicate_labels(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        assert len(DETECTION_LABELS) == len(set(DETECTION_LABELS))


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_contains_text(self):
        from trainr.core.annotate_detections import build_prompt

        prompt = build_prompt("hello world")
        assert "hello world" in prompt

    def test_prompt_contains_all_labels(self):
        from trainr.core.annotate_detections import build_prompt, DETECTION_LABELS

        prompt = build_prompt("some text")
        for label in DETECTION_LABELS:
            assert label in prompt, f"Label {label} missing from prompt"

    def test_prompt_requests_json(self):
        from trainr.core.annotate_detections import build_prompt

        prompt = build_prompt("some text")
        assert "JSON" in prompt or "json" in prompt.lower()

    def test_prompt_mentions_binary(self):
        from trainr.core.annotate_detections import build_prompt

        prompt = build_prompt("some text")
        # Should mention 0 or 1 values
        assert "0" in prompt and "1" in prompt


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_json_all_labels(self):
        from trainr.core.annotate_detections import parse_response, DETECTION_LABELS

        response = json.dumps({label: 0 for label in DETECTION_LABELS})
        result = parse_response(response)
        assert isinstance(result, dict)
        for label in DETECTION_LABELS:
            assert label in result
            assert result[label] in (0, 1)

    def test_valid_json_some_ones(self):
        from trainr.core.annotate_detections import parse_response, DETECTION_LABELS

        data = {label: 0 for label in DETECTION_LABELS}
        data["python"] = 1
        data["markdown"] = 1
        response = json.dumps(data)
        result = parse_response(response)
        assert result["python"] == 1
        assert result["markdown"] == 1
        assert result["plain"] == 0

    def test_malformed_json_returns_all_zeros(self):
        from trainr.core.annotate_detections import parse_response, DETECTION_LABELS

        result = parse_response("this is not json at all")
        assert isinstance(result, dict)
        for label in DETECTION_LABELS:
            assert result[label] == 0

    def test_missing_labels_default_to_zero(self):
        from trainr.core.annotate_detections import parse_response, DETECTION_LABELS

        # Only provide a subset of labels
        response = json.dumps({"python": 1, "markdown": 1})
        result = parse_response(response)
        assert result["python"] == 1
        assert result["markdown"] == 1
        assert result["plain"] == 0
        assert result["rust"] == 0
        assert len(result) == len(DETECTION_LABELS)

    def test_json_wrapped_in_markdown_code_block(self):
        from trainr.core.annotate_detections import parse_response, DETECTION_LABELS

        inner = json.dumps({label: 0 for label in DETECTION_LABELS})
        response = f"```json\n{inner}\n```"
        result = parse_response(response)
        for label in DETECTION_LABELS:
            assert label in result

    def test_non_binary_values_clamped(self):
        from trainr.core.annotate_detections import parse_response, DETECTION_LABELS

        data = {label: 0 for label in DETECTION_LABELS}
        data["python"] = 5
        data["markdown"] = -1
        response = json.dumps(data)
        result = parse_response(response)
        assert result["python"] == 1
        assert result["markdown"] == 0


# ---------------------------------------------------------------------------
# Output columns test
# ---------------------------------------------------------------------------


class TestOutputColumns:
    def test_output_has_det_columns(self):
        from trainr.core.annotate_detections import DETECTION_LABELS, annotate_dataframe

        df = pl.DataFrame({
            "text": ["print('hello')", "# Heading\nSome text"],
            "category": ["code", "prose"],
        })

        # Mock the LLM call to return known detections
        def mock_call_llm(text, model, api_key):
            if "print" in text:
                result = {label: 0 for label in DETECTION_LABELS}
                result["python"] = 1
                return result
            else:
                result = {label: 0 for label in DETECTION_LABELS}
                result["markdown"] = 1
                return result

        with patch("trainr.core.annotate_detections.call_llm", side_effect=mock_call_llm):
            result_df = annotate_dataframe(df, model="test-model", api_key="test-key")

        # Check that det_* columns are present
        expected_cols = [f"det_{label}" for label in DETECTION_LABELS]
        for col in expected_cols:
            assert col in result_df.columns, f"Missing column: {col}"

        # Check values
        assert result_df["det_python"][0] == 1
        assert result_df["det_python"][1] == 0
        assert result_df["det_markdown"][0] == 0
        assert result_df["det_markdown"][1] == 1

    def test_original_columns_preserved(self):
        from trainr.core.annotate_detections import DETECTION_LABELS, annotate_dataframe

        df = pl.DataFrame({
            "text": ["hello"],
            "category": ["prose"],
            "extra_col": [42],
        })

        def mock_call_llm(text, model, api_key):
            return {label: 0 for label in DETECTION_LABELS}

        with patch("trainr.core.annotate_detections.call_llm", side_effect=mock_call_llm):
            result_df = annotate_dataframe(df, model="test-model", api_key="test-key")

        assert "text" in result_df.columns
        assert "category" in result_df.columns
        assert "extra_col" in result_df.columns
        assert result_df["extra_col"][0] == 42


# ---------------------------------------------------------------------------
# CLI / main function test
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_reads_parquet_and_writes_parquet(self):
        from trainr.core.annotate_detections import DETECTION_LABELS, main

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = str(Path(tmpdir) / "input.parquet")
            output_path = str(Path(tmpdir) / "output.parquet")

            # Create input Parquet
            df = pl.DataFrame({
                "text": ["def foo(): pass", "# Title\nParagraph"],
                "category": ["code", "prose"],
            })
            df.write_parquet(input_path)

            def mock_call_llm(text, model, api_key):
                return {label: 0 for label in DETECTION_LABELS}

            with patch("trainr.core.annotate_detections.call_llm", side_effect=mock_call_llm), \
                 patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                main(["--input", input_path, "--output", output_path])

            result = pl.read_parquet(output_path)
            assert len(result) == 2
            expected_cols = [f"det_{label}" for label in DETECTION_LABELS]
            for col in expected_cols:
                assert col in result.columns
