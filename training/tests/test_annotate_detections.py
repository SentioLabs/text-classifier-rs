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
        # Current baseline: 40 labels (pre-iter16).
        # Phase 1 removes log_lines (→39), phases 3-5 add 3 semantic labels (→42).
        assert len(DETECTION_LABELS) == 40

    def test_expected_labels_present(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        expected = [
            "plain", "markdown", "rst", "latex",
            "python", "javascript", "typescript", "rust", "go", "java", "c_cpp", "objc",
            "csharp", "powershell", "ruby", "php", "swift", "kotlin", "r", "lua", "graphql",
            "sql", "shell", "css",
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
        from trainr.core.annotate_detections import SYSTEM_PROMPT, DETECTION_LABELS

        for label in DETECTION_LABELS:
            assert label in SYSTEM_PROMPT, f"Label {label} missing from system prompt"

    def test_prompt_requests_json(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT.lower()

    def test_prompt_mentions_binary(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        # Should mention 0 or 1 values
        assert "0" in SYSTEM_PROMPT and "1" in SYSTEM_PROMPT


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
        def mock_call_llm(text, model, api_key, **kwargs):
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

        def mock_call_llm(text, model, api_key, **kwargs):
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

            def mock_call_llm(text, model, api_key, **kwargs):
                return {label: 0 for label in DETECTION_LABELS}

            with patch("trainr.core.annotate_detections.call_llm", side_effect=mock_call_llm), \
                 patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                main(["--input", input_path, "--output", output_path])

            result = pl.read_parquet(output_path)
            assert len(result) == 2
            expected_cols = [f"det_{label}" for label in DETECTION_LABELS]
            for col in expected_cols:
                assert col in result.columns


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestRoutingTable:
    def test_routing_table_exists(self):
        from trainr.core.annotate_detections import ROUTING_TABLE

        assert isinstance(ROUTING_TABLE, dict)

    def test_routing_table_keys_are_valid_detection_labels(self):
        """ROUTING_TABLE is keyed by sub_type. Every key must be a known
        detection label that mirrors a sub_type (i.e., NOT a semantic
        detection-only label like log_content). Labels not in ROUTING_TABLE
        fall through to the default model in annotate_dataframe()."""
        from trainr.core.annotate_detections import DETECTION_LABELS, ROUTING_TABLE

        for sub_type in ROUTING_TABLE.keys():
            assert sub_type in DETECTION_LABELS, (
                f"ROUTING_TABLE key {sub_type!r} is not in DETECTION_LABELS"
            )

    def test_routing_table_values_are_model_backend_tuples(self):
        from trainr.core.annotate_detections import ROUTING_TABLE

        for sub_type, value in ROUTING_TABLE.items():
            assert isinstance(value, tuple), f"Value for {sub_type} is not a tuple"
            assert len(value) == 2, f"Tuple for {sub_type} has {len(value)} elements, expected 2"
            model, backend = value
            assert isinstance(model, str), f"Model for {sub_type} is not a string"
            assert isinstance(backend, str), f"Backend for {sub_type} is not a string"
            assert backend in ("openrouter", "anthropic"), (
                f"Backend for {sub_type} is '{backend}', expected 'openrouter' or 'anthropic'"
            )


class TestRoutingFlag:
    def test_parser_accepts_routing_flag(self):
        from trainr.core.annotate_detections import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--input", "in.parquet", "--output", "out.parquet", "--routing",
        ])
        assert args.routing is True

    def test_parser_defaults_routing_false(self):
        from trainr.core.annotate_detections import build_parser

        parser = build_parser()
        args = parser.parse_args(["--input", "in.parquet", "--output", "out.parquet"])
        assert args.routing is False


class TestRoutingAnnotation:
    def test_annotate_dataframe_accepts_routing_param(self):
        """annotate_dataframe should accept routing=True without error."""
        from trainr.core.annotate_detections import DETECTION_LABELS, annotate_dataframe

        df = pl.DataFrame({
            "text": ["print('hello')"],
            "sub_type": ["python"],
        })

        def mock_call_llm(text, model, api_key, **kwargs):
            return {label: 0 for label in DETECTION_LABELS}

        with patch("trainr.core.annotate_detections.call_llm", side_effect=mock_call_llm):
            result_df = annotate_dataframe(
                df, api_key="test-key", routing=True,
            )

        assert len(result_df) == len(df)

    def test_routing_dispatches_to_correct_model(self):
        """When routing=True, each sub_type should use its ROUTING_TABLE model."""
        from trainr.core.annotate_detections import (
            DETECTION_LABELS, ROUTING_TABLE, annotate_dataframe,
        )

        df = pl.DataFrame({
            "text": ["print('hello')", "<html></html>"],
            "sub_type": ["python", "html"],
        })

        calls: list[dict] = []

        def mock_call_llm(text, model, api_key, backend="openrouter"):
            calls.append({"model": model, "backend": backend})
            return {label: 0 for label in DETECTION_LABELS}

        with patch("trainr.core.annotate_detections.call_llm", side_effect=mock_call_llm):
            annotate_dataframe(df, api_key="test-key", routing=True)

        # Each row should have been called with its routing-table model
        python_model, python_backend = ROUTING_TABLE["python"]
        html_model, html_backend = ROUTING_TABLE["html"]

        python_calls = [c for c in calls if c["model"] == python_model and c["backend"] == python_backend]
        html_calls = [c for c in calls if c["model"] == html_model and c["backend"] == html_backend]

        assert len(python_calls) >= 1, f"Expected call with model={python_model}, got {calls}"
        assert len(html_calls) >= 1, f"Expected call with model={html_model}, got {calls}"

    def test_routing_unknown_subtype_uses_default(self):
        """Sub-types not in ROUTING_TABLE should fall back to default model."""
        from trainr.core.annotate_detections import (
            DEFAULT_MODEL, DETECTION_LABELS, annotate_dataframe,
        )

        df = pl.DataFrame({
            "text": ["some unknown content"],
            "sub_type": ["nonexistent_type"],
        })

        calls: list[dict] = []

        def mock_call_llm(text, model, api_key, backend="openrouter"):
            calls.append({"model": model, "backend": backend})
            return {label: 0 for label in DETECTION_LABELS}

        with patch("trainr.core.annotate_detections.call_llm", side_effect=mock_call_llm):
            annotate_dataframe(df, api_key="test-key", routing=True)

        assert len(calls) == 1
        assert calls[0]["model"] == DEFAULT_MODEL
        assert calls[0]["backend"] == "openrouter"

    def test_routing_uses_correct_api_keys(self):
        """When routing=True, anthropic-backed models should use anthropic_api_key."""
        from trainr.core.annotate_detections import (
            DETECTION_LABELS, ROUTING_TABLE, annotate_dataframe,
        )

        # Find a sub_type that uses anthropic backend
        anthropic_sub = None
        for sub, (model, backend) in ROUTING_TABLE.items():
            if backend == "anthropic":
                anthropic_sub = sub
                break
        if anthropic_sub is None:
            pytest.skip(
                "No anthropic entries in ROUTING_TABLE — test is a no-op "
                "until a future iteration reintroduces one."
            )

        df = pl.DataFrame({
            "text": ["some content"],
            "sub_type": [anthropic_sub],
        })

        calls: list[dict] = []

        def mock_call_llm(text, model, api_key, backend="openrouter"):
            calls.append({"api_key": api_key, "backend": backend})
            return {label: 0 for label in DETECTION_LABELS}

        with patch("trainr.core.annotate_detections.call_llm", side_effect=mock_call_llm):
            annotate_dataframe(
                df, api_key="openrouter-key",
                routing=True, anthropic_api_key="anthropic-key",
            )

        assert len(calls) == 1
        assert calls[0]["api_key"] == "anthropic-key"
        assert calls[0]["backend"] == "anthropic"
