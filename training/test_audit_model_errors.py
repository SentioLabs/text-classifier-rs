"""Tests for audit_model_errors.py."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from audit_model_errors import _parse_llm_answer


def _run(coro):
    """Helper to run an async coroutine in tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _parse_llm_answer
# ---------------------------------------------------------------------------


class TestParseLlmAnswer:
    def test_prose(self):
        assert _parse_llm_answer("prose") == "prose"

    def test_code(self):
        assert _parse_llm_answer("code") == "code"

    def test_structured(self):
        assert _parse_llm_answer("structured") == "structured"

    def test_whitespace_and_case(self):
        assert _parse_llm_answer("  Prose\n") == "prose"
        assert _parse_llm_answer("CODE ") == "code"

    def test_unknown_fallback(self):
        assert _parse_llm_answer("banana") == "unknown"
        assert _parse_llm_answer("") == "unknown"


# ---------------------------------------------------------------------------
# _compute_verdict  (4-way vote)
# ---------------------------------------------------------------------------


class TestComputeVerdict:
    """Test the 4-way vote logic for dual-LLM mode."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from audit_model_errors import _compute_verdict
        self.compute = _compute_verdict

    def test_3_1_majority_fix_label(self):
        # model, haiku, gpt all say "code"; label says "prose" -> FIX_LABEL
        verdict, winner, counts = self.compute("prose", "code", "code", "code")
        assert verdict == "FIX_LABEL"
        assert winner == "code"
        assert counts["code"] == 3

    def test_3_1_majority_keep_label(self):
        # label, haiku, gpt all say "prose"; model says "code" -> KEEP_LABEL
        verdict, winner, counts = self.compute("prose", "code", "prose", "prose")
        assert verdict == "KEEP_LABEL"
        assert winner == "prose"

    def test_unanimous_keep_label(self):
        # All four agree with label -> KEEP_LABEL
        verdict, winner, counts = self.compute("prose", "prose", "prose", "prose")
        assert verdict == "KEEP_LABEL"
        assert winner == "prose"
        assert counts["prose"] == 4

    def test_2_2_split_tie(self):
        # label+haiku say "prose", model+gpt say "code" -> TIE
        verdict, winner, counts = self.compute("prose", "code", "prose", "code")
        assert verdict == "TIE"

    def test_2_1_1_plurality_fix(self):
        # model+haiku say "code", label says "prose", gpt says "structured"
        verdict, winner, counts = self.compute("prose", "code", "code", "structured")
        assert verdict == "FIX_LABEL"
        assert winner == "code"

    def test_2_1_1_plurality_keep(self):
        # label+haiku say "prose", model says "code", gpt says "structured"
        verdict, winner, counts = self.compute("prose", "code", "prose", "structured")
        assert verdict == "KEEP_LABEL"
        assert winner == "prose"


# ---------------------------------------------------------------------------
# --filter-subtypes
# ---------------------------------------------------------------------------


class TestFilterSubtypes:
    """Test that --filter-subtypes restricts which errors are audited."""

    def _make_predictions_file(self, tmp_path):
        """Write a predictions JSONL with errors of different sub_types."""
        preds = [
            {"text": "a", "expected_category": "prose", "predicted_category": "code",
             "sub_type": "json"},
            {"text": "b", "expected_category": "prose", "predicted_category": "structured",
             "sub_type": "plain"},
            {"text": "c", "expected_category": "code", "predicted_category": "prose",
             "sub_type": "csv"},
            {"text": "d", "expected_category": "code", "predicted_category": "code",
             "sub_type": "json"},  # not an error -> should be skipped
        ]
        p = tmp_path / "preds.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in preds))
        return p

    def test_filter_subtypes_limits_audited_errors(self, tmp_path):
        from audit_model_errors import async_main

        preds = self._make_predictions_file(tmp_path)
        output = tmp_path / "votes.jsonl"

        # Mock the OpenRouter client to avoid real API calls
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="prose"))]
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("audit_model_errors.openai.AsyncOpenAI", return_value=mock_client), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            _run(async_main([
                "--predictions", str(preds),
                "--output", str(output),
                "--backend", "openrouter",
                "--filter-subtypes", "json,csv",
            ]))

        lines = [json.loads(l) for l in output.read_text().strip().splitlines()]
        sub_types = {r["sub_type"] for r in lines}
        # Only json and csv errors should appear; plain should be filtered out
        assert "plain" not in sub_types
        assert sub_types <= {"json", "csv"}

    def test_no_filter_subtypes_audits_all(self, tmp_path):
        from audit_model_errors import async_main

        preds = self._make_predictions_file(tmp_path)
        output = tmp_path / "votes.jsonl"

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="prose"))]
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("audit_model_errors.openai.AsyncOpenAI", return_value=mock_client), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            _run(async_main([
                "--predictions", str(preds),
                "--output", str(output),
                "--backend", "openrouter",
            ]))

        lines = [json.loads(l) for l in output.read_text().strip().splitlines()]
        # All 3 errors should be present (4th was not an error)
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# --dual-llm mode
# ---------------------------------------------------------------------------


class TestDualLlmMode:
    """Test that --dual-llm calls both models and applies 4-way vote."""

    def _make_predictions_file(self, tmp_path):
        preds = [
            {"text": "hello world", "expected_category": "prose",
             "predicted_category": "code", "sub_type": "plain"},
        ]
        p = tmp_path / "preds.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in preds))
        return p

    def test_dual_llm_produces_4way_vote(self, tmp_path):
        from audit_model_errors import async_main

        preds = self._make_predictions_file(tmp_path)
        output = tmp_path / "votes.jsonl"
        ties_output = tmp_path / "ties.jsonl"

        # Haiku says "code", GPT says "code" -> with model="code" that's 3 vs label "prose" -> FIX_LABEL
        mock_haiku = MagicMock()
        mock_haiku.choices = [MagicMock(message=MagicMock(content="code"))]
        mock_gpt = MagicMock()
        mock_gpt.choices = [MagicMock(message=MagicMock(content="code"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[mock_haiku, mock_gpt])

        with patch("audit_model_errors.openai.AsyncOpenAI", return_value=mock_client), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            _run(async_main([
                "--predictions", str(preds),
                "--output", str(output),
                "--ties-output", str(ties_output),
                "--dual-llm",
            ]))

        lines = [json.loads(l) for l in output.read_text().strip().splitlines()]
        assert len(lines) == 1
        rec = lines[0]
        assert rec["verdict"] == "CORRECT"  # FIX_LABEL remapped to CORRECT
        assert rec["haiku_category"] == "code"
        assert rec["gpt_category"] == "code"
        assert rec["winner"] == "code"
        assert rec["magika_category"] == "code"

    def test_dual_llm_ties_go_to_ties_output(self, tmp_path):
        from audit_model_errors import async_main

        preds = self._make_predictions_file(tmp_path)
        output = tmp_path / "votes.jsonl"
        ties_output = tmp_path / "ties.jsonl"

        # Haiku says "prose" (agrees with label), GPT says "code" (agrees with model) -> 2-2 TIE
        mock_haiku = MagicMock()
        mock_haiku.choices = [MagicMock(message=MagicMock(content="prose"))]
        mock_gpt = MagicMock()
        mock_gpt.choices = [MagicMock(message=MagicMock(content="code"))]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[mock_haiku, mock_gpt])

        with patch("audit_model_errors.openai.AsyncOpenAI", return_value=mock_client), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            _run(async_main([
                "--predictions", str(preds),
                "--output", str(output),
                "--ties-output", str(ties_output),
                "--dual-llm",
            ]))

        # Main output should be empty (only TIEs)
        assert output.read_text().strip() == ""
        # Ties output should have the TIE record
        tie_lines = [json.loads(l) for l in ties_output.read_text().strip().splitlines()]
        assert len(tie_lines) == 1
        assert tie_lines[0]["verdict"] == "TIE"


# ---------------------------------------------------------------------------
# backward compat: --backend/--model still works
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Existing single-LLM mode with --backend/--model must still work."""

    def test_single_llm_backend_still_works(self, tmp_path):
        from audit_model_errors import async_main

        preds_data = [
            {"text": "some text", "expected_category": "prose",
             "predicted_category": "code", "sub_type": "plain"},
        ]
        preds = tmp_path / "preds.jsonl"
        preds.write_text("\n".join(json.dumps(r) for r in preds_data))
        output = tmp_path / "votes.jsonl"

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="code"))]
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("audit_model_errors.openai.AsyncOpenAI", return_value=mock_client), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            _run(async_main([
                "--predictions", str(preds),
                "--output", str(output),
                "--backend", "openrouter",
                "--model", "openai/gpt-5.4-mini",
            ]))

        lines = [json.loads(l) for l in output.read_text().strip().splitlines()]
        assert len(lines) == 1
        # Single-LLM mode: has llm_category, not haiku_category/gpt_category
        assert "llm_category" in lines[0]
