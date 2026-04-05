"""Tests for trainr.core.vote_labels — tiered voting pipeline for label cleanup."""

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# VALID_SUB_TYPES tests
# ---------------------------------------------------------------------------


class TestValidSubTypes:
    def test_valid_sub_types_contains_all_detection_labels(self):
        from trainr.core.annotate_detections import DETECTION_LABELS
        from trainr.core.vote_labels import VALID_SUB_TYPES

        for label in DETECTION_LABELS:
            assert label in VALID_SUB_TYPES, f"Missing {label} from VALID_SUB_TYPES"

    def test_valid_sub_types_includes_unknown(self):
        from trainr.core.vote_labels import VALID_SUB_TYPES

        assert "unknown" in VALID_SUB_TYPES


# ---------------------------------------------------------------------------
# TIER2_ROUTING tests
# ---------------------------------------------------------------------------


class TestTier2Routing:
    def test_tier2_routing_defined(self):
        from trainr.core.vote_labels import TIER2_ROUTING

        assert isinstance(TIER2_ROUTING, dict)
        assert len(TIER2_ROUTING) > 0

    def test_tier2_routing_entries_are_valid(self):
        from trainr.core.vote_labels import TIER2_ROUTING

        valid_backends = {"openrouter", "anthropic"}
        for sub_type, (model_id, backend) in TIER2_ROUTING.items():
            assert isinstance(model_id, str), f"model_id for {sub_type} not a string"
            assert backend in valid_backends, f"bad backend for {sub_type}: {backend}"

    def test_tier2_routing_has_expected_sub_types(self):
        from trainr.core.vote_labels import TIER2_ROUTING

        expected = {
            "go", "html", "jsonl", "markdown", "shell",
            "javascript", "key_value", "pipe_table", "rst", "sgml", "toml", "tsv",
            "json", "makefile", "python", "xml",
        }
        assert set(TIER2_ROUTING.keys()) == expected

    def test_tier2_routing_differs_from_tier1_for_cheap_types(self):
        """Tier 2 should only exist for types where Tier 1 is a cheap model."""
        from trainr.core.vote_labels import TIER2_ROUTING
        from trainr.core.voting_pilot import TIER1_ROUTING

        for sub_type in TIER2_ROUTING:
            assert sub_type in TIER1_ROUTING, f"{sub_type} in T2 but not T1"


# ---------------------------------------------------------------------------
# decide_action tests
# ---------------------------------------------------------------------------


class TestDecideAction:
    def test_tier1_agrees_returns_keep(self):
        from trainr.core.vote_labels import decide_action

        action, new_st = decide_action("python", True, None, None)
        assert action == "keep"
        assert new_st is None

    def test_tier1_disagrees_tier2_agrees_returns_keep(self):
        from trainr.core.vote_labels import decide_action

        action, new_st = decide_action("python", False, True, None)
        assert action == "keep"
        assert new_st is None

    def test_tier1_disagrees_tier2_disagrees_with_relabel(self):
        from trainr.core.vote_labels import decide_action

        action, new_st = decide_action("python", False, False, "javascript")
        assert action == "relabel"
        assert new_st == "javascript"

    def test_tier1_disagrees_tier2_disagrees_invalid_relabel(self):
        from trainr.core.vote_labels import decide_action

        action, new_st = decide_action("python", False, False, "not_a_real_type")
        assert action == "drop"
        assert new_st is None

    def test_tier1_disagrees_tier2_disagrees_no_relabel(self):
        from trainr.core.vote_labels import decide_action

        action, new_st = decide_action("python", False, False, None)
        assert action == "drop"
        assert new_st is None

    def test_tier1_disagrees_no_tier2_available(self):
        from trainr.core.vote_labels import decide_action

        action, new_st = decide_action("csv", False, None, None)
        assert action == "drop"
        assert new_st is None

    def test_tier1_disagrees_tier2_disagrees_empty_string_relabel(self):
        from trainr.core.vote_labels import decide_action

        action, new_st = decide_action("python", False, False, "")
        assert action == "drop"
        assert new_st is None


# ---------------------------------------------------------------------------
# parse_relabel_response tests
# ---------------------------------------------------------------------------


class TestParseRelabelResponse:
    def test_valid_sub_type(self):
        from trainr.core.vote_labels import parse_relabel_response

        assert parse_relabel_response("python") == "python"

    def test_valid_with_whitespace(self):
        from trainr.core.vote_labels import parse_relabel_response

        assert parse_relabel_response("  markdown\n") == "markdown"

    def test_invalid_sub_type(self):
        from trainr.core.vote_labels import parse_relabel_response

        assert parse_relabel_response("not_a_type") is None

    def test_empty_string(self):
        from trainr.core.vote_labels import parse_relabel_response

        assert parse_relabel_response("") is None

    def test_multiword_response_extracts_first_valid(self):
        from trainr.core.vote_labels import parse_relabel_response

        # Model might say "The type is python" — we want to extract "python"
        assert parse_relabel_response("The type is python") == "python"

    def test_quoted_response(self):
        from trainr.core.vote_labels import parse_relabel_response

        assert parse_relabel_response('"json"') == "json"

    def test_pipe_table_with_underscore(self):
        from trainr.core.vote_labels import parse_relabel_response

        assert parse_relabel_response("pipe_table") == "pipe_table"


# ---------------------------------------------------------------------------
# build_voting_log tests
# ---------------------------------------------------------------------------


class TestBuildVotingLog:
    def test_voting_log_has_expected_columns(self):
        from trainr.core.vote_labels import build_voting_log

        records = [
            {
                "text": "x" * 300,
                "original_sub_type": "python",
                "tier1_model": "openai/gpt-5.4-nano",
                "tier1_agrees": True,
                "tier2_model": None,
                "tier2_agrees": None,
                "new_sub_type": None,
                "action": "keep",
            },
        ]
        log = build_voting_log(records)
        expected_cols = {
            "text", "original_sub_type", "tier1_model", "tier1_agrees",
            "tier2_model", "tier2_agrees", "new_sub_type", "action",
        }
        assert set(log.columns) == expected_cols

    def test_text_truncated_to_200_chars(self):
        from trainr.core.vote_labels import build_voting_log

        records = [
            {
                "text": "a" * 500,
                "original_sub_type": "python",
                "tier1_model": "m",
                "tier1_agrees": True,
                "tier2_model": None,
                "tier2_agrees": None,
                "new_sub_type": None,
                "action": "keep",
            },
        ]
        log = build_voting_log(records)
        assert len(log["text"][0]) == 200

    def test_multiple_records(self):
        from trainr.core.vote_labels import build_voting_log

        records = [
            {
                "text": "short",
                "original_sub_type": "python",
                "tier1_model": "m1",
                "tier1_agrees": True,
                "tier2_model": None,
                "tier2_agrees": None,
                "new_sub_type": None,
                "action": "keep",
            },
            {
                "text": "other",
                "original_sub_type": "rust",
                "tier1_model": "m2",
                "tier1_agrees": False,
                "tier2_model": "m3",
                "tier2_agrees": False,
                "new_sub_type": None,
                "action": "drop",
            },
        ]
        log = build_voting_log(records)
        assert log.height == 2

    def test_empty_records(self):
        from trainr.core.vote_labels import build_voting_log

        log = build_voting_log([])
        assert log.height == 0


# ---------------------------------------------------------------------------
# has_separate_tier2 tests
# ---------------------------------------------------------------------------


class TestHasSeparateTier2:
    def test_cheap_type_has_tier2(self):
        from trainr.core.vote_labels import has_separate_tier2

        # "python" has a cheap Tier 1 model and a different Tier 2 model
        assert has_separate_tier2("python") is True

    def test_premium_type_no_tier2(self):
        from trainr.core.vote_labels import has_separate_tier2

        # "plain" uses a cheap Tier 1 but is NOT in TIER2_ROUTING
        assert has_separate_tier2("plain") is False

    def test_type_not_in_either_routing(self):
        from trainr.core.vote_labels import has_separate_tier2

        assert has_separate_tier2("nonexistent_type") is False


# ---------------------------------------------------------------------------
# CLI parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_parser_defaults(self):
        from trainr.core.vote_labels import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.input == "golden_train.parquet"
        assert args.output == "golden_train.parquet"
        assert args.voting_log == "voting_log.parquet"
        assert args.tier1_only is False
        assert args.concurrency == 20
        assert args.dry_run is False

    def test_parser_custom_args(self):
        from trainr.core.vote_labels import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--input", "in.parquet",
            "--output", "out.parquet",
            "--voting-log", "log.parquet",
            "--tier1-only",
            "--concurrency", "5",
            "--dry-run",
        ])
        assert args.input == "in.parquet"
        assert args.output == "out.parquet"
        assert args.voting_log == "log.parquet"
        assert args.tier1_only is True
        assert args.concurrency == 5
        assert args.dry_run is True


# ---------------------------------------------------------------------------
# RELABEL_PROMPT tests
# ---------------------------------------------------------------------------


class TestRelabelPrompt:
    def test_relabel_prompt_contains_all_valid_sub_types(self):
        from trainr.core.vote_labels import RELABEL_PROMPT, VALID_SUB_TYPES

        for st in VALID_SUB_TYPES:
            assert st in RELABEL_PROMPT, f"Missing {st} in RELABEL_PROMPT"
