"""Tests for trainr.core.relabel_unknowns — 3-way voting for unknown sub_types."""

from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_text_len(self):
        from trainr.core.relabel_unknowns import MAX_TEXT_LEN

        assert MAX_TEXT_LEN == 2000

    def test_magika_to_subtype_mapping(self):
        from trainr.core.relabel_unknowns import MAGIKA_TO_SUBTYPE

        assert MAGIKA_TO_SUBTYPE["c"] == "c_cpp"
        assert MAGIKA_TO_SUBTYPE["cpp"] == "c_cpp"
        assert MAGIKA_TO_SUBTYPE["objectivec"] == "objc"
        assert MAGIKA_TO_SUBTYPE["txt"] == "prose_plain"
        assert MAGIKA_TO_SUBTYPE["text"] == "prose_plain"

    def test_classify_prompt_exists(self):
        from trainr.core.relabel_unknowns import CLASSIFY_PROMPT

        assert "c_cpp" in CLASSIFY_PROMPT
        assert "objc" in CLASSIFY_PROMPT
        assert "prose" in CLASSIFY_PROMPT
        assert "other" in CLASSIFY_PROMPT


# ---------------------------------------------------------------------------
# classify_heuristic tests
# ---------------------------------------------------------------------------


class TestClassifyHeuristic:
    def test_objc_interface(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic("@interface Foo : NSObject\n@end") == "objc"

    def test_objc_implementation(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic("@implementation Foo\n- (void)bar {}\n@end") == "objc"

    def test_objc_property(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic("@property (nonatomic) NSString *name;") == "objc"

    def test_objc_protocol(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic("@protocol MyDelegate\n@end") == "objc"

    def test_objc_synthesize(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic("@synthesize name = _name;") == "objc"

    def test_objc_class(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic("@class Foo;\n@class Bar;") == "objc"

    def test_c_include(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic('#include <stdio.h>\nint main() { return 0; }') == "c_cpp"

    def test_c_define(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic('#define MAX_SIZE 100\n#define MIN_SIZE 10') == "c_cpp"

    def test_c_ifndef(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic('#ifndef HEADER_H\n#define HEADER_H\n#endif') == "c_cpp"

    def test_c_function_sig(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic('int foo(int a, char *b) {\n  return 0;\n}') == "c_cpp"

    def test_c_struct(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic('struct Point {\n  int x;\n  int y;\n};') == "c_cpp"

    def test_license_text(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        text = (
            "Permission is hereby granted, free of charge, to any person obtaining a copy "
            "of this software and associated documentation files, to deal in the Software "
            "without restriction, including without limitation the rights to use, copy, "
            "modify, merge, publish, distribute, sublicense, and/or sell copies."
        )
        assert classify_heuristic(text) == "prose_plain"

    def test_code_like_braces_semicolons(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        text = "void foo() {\n  int x = 1;\n  int y = 2;\n  bar(x, y);\n}"
        assert classify_heuristic(text) == "c_cpp"

    def test_unknown_text(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic("hello world this is some random text") == "drop"

    def test_empty_text(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        assert classify_heuristic("") == "drop"


# ---------------------------------------------------------------------------
# classify_magika tests
# ---------------------------------------------------------------------------


class TestClassifyMagika:
    def test_mapped_label(self):
        from trainr.core.relabel_unknowns import classify_magika

        mock_magika = MagicMock()
        mock_result = MagicMock()
        mock_result.output.ct_label = "c"
        mock_magika.identify_bytes.return_value = mock_result

        assert classify_magika("some text", mock_magika) == "c_cpp"

    def test_unmapped_label(self):
        from trainr.core.relabel_unknowns import classify_magika

        mock_magika = MagicMock()
        mock_result = MagicMock()
        mock_result.output.ct_label = "javascript"
        mock_magika.identify_bytes.return_value = mock_result

        assert classify_magika("some text", mock_magika) == "unknown"

    def test_objectivec_label(self):
        from trainr.core.relabel_unknowns import classify_magika

        mock_magika = MagicMock()
        mock_result = MagicMock()
        mock_result.output.ct_label = "objectivec"
        mock_magika.identify_bytes.return_value = mock_result

        assert classify_magika("some text", mock_magika) == "objc"

    def test_txt_label(self):
        from trainr.core.relabel_unknowns import classify_magika

        mock_magika = MagicMock()
        mock_result = MagicMock()
        mock_result.output.ct_label = "txt"
        mock_magika.identify_bytes.return_value = mock_result

        assert classify_magika("some text", mock_magika) == "prose_plain"


# ---------------------------------------------------------------------------
# vote tests
# ---------------------------------------------------------------------------


class TestVote:
    def test_unanimous(self):
        from trainr.core.relabel_unknowns import vote

        label, method = vote("c_cpp", "c_cpp", "c_cpp")
        assert label == "c_cpp"
        assert method == "unanimous"

    def test_majority_heuristic_magika(self):
        from trainr.core.relabel_unknowns import vote

        label, method = vote("c_cpp", "c_cpp", "objc")
        assert label == "c_cpp"
        assert method == "majority"

    def test_majority_heuristic_llm(self):
        from trainr.core.relabel_unknowns import vote

        label, method = vote("objc", "c_cpp", "objc")
        assert label == "objc"
        assert method == "majority"

    def test_majority_magika_llm(self):
        from trainr.core.relabel_unknowns import vote

        label, method = vote("c_cpp", "objc", "objc")
        assert label == "objc"
        assert method == "majority"

    def test_tie_all_different(self):
        from trainr.core.relabel_unknowns import vote

        label, method = vote("c_cpp", "objc", "prose")
        assert label == "manual_review"
        assert method == "tie"

    def test_tie_with_drop(self):
        from trainr.core.relabel_unknowns import vote

        label, method = vote("drop", "unknown", "other")
        assert label == "manual_review"
        assert method == "tie"


# ---------------------------------------------------------------------------
# relabel_bulk tests
# ---------------------------------------------------------------------------


class TestRelabelBulk:
    def _make_df(self, rows: list[dict]) -> pl.DataFrame:
        """Build a DataFrame with the required columns."""
        return pl.DataFrame(rows)

    def test_stack_licenses_relabeled(self):
        from trainr.core.relabel_unknowns import relabel_bulk

        df = self._make_df([
            {"source": "real/the_stack_text_licenses", "category": "unknown", "sub_type": "unknown", "text": "MIT License"},
        ])
        result = relabel_bulk(df)
        assert result["category"][0] == "prose"
        assert result["sub_type"][0] == "plain"

    def test_generated_ocr_dropped(self):
        from trainr.core.relabel_unknowns import relabel_bulk

        df = self._make_df([
            {"source": "real/generated_ocr", "category": "unknown", "sub_type": "unknown", "text": "ocr text"},
            {"source": "real/the-stack-v2", "category": "code", "sub_type": "python", "text": "import os"},
        ])
        result = relabel_bulk(df)
        assert result.height == 1
        assert result["source"][0] == "real/the-stack-v2"

    def test_arxiv_relabeled(self):
        from trainr.core.relabel_unknowns import relabel_bulk

        df = self._make_df([
            {"source": "real/arxiv_summarization", "category": "unknown", "sub_type": "unknown", "text": "abstract"},
        ])
        result = relabel_bulk(df)
        assert result["category"][0] == "prose"
        assert result["sub_type"][0] == "plain"

    def test_finepdfs_relabeled(self):
        from trainr.core.relabel_unknowns import relabel_bulk

        df = self._make_df([
            {"source": "real/finepdfs", "category": "unknown", "sub_type": "unknown", "text": "pdf content"},
        ])
        result = relabel_bulk(df)
        assert result["category"][0] == "prose"
        assert result["sub_type"][0] == "plain"

    def test_generated_skip_dropped(self):
        from trainr.core.relabel_unknowns import relabel_bulk

        df = self._make_df([
            {"source": "real/generated_skip", "category": "unknown", "sub_type": "unknown", "text": "skip"},
        ])
        result = relabel_bulk(df)
        assert result.height == 0

    def test_unmatched_sources_unchanged(self):
        from trainr.core.relabel_unknowns import relabel_bulk

        df = self._make_df([
            {"source": "real/the-stack-v2", "category": "code", "sub_type": "unknown", "text": "code"},
        ])
        result = relabel_bulk(df)
        assert result.height == 1
        assert result["category"][0] == "code"
        assert result["sub_type"][0] == "unknown"


# ---------------------------------------------------------------------------
# _classify_one_llm tests
# ---------------------------------------------------------------------------


class TestClassifyOneLlm:
    def test_returns_valid_label(self):
        import asyncio

        from trainr.core.relabel_unknowns import _classify_one_llm

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "c_cpp"
        mock_client.chat.completions.create.return_value = mock_response

        sem = asyncio.Semaphore(1)
        result = asyncio.run(_classify_one_llm(mock_client, "some code", "model", sem))
        assert result == "c_cpp"

    def test_returns_other_on_error(self):
        import asyncio

        from trainr.core.relabel_unknowns import _classify_one_llm

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        sem = asyncio.Semaphore(1)
        result = asyncio.run(_classify_one_llm(mock_client, "some code", "model", sem))
        assert result == "other"

    def test_normalizes_response(self):
        import asyncio

        from trainr.core.relabel_unknowns import _classify_one_llm

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  OBJC  "
        mock_client.chat.completions.create.return_value = mock_response

        sem = asyncio.Semaphore(1)
        result = asyncio.run(_classify_one_llm(mock_client, "some code", "model", sem))
        assert result == "objc"


# ---------------------------------------------------------------------------
# classify_batch_llm tests
# ---------------------------------------------------------------------------


class TestClassifyBatchLlm:
    def test_batch_returns_list(self):
        import asyncio

        from trainr.core.relabel_unknowns import classify_batch_llm

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "c_cpp"
        mock_client.chat.completions.create.return_value = mock_response

        results = asyncio.run(classify_batch_llm(mock_client, ["text1", "text2"], "model", 5))
        assert len(results) == 2
        assert all(r == "c_cpp" for r in results)


# ---------------------------------------------------------------------------
# main / async_main tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_is_callable(self):
        from trainr.core.relabel_unknowns import main

        assert callable(main)

    def test_async_main_is_callable(self):
        from trainr.core.relabel_unknowns import async_main

        assert callable(async_main)

    def test_import_ok(self):
        from trainr.core.relabel_unknowns import main  # noqa: F401

        # Just verifying it imports without error
