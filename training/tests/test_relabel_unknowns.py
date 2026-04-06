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


class TestNormalizeLabel:
    """Tests for _normalize_label() — canonical label mapping."""

    def test_prose_plain_maps_to_prose(self):
        from trainr.core.relabel_unknowns import _normalize_label

        assert _normalize_label("prose_plain") == "prose"

    def test_other_maps_to_unknown(self):
        from trainr.core.relabel_unknowns import _normalize_label

        assert _normalize_label("other") == "unknown"

    def test_c_cpp_unchanged(self):
        from trainr.core.relabel_unknowns import _normalize_label

        assert _normalize_label("c_cpp") == "c_cpp"

    def test_objc_unchanged(self):
        from trainr.core.relabel_unknowns import _normalize_label

        assert _normalize_label("objc") == "objc"

    def test_prose_unchanged(self):
        from trainr.core.relabel_unknowns import _normalize_label

        assert _normalize_label("prose") == "prose"

    def test_drop_unchanged(self):
        from trainr.core.relabel_unknowns import _normalize_label

        assert _normalize_label("drop") == "drop"

    def test_unknown_unchanged(self):
        from trainr.core.relabel_unknowns import _normalize_label

        assert _normalize_label("unknown") == "unknown"

    def test_manual_review_unchanged(self):
        from trainr.core.relabel_unknowns import _normalize_label

        assert _normalize_label("manual_review") == "manual_review"


class TestCrossVoterNormalization:
    """Tests for cross-voter label agreement after normalization."""

    def test_heuristic_prose_plain_llm_prose_agree(self):
        """Heuristic 'prose_plain' and LLM 'prose' should agree after normalization."""
        from trainr.core.relabel_unknowns import _normalize_label, vote

        h = _normalize_label("prose_plain")  # -> "prose"
        m = _normalize_label("prose")         # -> "prose"
        l = _normalize_label("prose")         # -> "prose"
        label, method = vote(h, m, l)
        assert label == "prose"
        assert method == "unanimous"

    def test_heuristic_prose_plain_llm_prose_majority(self):
        """Heuristic 'prose_plain' + LLM 'prose' = majority even if magika differs."""
        from trainr.core.relabel_unknowns import _normalize_label, vote

        h = _normalize_label("prose_plain")  # -> "prose"
        m = _normalize_label("unknown")       # -> "unknown"
        l = _normalize_label("prose")         # -> "prose"
        label, method = vote(h, m, l)
        assert label == "prose"
        assert method == "majority"

    def test_magika_prose_plain_normalized_before_vote(self):
        """Magika 'prose_plain' should normalize to 'prose' before voting."""
        from trainr.core.relabel_unknowns import _normalize_label, vote

        h = _normalize_label("prose_plain")   # -> "prose"
        m = _normalize_label("prose_plain")    # -> "prose"
        l = _normalize_label("other")          # -> "unknown"
        label, method = vote(h, m, l)
        assert label == "prose"
        assert method == "majority"

    def test_llm_other_normalized_to_unknown(self):
        """LLM 'other' should normalize to 'unknown'."""
        from trainr.core.relabel_unknowns import _normalize_label, vote

        h = _normalize_label("c_cpp")
        m = _normalize_label("c_cpp")
        l = _normalize_label("other")  # -> "unknown"
        label, method = vote(h, m, l)
        assert label == "c_cpp"
        assert method == "majority"


class TestApplyVotedLabels:
    """Tests for apply_voted_labels() — mapping voted labels to (category, sub_type)."""

    def _make_df(self, rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows)

    def test_c_cpp_label(self):
        from trainr.core.relabel_unknowns import apply_voted_labels

        df = self._make_df([
            {"category": "code", "sub_type": "unknown", "text": "code"},
        ])
        indices = [0]
        labels = ["c_cpp"]
        result, _ = apply_voted_labels(df, indices, labels)
        assert result["category"][0] == "code"
        assert result["sub_type"][0] == "c_cpp"

    def test_objc_label(self):
        from trainr.core.relabel_unknowns import apply_voted_labels

        df = self._make_df([
            {"category": "code", "sub_type": "unknown", "text": "code"},
        ])
        indices = [0]
        labels = ["objc"]
        result, _ = apply_voted_labels(df, indices, labels)
        assert result["category"][0] == "code"
        assert result["sub_type"][0] == "objc"

    def test_prose_label_maps_to_plain_subtype(self):
        from trainr.core.relabel_unknowns import apply_voted_labels

        df = self._make_df([
            {"category": "unknown", "sub_type": "unknown", "text": "text"},
        ])
        indices = [0]
        labels = ["prose"]
        result, _ = apply_voted_labels(df, indices, labels)
        assert result["category"][0] == "prose"
        assert result["sub_type"][0] == "plain"

    def test_drop_label_removes_row(self):
        from trainr.core.relabel_unknowns import apply_voted_labels

        df = self._make_df([
            {"category": "code", "sub_type": "unknown", "text": "code1"},
            {"category": "code", "sub_type": "unknown", "text": "code2"},
            {"category": "code", "sub_type": "python", "text": "code3"},
        ])
        indices = [0, 1]
        labels = ["c_cpp", "drop"]
        result, _ = apply_voted_labels(df, indices, labels)
        assert result.height == 2
        assert result["sub_type"].to_list() == ["c_cpp", "python"]

    def test_manual_review_leaves_unknown(self):
        from trainr.core.relabel_unknowns import apply_voted_labels

        df = self._make_df([
            {"category": "code", "sub_type": "unknown", "text": "code"},
        ])
        indices = [0]
        labels = ["manual_review"]
        result, _ = apply_voted_labels(df, indices, labels)
        assert result["sub_type"][0] == "unknown"

    def test_manual_review_count_returned(self):
        from trainr.core.relabel_unknowns import apply_voted_labels

        df = self._make_df([
            {"category": "code", "sub_type": "unknown", "text": "code1"},
            {"category": "code", "sub_type": "unknown", "text": "code2"},
        ])
        indices = [0, 1]
        labels = ["manual_review", "manual_review"]
        _, manual_count = apply_voted_labels(df, indices, labels)
        assert manual_count == 2

    def test_mixed_labels(self):
        from trainr.core.relabel_unknowns import apply_voted_labels

        df = self._make_df([
            {"category": "code", "sub_type": "unknown", "text": "a"},
            {"category": "code", "sub_type": "unknown", "text": "b"},
            {"category": "code", "sub_type": "unknown", "text": "c"},
            {"category": "code", "sub_type": "unknown", "text": "d"},
            {"category": "code", "sub_type": "python", "text": "e"},
        ])
        indices = [0, 1, 2, 3]
        labels = ["c_cpp", "prose", "drop", "manual_review"]
        result, manual_count = apply_voted_labels(df, indices, labels)
        # "drop" removes 1 row, so 4 remain
        assert result.height == 4
        assert manual_count == 1


class TestHeuristicNarrowedBraces:
    """Tests that braces+semicolons heuristic requires C-specific indicators."""

    def test_javascript_not_classified_as_c(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        text = "function foo() {\n  let x = 1;\n  console.log(x);\n}"
        # No C-specific indicators, should not be c_cpp
        assert classify_heuristic(text) != "c_cpp"

    def test_java_not_classified_as_c(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        # Java code without C-specific keywords (no void/int/char function sigs)
        text = "public class Foo {\n  String name = \"hello\";\n  System.out.println(name);\n}"
        assert classify_heuristic(text) != "c_cpp"

    def test_c_with_pointer_still_classified(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        # Has braces, semicolons, AND pointer syntax
        text = "char *buf; int *ptr; foo() { bar(); }"
        assert classify_heuristic(text) == "c_cpp"

    def test_c_with_typedef_still_classified(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        text = "typedef int myint; struct { myint x; myint y; }"
        assert classify_heuristic(text) == "c_cpp"

    def test_c_with_null_still_classified(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        text = "if (ptr == NULL) { free(ptr); ptr = NULL; }"
        assert classify_heuristic(text) == "c_cpp"

    def test_c_with_nullptr_still_classified(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        text = "if (ptr == nullptr) { delete ptr; ptr = nullptr; }"
        assert classify_heuristic(text) == "c_cpp"

    def test_c_with_enum_brace_still_classified(self):
        from trainr.core.relabel_unknowns import classify_heuristic

        text = "enum Color { RED, GREEN, BLUE }; enum Size { S, M, L };"
        assert classify_heuristic(text) == "c_cpp"


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
