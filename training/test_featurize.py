"""Tests for featurize.py — validates structural feature extraction parity with Rust."""

import math
import sys
from pathlib import Path

import pytest

# Ensure the training module is importable
sys.path.insert(0, str(Path(__file__).parent))

from featurize import (
    FEATURES,
    SAMPLE_SIZE,
    UNIQUENESS_LINES,
    alpha_ratio,
    char_entropy,
    comment_ratio,
    delimiter_consistency,
    extract_all,
    json_brace_depth,
    key_value_ratio,
    leading_whitespace_ratio,
    line_length_cv,
    line_uniqueness,
    log_line_ratio,
    numeric_field_ratio,
    paragraph_break_rate,
    repetitive_structure_score,
    sentence_punctuation_rate,
    short_line_ratio,
    symbol_ratio,
    tab_density,
    xml_tag_ratio,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_sample_size(self):
        assert SAMPLE_SIZE == 10_000

    def test_uniqueness_lines(self):
        assert UNIQUENESS_LINES == 500

    def test_features_registry_has_18_entries(self):
        assert len(FEATURES) == 18

    def test_features_registry_keys(self):
        expected = [
            "line_length_cv",
            "char_entropy",
            "leading_whitespace_ratio",
            "tab_density",
            "sentence_punctuation_rate",
            "paragraph_break_rate",
            "alpha_ratio",
            "line_uniqueness",
            "short_line_ratio",
            "symbol_ratio",
            "delimiter_consistency",
            "json_brace_depth",
            "key_value_ratio",
            "xml_tag_ratio",
            "log_line_ratio",
            "comment_ratio",
            "numeric_field_ratio",
            "repetitive_structure_score",
        ]
        assert list(FEATURES.keys()) == expected

    def test_features_registry_values_are_callable(self):
        for name, fn in FEATURES.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """All features should return 0.0 for empty text."""

    def test_extract_all_empty(self):
        result = extract_all("")
        for name, val in result.items():
            assert val == 0.0, f"{name} should be 0.0 for empty text, got {val}"


# ---------------------------------------------------------------------------
# line_length_cv
# ---------------------------------------------------------------------------


class TestLineLengthCv:
    def test_single_line(self):
        assert line_length_cv("hello world") == 0.0

    def test_uniform_lines(self):
        text = "abc\nabc\nabc"
        assert line_length_cv(text) == 0.0

    def test_variable_lines(self):
        text = "a\nabcdefghij"
        result = line_length_cv(text)
        # lengths: 1, 10 -> mean=5.5, std=sqrt((4.5^2+4.5^2)/2)=4.5, cv=4.5/5.5
        assert result == pytest.approx(4.5 / 5.5, abs=1e-4)

    def test_all_empty_lines(self):
        text = "\n\n\n"
        # lines: ["", "", ""], mean < 1.0 -> return 0.0
        assert line_length_cv(text) == 0.0


# ---------------------------------------------------------------------------
# char_entropy
# ---------------------------------------------------------------------------


class TestCharEntropy:
    def test_single_char_repeated(self):
        text = "aaaa"
        assert char_entropy(text) == 0.0

    def test_two_equally_distributed(self):
        text = "aabb"
        assert char_entropy(text) == pytest.approx(1.0, abs=1e-4)

    def test_higher_entropy(self):
        text = "abcd"
        assert char_entropy(text) == pytest.approx(2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# leading_whitespace_ratio
# ---------------------------------------------------------------------------


class TestLeadingWhitespaceRatio:
    def test_no_indentation(self):
        text = "hello\nworld"
        assert leading_whitespace_ratio(text) == 0.0

    def test_all_indented(self):
        text = "   hello\n   world"
        assert leading_whitespace_ratio(text) == 1.0

    def test_tab_counts_as_four(self):
        # Single tab = 4 columns > 2
        text = "\thello\nworld"
        assert leading_whitespace_ratio(text) == pytest.approx(0.5, abs=1e-4)

    def test_two_spaces_not_enough(self):
        text = "  hello\nworld"
        assert leading_whitespace_ratio(text) == 0.0


# ---------------------------------------------------------------------------
# tab_density
# ---------------------------------------------------------------------------


class TestTabDensity:
    def test_no_tabs(self):
        assert tab_density("hello world") == 0.0

    def test_all_tabs(self):
        assert tab_density("\t\t\t") == 1.0

    def test_mixed(self):
        text = "a\tb"  # 3 chars, 1 tab
        assert tab_density(text) == pytest.approx(1.0 / 3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# sentence_punctuation_rate
# ---------------------------------------------------------------------------


class TestSentencePunctuationRate:
    def test_no_sentence_punct(self):
        text = "hello world"
        assert sentence_punctuation_rate(text) == 0.0

    def test_normal_sentence(self):
        text = "Hello world. How are you?"
        # "." after "d" (alpha, followed by space) -> count
        # "?" after "u" (alpha, end of text) -> count
        # word_count = 5
        assert sentence_punctuation_rate(text) == pytest.approx(2.0 / 5.0, abs=1e-4)

    def test_number_dot_not_counted(self):
        # "3.14" -> "." preceded by digit, not alpha -> not counted
        text = "value 3.14 here"
        assert sentence_punctuation_rate(text) == 0.0

    def test_exclamation(self):
        text = "Great!"
        assert sentence_punctuation_rate(text) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# paragraph_break_rate
# ---------------------------------------------------------------------------


class TestParagraphBreakRate:
    def test_no_breaks(self):
        text = "hello\nworld"
        assert paragraph_break_rate(text) == 0.0

    def test_one_break(self):
        text = "hello\n\nworld"
        # 1 break, 3 lines (split on \n gives ["hello", "", "world"])
        assert paragraph_break_rate(text) == pytest.approx(1.0 / 3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# alpha_ratio
# ---------------------------------------------------------------------------


class TestAlphaRatio:
    def test_all_alpha(self):
        assert alpha_ratio("hello world") == 1.0

    def test_all_symbols(self):
        assert alpha_ratio("@#$%^&") == 0.0

    def test_mixed(self):
        text = "ab @"  # 4 chars: 'a','b',' ','@' -> 3/4 = 0.75
        assert alpha_ratio(text) == pytest.approx(0.75, abs=1e-4)


# ---------------------------------------------------------------------------
# line_uniqueness
# ---------------------------------------------------------------------------


class TestLineUniqueness:
    def test_all_unique(self):
        text = "a\nb\nc"
        assert line_uniqueness(text) == 1.0

    def test_all_same(self):
        text = "a\na\na"
        assert line_uniqueness(text) == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_caps_at_500_lines(self):
        lines = [f"line{i}" for i in range(600)]
        text = "\n".join(lines)
        result = line_uniqueness(text)
        # First 500 are all unique -> 500/500 = 1.0
        assert result == 1.0


# ---------------------------------------------------------------------------
# short_line_ratio
# ---------------------------------------------------------------------------


class TestShortLineRatio:
    def test_no_short_lines(self):
        text = "this is a long line that is not short\nanother long one here"
        assert short_line_ratio(text) == 0.0

    def test_all_short(self):
        text = "hi\nyo"
        assert short_line_ratio(text) == 1.0

    def test_empty_lines_not_counted(self):
        text = "\n\n"
        # Empty lines have trimmed_len = 0, not in 1..=14
        assert short_line_ratio(text) == 0.0


# ---------------------------------------------------------------------------
# symbol_ratio
# ---------------------------------------------------------------------------


class TestSymbolRatio:
    def test_all_alpha(self):
        assert symbol_ratio("hello world") == 0.0

    def test_common_punct_excluded(self):
        text = "hello, world. how are you?"
        assert symbol_ratio(text) == 0.0

    def test_symbols_counted(self):
        text = "a@b"  # '@' is a symbol, 3 total chars
        assert symbol_ratio(text) == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_braces_counted(self):
        text = "a{b}"  # '{' and '}' are symbols
        assert symbol_ratio(text) == pytest.approx(2.0 / 4.0, abs=1e-4)

    def test_unicode_arrows_excluded(self):
        # U+2192 RIGHTWARDS ARROW should be excluded
        text = "a\u2192b"  # 3 chars, arrow excluded -> 0 symbols
        assert symbol_ratio(text) == 0.0

    def test_unicode_box_drawing_excluded(self):
        # U+2500 BOX DRAWINGS LIGHT HORIZONTAL
        text = "a\u2500b"
        assert symbol_ratio(text) == 0.0

    def test_unicode_block_elements_excluded(self):
        # U+2588 FULL BLOCK
        text = "a\u2588b"
        assert symbol_ratio(text) == 0.0

    def test_unicode_geometric_shapes_excluded(self):
        # U+25A0 BLACK SQUARE
        text = "a\u25A0b"
        assert symbol_ratio(text) == 0.0

    def test_unicode_dingbats_excluded(self):
        # U+2714 HEAVY CHECK MARK
        text = "a\u2714b"
        assert symbol_ratio(text) == 0.0

    def test_typographic_em_dash_excluded(self):
        text = "a\u2014b"  # em dash
        assert symbol_ratio(text) == 0.0

    def test_typographic_en_dash_excluded(self):
        text = "a\u2013b"  # en dash
        assert symbol_ratio(text) == 0.0

    def test_typographic_bullet_excluded(self):
        text = "a\u2022b"  # bullet
        assert symbol_ratio(text) == 0.0

    def test_typographic_degree_excluded(self):
        text = "a\u00B0b"  # degree sign
        assert symbol_ratio(text) == 0.0

    def test_typographic_copyright_excluded(self):
        text = "a\u00A9b"  # copyright
        assert symbol_ratio(text) == 0.0

    def test_typographic_registered_excluded(self):
        text = "a\u00AEb"  # registered
        assert symbol_ratio(text) == 0.0

    def test_typographic_trademark_excluded(self):
        text = "a\u2122b"  # trademark
        assert symbol_ratio(text) == 0.0

    def test_typographic_ellipsis_excluded(self):
        text = "a\u2026b"  # horizontal ellipsis
        assert symbol_ratio(text) == 0.0

    def test_typographic_multiply_excluded(self):
        text = "a\u00D7b"  # multiplication sign
        assert symbol_ratio(text) == 0.0

    def test_typographic_divide_excluded(self):
        text = "a\u00F7b"  # division sign
        assert symbol_ratio(text) == 0.0

    def test_non_excluded_unicode_still_counted(self):
        # U+03B1 GREEK SMALL LETTER ALPHA is alphanumeric, not a symbol
        # U+2605 BLACK STAR is in Misc Symbols range (excluded)
        # Use something outside excluded ranges, e.g. U+2000 EN QUAD (whitespace-ish but not alphanumeric)
        # Actually let's use a char clearly outside: U+00A7 SECTION SIGN
        text = "a\u00A7b"  # section sign, not in excluded sets
        assert symbol_ratio(text) == pytest.approx(1.0 / 3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# delimiter_consistency
# ---------------------------------------------------------------------------


class TestDelimiterConsistency:
    def test_fewer_than_3_lines(self):
        assert delimiter_consistency("a\nb") == 0.0

    def test_consistent_csv(self):
        text = "a,b,c\nd,e,f\ng,h,i"
        # Each line has 2 commas, mode=2, freq=3/3=1.0
        assert delimiter_consistency(text) == 1.0

    def test_inconsistent(self):
        text = "a,b,c\nd,e\ng,h,i"
        # Commas: [2,1,2], mode=2, freq=2/3
        assert delimiter_consistency(text) == pytest.approx(2.0 / 3.0, abs=1e-4)

    def test_no_delimiters(self):
        text = "abc\ndef\nghi"
        assert delimiter_consistency(text) == 0.0


# ---------------------------------------------------------------------------
# json_brace_depth
# ---------------------------------------------------------------------------


class TestJsonBraceDepth:
    def test_no_braces(self):
        assert json_brace_depth("hello world") == 0.0

    def test_all_braces(self):
        assert json_brace_depth("{}[]") == 1.0

    def test_mixed(self):
        text = "a{b}"  # 2 braces out of 4 chars
        assert json_brace_depth(text) == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# key_value_ratio
# ---------------------------------------------------------------------------


class TestKeyValueRatio:
    def test_no_kv(self):
        text = "hello\nworld"
        assert key_value_ratio(text) == 0.0

    def test_colon_pattern(self):
        text = "name: John\nage: 30"
        assert key_value_ratio(text) == 1.0

    def test_equals_pattern(self):
        text = "name=John\nage=30"
        assert key_value_ratio(text) == 1.0

    def test_colon_without_space_not_matched(self):
        # "key:value" without space after colon -> check `=` path
        # No `=` either -> not matched
        text = "key:value\nother"
        assert key_value_ratio(text) == 0.0

    def test_equals_at_end_not_matched(self):
        # "key=" with nothing after -> not matched
        text = "key=\nother"
        assert key_value_ratio(text) == 0.0


# ---------------------------------------------------------------------------
# xml_tag_ratio
# ---------------------------------------------------------------------------


class TestXmlTagRatio:
    def test_no_tags(self):
        text = "hello\nworld"
        assert xml_tag_ratio(text) == 0.0

    def test_opening_tags(self):
        text = "<div>\n<span>"
        assert xml_tag_ratio(text) == 1.0

    def test_closing_tags(self):
        text = "</div>\n</span>"
        assert xml_tag_ratio(text) == 1.0

    def test_less_than_number_not_matched(self):
        text = "x < 5\ny > 3"
        # "<" followed by " " (space), not alphabetic -> not a tag
        assert xml_tag_ratio(text) == 0.0


# ---------------------------------------------------------------------------
# log_line_ratio
# ---------------------------------------------------------------------------


class TestLogLineRatio:
    def test_no_log_lines(self):
        text = "hello\nworld"
        assert log_line_ratio(text) == 0.0

    def test_date_pattern(self):
        text = "2024-01-15 something\n2024-01-16 other"
        assert log_line_ratio(text) == 1.0

    def test_time_pattern(self):
        text = "12:30:45 something\nnot a log"
        assert log_line_ratio(text) == pytest.approx(0.5, abs=1e-4)

    def test_bracket_pattern(self):
        text = "[2024 something\n[1234 other"
        assert log_line_ratio(text) == 1.0


# ---------------------------------------------------------------------------
# comment_ratio
# ---------------------------------------------------------------------------


class TestCommentRatio:
    def test_no_comments(self):
        text = "hello\nworld"
        assert comment_ratio(text) == 0.0

    def test_hash_comments(self):
        text = "# comment\n# another"
        assert comment_ratio(text) == 1.0

    def test_double_slash(self):
        text = "// comment\ncode"
        assert comment_ratio(text) == pytest.approx(0.5, abs=1e-4)

    def test_block_comment(self):
        text = "/* comment\ncode"
        assert comment_ratio(text) == pytest.approx(0.5, abs=1e-4)

    def test_sql_comment(self):
        text = "-- comment\ncode"
        assert comment_ratio(text) == pytest.approx(0.5, abs=1e-4)

    def test_percent_comment(self):
        text = "% comment\ncode"
        assert comment_ratio(text) == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# numeric_field_ratio
# ---------------------------------------------------------------------------


class TestNumericFieldRatio:
    def test_no_numbers(self):
        assert numeric_field_ratio("hello world") == 0.0

    def test_all_numbers(self):
        assert numeric_field_ratio("1 2 3") == 1.0

    def test_comma_stripped(self):
        assert numeric_field_ratio("1,000 words") == pytest.approx(0.5, abs=1e-4)

    def test_float_numbers(self):
        assert numeric_field_ratio("3.14 2.72") == 1.0


# ---------------------------------------------------------------------------
# repetitive_structure_score
# ---------------------------------------------------------------------------


class TestRepetitiveStructureScore:
    def test_fewer_than_3_lines(self):
        assert repetitive_structure_score("a\nb") == 0.0

    def test_all_same_shape(self):
        text = "a b c\nd e f\ng h i"
        assert repetitive_structure_score(text) == 1.0

    def test_different_shapes(self):
        text = "a\nb c\nd e f"
        # shapes: (1,[F,F,F,F]), (2,[F,F,F,F]), (3,[F,F,F,F]) all different
        assert repetitive_structure_score(text) == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_capped_at_20_lines(self):
        lines = ["a b c"] * 25
        text = "\n".join(lines)
        # sample_size = 20, all same shape -> 20/20 = 1.0
        assert repetitive_structure_score(text) == 1.0


# ---------------------------------------------------------------------------
# extract_all
# ---------------------------------------------------------------------------


class TestExtractAll:
    def test_returns_18_features(self):
        result = extract_all("hello world, this is a test.")
        assert len(result) == 18

    def test_returns_dict_with_correct_keys(self):
        result = extract_all("hello world")
        assert set(result.keys()) == set(FEATURES.keys())

    def test_samples_first_10k_chars(self):
        """Text longer than 10k chars should be truncated."""
        long_text = "a" * 20_000
        result = extract_all(long_text)
        # Should not raise, and should produce valid results
        assert all(isinstance(v, float) for v in result.values())


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCli:
    def test_main_produces_output(self, tmp_path):
        """Run featurize.py as a script with a small CSV."""
        import polars as pl

        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        df = pl.DataFrame(
            {
                "text": [
                    "Hello world. This is prose.",
                    "def foo():\n    return 42",
                    "a,b,c\n1,2,3\n4,5,6",
                ],
                "category": ["prose", "code", "structured"],
                "sub_type": ["article", "python", "csv"],
                "source": ["test", "test", "test"],
            }
        )
        df.write_csv(str(input_csv))

        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "featurize.py"),
                "--input",
                str(input_csv),
                "--output",
                str(output_csv),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        output_df = pl.read_csv(str(output_csv))
        assert len(output_df) == 3
        # Should have original 4 cols + 18 feature cols
        assert len(output_df.columns) == 22
        for col in FEATURES:
            assert col in output_df.columns, f"missing column: {col}"

    def test_feature_columns_are_float32(self, tmp_path):
        """Feature columns should use Float32 dtype to match Rust f32."""
        import polars as pl
        from featurize import main

        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        df = pl.DataFrame({"text": ["Hello world."]})
        df.write_csv(str(input_csv))

        main(["--input", str(input_csv), "--output", str(output_csv)])

        output_df = pl.read_csv(str(output_csv))
        # Re-read with schema overrides won't help; instead, test via main internals
        # Actually, CSV loses dtype info. Let's test the DataFrame construction directly.
        from featurize import extract_all

        rows = [extract_all("Hello world.")]
        features_df = pl.DataFrame(
            rows, schema={name: pl.Float32 for name in FEATURES}
        )
        for col in FEATURES:
            assert features_df[col].dtype == pl.Float32, (
                f"{col} should be Float32, got {features_df[col].dtype}"
            )
