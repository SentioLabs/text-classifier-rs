"""Tests for trainr.core.audit_heuristic — heuristic audit for label quality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Unit tests for individual heuristic functions
# ---------------------------------------------------------------------------


class TestCodeHeuristics:
    """Test code sub_type heuristics."""

    def test_python_with_def(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["python"]("def foo():\n    pass") is True

    def test_python_with_import(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["python"]("import os\nimport sys") is True

    def test_python_with_class(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["python"]("class Foo:\n    pass") is True

    def test_python_with_print(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["python"]("print('hello')") is True

    def test_python_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["python"]("The quick brown fox jumps.") is False

    def test_javascript_with_function(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["javascript"]("function foo() { return 1; }") is True

    def test_javascript_with_const(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["javascript"]("const x = 5;") is True

    def test_javascript_with_arrow(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["javascript"]("const f = (x) => x + 1;") is True

    def test_javascript_with_require(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["javascript"]("const fs = require('fs')") is True

    def test_javascript_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["javascript"]("Just some plain text here.") is False

    def test_typescript_same_as_javascript(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["typescript"]("const x: number = 5;") is True

    def test_rust_with_fn(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["rust"]("fn main() {\n    println!(\"hi\");\n}") is True

    def test_rust_with_struct(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["rust"]("struct Foo {\n    bar: i32,\n}") is True

    def test_rust_with_impl(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["rust"]("impl Foo {\n    fn new() -> Self {}\n}") is True

    def test_rust_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["rust"]("This is not Rust code.") is False

    def test_go_with_func(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["go"]("func main() {\n    fmt.Println(\"hi\")\n}") is True

    def test_go_with_package(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["go"]("package main\n\nimport \"fmt\"") is True

    def test_go_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["go"]("No Go code here.") is False

    def test_java_with_class(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["java"]("public class Foo {\n}") is True

    def test_java_with_import(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["java"]("import java.util.List;") is True

    def test_java_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["java"]("No Java code at all.") is False

    def test_sql_select(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["sql"]("SELECT * FROM users WHERE id = 1;") is True

    def test_sql_case_insensitive(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["sql"]("select count(*) from orders") is True

    def test_sql_create(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["sql"]("CREATE TABLE foo (id INT);") is True

    def test_sql_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["sql"]("This is normal text.") is False

    def test_shell_shebang(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["shell"]("#!/bin/bash\necho hello") is True

    def test_shell_if_bracket(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["shell"]("if [ -f foo ]; then\n  echo yes\nfi") is True

    def test_shell_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["shell"]("Some random prose.") is False

    def test_html_div(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["html"]("<div>hello world</div>") is True

    def test_html_case_insensitive(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["html"]("<HTML><BODY>hi</BODY></HTML>") is True

    def test_html_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["html"]("Just text, no tags.") is False

    def test_xml_processing_instruction(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["xml"]("<?xml version='1.0'?>\n<root/>") is True

    def test_xml_tags(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["xml"]("<item>data</item>") is True

    def test_xml_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["xml"]("No XML here.") is False

    def test_css_with_color(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["css"]("body { color: red; }") is True

    def test_css_with_margin(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["css"]("div { margin: 10px; }") is True

    def test_css_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["css"]("No CSS here.") is False

    def test_yaml_key_value(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["yaml"]("name: John\nage: 30") is True

    def test_yaml_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["yaml"]("No YAML here at all") is False

    def test_toml_section_and_key(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["toml"]("[package]\nname = \"foo\"\nversion = \"1.0\"") is True

    def test_toml_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["toml"]("No TOML here.") is False

    def test_ini_section_and_key(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["ini"]("[section]\nkey=value") is True

    def test_ini_with_spaces(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["ini"]("[database]\nhost = localhost\nport = 5432") is True

    def test_ini_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["ini"]("Not an ini file.") is False

    def test_dockerfile_from(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["dockerfile"]("FROM python:3.11\nRUN pip install flask") is True

    def test_dockerfile_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["dockerfile"]("No docker here.") is False

    def test_makefile_target(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["makefile"]("build:\n\tgcc -o main main.c") is True

    def test_makefile_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["makefile"]("No makefile here.") is False

    def test_markdown_heading(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["markdown"]("# Title\n\nSome text.") is True

    def test_markdown_bold(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["markdown"]("This is **bold** text.") is True

    def test_markdown_code_fence(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["markdown"]("```python\nprint('hi')\n```") is True

    def test_markdown_link(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["markdown"]("See [docs](https://example.com).") is True

    def test_markdown_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["markdown"]("No markdown syntax here at all ever") is False

    def test_rst_directive(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["rst"](".. note::\n   This is important.") is True

    def test_rst_heading_underline(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["rst"]("Title\n=====\n\nSome text.") is True

    def test_rst_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["rst"]("No rst here at all ever") is False

    def test_latex_begin(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["latex"]("\\begin{document}\nHello.\n\\end{document}") is True

    def test_latex_section(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["latex"]("\\section{Introduction}") is True

    def test_latex_negative(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["latex"]("No LaTeX here.") is False


class TestAlwaysTrueHeuristics:
    """Test sub_types that always pass."""

    def test_plain_always_true(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["plain"]("Anything at all") is True

    def test_unknown_always_true(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["unknown"]("Literally anything") is True


class TestPatternHeuristics:
    """Test key_value and log_lines heuristics."""

    def test_key_value_passes(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "name=John\nage=30\ncity=NYC\nfoo bar baz"
        assert HEURISTICS["key_value"](text) is True

    def test_key_value_with_colon_space(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "name: John\nage: 30\ncity: NYC"
        assert HEURISTICS["key_value"](text) is True

    def test_key_value_fails(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "Just a bunch\nof random lines\nwith no patterns"
        assert HEURISTICS["key_value"](text) is False

    def test_log_lines_passes(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "2024-01-15 10:30:00 INFO Starting\n2024-01-15 10:30:01 DEBUG Loaded\n2024-01-15 10:30:02 INFO Done"
        assert HEURISTICS["log_lines"](text) is True

    def test_log_lines_timestamp_colon(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "10:30:00 msg1\n10:30:01 msg2\n10:30:02 msg3"
        assert HEURISTICS["log_lines"](text) is True

    def test_log_lines_fails(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "No timestamps\nHere at all\nJust text"
        assert HEURISTICS["log_lines"](text) is False


class TestStructuredHeuristics:
    """Test that structured sub_types delegate to pull_real_data validators."""

    def test_csv_valid(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "name,age,city\nAlice,30,NYC\nBob,25,LA"
        assert HEURISTICS["csv"](text) is True

    def test_csv_invalid(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["csv"]("Not CSV at all.") is False

    def test_tsv_valid(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA"
        assert HEURISTICS["tsv"](text) is True

    def test_jsonl_valid(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = '{"a": 1}\n{"a": 2}'
        assert HEURISTICS["jsonl"](text) is True

    def test_json_valid(self):
        from trainr.core.audit_heuristic import HEURISTICS

        assert HEURISTICS["json"]('{"key": "value"}') is True

    def test_pipe_table_valid(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |"
        assert HEURISTICS["pipe_table"](text) is True

    def test_fixed_width_valid(self):
        from trainr.core.audit_heuristic import HEURISTICS

        text = "Name      Age  City\nAlice     30   NYC\nBob       25   LA\nCharlie   35   CHI"
        assert HEURISTICS["fixed_width"](text) is True


# ---------------------------------------------------------------------------
# Integration test: run_audit on a small parquet
# ---------------------------------------------------------------------------


class TestRunAudit:
    """Integration test for run_audit function."""

    def test_run_audit_adds_heuristic_pass_column(self):
        from trainr.core.audit_heuristic import run_audit

        df = pl.DataFrame(
            {
                "text": [
                    "def foo():\n    return 42",
                    "The quick brown fox.",
                    "SELECT * FROM users;",
                ],
                "sub_type": ["python", "python", "sql"],
                "expected_category": ["code", "code", "code"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "output.parquet"
            df.write_parquet(input_path)

            run_audit(str(input_path), str(output_path))

            result = pl.read_parquet(output_path)
            assert "heuristic_pass" in result.columns
            assert result.shape[0] == 3
            # Row 0: python with def -> True
            assert result["heuristic_pass"][0] is True
            # Row 1: python with plain text -> False
            assert result["heuristic_pass"][1] is False
            # Row 2: sql with SELECT -> True
            assert result["heuristic_pass"][2] is True

    def test_run_audit_handles_unknown_sub_type(self):
        """Unknown sub_types not in HEURISTICS should default to True."""
        from trainr.core.audit_heuristic import run_audit

        df = pl.DataFrame(
            {
                "text": ["some random text"],
                "sub_type": ["nonexistent_type"],
                "expected_category": ["code"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "output.parquet"
            df.write_parquet(input_path)

            run_audit(str(input_path), str(output_path))

            result = pl.read_parquet(output_path)
            assert result["heuristic_pass"][0] is True

    def test_run_audit_preserves_original_columns(self):
        from trainr.core.audit_heuristic import run_audit

        df = pl.DataFrame(
            {
                "text": ["def foo(): pass"],
                "sub_type": ["python"],
                "expected_category": ["code"],
                "extra_col": ["keep_me"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "output.parquet"
            df.write_parquet(input_path)

            run_audit(str(input_path), str(output_path))

            result = pl.read_parquet(output_path)
            assert "extra_col" in result.columns
            assert result["extra_col"][0] == "keep_me"
