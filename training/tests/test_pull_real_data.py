"""Tests for trainr.core.pull_real_data — helper functions only.

Does NOT test actual HuggingFace streaming (integration test).
Tests extension mapping, row construction, and size filtering.
"""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from trainr.core.pull_real_data import (
    FEATURE_COLUMNS,
    PHASE_SUB_TYPES,
    STACK_DATA_DIR,
    SUB_TYPE_CATEGORY,
    TARGET_COUNTS,
    append_to_parquet,
    build_row,
    passes_size_filter,
    sub_type_for_extension,
)


# ---------------------------------------------------------------------------
# Extension mapping
# ---------------------------------------------------------------------------


class TestSubTypeForExtension:
    """Test extension-to-sub_type mapping."""

    def test_python_extension(self):
        assert sub_type_for_extension("py") == "python"

    def test_javascript_extension(self):
        assert sub_type_for_extension("js") == "javascript"

    def test_typescript_extension(self):
        assert sub_type_for_extension("ts") == "typescript"

    def test_rust_extension(self):
        assert sub_type_for_extension("rs") == "rust"

    def test_go_extension(self):
        assert sub_type_for_extension("go") == "go"

    def test_java_extension(self):
        assert sub_type_for_extension("java") == "java"

    def test_sql_extension(self):
        assert sub_type_for_extension("sql") == "sql"

    def test_shell_extensions(self):
        assert sub_type_for_extension("sh") == "shell"
        assert sub_type_for_extension("bash") == "shell"

    def test_css_extension(self):
        assert sub_type_for_extension("css") == "css"

    def test_html_extensions(self):
        assert sub_type_for_extension("html") == "html"
        assert sub_type_for_extension("htm") == "html"

    def test_xml_extensions(self):
        assert sub_type_for_extension("xml") == "xml"
        assert sub_type_for_extension("xsl") == "xml"

    def test_unknown_extension_returns_none(self):
        assert sub_type_for_extension("zzzzz") is None

    def test_dockerfile_filename(self):
        """Dockerfile is matched by filename, not extension."""
        assert sub_type_for_extension("dockerfile") == "dockerfile"

    def test_makefile_filename(self):
        """Makefile matched by extension or convention."""
        assert sub_type_for_extension("makefile") == "makefile"
        assert sub_type_for_extension("mk") == "makefile"


# ---------------------------------------------------------------------------
# Size filtering
# ---------------------------------------------------------------------------


class TestPassesSizeFilter:
    """Test text size filtering (skip <50 bytes or >50KB)."""

    def test_empty_string_rejected(self):
        assert passes_size_filter("") is False

    def test_too_short_rejected(self):
        assert passes_size_filter("x" * 49) is False

    def test_exactly_50_bytes_accepted(self):
        assert passes_size_filter("x" * 50) is True

    def test_normal_size_accepted(self):
        assert passes_size_filter("x" * 1000) is True

    def test_exactly_50kb_accepted(self):
        assert passes_size_filter("x" * 50_000) is True

    def test_over_50kb_rejected(self):
        assert passes_size_filter("x" * 50_001) is False


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


class TestBuildRow:
    """Test row construction for parquet append."""

    def test_row_has_all_required_columns(self):
        row = build_row(text="hello world", sub_type="python")
        assert "text" in row
        assert "category" in row
        assert "sub_type" in row
        assert "source" in row
        assert "model" in row

    def test_row_has_correct_values(self):
        row = build_row(text="print('hi')", sub_type="python")
        assert row["text"] == "print('hi')"
        assert row["category"] == "code"
        assert row["sub_type"] == "python"
        assert row["source"] == "real/the-stack-v2"
        assert row["model"] == "real/the-stack-v2"

    def test_feature_columns_are_null(self):
        row = build_row(text="some code", sub_type="rust")
        for col in FEATURE_COLUMNS:
            assert row[col] is None, f"Feature column {col} should be None"

    def test_row_has_exactly_43_keys(self):
        row = build_row(text="code", sub_type="go")
        assert len(row) == 43

    def test_row_produces_valid_dataframe(self):
        """Row dict can be loaded into a polars DataFrame with correct schema."""
        row = build_row(text="fn main() {}", sub_type="rust")
        df = pl.DataFrame([row])
        assert df.shape == (1, 43)
        assert df["text"].dtype == pl.Utf8
        assert df["category"].dtype == pl.Utf8


# ---------------------------------------------------------------------------
# Phase sub_type groups
# ---------------------------------------------------------------------------


class TestPhaseSubTypes:
    """Test phase -> sub_type mapping."""

    def test_code_phase_has_14_sub_types(self):
        assert len(PHASE_SUB_TYPES["code"]) == 14

    def test_code_phase_includes_python(self):
        assert "python" in PHASE_SUB_TYPES["code"]

    def test_code_phase_includes_dockerfile(self):
        assert "dockerfile" in PHASE_SUB_TYPES["code"]

    def test_code_phase_includes_makefile(self):
        assert "makefile" in PHASE_SUB_TYPES["code"]


# ---------------------------------------------------------------------------
# Target counts
# ---------------------------------------------------------------------------


class TestTargetCounts:
    """Test target count configuration."""

    def test_all_code_sub_types_have_targets(self):
        for st in PHASE_SUB_TYPES["code"]:
            assert st in TARGET_COUNTS, f"Missing target count for {st}"

    def test_total_code_target_approximately_37k(self):
        total = sum(TARGET_COUNTS[st] for st in PHASE_SUB_TYPES["code"])
        assert 30_000 <= total <= 45_000, f"Total code target {total} not ~37K"


# ---------------------------------------------------------------------------
# Stack data dir mapping
# ---------------------------------------------------------------------------


class TestStackDataDir:
    """Test the mapping from sub_type to Stack v1 data_dir."""

    def test_python_data_dir(self):
        assert STACK_DATA_DIR["python"] == "data/python"

    def test_javascript_data_dir(self):
        assert STACK_DATA_DIR["javascript"] == "data/javascript"

    def test_all_code_sub_types_have_data_dir(self):
        for st in PHASE_SUB_TYPES["code"]:
            assert st in STACK_DATA_DIR, f"Missing data_dir for {st}"


# ---------------------------------------------------------------------------
# Parquet append
# ---------------------------------------------------------------------------


class TestAppendToParquet:
    """Test appending new rows to an existing parquet file."""

    def _make_existing_parquet(self, path: Path, n_rows: int = 5) -> None:
        """Create a minimal parquet matching golden_train schema."""
        data: dict = {
            "text": [f"sample {i}" for i in range(n_rows)],
            "category": ["code"] * n_rows,
            "sub_type": ["python"] * n_rows,
            "source": ["real/test"] * n_rows,
            "model": ["real/test"] * n_rows,
        }
        for col in FEATURE_COLUMNS:
            data[col] = [None] * n_rows
        schema = {
            "text": pl.Utf8,
            "category": pl.Utf8,
            "sub_type": pl.Utf8,
            "source": pl.Utf8,
            "model": pl.Utf8,
        }
        for col in FEATURE_COLUMNS:
            schema[col] = pl.Float32
        df = pl.DataFrame(data, schema=schema)
        df.write_parquet(path)

    def test_append_increases_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            pq_path = Path(tmp) / "test.parquet"
            self._make_existing_parquet(pq_path, n_rows=5)

            new_rows = [build_row("new code", "rust") for _ in range(3)]
            appended = append_to_parquet(new_rows, parquet_path=str(pq_path))

            assert appended == 3
            df = pl.read_parquet(pq_path)
            assert df.shape[0] == 8

    def test_append_preserves_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            pq_path = Path(tmp) / "test.parquet"
            self._make_existing_parquet(pq_path, n_rows=2)

            new_rows = [build_row("new code", "go")]
            append_to_parquet(new_rows, parquet_path=str(pq_path))

            df = pl.read_parquet(pq_path)
            assert df["text"][0] == "sample 0"
            assert df["text"][1] == "sample 1"
            assert df["text"][2] == "new code"

    def test_append_preserves_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            pq_path = Path(tmp) / "test.parquet"
            self._make_existing_parquet(pq_path, n_rows=1)

            new_rows = [build_row("code", "java")]
            append_to_parquet(new_rows, parquet_path=str(pq_path))

            df = pl.read_parquet(pq_path)
            assert len(df.columns) == 43
            assert df["line_length_cv"].dtype == pl.Float32


# ---------------------------------------------------------------------------
# Multibyte size filtering
# ---------------------------------------------------------------------------


class TestPassesSizeFilterMultibyte:
    """Test size filter counts bytes, not characters."""

    def test_multibyte_chars_counted_correctly(self):
        # 20 CJK chars = 60 bytes in UTF-8, should pass
        text = "\u4e00" * 20
        assert passes_size_filter(text) is True

    def test_short_multibyte_rejected(self):
        # 10 CJK chars = 30 bytes, should fail
        text = "\u4e00" * 10
        assert passes_size_filter(text) is False


# ---------------------------------------------------------------------------
# Config and prose extension mappings
# ---------------------------------------------------------------------------


class TestConfigProseExtensions:
    """Test extension mapping for config and prose sub_types."""

    def test_yaml_extension(self):
        assert sub_type_for_extension("yaml") == "yaml"

    def test_yml_extension(self):
        assert sub_type_for_extension("yml") == "yaml"

    def test_toml_extension(self):
        assert sub_type_for_extension("toml") == "toml"

    def test_ini_extension(self):
        assert sub_type_for_extension("ini") == "ini"

    def test_cfg_extension(self):
        assert sub_type_for_extension("cfg") == "ini"

    def test_markdown_extensions(self):
        assert sub_type_for_extension("md") == "markdown"
        assert sub_type_for_extension("markdown") == "markdown"

    def test_rst_extension(self):
        assert sub_type_for_extension("rst") == "rst"

    def test_latex_extensions(self):
        assert sub_type_for_extension("tex") == "latex"
        assert sub_type_for_extension("latex") == "latex"


# ---------------------------------------------------------------------------
# SUB_TYPE_CATEGORY mapping
# ---------------------------------------------------------------------------


class TestSubTypeCategory:
    """Test sub_type -> category mapping."""

    def test_code_sub_types_have_code_category(self):
        for st in ["python", "javascript", "typescript", "rust", "go", "java",
                    "sql", "shell", "css", "html", "xml", "dockerfile",
                    "makefile", "unknown"]:
            assert SUB_TYPE_CATEGORY[st] == "code", f"{st} should be 'code'"

    def test_config_sub_types_have_code_category(self):
        """Config files (yaml/toml/ini) are categorized as code per types.rs."""
        for st in ["yaml", "toml", "ini"]:
            assert SUB_TYPE_CATEGORY[st] == "code", f"{st} should be 'code'"

    def test_prose_sub_types_have_prose_category(self):
        for st in ["markdown", "rst", "latex"]:
            assert SUB_TYPE_CATEGORY[st] == "prose", f"{st} should be 'prose'"

    def test_all_phase_sub_types_have_category(self):
        """Every sub_type in any phase must have a category entry."""
        for phase_list in PHASE_SUB_TYPES.values():
            for st in phase_list:
                assert st in SUB_TYPE_CATEGORY, f"Missing category for {st}"


# ---------------------------------------------------------------------------
# Config and prose phases
# ---------------------------------------------------------------------------


class TestConfigProsePhases:
    """Test phase configuration for config and prose."""

    def test_config_phase_exists(self):
        assert "config" in PHASE_SUB_TYPES

    def test_config_phase_sub_types(self):
        assert PHASE_SUB_TYPES["config"] == ["yaml", "toml", "ini"]

    def test_prose_phase_exists(self):
        assert "prose" in PHASE_SUB_TYPES

    def test_prose_phase_sub_types(self):
        assert PHASE_SUB_TYPES["prose"] == ["markdown", "rst", "latex"]

    def test_all_config_sub_types_have_targets(self):
        for st in PHASE_SUB_TYPES["config"]:
            assert st in TARGET_COUNTS, f"Missing target count for {st}"

    def test_all_prose_sub_types_have_targets(self):
        for st in PHASE_SUB_TYPES["prose"]:
            assert st in TARGET_COUNTS, f"Missing target count for {st}"

    def test_all_config_sub_types_have_data_dir(self):
        for st in PHASE_SUB_TYPES["config"]:
            assert st in STACK_DATA_DIR, f"Missing data_dir for {st}"

    def test_all_prose_sub_types_have_data_dir(self):
        for st in PHASE_SUB_TYPES["prose"]:
            assert st in STACK_DATA_DIR, f"Missing data_dir for {st}"


# ---------------------------------------------------------------------------
# Config and prose target counts
# ---------------------------------------------------------------------------


class TestConfigProseTargetCounts:
    """Test target counts for config and prose phases."""

    def test_yaml_target(self):
        assert TARGET_COUNTS["yaml"] == 1000

    def test_toml_target(self):
        assert TARGET_COUNTS["toml"] == 1000

    def test_ini_target(self):
        assert TARGET_COUNTS["ini"] == 500

    def test_markdown_target(self):
        assert TARGET_COUNTS["markdown"] == 2500

    def test_rst_target(self):
        assert TARGET_COUNTS["rst"] == 1500

    def test_latex_target(self):
        assert TARGET_COUNTS["latex"] == 2000

    def test_total_config_target(self):
        total = sum(TARGET_COUNTS[st] for st in PHASE_SUB_TYPES["config"])
        assert total == 2500

    def test_total_prose_target(self):
        total = sum(TARGET_COUNTS[st] for st in PHASE_SUB_TYPES["prose"])
        assert total == 6000


# ---------------------------------------------------------------------------
# build_row with category parameter
# ---------------------------------------------------------------------------


class TestBuildRowCategory:
    """Test that build_row accepts and uses the category parameter."""

    def test_default_category_is_code(self):
        row = build_row(text="hello", sub_type="python")
        assert row["category"] == "code"

    def test_explicit_prose_category(self):
        row = build_row(text="# Hello World", sub_type="markdown", category="prose")
        assert row["category"] == "prose"

    def test_explicit_code_category(self):
        row = build_row(text="key: val", sub_type="yaml", category="code")
        assert row["category"] == "code"

    def test_prose_row_has_all_columns(self):
        row = build_row(text="Some text", sub_type="rst", category="prose")
        assert len(row) == 43
        assert row["sub_type"] == "rst"
        for col in FEATURE_COLUMNS:
            assert row[col] is None


# ---------------------------------------------------------------------------
# Stack data dir for config/prose
# ---------------------------------------------------------------------------


class TestConfigProseStackDataDir:
    """Test Stack v1 data_dir mapping for config and prose."""

    def test_yaml_data_dir(self):
        assert STACK_DATA_DIR["yaml"] == "data/yaml"

    def test_toml_data_dir(self):
        assert STACK_DATA_DIR["toml"] == "data/toml"

    def test_ini_data_dir(self):
        assert STACK_DATA_DIR["ini"] == "data/ini"

    def test_markdown_data_dir(self):
        assert STACK_DATA_DIR["markdown"] == "data/markdown"

    def test_rst_data_dir(self):
        assert STACK_DATA_DIR["rst"] == "data/restructuredtext"

    def test_latex_data_dir(self):
        assert STACK_DATA_DIR["latex"] == "data/tex"
