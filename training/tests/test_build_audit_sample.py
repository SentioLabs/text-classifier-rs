"""Smoke tests for trainr.core.build_audit_sample."""

import tempfile
from pathlib import Path

import polars as pl


def _make_fake_corpus(n_rows: int = 200) -> pl.DataFrame:
    """Minimal corpus DataFrame with the columns build_audit_sample needs."""
    import random

    random.seed(42)
    rows = []
    sub_types = ["markdown", "python", "plain", "json", "log_lines"]

    # Plain filler
    for i in range(n_rows - 6):
        rows.append({
            "text": f"filler text sample {i} with enough content to exist",
            "sub_type": sub_types[i % len(sub_types)],
        })

    # Inject 2 known positives per label
    rows.append({
        "text": (
            "Traceback (most recent call last):\n"
            '  File "foo.py", line 10, in main\n'
            "    raise ValueError('oops')\n"
            "ValueError: oops"
        ),
        "sub_type": "plain",
    })
    rows.append({
        "text": (
            'Exception in thread "main" java.lang.NullPointerException\n'
            "    at com.foo.Bar.baz(Bar.java:42)\n"
            "    at com.foo.App.main(App.java:12)"
        ),
        "sub_type": "plain",
    })
    rows.append({
        "text": (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-old line\n"
            "+new line\n"
            " context"
        ),
        "sub_type": "plain",
    })
    rows.append({
        "text": (
            "# Patch\n"
            "@@ -10,4 +10,4 @@ def foo():\n"
            "-    return 1\n"
            "+    return 2\n"
        ),
        "sub_type": "markdown",
    })
    rows.append({
        "text": (
            "[2024-01-15T10:23:45Z] INFO request received\n"
            "[2024-01-15T10:23:46Z] WARN slow query\n"
            "[2024-01-15T10:23:47Z] ERROR timeout"
        ),
        "sub_type": "plain",
    })
    rows.append({
        "text": (
            "Here is a log example:\n\n"
            "2024-01-15 10:23:45 INFO app starting\n"
            "2024-01-15 10:23:46 INFO listening on :8080\n"
        ),
        "sub_type": "markdown",
    })

    return pl.DataFrame(rows)


def test_build_audit_sample_stratified_count():
    from trainr.core.build_audit_sample import build_audit_sample

    corpus = _make_fake_corpus(n_rows=200)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "corpus.parquet"
        output_path = Path(tmpdir) / "audit_input.parquet"
        corpus.write_parquet(input_path)

        build_audit_sample(
            input_path=str(input_path),
            output_path=str(output_path),
            stratified_n=100,
            injection_per_label=3,
            seed=42,
        )

        result = pl.read_parquet(output_path)
        # stratified 100 + up to 3*3=9 injected (some may overlap)
        assert 100 <= len(result) <= 112
        assert "audit_source" in result.columns


def test_build_audit_sample_injection_tags():
    """Injected rows must be tagged so the audit can exclude them from
    the agreement metric and measure recall on them separately."""
    from trainr.core.build_audit_sample import build_audit_sample

    corpus = _make_fake_corpus(n_rows=200)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "corpus.parquet"
        output_path = Path(tmpdir) / "audit_input.parquet"
        corpus.write_parquet(input_path)

        build_audit_sample(
            input_path=str(input_path),
            output_path=str(output_path),
            stratified_n=50,
            injection_per_label=3,
            seed=42,
        )

        result = pl.read_parquet(output_path)
        sources = set(result["audit_source"].to_list())
        assert "stratified" in sources
        assert "inject_stack_trace" in sources
        assert "inject_diff_patch" in sources
        assert "inject_log_content" in sources


def test_build_audit_sample_injection_regexes_match_positives():
    """Verify the known-positive fixture rows are selected by the
    injection regexes (not passed over)."""
    from trainr.core.build_audit_sample import build_audit_sample

    corpus = _make_fake_corpus(n_rows=200)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "corpus.parquet"
        output_path = Path(tmpdir) / "audit_input.parquet"
        corpus.write_parquet(input_path)

        build_audit_sample(
            input_path=str(input_path),
            output_path=str(output_path),
            stratified_n=10,
            injection_per_label=10,  # Enough to catch all fixture positives
            seed=42,
        )

        result = pl.read_parquet(output_path)
        injected = result.filter(pl.col("audit_source") != "stratified")
        # All 6 injected fixture rows should be found
        inject_sources = injected["audit_source"].to_list()
        assert inject_sources.count("inject_stack_trace") == 2
        assert inject_sources.count("inject_diff_patch") == 2
        assert inject_sources.count("inject_log_content") == 2
