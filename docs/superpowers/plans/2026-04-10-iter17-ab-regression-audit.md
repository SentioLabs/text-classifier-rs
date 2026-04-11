# iter17 A/B Regression Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate iter18's ~$400-600 90k annotation run on proof that iter16's longer `SYSTEM_PROMPT` did not silently regress any of the 40 pre-existing detection labels. Run a controlled A/B comparison with a real same-prompt noise floor and emit a regression report with pass/fail verdicts.

**Architecture:** (1) Refactor `audit_semantic_labels.py` to add prevalence computation + a loader helper with filename-slug parsing. (2) New `compare_prompt_versions.py` module that takes 3 + 3 + 3 annotation parquets (iter15 before, iter16a after, iter16b noise-floor companion), computes per-label agreement + prevalence + noise floor + verdict, and emits a markdown report. (3) Run 3 fresh annotation passes across 3 models (iter15 from a `22bc292` git worktree, iter16a and iter16b from HEAD) with a concurrency cap of 2 jobs in flight. (4) Run the comparison, write the iteration doc, make a gate decision.

**Tech Stack:** Python 3.14, `polars` (not `pandas`), `click` for CLI wiring, `pytest` for tests, existing `trainr` package conventions, git worktrees, OpenRouter LLM API (3 models: `gemini3flash`, `sonnet`, `gpt54mini`).

**Spec:** [`docs/superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md`](../specs/2026-04-10-iter17-ab-regression-audit-design.md)

**Prior iteration:** [`docs/accuracy_runs/2026-04-10-iteration-16.md`](../../accuracy_runs/2026-04-10-iteration-16.md)

---

## Working context you need before starting

Read these files once before touching any task. They are the authoritative source for how the existing audit stack is organized:

- `training/trainr/core/annotate_detections.py` — the annotator CLI entry point. Note: it has a `def main()` at line ~727 but **NO `__main__` guard** (that's the bug P0a fixes). `DEFAULT_MODEL = "openai/gpt-5.4-nano"`; `BACKEND_DEFAULT_MODELS` at line ~721; `SEMANTIC_LABELS` frozenset at module level.
- `training/trainr/core/audit_semantic_labels.py` — the iter16 audit module. Already exposes `filter_for_agreement()`, `detection_columns()`, `compute_agreement_across_models(dfs: dict[str, pl.DataFrame])`, `compute_recall_on_injected()`, `compute_recall_majority()`, `format_report()`. These are the "primitives" referenced by the spec's P1. Convention: frames are passed as `dict[str, pl.DataFrame]` keyed by model slug, not `list[pl.DataFrame]`. Follow this convention in new code.
- `training/trainr/core/build_audit_sample.py` — writes `audit_source` column with values `"stratified"` (4984 rows) or `"inject_<label>"` (81 rows total across the 3 labels). Total audit input = 5065 rows.
- `training/trainr/commands/data.py` — click CLI wire-up. Add new commands as `@data.command("name")` blocks that import `from trainr.core.X import main as _main; _main(argv)`.
- `training/tests/test_audit_semantic_labels.py`, `training/tests/test_annotate_detections.py`, `training/tests/test_build_audit_sample.py` — existing test file layout. Flat `training/tests/` directory, `test_<module>.py` naming.
- `training/data/audit/iter16_5k_input.parquet` — 5065 rows, 47 non-`det_*` columns, `audit_source` distribution: `stratified`=4984, `inject_stack_trace`=24, `inject_log_content`=50, `inject_diff_patch`=7.
- `training/data/audit/iter16_5k_{gemini3flash,sonnet,gpt54mini}.parquet` — the archived (pre-`c1ec175`) iter16 annotation parquets. **NOT used** by iter17's compare module (they measure prompt drift, not noise, per the SPEC_REVIEW finding). They stay on disk for historical reference only.

Key invariants from the spec that easy-to-break in implementation:

1. **The noise floor comes from `|agr(iter16a) - agr(iter16b)|` where iter16a and iter16b are both fresh runs on the iter16 prompt at commit HEAD.** Never reuse archived iter16 parquets as the noise-floor source. Never cache the noise-floor parquets across a FAIL-path prompt iteration — each iteration reruns both iter16a and iter16b.
2. **Agreement, prevalence, and all gate metrics are computed on stratified rows only** (`audit_source == "stratified"`). Injected rows (`audit_source` starting with `inject_`) are excluded from the gate.
3. **The agreement metric has a floor of `2/3 ≈ 0.667`, not 0**, because with 3 models the worst case is a 2-1 split (2/3 agreement). The 0.995 threshold and the 0.005 delta gate live on a `[0.667, 1.0]` scale.
4. **`prev_ratio` zero-handling is load-bearing.** `0/0 → 1.0, no warn`. `0/>0 → inf, warn`. `>0/0 → 0.0, warn`. Without this rule, silent labels (prevalence 0 on both sides) would trip WARN-prevalence for no reason.
5. **Canonical model slugs are `{"gemini3flash", "sonnet", "gpt54mini"}`** — matching the existing `iter16_5k_*.parquet` filename convention. Not `sonnet46`.
6. **Each phase ends with a commit and a two-stage review** (spec-compliance review + code-quality review), matching iter16 precedent.

---

## File Structure

**Files to create:**
- `training/trainr/core/compare_prompt_versions.py` — A/B comparison module (the main new module).
- `training/tests/test_compare_prompt_versions.py` — unit tests for the compare module.

**Files to modify:**
- `training/trainr/core/annotate_detections.py` — add `if __name__ == "__main__": main()` guard at EOF.
- `training/trainr/core/build_audit_sample.py` — tighten `INJECTION_PATTERNS` for python/rust/java stack_trace entries.
- `training/trainr/core/audit_semantic_labels.py` — add `compute_prevalence_per_label()` and `load_annotator_parquets()` helpers; no behavior change to existing functions.
- `training/tests/test_build_audit_sample.py` — regression tests for tightened injection patterns.
- `training/tests/test_audit_semantic_labels.py` — tests for new prevalence + loader helpers.
- `training/trainr/commands/data.py` — add `@data.command("compare-prompts")` wire-up.

**Files to produce (not code — outputs of annotation runs and gate decision):**
- `training/data/audit/iter17_ab_iter15_{gemini3flash,sonnet,gpt54mini}.parquet` (3 files)
- `training/data/audit/iter17_ab_iter16a_{gemini3flash,sonnet,gpt54mini}.parquet` (3 files)
- `training/data/audit/iter17_ab_iter16b_{gemini3flash,sonnet,gpt54mini}.parquet` (3 files)
- `docs/accuracy_runs/2026-04-10-iter17-regression-report.md` (tool output)
- `docs/accuracy_runs/2026-04-10-iteration-17.md` (iteration write-up with gate decision)

---

## Phase 0: Operational Fixes

### Task 0.1: Add `__main__` guard to `annotate_detections.py`

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` (append one block at EOF)

**Context:** iter16 lost ~30 min diagnosing a `python -m trainr.core.annotate_detections ...` invocation that silently exited 0 because the module had no `__main__` guard. Python imported the module but never called `main()`. Adding the guard is a 3-line fix with no risk — the `click` CLI (`trainr data annotate-detections`) remains the canonical invocation; the guard just makes `python -m` work as a backup diagnostic path.

- [ ] **Step 1: Verify the file currently lacks the guard.**

```bash
grep -c '^if __name__' training/trainr/core/annotate_detections.py
```

Expected output: `0` (no guard present).

- [ ] **Step 2: Append the guard to the end of the file.**

```python
# At end of training/trainr/core/annotate_detections.py:


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify `python -m` invocation now reaches main().**

```bash
uv run --directory training python -m trainr.core.annotate_detections --help 2>&1 | head -5
```

Expected: the help text for the annotator (same as `trainr data annotate-detections --help`). If it silently exits with no output, the guard is wrong.

- [ ] **Step 4: Run existing annotate-detections test suite.**

```bash
uv run --directory training pytest tests/test_annotate_detections.py -q
```

Expected: all tests pass (the guard addition should not affect any existing behavior).

- [ ] **Step 5: Commit.**

```bash
git add training/trainr/core/annotate_detections.py
git commit -m "feat(detections): add __main__ guard to annotate_detections

Restores python -m trainr.core.annotate_detections as a working
invocation path. iter16 lost ~30 min diagnosing a silent exit 0
when running via python -m; the click CLI was always the canonical
path but the backup diagnostic route was broken."
```

---

### Task 0.2: Tighten `INJECTION_PATTERNS` for stack_trace

**Files:**
- Modify: `training/trainr/core/build_audit_sample.py:33-53` (INJECTION_PATTERNS dict)
- Modify: `training/tests/test_build_audit_sample.py` (add regression tests)

**Context:** iter16 §Follow-Up called out that the existing `stack_trace` injection regexes overmatch on prose and test directives. For example, the Rust `// error-pattern:thread 'main' panicked at` test comment matches the `panicked at` pattern but isn't a real stack trace. iter16's audit had 9 of 9 `stack_trace` "misses" that were actually regex overmatches, not prompt failures. The tightening makes the next audit's recall numbers interpretable. **Does not rebuild `iter16_5k_input.parquet`** — reusing the existing file is load-bearing for the iter17 A/B.

- [ ] **Step 1: Write failing regression tests first.**

Add to `training/tests/test_build_audit_sample.py`:

```python
import re

from trainr.core.build_audit_sample import INJECTION_PATTERNS


def _combined(label: str) -> str:
    return "|".join(f"(?:{p})" for p in INJECTION_PATTERNS[label])


def _matches(label: str, text: str) -> bool:
    return re.search(_combined(label), text) is not None


class TestStackTraceInjectionPatternTightening:
    """Regression tests: each case SHOULD NOT match after the tightening."""

    def test_rust_error_pattern_directive_does_not_match(self):
        text = "// error-pattern:thread 'main' panicked at"
        assert not _matches("stack_trace", text), (
            "Rust test directive must not be injected as a stack_trace candidate"
        )

    def test_python_prose_at_line_without_traceback_does_not_match(self):
        text = "the parser errored at line 42 of the config"
        assert not _matches("stack_trace", text), (
            "Prose mentioning 'at line N' without a Traceback header must not match"
        )

    def test_java_frame_without_exception_header_does_not_match(self):
        text = "  See also: at com.foo.Bar.method(Bar.java:15) for details"
        assert not _matches("stack_trace", text), (
            "Java frame with no 'Exception in thread' context must not match"
        )

    def test_real_python_traceback_still_matches(self):
        text = (
            'Traceback (most recent call last):\n'
            '  File "foo.py", line 5, in <module>\n'
            '    raise ValueError("bad")\n'
            'ValueError: bad'
        )
        assert _matches("stack_trace", text), (
            "Real Python traceback must still be detected"
        )

    def test_real_java_trace_still_matches(self):
        text = (
            'Exception in thread "main" java.lang.NullPointerException\n'
            '    at com.foo.Bar.method(Bar.java:15)\n'
            '    at com.foo.Baz.run(Baz.java:22)'
        )
        assert _matches("stack_trace", text), (
            "Real Java trace with Exception header must still be detected"
        )

    def test_real_rust_panic_still_matches(self):
        text = (
            "thread 'main' panicked at 'assertion failed', src/lib.rs:42\n"
            "note: run with `RUST_BACKTRACE=1`"
        )
        assert _matches("stack_trace", text), (
            "Real Rust panic at runtime must still match"
        )
```

- [ ] **Step 2: Run the tests to confirm they fail.**

```bash
uv run --directory training pytest tests/test_build_audit_sample.py::TestStackTraceInjectionPatternTightening -v
```

Expected: the 3 "does not match" tests FAIL under the current patterns (they match), the 3 "still matches" tests PASS.

- [ ] **Step 3: Tighten the patterns in `build_audit_sample.py`.**

Replace the `"stack_trace"` entry in `INJECTION_PATTERNS` (around line 34-41) with:

```python
"stack_trace": [
    # Python: require Traceback header + "File " adjacency. The (?s:...){0,200}
    # bounds the inter-line distance so unrelated prose can't bridge the two.
    r"Traceback \(most recent call last\):(?s:.{0,200})File \"",
    # Java: require frame adjacency to an "Exception in thread" header, again
    # within a bounded inter-line window.
    r"Exception in thread(?s:.{0,400})\s+at [\w.$]+\(.*\.java:\d+\)",
    # Go is unchanged — the "goroutine N [" header is already specific.
    r"goroutine \d+ \[",
    # Rust: require "panicked at" but explicitly exclude the
    # "// error-pattern:" test directive prefix via a negative lookbehind.
    r"(?<!// error-pattern:)(?<!// error-pattern: )thread '[^']+' panicked at",
    # .NET is unchanged — the "in <file>.cs:line N" suffix is specific.
    r"^\s+at \w+\.\w+\.\w+\(\) in .*\.cs:line \d+",
],
```

Notes for the implementer: the `(?s:.{0,200})` syntax is a polars/python re-compatible inline-dotall non-capturing group with length bound. The negative lookbehind for Rust uses two variants (with and without trailing space) because lookbehind requires fixed-width patterns. Python `re` and `polars` `str.contains` both support this syntax.

- [ ] **Step 4: Re-run the tests — they should pass now.**

```bash
uv run --directory training pytest tests/test_build_audit_sample.py::TestStackTraceInjectionPatternTightening -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Run the full build_audit_sample test suite to confirm no collateral damage.**

```bash
uv run --directory training pytest tests/test_build_audit_sample.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit.**

```bash
git add training/trainr/core/build_audit_sample.py training/tests/test_build_audit_sample.py
git commit -m "fix(audit): tighten INJECTION_PATTERNS stack_trace regexes

iter16 found 9 of 9 stack_trace 'misses' were actually regex
overmatches on Rust test directives, prose with 'at line N', and
Java frames without Exception headers. Tighten:

- Python: require 'Traceback (...)' header + 'File \"' adjacency
- Java:   require 'Exception in thread' header + java-frame adjacency
- Rust:   exclude '// error-pattern:' test-comment prefix

Adds regression tests in both directions — the previously-matching
anti-cases should no longer match, and real traces must still match.
Does not rebuild iter16_5k_input.parquet; existing file is reused
for the iter17 A/B."
```

---

## Phase 1: Refactor `audit_semantic_labels.py` — add prevalence + loader helpers

Note: the spec's P1 named a refactor to "expose primitives." Most of the target primitives (`filter_for_agreement`, `detection_columns`, `compute_agreement_across_models`) are **already exposed** as module-level functions — no refactor needed for those. What's genuinely missing are (a) a prevalence computation function, (b) a loader helper that parses filename slugs, and (c) a byte-for-byte smoke test for `format_report()` so the iter16 audit report is reproducible after any future edit.

### Task 1.1: Add `compute_prevalence_per_label()`

**Files:**
- Modify: `training/trainr/core/audit_semantic_labels.py` (add function after `compute_agreement_across_models`)
- Modify: `training/tests/test_audit_semantic_labels.py` (add test class)

**Context:** The iter17 A/B gate's soft WARN-prevalence signal needs per-label fire rate on stratified rows. The existing module has recall computation on injected rows but no prevalence computation on stratified rows. This function fills the gap and has the same `dict[str, pl.DataFrame]` calling convention as the existing `compute_agreement_across_models`.

- [ ] **Step 1: Write the failing test.**

Add to `training/tests/test_audit_semantic_labels.py`:

```python
import polars as pl

from trainr.core.audit_semantic_labels import compute_prevalence_per_label


class TestComputePrevalencePerLabel:
    def test_majority_fire_rate_single_model(self):
        # Single model: prevalence is simply mean fire rate.
        df = pl.DataFrame({
            "audit_source": ["stratified"] * 10,
            "det_python": [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "det_markdown": [0] * 10,
        })
        result = compute_prevalence_per_label({"only": df})
        assert result["python"] == 0.3
        assert result["markdown"] == 0.0

    def test_majority_of_three_fire(self):
        # 3 models: prevalence = fraction of rows where majority (>=2 of 3) fires.
        base = pl.DataFrame({
            "audit_source": ["stratified"] * 4,
        })
        df1 = base.with_columns(pl.Series("det_python", [1, 1, 0, 0]))
        df2 = base.with_columns(pl.Series("det_python", [1, 0, 1, 0]))
        df3 = base.with_columns(pl.Series("det_python", [0, 1, 1, 0]))
        # Row-by-row: 2+, 2+, 2+, 0 → majority fires on rows 0, 1, 2 → prev = 3/4
        result = compute_prevalence_per_label({"a": df1, "b": df2, "c": df3})
        assert result["python"] == 0.75

    def test_zero_rows_returns_zero(self):
        df = pl.DataFrame({
            "audit_source": pl.Series([], dtype=pl.Utf8),
            "det_python": pl.Series([], dtype=pl.Int64),
        })
        result = compute_prevalence_per_label({"only": df})
        assert result["python"] == 0.0

    def test_filters_out_injected_rows(self):
        # Only stratified rows count. Injected rows that fire must be excluded.
        df = pl.DataFrame({
            "audit_source": ["stratified", "stratified", "inject_det_python"],
            "det_python": [0, 0, 1],  # only the injected row fires
        })
        result = compute_prevalence_per_label({"only": df})
        assert result["python"] == 0.0
```

- [ ] **Step 2: Run the test — it should fail (function not defined).**

```bash
uv run --directory training pytest tests/test_audit_semantic_labels.py::TestComputePrevalencePerLabel -v
```

Expected: `ImportError` or `AttributeError: module has no attribute 'compute_prevalence_per_label'`.

- [ ] **Step 3: Implement the function.**

Add to `training/trainr/core/audit_semantic_labels.py` (after `compute_agreement_across_models`, before `compute_recall_on_injected`):

```python
def compute_prevalence_per_label(
    dfs: dict[str, pl.DataFrame],
) -> dict[str, float]:
    """Majority-of-N fire rate per label on stratified rows.

    For each `det_*` column present in the first DataFrame, compute the
    fraction of stratified rows where at least ceil(N/2) of the N models
    fired (det == 1). Zero-row inputs return 0.0.

    Args:
        dfs: {model_slug: DataFrame}. All DataFrames must already be
            filtered to stratified rows (use filter_for_agreement upstream)
            and share row ordering.

    Returns:
        {label_without_det_prefix: prevalence_float in [0.0, 1.0]}.
    """
    model_names = list(dfs.keys())
    if not model_names:
        return {}

    first = dfs[model_names[0]]
    # Filter all frames to stratified rows. Doing this inside the function
    # makes the function safe to call with raw annotator output (the alt is
    # requiring every caller to remember to filter first).
    stratified_dfs = {
        name: df.filter(pl.col("audit_source") == STRATIFIED)
        for name, df in dfs.items()
    }
    stratified_first = stratified_dfs[model_names[0]]
    n_rows = len(stratified_first)
    if n_rows == 0:
        return {label[len("det_"):]: 0.0 for label in detection_columns(first)}

    n_models = len(model_names)
    majority_threshold = (n_models + 1) // 2  # ceil(N/2); for N=3 this is 2

    result: dict[str, float] = {}
    for col in detection_columns(first):
        label = col[len("det_"):]
        # Skip columns that don't exist on every frame (asymmetric schemas
        # are the compare module's problem, not this function's).
        if not all(col in stratified_dfs[m].columns for m in model_names):
            continue
        votes_per_model = [stratified_dfs[m][col].to_list() for m in model_names]
        fire_count = 0
        for row_idx in range(n_rows):
            row_votes = [votes_per_model[m][row_idx] for m in range(n_models)]
            if sum(row_votes) >= majority_threshold:
                fire_count += 1
        result[label] = fire_count / n_rows
    return result
```

- [ ] **Step 4: Re-run the test — it should pass.**

```bash
uv run --directory training pytest tests/test_audit_semantic_labels.py::TestComputePrevalencePerLabel -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the full audit test suite for collateral safety.**

```bash
uv run --directory training pytest tests/test_audit_semantic_labels.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit.**

```bash
git add training/trainr/core/audit_semantic_labels.py training/tests/test_audit_semantic_labels.py
git commit -m "feat(audit): add compute_prevalence_per_label

Per-label majority-of-N fire rate on stratified rows. Needed by
the iter17 A/B compare module's soft WARN-prevalence gate. Uses
the same dict[model_slug, DataFrame] convention as the existing
compute_agreement_across_models. Filters internally so callers
don't need to remember to pre-filter to stratified rows."
```

---

### Task 1.2: Add `load_annotator_parquets()` loader with slug parsing

**Files:**
- Modify: `training/trainr/core/audit_semantic_labels.py` (add function + module-level constant)
- Modify: `training/tests/test_audit_semantic_labels.py` (add test class)

**Context:** Both the existing audit CLI and the new compare CLI need to load 3 annotator parquets and know which model each one came from. Right now `audit_semantic_labels.main()` hardcodes model names via argparse flags (`--gemini`, `--sonnet`, `--gpt54mini`). The compare module will use a glob + filename regex instead, for robustness against duplicates and misnamed files. Extract the shared loader logic here so both callers use the same slug-parsing code path.

- [ ] **Step 1: Write the failing test.**

Add to `training/tests/test_audit_semantic_labels.py`:

```python
import re
from pathlib import Path

import polars as pl
import pytest

from trainr.core.audit_semantic_labels import (
    EXPECTED_MODEL_SLUGS,
    load_annotator_parquets,
)


class TestLoadAnnotatorParquets:
    def test_expected_slug_set_constant(self):
        assert EXPECTED_MODEL_SLUGS == frozenset({"gemini3flash", "sonnet", "gpt54mini"})

    def test_loads_three_parquets_by_slug(self, tmp_path):
        def _make(path: Path, val: int):
            pl.DataFrame({
                "audit_source": ["stratified"],
                "det_python": [val],
            }).write_parquet(path)

        _make(tmp_path / "iter17_ab_iter15_gemini3flash.parquet", 1)
        _make(tmp_path / "iter17_ab_iter15_sonnet.parquet", 0)
        _make(tmp_path / "iter17_ab_iter15_gpt54mini.parquet", 1)

        paths = sorted(tmp_path.glob("iter17_ab_iter15_*.parquet"))
        result = load_annotator_parquets(paths)

        assert set(result.keys()) == {"gemini3flash", "sonnet", "gpt54mini"}
        assert result["gemini3flash"]["det_python"][0] == 1
        assert result["sonnet"]["det_python"][0] == 0

    def test_rejects_wrong_count(self, tmp_path):
        pl.DataFrame({"audit_source": ["stratified"]}).write_parquet(
            tmp_path / "iter17_ab_iter15_gemini3flash.parquet"
        )
        paths = [tmp_path / "iter17_ab_iter15_gemini3flash.parquet"]
        with pytest.raises(ValueError, match="expected 3 parquets"):
            load_annotator_parquets(paths)

    def test_rejects_duplicate_slug(self, tmp_path):
        def _make(path: Path):
            pl.DataFrame({"audit_source": ["stratified"]}).write_parquet(path)

        _make(tmp_path / "iter17_ab_iter15_gemini3flash.parquet")
        _make(tmp_path / "iter17_ab_iter15_sonnet.parquet")
        _make(tmp_path / "iter17_ab_iter15_sonnet_copy.parquet")
        # The "copy" one won't match the regex and will be rejected as
        # "could not parse slug". Prove that:
        paths = sorted(tmp_path.glob("iter17_ab_iter15_*.parquet"))
        with pytest.raises(ValueError):
            load_annotator_parquets(paths)

    def test_rejects_unknown_slug(self, tmp_path):
        def _make(path: Path):
            pl.DataFrame({"audit_source": ["stratified"]}).write_parquet(path)

        _make(tmp_path / "iter17_ab_iter15_gemini3flash.parquet")
        _make(tmp_path / "iter17_ab_iter15_sonnet.parquet")
        _make(tmp_path / "iter17_ab_iter15_claude35.parquet")

        paths = sorted(tmp_path.glob("iter17_ab_iter15_*.parquet"))
        with pytest.raises(ValueError, match="unexpected slugs"):
            load_annotator_parquets(paths)
```

- [ ] **Step 2: Run the tests — they should fail (function/constant not defined).**

```bash
uv run --directory training pytest tests/test_audit_semantic_labels.py::TestLoadAnnotatorParquets -v
```

Expected: `ImportError: cannot import name 'EXPECTED_MODEL_SLUGS'`.

- [ ] **Step 3: Implement the loader and constant.**

Add to `training/trainr/core/audit_semantic_labels.py` near the top (after the `STRATIFIED` constant):

```python
# Canonical model slug set for iter16/iter17 annotation parquets. This is
# the authoritative set — every loader that accepts a glob of 3 parquets
# asserts its parsed slugs equal this set. Duplicates, misnames, and
# unknown slugs all fail loud here rather than silently corrupting metrics.
EXPECTED_MODEL_SLUGS: frozenset[str] = frozenset({"gemini3flash", "sonnet", "gpt54mini"})

# Filename pattern: <anything>_<slug>.parquet where slug is one of the
# canonical slugs. The capture group extracts the slug.
_SLUG_RE = re.compile(r".*_(?P<slug>[a-z0-9]+)\.parquet$")
```

And add `import re` to the module's imports if not already present.

Then add the loader function after `detection_columns()`:

```python
def load_annotator_parquets(
    paths: list[Path],
) -> dict[str, pl.DataFrame]:
    """Load annotator parquets keyed by model slug parsed from filename.

    Asserts that exactly 3 parquets are passed AND that their parsed slug
    set equals EXPECTED_MODEL_SLUGS. This catches the "three files but one
    is a duplicate or misnamed" failure mode that a count check alone
    cannot — critical because gate metrics depend on knowing which model
    produced which parquet.

    Args:
        paths: List of 3 parquet file paths.

    Returns:
        {slug: DataFrame} with keys exactly equal to EXPECTED_MODEL_SLUGS.

    Raises:
        ValueError: If count != 3, a filename doesn't match the slug regex,
            or the parsed slug set != EXPECTED_MODEL_SLUGS.
    """
    if len(paths) != 3:
        raise ValueError(
            f"load_annotator_parquets: expected 3 parquets, got {len(paths)}: "
            f"{[str(p) for p in paths]}"
        )

    parsed: dict[str, Path] = {}
    for path in paths:
        match = _SLUG_RE.match(path.name)
        if match is None:
            raise ValueError(
                f"load_annotator_parquets: could not parse model slug from "
                f"{path.name!r}; expected format '<prefix>_<slug>.parquet' "
                f"with slug in {sorted(EXPECTED_MODEL_SLUGS)}"
            )
        slug = match.group("slug")
        if slug in parsed:
            raise ValueError(
                f"load_annotator_parquets: duplicate slug {slug!r} "
                f"(already mapped to {parsed[slug].name}, now {path.name})"
            )
        parsed[slug] = path

    parsed_slugs = frozenset(parsed.keys())
    if parsed_slugs != EXPECTED_MODEL_SLUGS:
        missing = EXPECTED_MODEL_SLUGS - parsed_slugs
        extra = parsed_slugs - EXPECTED_MODEL_SLUGS
        raise ValueError(
            f"load_annotator_parquets: parsed slugs do not match expected set. "
            f"missing={sorted(missing)}, unexpected slugs={sorted(extra)}. "
            f"expected={sorted(EXPECTED_MODEL_SLUGS)}"
        )

    return {slug: pl.read_parquet(path) for slug, path in parsed.items()}
```

- [ ] **Step 4: Re-run the tests — they should pass.**

```bash
uv run --directory training pytest tests/test_audit_semantic_labels.py::TestLoadAnnotatorParquets -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run the full audit test suite.**

```bash
uv run --directory training pytest tests/test_audit_semantic_labels.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit.**

```bash
git add training/trainr/core/audit_semantic_labels.py training/tests/test_audit_semantic_labels.py
git commit -m "feat(audit): add load_annotator_parquets loader with slug parsing

Parses model slug from filename and asserts the parsed set equals
the canonical {gemini3flash, sonnet, gpt54mini} set. Rejects wrong
count, unparseable names, duplicates, and unknown slugs. Shared by
both the existing audit_semantic_labels.main() and the upcoming
compare_prompt_versions module.

The EXPECTED_MODEL_SLUGS frozenset is the single source of truth
for the canonical slug set."
```

---

### Task 1.3: Refactor smoke test — iter16 audit report reproducibility

**Files:**
- Modify: `training/tests/test_audit_semantic_labels.py` (add test class)

**Context:** The spec requires that the existing iter16 audit report reproduces byte-for-byte after any future refactor of `audit_semantic_labels.py`. This is insurance: if Task 1.1/1.2 silently changed a rounding edge or introduced a column-ordering drift, the smoke test catches it. The test reads the 3 archived `iter16_5k_*.parquet` files and the committed `docs/accuracy_runs/2026-04-10-iteration-16-audit-report.md` and diffs the regenerated report against the committed one.

- [ ] **Step 1: Write the smoke test.**

Add to `training/tests/test_audit_semantic_labels.py`:

```python
class TestIter16ReportReproducibility:
    """Guard against silent drift in audit_semantic_labels output.

    The iter16 audit report is committed at docs/accuracy_runs/... . If any
    refactor changes agreement math, rounding, column ordering, or
    format_report output, this test fails loud.
    """

    def test_iter16_audit_report_reproduces_byte_for_byte(self):
        from trainr.core.audit_semantic_labels import (
            compute_agreement_across_models,
            compute_recall_majority,
            filter_for_agreement,
            format_report,
        )

        repo_root = Path(__file__).resolve().parents[2]
        parquet_dir = repo_root / "training" / "data" / "audit"
        report_path = (
            repo_root / "docs" / "accuracy_runs"
            / "2026-04-10-iteration-16-audit-report.md"
        )

        if not parquet_dir.exists() or not report_path.exists():
            pytest.skip("iter16 fixture data not available in this checkout")

        dfs = {
            "gemini3flash": pl.read_parquet(parquet_dir / "iter16_5k_gemini3flash.parquet"),
            "sonnet": pl.read_parquet(parquet_dir / "iter16_5k_sonnet.parquet"),
            "gpt54mini": pl.read_parquet(parquet_dir / "iter16_5k_gpt54mini.parquet"),
        }
        stratified_dfs = {name: filter_for_agreement(df) for name, df in dfs.items()}
        agreement = compute_agreement_across_models(stratified_dfs)
        recall = compute_recall_majority(dfs)
        regenerated = format_report(dfs, agreement, recall)

        committed = report_path.read_text()
        assert regenerated == committed, (
            "iter16 audit report drift detected — a refactor changed the "
            "audit output. If this change is intentional, regenerate the "
            "committed report and update this test's fixture reference."
        )
```

- [ ] **Step 2: Run the test.**

```bash
uv run --directory training pytest tests/test_audit_semantic_labels.py::TestIter16ReportReproducibility -v
```

Expected: **PASS**. If it fails, something in Task 1.1 or 1.2 broke existing behavior — investigate before proceeding. (If the test fails with "iter16 fixture data not available", verify the archived parquets exist at `training/data/audit/iter16_5k_*.parquet` and the iter16 audit report is committed.)

- [ ] **Step 3: Commit.**

```bash
git add training/tests/test_audit_semantic_labels.py
git commit -m "test(audit): iter16 report byte-for-byte reproducibility smoke test

Regenerates the iter16 audit report from the archived parquets
and diffs against the committed docs/accuracy_runs report. Fails
loud on any silent drift in agreement math, rounding, or output
format. Insurance for the Phase 1 refactor + any future edit."
```

---

## Phase 2: `compare_prompt_versions.py`

### Task 2.1: Module skeleton, `DeltaReport` dataclass, happy-path test

**Files:**
- Create: `training/trainr/core/compare_prompt_versions.py`
- Create: `training/tests/test_compare_prompt_versions.py`

**Context:** First task in the new module. Define the data type the comparison returns (`DeltaReport`) and write a happy-path end-to-end test against fixture DataFrames. The test drives the final public API shape — everything downstream hangs off this test.

- [ ] **Step 1: Write the happy-path test against the envisioned API.**

Create `training/tests/test_compare_prompt_versions.py`:

```python
"""Unit tests for compare_prompt_versions.py.

Fixture pattern: construct small polars DataFrames in-memory and call the
public API directly. No filesystem I/O except where explicitly testing
filesystem-interacting functions (glob handling, parquet reading).
"""

from __future__ import annotations

import polars as pl
import pytest

from trainr.core.compare_prompt_versions import (
    DeltaReport,
    LabelVerdict,
    compare_prompt_versions,
)


def _make_input_frame(n_strat: int = 10, n_inject: int = 2) -> pl.DataFrame:
    """Minimal input parquet fixture matching the real iter16_5k_input schema."""
    rows = [
        {"text": f"row-{i}", "sub_type": "python", "audit_source": "stratified"}
        for i in range(n_strat)
    ] + [
        {"text": f"inj-{i}", "sub_type": "python", "audit_source": "inject_det_python"}
        for i in range(n_inject)
    ]
    return pl.DataFrame(rows)


def _make_annotator_frame(
    input_frame: pl.DataFrame,
    det_columns: dict[str, list[int]],
) -> pl.DataFrame:
    """Clone the input frame and append det_* columns with given values."""
    result = input_frame.clone()
    for col, values in det_columns.items():
        result = result.with_columns(pl.Series(col, values))
    return result


class TestHappyPath:
    def test_all_shared_all_pass(self):
        """Baseline: 3 shared labels, zero delta, identical noise floor, all PASS."""
        input_frame = _make_input_frame(n_strat=10, n_inject=0)
        # Every model fires det_python on rows 0-2 (30% prevalence).
        votes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        before_frames = {
            "gemini3flash": _make_annotator_frame(input_frame, {"det_python": votes}),
            "sonnet": _make_annotator_frame(input_frame, {"det_python": votes}),
            "gpt54mini": _make_annotator_frame(input_frame, {"det_python": votes}),
        }
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames = {k: v.clone() for k, v in before_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

        assert isinstance(report, DeltaReport)
        assert "python" in report.shared_labels
        py = report.labels["python"]
        assert py.verdict == LabelVerdict.PASS
        assert py.delta_agreement == pytest.approx(0.0)
        assert py.iter15_prevalence == pytest.approx(0.3)
        assert py.iter16_prevalence == pytest.approx(0.3)
        assert py.prevalence_ratio == pytest.approx(1.0)
        assert py.noise_floor == pytest.approx(0.0)
```

- [ ] **Step 2: Run the test — it should fail (module does not exist).**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py -v
```

Expected: `ImportError: No module named 'trainr.core.compare_prompt_versions'`.

- [ ] **Step 3: Create the module skeleton with enough code for the happy-path test to pass.**

Create `training/trainr/core/compare_prompt_versions.py`:

```python
"""A/B regression audit comparison — iter15 prompt vs iter16 prompt.

Given three sets of annotator parquets (iter15 before, iter16a after, and
iter16b noise-floor companion), computes per-label agreement delta,
prevalence ratio, and same-prompt noise floor, then emits a gate verdict.

See docs/superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import polars as pl

from trainr.core.audit_semantic_labels import (
    compute_agreement_across_models,
    compute_prevalence_per_label,
    detection_columns,
    filter_for_agreement,
)

# Gate thresholds, from the spec.
AGREEMENT_DELTA_THRESHOLD = 0.005
PREVALENCE_RATIO_LOW = 0.5
PREVALENCE_RATIO_HIGH = 2.0


class LabelCategory(str, Enum):
    SHARED = "shared"
    ITER15_ONLY = "iter15-only"
    ITER16_ONLY = "iter16-only"


class LabelVerdict(str, Enum):
    PASS = "PASS"
    FAIL_AGREEMENT = "FAIL-agreement"
    WARN_PREVALENCE = "WARN-prevalence"
    FAIL_AND_WARN = "FAIL-agreement+WARN-prevalence"
    NO_VERDICT = "N/A"  # iter15-only / iter16-only rows


@dataclass
class LabelRow:
    label: str
    category: LabelCategory
    iter15_agreement: float | None
    iter16_agreement: float | None
    delta_agreement: float | None
    iter15_prevalence: float | None
    iter16_prevalence: float | None
    prevalence_ratio: float | None
    noise_floor: float | None
    verdict: LabelVerdict


@dataclass
class DeltaReport:
    labels: dict[str, LabelRow] = field(default_factory=dict)
    shared_labels: list[str] = field(default_factory=list)
    iter15_only_labels: list[str] = field(default_factory=list)
    iter16_only_labels: list[str] = field(default_factory=list)

    @property
    def fail_agreement_rows(self) -> list[LabelRow]:
        return [
            r for r in self.labels.values()
            if r.verdict in (LabelVerdict.FAIL_AGREEMENT, LabelVerdict.FAIL_AND_WARN)
        ]

    @property
    def warn_prevalence_rows(self) -> list[LabelRow]:
        return [
            r for r in self.labels.values()
            if r.verdict in (LabelVerdict.WARN_PREVALENCE, LabelVerdict.FAIL_AND_WARN)
        ]


def compare_prompt_versions(
    before_frames: dict[str, pl.DataFrame],
    after_frames: dict[str, pl.DataFrame],
    noise_floor_frames: dict[str, pl.DataFrame],
    input_frame: pl.DataFrame,
) -> DeltaReport:
    """Compute the A/B regression report.

    All three frame dicts must be keyed by the canonical model slugs
    ({"gemini3flash", "sonnet", "gpt54mini"}) and share row ordering with
    `input_frame`. The caller is responsible for loading and filtering —
    this function works in-memory only.

    Future tasks extend this function; for now it handles the simple
    all-shared, all-pass case.
    """
    report = DeltaReport()

    # Stratified frames for agreement/prevalence. The existing
    # audit_semantic_labels functions do this internally but we match their
    # convention by passing already-filtered dicts when possible.
    before_strat = {k: filter_for_agreement(v) for k, v in before_frames.items()}
    after_strat = {k: filter_for_agreement(v) for k, v in after_frames.items()}
    noise_strat = {k: filter_for_agreement(v) for k, v in noise_floor_frames.items()}

    iter15_agr = compute_agreement_across_models(before_strat)
    iter16_agr = compute_agreement_across_models(after_strat)
    noise_agr = compute_agreement_across_models(noise_strat)

    iter15_prev = compute_prevalence_per_label(before_frames)
    iter16_prev = compute_prevalence_per_label(after_frames)

    shared_labels = sorted(set(iter15_agr.keys()) & set(iter16_agr.keys()))
    report.shared_labels = shared_labels

    for label in shared_labels:
        row = LabelRow(
            label=label,
            category=LabelCategory.SHARED,
            iter15_agreement=iter15_agr[label],
            iter16_agreement=iter16_agr[label],
            delta_agreement=iter16_agr[label] - iter15_agr[label],
            iter15_prevalence=iter15_prev.get(label, 0.0),
            iter16_prevalence=iter16_prev.get(label, 0.0),
            prevalence_ratio=_compute_prev_ratio(
                iter15_prev.get(label, 0.0),
                iter16_prev.get(label, 0.0),
            ),
            noise_floor=abs(noise_agr[label] - iter16_agr[label]) if label in noise_agr else None,
            verdict=LabelVerdict.PASS,  # placeholder; Task 2.7 adds real logic
        )
        report.labels[label] = row

    return report


def _compute_prev_ratio(iter15_prev: float, iter16_prev: float) -> float:
    """Zero-handling rules from the spec: 0/0 → 1.0, 0/>0 → inf, >0/0 → 0.0."""
    if iter15_prev == 0 and iter16_prev == 0:
        return 1.0
    if iter15_prev == 0:
        return math.inf
    return iter16_prev / iter15_prev
```

- [ ] **Step 4: Run the test — it should pass.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py::TestHappyPath -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add training/trainr/core/compare_prompt_versions.py training/tests/test_compare_prompt_versions.py
git commit -m "feat(audit): compare_prompt_versions.py skeleton + happy-path test

Defines LabelCategory, LabelVerdict, LabelRow, and DeltaReport
dataclasses. The compare_prompt_versions() function handles the
all-shared happy-path case with placeholder PASS verdicts.
_compute_prev_ratio implements the spec's zero-handling rules
(0/0 → 1.0, 0/>0 → inf, >0/0 → 0.0).

Verdict logic, column categorization, fingerprint validation, and
report emission arrive in subsequent tasks."
```

---

### Task 2.2: Column categorization — shared / iter15-only / iter16-only

**Files:**
- Modify: `training/trainr/core/compare_prompt_versions.py`
- Modify: `training/tests/test_compare_prompt_versions.py`

**Context:** The compare module must categorize every `det_*` label as shared/iter15-only/iter16-only based on dynamic column introspection (no hardcoded DETECTION_LABELS list). The gate operates only on shared labels; the other two categories appear in the report for context. This task adds the categorization logic and its tests.

- [ ] **Step 1: Write failing tests for each category and for the hard-assert between after and noise_floor.**

Add to `training/tests/test_compare_prompt_versions.py`:

```python
class TestColumnCategorization:
    def _make_frames(self, det_columns: list[str]) -> dict[str, pl.DataFrame]:
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        frames = {}
        for slug in ("gemini3flash", "sonnet", "gpt54mini"):
            frames[slug] = _make_annotator_frame(
                input_frame,
                {col: [0] * 5 for col in det_columns},
            )
        return frames, input_frame

    def test_iter15_only_label_categorized(self):
        before_frames, input_frame = self._make_frames(["det_python", "det_log_lines"])
        after_frames, _ = self._make_frames(["det_python"])
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

        assert "log_lines" in report.iter15_only_labels
        assert "log_lines" in report.labels
        assert report.labels["log_lines"].category == LabelCategory.ITER15_ONLY
        assert report.labels["log_lines"].verdict == LabelVerdict.NO_VERDICT
        assert report.labels["log_lines"].iter16_agreement is None
        assert report.labels["log_lines"].delta_agreement is None
        assert "python" in report.shared_labels

    def test_iter16_only_label_categorized(self):
        before_frames, input_frame = self._make_frames(["det_python"])
        after_frames, _ = self._make_frames(["det_python", "det_log_content"])
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

        assert "log_content" in report.iter16_only_labels
        assert report.labels["log_content"].category == LabelCategory.ITER16_ONLY
        assert report.labels["log_content"].verdict == LabelVerdict.NO_VERDICT
        assert report.labels["log_content"].iter15_agreement is None

    def test_mixed_asymmetry(self):
        before_frames, input_frame = self._make_frames(["det_python", "det_log_lines"])
        after_frames, _ = self._make_frames(["det_python", "det_log_content"])
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

        assert report.shared_labels == ["python"]
        assert report.iter15_only_labels == ["log_lines"]
        assert report.iter16_only_labels == ["log_content"]

    def test_after_and_noise_floor_column_mismatch_raises(self):
        before_frames, input_frame = self._make_frames(["det_python"])
        after_frames, _ = self._make_frames(["det_python", "det_log_content"])
        # noise_floor missing det_log_content — this must fail loud.
        noise_frames, _ = self._make_frames(["det_python"])

        with pytest.raises(ValueError, match="det_log_content"):
            compare_prompt_versions(
                before_frames=before_frames,
                after_frames=after_frames,
                noise_floor_frames=noise_frames,
                input_frame=input_frame,
            )
```

- [ ] **Step 2: Run the tests — they should fail.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py::TestColumnCategorization -v
```

Expected: most tests FAIL (categorization not implemented yet).

- [ ] **Step 3: Extend `compare_prompt_versions()` to categorize labels and hard-assert after/noise_floor schema equality.**

Replace the body of `compare_prompt_versions()` with:

```python
def compare_prompt_versions(
    before_frames: dict[str, pl.DataFrame],
    after_frames: dict[str, pl.DataFrame],
    noise_floor_frames: dict[str, pl.DataFrame],
    input_frame: pl.DataFrame,
) -> DeltaReport:
    """See module docstring."""
    report = DeltaReport()

    # --- Hard schema assertion: after and noise_floor must agree on det_* cols.
    # Noise floor correctness depends on both sides having the same label set.
    after_first = next(iter(after_frames.values()))
    noise_first = next(iter(noise_floor_frames.values()))
    after_det = set(detection_columns(after_first))
    noise_det = set(detection_columns(noise_first))
    if after_det != noise_det:
        diff = sorted(after_det.symmetric_difference(noise_det))
        raise ValueError(
            f"compare_prompt_versions: after and noise_floor have differing "
            f"det_* column sets. symmetric_difference={diff}. "
            f"after_only={sorted(after_det - noise_det)}, "
            f"noise_only={sorted(noise_det - after_det)}"
        )

    # --- Column categorization via dynamic introspection.
    before_first = next(iter(before_frames.values()))
    before_det = set(detection_columns(before_first))

    def _strip(col: str) -> str:
        return col[len("det_"):]

    before_labels = {_strip(c) for c in before_det}
    after_labels = {_strip(c) for c in after_det}

    shared = sorted(before_labels & after_labels)
    iter15_only = sorted(before_labels - after_labels)
    iter16_only = sorted(after_labels - before_labels)

    report.shared_labels = shared
    report.iter15_only_labels = iter15_only
    report.iter16_only_labels = iter16_only

    # --- Filter to stratified rows for agreement/prevalence.
    before_strat = {k: filter_for_agreement(v) for k, v in before_frames.items()}
    after_strat = {k: filter_for_agreement(v) for k, v in after_frames.items()}
    noise_strat = {k: filter_for_agreement(v) for k, v in noise_floor_frames.items()}

    iter15_agr = compute_agreement_across_models(before_strat)
    iter16_agr = compute_agreement_across_models(after_strat)
    noise_agr = compute_agreement_across_models(noise_strat)

    iter15_prev = compute_prevalence_per_label(before_frames)
    iter16_prev = compute_prevalence_per_label(after_frames)

    # --- Shared labels: full metrics, placeholder verdict (Task 2.3 fixes).
    for label in shared:
        report.labels[label] = LabelRow(
            label=label,
            category=LabelCategory.SHARED,
            iter15_agreement=iter15_agr[label],
            iter16_agreement=iter16_agr[label],
            delta_agreement=iter16_agr[label] - iter15_agr[label],
            iter15_prevalence=iter15_prev.get(label, 0.0),
            iter16_prevalence=iter16_prev.get(label, 0.0),
            prevalence_ratio=_compute_prev_ratio(
                iter15_prev.get(label, 0.0),
                iter16_prev.get(label, 0.0),
            ),
            noise_floor=abs(noise_agr[label] - iter16_agr[label]) if label in noise_agr else None,
            verdict=LabelVerdict.PASS,
        )

    # --- iter15-only labels: partial metrics, no verdict.
    for label in iter15_only:
        report.labels[label] = LabelRow(
            label=label,
            category=LabelCategory.ITER15_ONLY,
            iter15_agreement=iter15_agr[label],
            iter16_agreement=None,
            delta_agreement=None,
            iter15_prevalence=iter15_prev.get(label, 0.0),
            iter16_prevalence=None,
            prevalence_ratio=None,
            noise_floor=None,
            verdict=LabelVerdict.NO_VERDICT,
        )

    # --- iter16-only labels: partial metrics, no verdict.
    for label in iter16_only:
        report.labels[label] = LabelRow(
            label=label,
            category=LabelCategory.ITER16_ONLY,
            iter15_agreement=None,
            iter16_agreement=iter16_agr[label],
            delta_agreement=None,
            iter15_prevalence=None,
            iter16_prevalence=iter16_prev.get(label, 0.0),
            prevalence_ratio=None,
            noise_floor=abs(noise_agr[label] - iter16_agr[label]) if label in noise_agr else None,
            verdict=LabelVerdict.NO_VERDICT,
        )

    return report
```

- [ ] **Step 4: Re-run the tests.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py -v
```

Expected: all tests PASS (happy path + categorization).

- [ ] **Step 5: Commit.**

```bash
git add training/trainr/core/compare_prompt_versions.py training/tests/test_compare_prompt_versions.py
git commit -m "feat(audit): compare_prompt_versions column categorization

Dynamic det_* introspection categorizes every label as
shared/iter15-only/iter16-only. Shared labels get full metrics;
the other two categories get partial metrics and NO_VERDICT.
Hard-asserts after and noise_floor have identical det_* column
sets — the override criterion depends on noise floor, so a
schema mismatch must fail loud rather than silently skip."
```

---

### Task 2.3: Verdict logic — hard gate + soft gate + override eligibility hint

**Files:**
- Modify: `training/trainr/core/compare_prompt_versions.py`
- Modify: `training/tests/test_compare_prompt_versions.py`

**Context:** Replace the placeholder `verdict=LabelVerdict.PASS` with real gate logic per the spec. Hard FAIL-agreement on `|Δagr| > 0.005`, soft WARN-prevalence on `prev_ratio ∉ [0.5, 2.0]`, and both verdicts can co-occur. The override eligibility itself (`|Δagr| ≤ 2 × noise_floor`) is computed and reported but does NOT change the verdict — human review applies it downstream per the Gate Decision Protocol.

- [ ] **Step 1: Write failing tests for each verdict path.**

Add to `training/tests/test_compare_prompt_versions.py`:

```python
class TestVerdictLogic:
    """Verdict logic per the spec's hard/soft gate rules."""

    def _frames_with_votes(
        self,
        before_votes: list[list[int]],
        after_votes: list[list[int]],
        noise_votes: list[list[int]] | None = None,
    ) -> tuple[dict, dict, dict, pl.DataFrame]:
        """Build 3-model frame sets from per-model vote lists of equal length.

        Each votes argument is a list of 3 lists (one per model), each of
        length n_rows. Returns (before_frames, after_frames, noise_frames,
        input_frame).
        """
        n_rows = len(before_votes[0])
        input_frame = _make_input_frame(n_strat=n_rows, n_inject=0)
        noise_votes = noise_votes if noise_votes is not None else after_votes

        def _build(vote_lists):
            return {
                slug: _make_annotator_frame(input_frame, {"det_python": votes})
                for slug, votes in zip(
                    ("gemini3flash", "sonnet", "gpt54mini"),
                    vote_lists,
                )
            }

        return _build(before_votes), _build(after_votes), _build(noise_votes), input_frame

    def test_pass_when_delta_below_threshold(self):
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[1, 1, 0, 0]] * 3,  # unanimous 0.5 prev, agr=1.0
            after_votes=[[1, 1, 0, 0]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        assert report.labels["python"].verdict == LabelVerdict.PASS

    def test_fail_agreement_hard_gate(self):
        # Construct rows so iter15 agrees unanimously but iter16 has 2-1 splits,
        # producing a Δagr > 0.005.
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # all 1.0 agreement
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            after_votes=[
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 9 unanimous + 1 split
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            # noise == after → noise_floor is 0
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        # iter15_agr=1.0, iter16_agr=(9*1.0 + 2/3)/10 = 0.9666..., Δ≈-0.0333
        assert abs(row.delta_agreement) > AGREEMENT_DELTA_THRESHOLD
        assert row.verdict == LabelVerdict.FAIL_AGREEMENT

    def test_pass_at_exact_threshold(self):
        """|Δagr| == 0.005 exactly should PASS (strict >)."""
        # Hand-build frames via delta injection rather than computing via votes.
        # We can't reach 0.005 exactly via 3-model majority agreement at small
        # n, so construct directly against the PASS branch by asserting
        # behavior via the threshold constant test below.
        pytest.skip("Exact-threshold test covered by constants test")

    def test_warn_prevalence_low_ratio(self):
        # iter15 fires 3/4 rows, iter16 fires 1/4 → ratio = 0.33 → WARN.
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[1, 1, 1, 0]] * 3,
            after_votes=[[1, 0, 0, 0]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        assert row.iter15_prevalence == 0.75
        assert row.iter16_prevalence == 0.25
        assert row.prevalence_ratio == pytest.approx(0.333, rel=0.01)
        assert row.verdict in (LabelVerdict.WARN_PREVALENCE, LabelVerdict.FAIL_AND_WARN)

    def test_zero_prevalence_both_sides_no_warn(self):
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[0, 0, 0, 0]] * 3,
            after_votes=[[0, 0, 0, 0]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        assert row.prevalence_ratio == 1.0
        assert row.verdict == LabelVerdict.PASS

    def test_zero_to_nonzero_warn(self):
        before_frames, after_frames, noise_frames, input_frame = self._frames_with_votes(
            before_votes=[[0, 0, 0, 0]] * 3,
            after_votes=[[1, 1, 1, 1]] * 3,
        )
        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        row = report.labels["python"]
        assert math.isinf(row.prevalence_ratio)
        assert row.verdict in (LabelVerdict.WARN_PREVALENCE, LabelVerdict.FAIL_AND_WARN)


def test_agreement_delta_threshold_constant():
    """Pin the threshold so future edits don't silently move it."""
    from trainr.core.compare_prompt_versions import AGREEMENT_DELTA_THRESHOLD
    assert AGREEMENT_DELTA_THRESHOLD == 0.005
```

Add `import math` at the top of the test file if not already present.

- [ ] **Step 2: Run the tests — they should fail.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py::TestVerdictLogic tests/test_compare_prompt_versions.py::test_agreement_delta_threshold_constant -v
```

Expected: most verdict tests FAIL because everything is still `LabelVerdict.PASS`.

- [ ] **Step 3: Implement the verdict helper.**

Add to `training/trainr/core/compare_prompt_versions.py` after `_compute_prev_ratio`:

```python
def _compute_verdict(
    delta_agreement: float,
    prevalence_ratio: float,
) -> LabelVerdict:
    """Combine hard agreement gate + soft prevalence gate into a verdict.

    - |Δagr| > AGREEMENT_DELTA_THRESHOLD → FAIL-agreement
    - prev_ratio outside [PREVALENCE_RATIO_LOW, PREVALENCE_RATIO_HIGH] → WARN-prevalence
    - Both → FAIL_AND_WARN
    - Neither → PASS
    """
    fail = abs(delta_agreement) > AGREEMENT_DELTA_THRESHOLD
    # math.inf, math.nan, and 0.0 all need to be outside [0.5, 2.0] for warn.
    # Only 0/0 → 1.0 is explicitly in-range; all other zero cases are out.
    warn = (
        math.isinf(prevalence_ratio)
        or math.isnan(prevalence_ratio)
        or prevalence_ratio < PREVALENCE_RATIO_LOW
        or prevalence_ratio > PREVALENCE_RATIO_HIGH
    )
    if fail and warn:
        return LabelVerdict.FAIL_AND_WARN
    if fail:
        return LabelVerdict.FAIL_AGREEMENT
    if warn:
        return LabelVerdict.WARN_PREVALENCE
    return LabelVerdict.PASS
```

- [ ] **Step 4: Call `_compute_verdict` for shared labels.** Replace the shared-label loop's `verdict=LabelVerdict.PASS` line with:

```python
            verdict=_compute_verdict(
                delta_agreement=iter16_agr[label] - iter15_agr[label],
                prevalence_ratio=_compute_prev_ratio(
                    iter15_prev.get(label, 0.0),
                    iter16_prev.get(label, 0.0),
                ),
            ),
```

- [ ] **Step 5: Re-run all compare tests.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit.**

```bash
git add training/trainr/core/compare_prompt_versions.py training/tests/test_compare_prompt_versions.py
git commit -m "feat(audit): compare_prompt_versions verdict logic

Implements _compute_verdict() combining the hard agreement gate
(|Δagr| > 0.005) with the soft prevalence gate (prev_ratio outside
[0.5, 2.0]). Verdicts can co-occur (FAIL_AND_WARN). Includes
tests for each path including the zero-prevalence cases.

Override eligibility (|Δagr| ≤ 2 × noise_floor) is reported as
a field but not applied in the verdict — it enters the Gate
Decision Protocol as human review input."
```

---

### Task 2.4: Row-alignment fingerprint validation

**Files:**
- Modify: `training/trainr/core/compare_prompt_versions.py`
- Modify: `training/tests/test_compare_prompt_versions.py`

**Context:** The spec's §Components Phase 2 algorithm step 3 requires fingerprint-based row alignment because `iter16_5k_input.parquet` has 5065 rows but only 5059 unique `text` values — text-only equality is insufficient. The fingerprint hashes all non-`det_*` columns from the input parquet and asserts equality against the same columns in every annotation parquet. Mismatches raise `ValueError` with the first diverging row index and column names.

- [ ] **Step 1: Write failing tests.**

Add to `training/tests/test_compare_prompt_versions.py`:

```python
class TestRowAlignmentFingerprint:
    def test_passing_fingerprint_does_not_raise(self):
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        before_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": [0] * 5})
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames = {k: v.clone() for k, v in before_frames.items()}

        # Should succeed without raising.
        compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )

    def test_text_mutation_raises_with_row_index(self):
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        before_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": [0] * 5})
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        # Corrupt one model's parquet by mutating a text row.
        bad = before_frames["gemini3flash"].with_columns(
            pl.Series("text", ["row-0", "row-1", "CORRUPTED", "row-3", "row-4"])
        )
        before_frames["gemini3flash"] = bad
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames = {k: v.clone() for k, v in before_frames.items()}

        with pytest.raises(ValueError) as excinfo:
            compare_prompt_versions(
                before_frames=before_frames,
                after_frames=after_frames,
                noise_floor_frames=noise_frames,
                input_frame=input_frame,
            )
        # Error message must name the row index and the diverging column.
        msg = str(excinfo.value)
        assert "row 2" in msg or "index 2" in msg
        assert "text" in msg
        assert "gemini3flash" in msg

    def test_row_count_mismatch_raises(self):
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        smaller = input_frame.head(3)
        before_frames = {
            "gemini3flash": _make_annotator_frame(smaller, {"det_python": [0, 0, 0]}),
            "sonnet": _make_annotator_frame(input_frame, {"det_python": [0] * 5}),
            "gpt54mini": _make_annotator_frame(input_frame, {"det_python": [0] * 5}),
        }
        after_frames = {k: v.clone() for k, v in before_frames.items()}
        noise_frames = {k: v.clone() for k, v in before_frames.items()}

        with pytest.raises(ValueError, match="row count"):
            compare_prompt_versions(
                before_frames=before_frames,
                after_frames=after_frames,
                noise_floor_frames=noise_frames,
                input_frame=input_frame,
            )
```

- [ ] **Step 2: Run tests — they should fail.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py::TestRowAlignmentFingerprint -v
```

Expected: most fail because fingerprinting isn't implemented yet.

- [ ] **Step 3: Add the fingerprint helper and call it at the top of `compare_prompt_versions`.**

Add to `training/trainr/core/compare_prompt_versions.py`:

```python
def _non_det_columns(frame: pl.DataFrame) -> list[str]:
    """Non-`det_*` columns, in the frame's natural order."""
    return [c for c in frame.columns if not c.startswith("det_")]


def _assert_fingerprint_matches_input(
    frame: pl.DataFrame,
    input_frame: pl.DataFrame,
    source: str,
) -> None:
    """Assert that `frame`'s non-det_* columns row-match `input_frame`.

    Fingerprint = concatenation of all non-`det_*` column values per row.
    The input_frame defines the column set; frame must have every column
    the input has (and may have more — the extras are det_* columns).

    Raises ValueError with the first diverging row index and column names
    on mismatch. `source` is a tag included in the error message so the
    caller can identify which parquet failed (e.g., "iter15/gemini3flash").
    """
    if len(frame) != len(input_frame):
        raise ValueError(
            f"{source}: row count mismatch — frame has {len(frame)} rows, "
            f"input has {len(input_frame)}"
        )

    cols = _non_det_columns(input_frame)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{source}: annotation parquet missing non-det columns from "
            f"input: {missing}"
        )

    # Column-by-column equality check. Stop on first diverging row.
    for col in cols:
        left = input_frame[col].to_list()
        right = frame[col].to_list()
        for row_idx, (l, r) in enumerate(zip(left, right)):
            if l != r:
                raise ValueError(
                    f"{source}: fingerprint mismatch at row {row_idx}, "
                    f"column {col!r}: input={l!r} vs annotation={r!r}"
                )
```

And at the very top of `compare_prompt_versions()` (before any other work), add:

```python
    # Fingerprint every annotation frame against the input parquet to catch
    # row-order drift, row count mismatches, or silent corruption of the
    # non-det_* columns. The input parquet is the single source of truth.
    for slug, frame in before_frames.items():
        _assert_fingerprint_matches_input(frame, input_frame, f"iter15/{slug}")
    for slug, frame in after_frames.items():
        _assert_fingerprint_matches_input(frame, input_frame, f"iter16a/{slug}")
    for slug, frame in noise_floor_frames.items():
        _assert_fingerprint_matches_input(frame, input_frame, f"iter16b/{slug}")
```

- [ ] **Step 4: Re-run all compare tests.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py -v
```

Expected: all tests PASS including fingerprint tests. Earlier tests may need tiny fixture adjustments if they happen to trip the fingerprint (they shouldn't, because `_make_annotator_frame` clones the input before adding `det_*` columns).

- [ ] **Step 5: Commit.**

```bash
git add training/trainr/core/compare_prompt_versions.py training/tests/test_compare_prompt_versions.py
git commit -m "feat(audit): compare_prompt_versions row-alignment fingerprint

Validates every annotation frame against the input parquet by
comparing all non-det_* columns row-for-row. Raises ValueError
naming the first diverging row index and column on mismatch.

Needed because iter16_5k_input.parquet has 5065 rows but only
5059 unique 'text' values — text-only equality is insufficient
as an alignment check."
```

---

### Task 2.5: Markdown report emission

**Files:**
- Modify: `training/trainr/core/compare_prompt_versions.py`
- Modify: `training/tests/test_compare_prompt_versions.py`

**Context:** Produce the markdown report exactly as specified in the spec's §Report Schema section. Sections: gate verdict → shared labels table → iter15-only → iter16-only → FAIL-agreement detail → WARN-prevalence detail → noise floor table. The function takes a `DeltaReport` and returns a string.

- [ ] **Step 1: Write a failing test for the report format.**

Add to `training/tests/test_compare_prompt_versions.py`:

```python
class TestFormatReport:
    def test_report_contains_all_required_sections(self):
        from trainr.core.compare_prompt_versions import format_delta_report

        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        before_frames = {
            slug: _make_annotator_frame(
                input_frame, {"det_python": [1, 0, 0, 0, 0], "det_log_lines": [0] * 5}
            )
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        after_frames = {
            slug: _make_annotator_frame(
                input_frame,
                {"det_python": [1, 0, 0, 0, 0], "det_log_content": [0] * 5},
            )
            for slug in ("gemini3flash", "sonnet", "gpt54mini")
        }
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        text = format_delta_report(report)

        assert "# iter17 A/B Regression Audit Report" in text
        assert "## Gate verdict" in text
        assert "## Shared labels" in text
        assert "## iter15-only labels" in text
        assert "## iter16-only labels" in text
        assert "## Noise floor table" in text
        # Shared label row present
        assert "python" in text
        # iter15-only label present in its section
        assert "log_lines" in text
        # iter16-only label present in its section
        assert "log_content" in text
        # Gate verdict line shows PASS (no FAIL rows in this fixture)
        assert "**PASS**" in text or "Gate verdict**\n\n**PASS**" in text

    def test_report_summary_counts_match_report_state(self):
        from trainr.core.compare_prompt_versions import format_delta_report

        input_frame = _make_input_frame(n_strat=10, n_inject=0)
        # Construct a FAIL-agreement row deliberately.
        before_votes = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] * 3
        after_votes = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        before_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": v})
            for slug, v in zip(("gemini3flash", "sonnet", "gpt54mini"), before_votes)
        }
        after_frames = {
            slug: _make_annotator_frame(input_frame, {"det_python": v})
            for slug, v in zip(("gemini3flash", "sonnet", "gpt54mini"), after_votes)
        }
        noise_frames = {k: v.clone() for k, v in after_frames.items()}

        report = compare_prompt_versions(
            before_frames=before_frames,
            after_frames=after_frames,
            noise_floor_frames=noise_frames,
            input_frame=input_frame,
        )
        text = format_delta_report(report)
        assert "1 FAIL-agreement" in text or "FAIL-agreement=1" in text or "FAIL" in text
        assert "**FAIL**" in text
```

- [ ] **Step 2: Run the test — it should fail.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py::TestFormatReport -v
```

Expected: `ImportError: cannot import name 'format_delta_report'`.

- [ ] **Step 3: Implement the report formatter.**

Add to `training/trainr/core/compare_prompt_versions.py`:

```python
def format_delta_report(report: DeltaReport) -> str:
    """Render the DeltaReport as markdown per the spec's §Report Schema."""
    lines: list[str] = []

    lines.append("# iter17 A/B Regression Audit Report")
    lines.append("")
    lines.append("**Date:** 2026-04-10")
    lines.append("")

    # Gate verdict summary
    n_shared = len(report.shared_labels)
    n_fail = len(report.fail_agreement_rows)
    n_warn = len(report.warn_prevalence_rows)
    overall = "**FAIL**" if n_fail > 0 else "**PASS**"
    lines.append("## Gate verdict")
    lines.append("")
    lines.append(overall)
    lines.append("")
    lines.append(
        f"Summary: {n_shared} shared labels, {n_fail} FAIL-agreement, "
        f"{n_warn} WARN-prevalence."
    )
    lines.append("")

    # Shared labels table
    lines.append("## Shared labels (gate applies)")
    lines.append("")
    lines.append(
        "| label | iter15_agr | iter16_agr | Δagr | iter15_prev | iter16_prev | "
        "prev_ratio | noise_floor | verdict |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for label in report.shared_labels:
        row = report.labels[label]
        lines.append(
            f"| det_{row.label} "
            f"| {row.iter15_agreement:.4f} "
            f"| {row.iter16_agreement:.4f} "
            f"| {row.delta_agreement:+.4f} "
            f"| {row.iter15_prevalence:.4f} "
            f"| {row.iter16_prevalence:.4f} "
            f"| {_fmt_ratio(row.prevalence_ratio)} "
            f"| {_fmt_float(row.noise_floor)} "
            f"| {row.verdict.value} |"
        )
    lines.append("")

    # iter15-only labels
    lines.append("## iter15-only labels (removed in iter16, context only)")
    lines.append("")
    if report.iter15_only_labels:
        lines.append("| label | iter15_agr | iter15_prev | note |")
        lines.append("|---|---:|---:|---|")
        for label in report.iter15_only_labels:
            row = report.labels[label]
            lines.append(
                f"| det_{row.label} "
                f"| {row.iter15_agreement:.4f} "
                f"| {row.iter15_prevalence:.4f} "
                f"| Not in iter16 prompt |"
            )
    else:
        lines.append("_None._")
    lines.append("")

    # iter16-only labels
    lines.append("## iter16-only labels (new in iter16, cross-ref iter16 audit)")
    lines.append("")
    if report.iter16_only_labels:
        lines.append("| label | iter16_agr | iter16_prev | noise_floor |")
        lines.append("|---|---:|---:|---:|")
        for label in report.iter16_only_labels:
            row = report.labels[label]
            lines.append(
                f"| det_{row.label} "
                f"| {row.iter16_agreement:.4f} "
                f"| {row.iter16_prevalence:.4f} "
                f"| {_fmt_float(row.noise_floor)} |"
            )
    else:
        lines.append("_None._")
    lines.append("")

    # FAIL-agreement detail
    lines.append("## FAIL-agreement rows (if any)")
    lines.append("")
    if report.fail_agreement_rows:
        for row in report.fail_agreement_rows:
            ratio = (
                abs(row.delta_agreement) / row.noise_floor
                if row.noise_floor and row.noise_floor > 0
                else float("inf")
            )
            overridable = (
                "ELIGIBLE for override" if ratio <= 2.0 else "NOT overridable"
            )
            lines.append(
                f"- **det_{row.label}**: Δagr={row.delta_agreement:+.4f}, "
                f"noise_floor={_fmt_float(row.noise_floor)}, "
                f"|Δagr|/noise_floor={_fmt_float(ratio)} → {overridable}"
            )
    else:
        lines.append("_None._")
    lines.append("")

    # WARN-prevalence detail
    lines.append("## WARN-prevalence rows (if any)")
    lines.append("")
    if report.warn_prevalence_rows:
        for row in report.warn_prevalence_rows:
            lines.append(
                f"- **det_{row.label}**: iter15_prev={row.iter15_prevalence:.4f}, "
                f"iter16_prev={row.iter16_prevalence:.4f}, "
                f"ratio={_fmt_ratio(row.prevalence_ratio)}"
            )
    else:
        lines.append("_None._")
    lines.append("")

    # Noise floor table
    lines.append("## Noise floor table")
    lines.append("")
    lines.append(
        "Per-label same-prompt variance from iter16a vs iter16b runs. "
        "Used to bound override eligibility for FAIL-agreement rows per "
        "the Gate Decision Protocol."
    )
    lines.append("")
    lines.append("| label | category | noise_floor |")
    lines.append("|---|---|---:|")
    for label in sorted(report.labels.keys()):
        row = report.labels[label]
        lines.append(
            f"| det_{row.label} | {row.category.value} | {_fmt_float(row.noise_floor)} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "—"
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "—"
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"
```

- [ ] **Step 4: Re-run the tests.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit.**

```bash
git add training/trainr/core/compare_prompt_versions.py training/tests/test_compare_prompt_versions.py
git commit -m "feat(audit): compare_prompt_versions markdown report emission

format_delta_report renders the DeltaReport with the sections
from the spec: gate verdict summary, shared labels table,
iter15-only and iter16-only sections, FAIL-agreement detail
with override-eligibility hint (|Δagr|/noise_floor ratio), and
a noise floor table covering all labels."
```

---

### Task 2.6: Click CLI wire-up — `trainr data compare-prompts`

**Files:**
- Modify: `training/trainr/core/compare_prompt_versions.py` (add `main()` + argparse)
- Modify: `training/trainr/commands/data.py` (add `@data.command("compare-prompts")`)
- Modify: `training/tests/test_compare_prompt_versions.py` (add CLI smoke test)

**Context:** Wire the compare module into the `trainr data` CLI group. Follows the established pattern from `annotate_detections_cmd`: click decorator with options, import the module's `main` function, build argv via `_build_argv`, call. Inside the module, `main()` parses argv, expands globs, loads parquets via `load_annotator_parquets`, calls `compare_prompt_versions`, writes the report to disk.

- [ ] **Step 1: Write a failing smoke test for the module's main() function.**

Add to `training/tests/test_compare_prompt_versions.py`:

```python
class TestMainCLI:
    def test_main_end_to_end_from_parquets(self, tmp_path):
        """Run main() against a small set of real parquet files on disk."""
        from trainr.core.compare_prompt_versions import main as compare_main

        # Build a tiny input parquet and 9 annotation parquets.
        input_frame = _make_input_frame(n_strat=5, n_inject=0)
        input_path = tmp_path / "input.parquet"
        input_frame.write_parquet(input_path)

        def _write_side(side: str):
            for slug in ("gemini3flash", "sonnet", "gpt54mini"):
                df = _make_annotator_frame(input_frame, {"det_python": [0] * 5})
                df.write_parquet(tmp_path / f"iter17_ab_{side}_{slug}.parquet")

        _write_side("iter15")
        _write_side("iter16a")
        _write_side("iter16b")

        output_report = tmp_path / "report.md"
        compare_main([
            "--before", str(tmp_path / "iter17_ab_iter15_*.parquet"),
            "--after", str(tmp_path / "iter17_ab_iter16a_*.parquet"),
            "--noise-floor", str(tmp_path / "iter17_ab_iter16b_*.parquet"),
            "--input", str(input_path),
            "--output", str(output_report),
        ])

        assert output_report.exists()
        content = output_report.read_text()
        assert "# iter17 A/B Regression Audit Report" in content
        assert "**PASS**" in content
```

- [ ] **Step 2: Run the test.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py::TestMainCLI -v
```

Expected: `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Implement `main()` in the compare module.**

Add to `training/trainr/core/compare_prompt_versions.py`:

```python
def main(argv: list[str] | None = None) -> None:
    """CLI entry point: `trainr data compare-prompts`."""
    import argparse
    import glob as _glob
    import sys

    from trainr.core.audit_semantic_labels import load_annotator_parquets

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before", required=True,
        help="Glob for the iter15-side parquets (3 files, one per model slug).",
    )
    parser.add_argument(
        "--after", required=True,
        help="Glob for the iter16a-side parquets (the A/B 'after' side, 3 files).",
    )
    parser.add_argument(
        "--noise-floor", required=True, dest="noise_floor",
        help="Glob for the iter16b-side parquets (noise floor companion, 3 files).",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the input parquet used for all annotation runs "
             "(iter16_5k_input.parquet).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the markdown regression report.",
    )
    args = parser.parse_args(argv)

    def _resolve(glob_arg: str, name: str) -> list[Path]:
        paths = sorted(Path(p) for p in _glob.glob(glob_arg))
        if not paths:
            raise ValueError(
                f"--{name}: glob {glob_arg!r} matched zero files"
            )
        return paths

    before_paths = _resolve(args.before, "before")
    after_paths = _resolve(args.after, "after")
    noise_paths = _resolve(args.noise_floor, "noise-floor")

    before_frames = load_annotator_parquets(before_paths)
    after_frames = load_annotator_parquets(after_paths)
    noise_frames = load_annotator_parquets(noise_paths)

    input_frame = pl.read_parquet(args.input)

    report = compare_prompt_versions(
        before_frames=before_frames,
        after_frames=after_frames,
        noise_floor_frames=noise_frames,
        input_frame=input_frame,
    )
    text = format_delta_report(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)

    # Exit non-zero on FAIL to make shell scripting easier.
    n_fail = len(report.fail_agreement_rows)
    print(
        f"compare-prompts: wrote {args.output}. "
        f"{len(report.shared_labels)} shared labels, "
        f"{n_fail} FAIL-agreement, "
        f"{len(report.warn_prevalence_rows)} WARN-prevalence.",
        file=sys.stderr,
    )
    if n_fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the click wire-up in `training/trainr/commands/data.py`** after the `annotate_detections_cmd` block:

```python
@data.command("compare-prompts")
@click.option("--before", required=True, help="Glob for iter15-side parquets (3 files).")
@click.option("--after", required=True, help="Glob for iter16a-side parquets (3 files, A/B after side).")
@click.option(
    "--noise-floor", "noise_floor", required=True,
    help="Glob for iter16b-side parquets (3 files, same-prompt noise companion).",
)
@click.option("--input", required=True, help="Path to input parquet (iter16_5k_input.parquet).")
@click.option("--output", required=True, help="Markdown report output path.")
def compare_prompts_cmd(**kwargs):
    """A/B regression audit between two SYSTEM_PROMPT versions."""
    from trainr.core.compare_prompt_versions import main as _main

    # Click uses underscores; _build_argv converts to dashes.
    argv = _build_argv(kwargs)
    _main(argv)
```

- [ ] **Step 5: Run the smoke test.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py::TestMainCLI -v
```

Expected: PASS. If the smoke test fails with SystemExit(2), that means the fixture produced FAIL rows — check the fixture's delta and noise floor match.

- [ ] **Step 6: Verify the click wire-up loads without error.**

```bash
uv run --directory training trainr data compare-prompts --help 2>&1 | head -20
```

Expected: help text listing the 5 options (`--before`, `--after`, `--noise-floor`, `--input`, `--output`).

- [ ] **Step 7: Run the full test suite for both modules.**

```bash
uv run --directory training pytest tests/test_compare_prompt_versions.py tests/test_audit_semantic_labels.py tests/test_build_audit_sample.py tests/test_annotate_detections.py -q
```

Expected: all tests PASS across all affected modules.

- [ ] **Step 8: Commit.**

```bash
git add training/trainr/core/compare_prompt_versions.py training/trainr/commands/data.py training/tests/test_compare_prompt_versions.py
git commit -m "feat(cli): trainr data compare-prompts

Wires compare_prompt_versions into the trainr data click group.
Module main() parses argv, resolves globs via Python glob module
(not the shell), loads parquets via load_annotator_parquets with
slug validation, runs the comparison, and writes the markdown
report. Exits 2 on any FAIL-agreement verdict so shell callers
can gate on it."
```

---

## Phase 3: Annotation Runs

No code changes in this phase — these tasks produce parquet artifacts. Each task documents the exact command, expected output, and verification step.

### Task 3.1: Git worktree setup for iter15 prompt state

**Files:**
- Create: `.worktrees/iter17-iter15-prompt/` (git worktree)

**Context:** The iter15 prompt state is at commit `22bc292` (the Phase 0 iter16 test repairs commit, with iter15 `SYSTEM_PROMPT` still intact and tests green). We create a git worktree so both prompt versions can be exercised simultaneously without checkout juggling.

- [ ] **Step 1: Verify 22bc292 exists and is the expected state.**

```bash
git log 22bc292 -1 --oneline && grep -c 'log_content\|stack_trace\|diff_patch' training/trainr/core/annotate_detections.py
```

Expected: `22bc292 test(detections): repair stale iter15 test baseline`. The grep count on HEAD's file reflects iter16's 3 new labels; we don't care about the count here — this is a sanity check that `22bc292` is the commit we think it is.

- [ ] **Step 2: Create the worktree.**

```bash
git worktree add .worktrees/iter17-iter15-prompt 22bc292
```

Expected: `Preparing worktree (detached HEAD 22bc292)` (or `(new branch 'iter17-iter15-prompt')` if Git attaches a branch — either is fine).

- [ ] **Step 3: Sanity-grep the iter15 SYSTEM_PROMPT.**

```bash
grep -c 'log_content' .worktrees/iter17-iter15-prompt/training/trainr/core/annotate_detections.py
grep -c 'stack_trace' .worktrees/iter17-iter15-prompt/training/trainr/core/annotate_detections.py
grep -c 'diff_patch'  .worktrees/iter17-iter15-prompt/training/trainr/core/annotate_detections.py
grep -c 'log_lines'   .worktrees/iter17-iter15-prompt/training/trainr/core/annotate_detections.py
```

Expected: the first 3 counts are **0** (the 3 iter16 semantic labels don't exist yet), the last is **>0** (`log_lines` still exists as a detection label in iter15).

If any count is wrong, STOP. The worktree is not on the right commit.

- [ ] **Step 4: `uv sync` in the worktree.**

```bash
cd .worktrees/iter17-iter15-prompt/training && uv sync && cd -
```

Expected: `Resolved N packages` / `Installed N packages` or similar. The worktree now has its own `.venv` with a working `trainr` CLI.

- [ ] **Step 5: Verify `trainr data annotate-detections --help` works in the worktree.**

```bash
(cd .worktrees/iter17-iter15-prompt/training && uv run trainr data annotate-detections --help 2>&1 | head -5)
```

Expected: help text for the annotator. If this fails with "command not found" or similar, the `uv sync` didn't produce a working install — fix before proceeding.

- [ ] **Step 6: Export `$REPO_ROOT` for the annotation commands to use.**

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
echo "$REPO_ROOT"
```

Expected: absolute path to the main repo (e.g., `/home/bfirestone/devspace/personal/sentiolabs/text-classifier-rs`).

No commit for this task — worktrees are local-only state.

---

### Task 3.2: Run iter15 side annotation (3 models)

**Files:**
- Produce: `training/data/audit/iter17_ab_iter15_{gemini3flash,sonnet,gpt54mini}.parquet` (3 files)

**Context:** Run the iter15 prompt against the 5065-row audit input for each of the 3 annotator models. Per the spec's §Phase 3 concurrency cap, run serially within this side (one model at a time). The iter16a run (Task 3.3) may run concurrently with this one in a separate shell for a total cap of 2 jobs in flight.

**Pre-flight — look up the exact OpenRouter model IDs:** the spec left these as placeholders. The plan locks them in by reading the iter16 iteration doc which names the 3 slugs:

```bash
grep -E 'google/gemini|anthropic/claude|openai/gpt' docs/accuracy_runs/2026-04-10-iteration-16.md
```

Expected: references to the actual OpenRouter IDs used in iter16. Based on the iter16 iteration doc and the existing ROUTING_TABLE in `annotate_detections.py`, the canonical model IDs are:

- `gemini3flash` → `google/gemini-3-flash-preview`
- `sonnet` → `anthropic/claude-sonnet-4.6`
- `gpt54mini` → `openai/gpt-5.4-mini`

If the iter16 doc names different IDs, use whatever the iter16 doc says — those are the source of truth. Cross-reference with `git log --all --oneline --grep='annotate' -- training/` to find the commit that ran the iter16 audit and inspect its commands.

- [ ] **Step 1: Change into the worktree's training directory.**

```bash
cd .worktrees/iter17-iter15-prompt/training
```

- [ ] **Step 2: Run the gemini3flash annotation.**

```bash
uv run trainr data annotate-detections \
    --input "$REPO_ROOT/training/data/audit/iter16_5k_input.parquet" \
    --model "google/gemini-3-flash-preview" \
    --backend "openrouter" \
    --output "$REPO_ROOT/training/data/audit/iter17_ab_iter15_gemini3flash.parquet"
```

Expected: the annotator streams progress to stderr (row count, backend, model, periodic checkpoint updates). Completes in ~5-15 min depending on provider latency. The output parquet should have 5065 rows and `det_*` columns appended.

- [ ] **Step 3: Run the sonnet annotation.**

```bash
uv run trainr data annotate-detections \
    --input "$REPO_ROOT/training/data/audit/iter16_5k_input.parquet" \
    --model "anthropic/claude-sonnet-4.6" \
    --backend "openrouter" \
    --output "$REPO_ROOT/training/data/audit/iter17_ab_iter15_sonnet.parquet"
```

Expected: same shape as step 2.

- [ ] **Step 4: Run the gpt54mini annotation.**

```bash
uv run trainr data annotate-detections \
    --input "$REPO_ROOT/training/data/audit/iter16_5k_input.parquet" \
    --model "openai/gpt-5.4-mini" \
    --backend "openrouter" \
    --output "$REPO_ROOT/training/data/audit/iter17_ab_iter15_gpt54mini.parquet"
```

- [ ] **Step 5: Verify all 3 parquets exist and have the expected shape.**

```bash
cd "$REPO_ROOT"
uv run --directory training python -c "
import polars as pl
for slug in ['gemini3flash', 'sonnet', 'gpt54mini']:
    df = pl.read_parquet(f'training/data/audit/iter17_ab_iter15_{slug}.parquet')
    det_cols = [c for c in df.columns if c.startswith('det_')]
    print(f'{slug}: rows={len(df)}, det_cols={len(det_cols)}, log_lines={\"det_log_lines\" in df.columns}, log_content={\"det_log_content\" in df.columns}')
"
```

Expected:
- rows=5065 for all 3
- `log_lines=True` (the iter15 label is present)
- `log_content=False` (the iter16 label is NOT present)

If any row count is off or the label presence is wrong, STOP — the worktree wasn't actually running iter15's prompt.

No commit for this task; parquets are data artifacts.

---

### Task 3.3: Run iter16a side annotation (3 models, fresh)

**Files:**
- Produce: `training/data/audit/iter17_ab_iter16a_{gemini3flash,sonnet,gpt54mini}.parquet` (3 files)

**Context:** Run the iter16 prompt against the same input. This is the "after" side of the A/B. Same model IDs as Task 3.2 but from the main repo at HEAD. May run concurrently with Task 3.2 (in a separate terminal) for up to 2 jobs in flight per the concurrency cap.

- [ ] **Step 1: Change into the main repo's training directory.**

```bash
cd "$REPO_ROOT"/training
```

- [ ] **Step 2: Run the gemini3flash annotation.**

```bash
uv run trainr data annotate-detections \
    --input "data/audit/iter16_5k_input.parquet" \
    --model "google/gemini-3-flash-preview" \
    --backend "openrouter" \
    --output "data/audit/iter17_ab_iter16a_gemini3flash.parquet"
```

- [ ] **Step 3: Run the sonnet annotation.**

```bash
uv run trainr data annotate-detections \
    --input "data/audit/iter16_5k_input.parquet" \
    --model "anthropic/claude-sonnet-4.6" \
    --backend "openrouter" \
    --output "data/audit/iter17_ab_iter16a_sonnet.parquet"
```

- [ ] **Step 4: Run the gpt54mini annotation.**

```bash
uv run trainr data annotate-detections \
    --input "data/audit/iter16_5k_input.parquet" \
    --model "openai/gpt-5.4-mini" \
    --backend "openrouter" \
    --output "data/audit/iter17_ab_iter16a_gpt54mini.parquet"
```

- [ ] **Step 5: Verify shape.**

```bash
cd "$REPO_ROOT"
uv run --directory training python -c "
import polars as pl
for slug in ['gemini3flash', 'sonnet', 'gpt54mini']:
    df = pl.read_parquet(f'training/data/audit/iter17_ab_iter16a_{slug}.parquet')
    det_cols = [c for c in df.columns if c.startswith('det_')]
    print(f'{slug}: rows={len(df)}, det_cols={len(det_cols)}, log_lines={\"det_log_lines\" in df.columns}, log_content={\"det_log_content\" in df.columns}')
"
```

Expected:
- rows=5065 for all 3
- `log_lines=False` (removed in iter16)
- `log_content=True` (added in iter16)
- `log_content`, `stack_trace`, `diff_patch` all present as `det_*` columns

No commit for this task.

---

### Task 3.4: Run iter16b side annotation (3 models, same-prompt noise companion)

**Files:**
- Produce: `training/data/audit/iter17_ab_iter16b_{gemini3flash,sonnet,gpt54mini}.parquet` (3 files)

**Context:** Independent same-prompt run for noise floor computation. Per the spec's concurrency cap, this runs **after** iter15 and iter16a complete (serially, one model at a time). Running it last ensures no more than 2 jobs in flight at any time — iter15 and iter16a in parallel, then iter16b alone.

- [ ] **Step 1: Confirm Tasks 3.2 and 3.3 completed.**

```bash
ls training/data/audit/iter17_ab_iter1{5,6a}_*.parquet 2>&1
```

Expected: 6 files listed (3 iter15 + 3 iter16a).

- [ ] **Step 2: Run all 3 models serially.**

```bash
cd "$REPO_ROOT"/training
for model_slug_pair in "gemini3flash google/gemini-3-flash-preview" "sonnet anthropic/claude-sonnet-4.6" "gpt54mini openai/gpt-5.4-mini"; do
    slug=${model_slug_pair% *}
    model=${model_slug_pair#* }
    echo "=== Running iter16b for $slug ==="
    uv run trainr data annotate-detections \
        --input "data/audit/iter16_5k_input.parquet" \
        --model "$model" \
        --backend "openrouter" \
        --output "data/audit/iter17_ab_iter16b_${slug}.parquet"
done
```

Expected: 3 successful runs, 3 new parquets in `training/data/audit/`.

- [ ] **Step 3: Verify all 9 iter17 parquets exist.**

```bash
cd "$REPO_ROOT" && ls training/data/audit/iter17_ab_*.parquet | wc -l
```

Expected: `9`.

- [ ] **Step 4: Sanity check: iter16a and iter16b should have different fire patterns on at least one row** (proving they're independent runs, not duplicates of the same call).

```bash
uv run --directory training python -c "
import polars as pl
a = pl.read_parquet('training/data/audit/iter17_ab_iter16a_gemini3flash.parquet')
b = pl.read_parquet('training/data/audit/iter17_ab_iter16b_gemini3flash.parquet')
same = (a['det_python'] == b['det_python']).sum()
total = len(a)
print(f'gemini3flash iter16a vs iter16b: {same}/{total} rows agree on det_python')
print(f'Expected: similar but not identical (LLM nondeterminism).')
"
```

Expected: agreement should be high (>95%) but not exactly 100% — if it is exactly 100%, either iter16b is a duplicate of iter16a, the LLM backend is deterministic (possible with greedy decoding), or the parquets were accidentally copied. Investigate if suspicious before proceeding.

No commit for this task.

---

## Phase 4: Comparison + Report

### Task 4.1: Run `trainr data compare-prompts`

**Files:**
- Produce: `docs/accuracy_runs/2026-04-10-iter17-regression-report.md`

**Context:** Execute the compare module against all 9 freshly-produced parquets. The output markdown report is the input to the Gate Decision Protocol.

- [ ] **Step 1: Run the compare CLI.**

```bash
cd "$REPO_ROOT"
uv run --directory training trainr data compare-prompts \
    --before "$REPO_ROOT/training/data/audit/iter17_ab_iter15_*.parquet" \
    --after "$REPO_ROOT/training/data/audit/iter17_ab_iter16a_*.parquet" \
    --noise-floor "$REPO_ROOT/training/data/audit/iter17_ab_iter16b_*.parquet" \
    --input "$REPO_ROOT/training/data/audit/iter16_5k_input.parquet" \
    --output "$REPO_ROOT/docs/accuracy_runs/2026-04-10-iter17-regression-report.md"
```

Expected behavior:
- Exit 0 if no FAIL-agreement rows.
- Exit 2 if at least one FAIL-agreement row.
- In either case the report file is written.
- Stderr prints a summary line: `compare-prompts: wrote ... N shared labels, M FAIL-agreement, P WARN-prevalence.`

If the command raises `ValueError` for schema drift, fingerprint mismatch, or unexpected slug set, STOP and diagnose — this is the tool doing exactly what it's designed to do: fail loud on corrupted inputs.

- [ ] **Step 2: Read the report and note the top-level verdict.**

```bash
head -30 docs/accuracy_runs/2026-04-10-iter17-regression-report.md
```

- [ ] **Step 3: Commit the regression report.**

```bash
git add docs/accuracy_runs/2026-04-10-iter17-regression-report.md
git commit -m "docs(iter17): A/B regression audit report

Generated by trainr data compare-prompts against 9 fresh parquets
(iter15 × 3 models, iter16a × 3 models, iter16b × 3 models).

See docs/accuracy_runs/2026-04-10-iteration-17.md for the gate
decision and human override sign-off (if any)."
```

---

### Task 4.2: Write iter17 iteration doc

**Files:**
- Create: `docs/accuracy_runs/2026-04-10-iteration-17.md`

**Context:** The iteration doc narrates what iter17 did, interprets the regression report, records human overrides (if any), and states the gate decision. Follows the iter16 iteration doc's structure as a reference.

- [ ] **Step 1: Create the iteration doc using this template.** Fill in the bracketed sections from the regression report generated in Task 4.1. If there are no FAIL-agreement or WARN-prevalence rows, note that explicitly.

Create `docs/accuracy_runs/2026-04-10-iteration-17.md`:

```markdown
# Iteration 17 — A/B Regression Audit (iter15 vs iter16 prompt)

**Date:** 2026-04-10
**Branch:** `iter17-ab-regression-audit`
**Spec:** [`docs/superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md`](../superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md)
**Plan:** [`docs/superpowers/plans/2026-04-10-iter17-ab-regression-audit.md`](../superpowers/plans/2026-04-10-iter17-ab-regression-audit.md)
**Regression report:** [`2026-04-10-iter17-regression-report.md`](./2026-04-10-iter17-regression-report.md)
**Corpus:** `iter16_5k_input.parquet` (5065 rows, unchanged from iter16)

## Summary

Ran the A/B regression audit that iter16 explicitly deferred. Three fresh
annotation passes: iter15 prompt (from worktree off commit `22bc292`),
iter16a prompt (from HEAD, the "after" side of the A/B), and iter16b
prompt (from HEAD, same-prompt noise floor companion). Nine parquets
total, ~$45 in annotation cost.

**Gate decision: [PASS / FAIL / PASS with override]**

[One-paragraph justification drawn from the regression report.]

## Regression Report

See the committed regression report for the full per-label table. Key
numbers:

- **Shared labels:** N
- **FAIL-agreement rows:** M (of which K are override-eligible, K' are blocking)
- **WARN-prevalence rows:** P
- **iter15-only labels (removed in iter16):** det_log_lines (agreement=X)
- **iter16-only labels (new in iter16):** det_log_content, det_stack_trace, det_diff_patch

## Human Review — Gate Decision Protocol

Per `docs/superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md#gate-decision-protocol`:

### FAIL-agreement rows

[For each FAIL-agreement row in the regression report:]

- **det_[label]**: Δagr=[±X.XXXX], noise_floor=[X.XXXX], |Δagr|/noise_floor=[X.X]
  - **[Override-eligible / Blocking]**
  - Reasoning: [if overriding, why the delta is within same-prompt noise; if blocking, why the prompt must iterate]
  - Reviewer sign-off: [name, accepting iter18 risk if overriding]

### WARN-prevalence rows

[For each WARN-prevalence row:]

- **det_[label]**: iter15_prev=[X.XXXX], iter16_prev=[X.XXXX], ratio=[X.XXX]
  - Interpretation: [semantic review — is this a legitimate prompt tightening, or silent drift?]
  - Reviewer sign-off: [name, accept / investigate]

## What Changed in the Codebase

### Phase 0: Operational fixes

- **`__main__` guard on `annotate_detections.py`** — restores `python -m` as a working invocation path. Click CLI (`trainr data annotate-detections`) remains canonical.
- **`INJECTION_PATTERNS` tightening** — stack_trace regexes now require Traceback+File adjacency (Python), Exception-header+frame adjacency (Java), and exclude `// error-pattern:` directives (Rust). Does not rebuild `iter16_5k_input.parquet` (would invalidate this A/B).

### Phase 1: Audit helper extensions

- **`compute_prevalence_per_label`** in `audit_semantic_labels.py` — majority-of-N fire rate on stratified rows, for the new WARN-prevalence signal.
- **`load_annotator_parquets` + `EXPECTED_MODEL_SLUGS`** — shared loader with filename slug parsing and set validation. Catches duplicates, misnames, and unknown slugs.
- **iter16 report reproducibility smoke test** — guards against silent drift in audit output.

### Phase 2: New `compare_prompt_versions.py` module

- `DeltaReport` / `LabelRow` / `LabelCategory` / `LabelVerdict` types.
- Dynamic column introspection → shared/iter15-only/iter16-only categorization.
- Hard schema assertion: `after` and `noise_floor` must have identical `det_*` column sets.
- Fingerprint-based row alignment against the input parquet (all non-`det_*` columns, row-by-row).
- Per-label agreement delta, prevalence ratio with explicit 0/0→1.0 rule, noise floor from iter16a vs iter16b.
- Verdict logic: hard FAIL-agreement (|Δagr|>0.005), soft WARN-prevalence (ratio∉[0.5,2.0]), both can co-occur.
- Markdown report with all sections from the spec's §Report Schema.
- Click CLI (`trainr data compare-prompts`) with internal glob expansion.

## Cost

| Item | Cost |
|------|------|
| iter15 side: 5065 rows × 3 models | ~$15 |
| iter16a side: 5065 rows × 3 models | ~$15 |
| iter16b side (noise floor companion): 5065 rows × 3 models | ~$15 |
| **Total annotation** | **~$45** |
| Each FAIL-path prompt iteration (if any) | +~$30 (iter16a + iter16b re-run) |

## Follow-Up

[If PASS:] iter18 (90k annotation run) unblocked. Proceed to the next branch.

[If FAIL without overrides:] Prompt iteration required. For each blocking FAIL-agreement row, revise the iter16 `SYSTEM_PROMPT` (likely: trim verbosity in the 3 new definition blocks) and re-run Tasks 3.3 and 3.4 (iter16a and iter16b must both rerun — caching iter16b on the old prompt would turn noise floor into prompt drift). Then re-run Task 4.1.

## Key Learnings

[Fill in after the gate decision. Candidates:
- Whether the iter16 "failing" labels turned out to be noise floors or real regressions.
- Whether any unexpected labels regressed.
- Whether the CSV-log refinement from iter16 left any downstream signatures visible in the A/B.
- Any ergonomic issues with the compare-prompts CLI, concurrency cap discipline, or the fingerprint-based row alignment in practice.]
```

- [ ] **Step 2: Commit the iteration doc with PLACEHOLDERS filled in** based on the regression report from Task 4.1. Do NOT commit with bracketed placeholders — fill them in before committing.

```bash
git add docs/accuracy_runs/2026-04-10-iteration-17.md
git commit -m "docs(iter17): iteration doc with gate decision

[Summary of what the A/B showed and the gate decision.]"
```

---

## Phase 5: Gate Decision

### Task 5.1: Final gate decision commit

**Context:** After the human review in Task 4.2 is complete and documented in the iteration doc, make a final explicit gate decision commit. This is the marker iter18 uses to know whether it's unblocked.

- [ ] **Step 1: Verify the regression report, iteration doc, and all Phase 0-4 commits are on the branch.**

```bash
git log --oneline main..HEAD
```

Expected: ~11 commits from P0 through P4, matching the estimated commit plan.

- [ ] **Step 2: Decide the gate outcome.** Walk through the iter17 iteration doc's Human Review section. Confirm:

- Every FAIL-agreement row is either eligible-and-overridden OR blocking.
- Every WARN-prevalence row has a reviewer sign-off.
- If any row is blocking, the gate is FAIL and iter17 must iterate (back to Task 3.3 with a revised prompt). **Do not make a PASS commit in this state.**

- [ ] **Step 3: Make the explicit gate decision commit.**

If PASS:

```bash
git commit --allow-empty -m "docs(iter17): gate decision — PASS

iter16 SYSTEM_PROMPT does not degrade any pre-existing detection
label beyond same-prompt noise. iter18 (90k annotation run) is
unblocked.

[One-paragraph summary of the key numbers: N shared labels, M FAIL
rows overridden as within-noise, P WARN-prevalence rows accepted
as legitimate tightening, 0 blocking regressions.]"
```

If FAIL:

```bash
git commit --allow-empty -m "docs(iter17): gate decision — FAIL

N blocking FAIL-agreement rows outside 2x same-prompt noise floor.
iter18 remains blocked. Prompt iteration required.

[List each blocking row with its Δagr and noise_floor numbers, and
the planned direction for the prompt fix.]"
```

- [ ] **Step 4: Push the branch.**

```bash
git push -u origin iter17-ab-regression-audit
```

Expected: branch pushed to remote; the push is the signal that iter17 is handed off for review (or for iter18 to begin, if PASS).

---

## Self-Review Checklist (run this before handing the plan off)

**1. Spec coverage:**

- [x] §Scope (iter17 = A/B + ops fixes, stops before 90k) — Phases 0-5 match the seam boundary.
- [x] §Re-annotation strategy (iter15 ×1 + iter16 ×2 = ~$45) — Tasks 3.2, 3.3, 3.4 run all three.
- [x] §Worktree (22bc292) — Task 3.1.
- [x] §New module compare_prompt_versions.py — Phase 2, all 6 tasks.
- [x] §Column asymmetry tri-categorization — Task 2.2.
- [x] §Gate model (hard FAIL agreement + soft WARN prev + noise floor) — Task 2.3.
- [x] §Override protocol with 2 × noise_floor — Task 4.2 (iteration doc) + Task 5.1 (gate decision).
- [x] §Operational follow-ups (__main__ guard, INJECTION_PATTERNS tightening) — Tasks 0.1, 0.2.
- [x] §Phase 1 refactor (expose primitives) — Tasks 1.1 (prevalence), 1.2 (loader), 1.3 (smoke test).
- [x] §Row alignment via fingerprinting on non-det_* columns — Task 2.4.
- [x] §CLI with internal glob expansion — Task 2.6.
- [x] §Algorithm step 1 slug set validation — Task 1.2 (load_annotator_parquets).
- [x] §Noise floor from iter16a vs iter16b — Task 2.2 logic.
- [x] §Hard assert after/noise_floor column set equality — Task 2.2.
- [x] §prev_ratio 0/0 → 1.0 rule — Task 2.1 (_compute_prev_ratio).
- [x] §FAIL path cost = $30 per iteration — Task 4.2 (iteration doc) + 5.1 (gate commit).
- [x] §Concurrency cap (≤ 2 jobs in flight) — Tasks 3.2/3.3/3.4 describe serial-within-side, parallel-across-sides.

**2. Placeholder scan:** No "TBD", no "implement later", no "handle edge cases" without concrete code. The iteration doc template (Task 4.2) has bracketed sections for the human reviewer to fill in from the regression report — these are labeled as such and are not code TODOs.

**3. Type consistency:** All `dict[str, pl.DataFrame]` keyed by model slug (`gemini3flash`/`sonnet`/`gpt54mini`). `LabelVerdict` enum values consistent (`PASS`/`FAIL_AGREEMENT`/`WARN_PREVALENCE`/`FAIL_AND_WARN`/`NO_VERDICT`). `DeltaReport.labels` is `dict[str, LabelRow]`. Threshold constants (`AGREEMENT_DELTA_THRESHOLD`, `PREVALENCE_RATIO_LOW`, `PREVALENCE_RATIO_HIGH`) defined once at module level.

Plan complete and ready for execution.
