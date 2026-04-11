# Semantic Detection Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3 cross-cutting semantic detection labels (`log_content`, `stack_trace`, `diff_patch`), remove `det_log_lines`, and validate via 5k audit before the downstream $400-600 90k annotation run.

**Architecture:** Prompt-only change to `training/trainr/core/annotate_detections.py`. The 3 new labels are the first detection-only labels (not mirrored by `ContentSubType`). Validation is a 5k stratified audit + ~150 targeted positive injection with strict pass criteria: inter-annotator agreement ≥0.995, recall ≥0.90 on injected positives, zero rule violations in spot-check.

**Tech Stack:** Python 3.10+, uv, polars, openai (OpenRouter), anthropic, pytest. Runs under `training/` workstream only — no Rust/`src/` changes.

**Spec:** `docs/superpowers/specs/2026-04-10-semantic-detection-labels-design.md`

**Working directory for all commands below:** `training/` (the Python workstream lives there). When running `uv` commands, `cd training/` first unless noted.

---

## File Inventory

**Modified:**
- `training/trainr/core/annotate_detections.py` — `DETECTION_LABELS` list, `SYSTEM_PROMPT` text, JSON template; new `SEMANTIC_LABELS` module constant
- `training/tests/test_annotate_detections.py` — fix 3 stale iter15 tests, add regression-guard tests for new labels

**Created:**
- `training/trainr/core/build_audit_sample.py` — builds the 5k stratified + injection-pool input parquet
- `training/trainr/core/audit_semantic_labels.py` — computes agreement + recall metrics, outputs markdown report
- `training/tests/test_build_audit_sample.py` — smoke tests for the builder
- `docs/accuracy_runs/2026-04-10-iteration-16.md` — iteration report (written after the audit run)

---

## Phase 0 — Repair Baseline Tests (prep)

The existing `test_annotate_detections.py` has 3 failing tests carried over from iter15. They reference pre-iter15 label counts and ROUTING_TABLE assumptions that no longer hold. Fix them first so the baseline is green and subsequent TDD is not muddied by pre-existing noise.

### Task 0.1: Verify baseline failure state

**Files:**
- Read: `training/tests/test_annotate_detections.py`

- [ ] **Step 1: Capture current failing tests**

Run from `training/`:
```bash
uv run pytest tests/test_annotate_detections.py --tb=no -q
```

Expected: exactly 3 failures:
- `TestDetectionLabels::test_labels_defined`
- `TestRoutingTable::test_routing_table_covers_all_detection_labels`
- `TestRoutingAnnotation::test_routing_uses_correct_api_keys`

If any other tests fail, STOP and investigate — the plan assumes these 3 as the starting state.

### Task 0.2: Fix `test_labels_defined` (stale label count)

**Files:**
- Modify: `training/tests/test_annotate_detections.py` around the `TestDetectionLabels::test_labels_defined` method

- [ ] **Step 1: Replace the stale count assertion**

Replace:
```python
    def test_labels_defined(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        assert isinstance(DETECTION_LABELS, list)
        assert len(DETECTION_LABELS) == 29
```

With:
```python
    def test_labels_defined(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        assert isinstance(DETECTION_LABELS, list)
        # Current baseline: 40 labels (pre-iter16).
        # Phase 1 removes log_lines (→39), phases 3-5 add 3 semantic labels (→42).
        assert len(DETECTION_LABELS) == 40
```

- [ ] **Step 2: Run the single test to verify it passes**

```bash
uv run pytest tests/test_annotate_detections.py::TestDetectionLabels::test_labels_defined -v
```

Expected: PASS.

### Task 0.3: Fix `test_expected_labels_present` (stale label list)

**Files:**
- Modify: `training/tests/test_annotate_detections.py` — `test_expected_labels_present` method

- [ ] **Step 1: Update the expected list to match iter15's state**

Replace the existing `expected = [...]` block with:
```python
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
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_annotate_detections.py::TestDetectionLabels::test_expected_labels_present -v
```

Expected: PASS.

### Task 0.4: Fix `test_routing_table_covers_all_detection_labels` (architectural mismatch)

**Files:**
- Modify: `training/tests/test_annotate_detections.py` — `TestRoutingTable::test_routing_table_covers_all_detection_labels`

**Why:** `ROUTING_TABLE` is keyed by `sub_type`, not detection label. The iter15 labels (`csharp`, `swift`, ...) were added to `DETECTION_LABELS` but not to `ROUTING_TABLE` because they routed through fallback. The assertion was backwards — it should check that ROUTING_TABLE keys are a subset of sub_type-mirroring detection labels, NOT that every detection label has a routing entry. Phase 3-5 will add `log_content`/`stack_trace`/`diff_patch` which are explicitly detection-only and will never be in ROUTING_TABLE.

- [ ] **Step 1: Replace the assertion direction**

Replace:
```python
    def test_routing_table_covers_all_detection_labels(self):
        from trainr.core.annotate_detections import DETECTION_LABELS, ROUTING_TABLE

        for label in DETECTION_LABELS:
            assert label in ROUTING_TABLE, f"ROUTING_TABLE missing entry for: {label}"
```

With:
```python
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
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_annotate_detections.py::TestRoutingTable::test_routing_table_keys_are_valid_detection_labels -v
```

Expected: PASS.

### Task 0.5: Fix `test_routing_uses_correct_api_keys` (no anthropic entries)

**Files:**
- Modify: `training/tests/test_annotate_detections.py` — `TestRoutingAnnotation::test_routing_uses_correct_api_keys`

**Why:** iter15 migrated all ROUTING_TABLE entries to `openrouter` backend. The test searches for an anthropic entry and asserts one exists. That's a stale assumption. Make the test skip cleanly when no anthropic entry exists, so it becomes a no-op until a future iteration reintroduces one.

- [ ] **Step 1: Replace the hard assertion with a conditional skip**

Replace:
```python
        anthropic_sub = None
        for sub, (model, backend) in ROUTING_TABLE.items():
            if backend == "anthropic":
                anthropic_sub = sub
                break
        assert anthropic_sub is not None, "No anthropic entries in ROUTING_TABLE"
```

With:
```python
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
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_annotate_detections.py::TestRoutingAnnotation::test_routing_uses_correct_api_keys -v
```

Expected: SKIPPED.

### Task 0.6: Verify full baseline is green

- [ ] **Step 1: Run the full test file**

```bash
uv run pytest tests/test_annotate_detections.py -v
```

Expected: all green (some SKIPPED is acceptable).

- [ ] **Step 2: Commit**

```bash
git add training/tests/test_annotate_detections.py
git commit -m "test(detections): repair stale iter15 test baseline

- Update label count to 40 (was 29, pre-iter15).
- Refresh expected label list to include iter15's 9 language labels.
- Invert routing-table coverage assertion: ROUTING_TABLE keys must be
  in DETECTION_LABELS, not the reverse. Sub_types without an entry
  fall through to the default model by design, and iter16 introduces
  detection-only semantic labels that will never have routing entries.
- Skip test_routing_uses_correct_api_keys when ROUTING_TABLE has no
  anthropic entries (iter15 migrated everything to openrouter)."
```

---

## Phase 1 — Remove `log_lines` from Detection Labels

### Task 1.1: Write the failing tests for log_lines removal

**Files:**
- Modify: `training/tests/test_annotate_detections.py` — new test methods in `TestDetectionLabels`

- [ ] **Step 1: Append these tests to `class TestDetectionLabels`**

```python
    def test_log_lines_removed_from_detection_labels(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        assert "log_lines" not in DETECTION_LABELS, (
            "log_lines must be sub_type-only (not a detection label) after "
            "iter16. See docs/superpowers/specs/"
            "2026-04-10-semantic-detection-labels-design.md"
        )

    def test_log_lines_removed_from_system_prompt_label_list(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        # The inline label list line in SYSTEM_PROMPT begins with "Labels:"
        label_list_line = next(
            line for line in SYSTEM_PROMPT.splitlines() if line.startswith("Labels:")
        )
        assert "log_lines" not in label_list_line

    def test_log_lines_removed_from_json_template(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        # The JSON example template at the bottom should not include log_lines
        assert '"log_lines":' not in SYSTEM_PROMPT
```

- [ ] **Step 2: Also update `test_labels_defined` count expectation**

Change the assertion in `test_labels_defined` from `== 40` to `== 39`:
```python
        # After phase 1: log_lines removed → 39.
        # After phases 3-5: 3 semantic labels added → 42.
        assert len(DETECTION_LABELS) == 39
```

- [ ] **Step 3: Also update `test_expected_labels_present`**

Remove `"log_lines",` from the expected list inside `test_expected_labels_present`.

- [ ] **Step 4: Run the tests and verify they FAIL**

```bash
uv run pytest tests/test_annotate_detections.py::TestDetectionLabels -v
```

Expected: 4 failures (the three new tests plus `test_labels_defined`) because `log_lines` is still in the implementation.

### Task 1.2: Remove `log_lines` from `DETECTION_LABELS`

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — `DETECTION_LABELS` list (around lines 31-42)

- [ ] **Step 1: Delete `log_lines` from the list**

Change:
```python
    "json", "jsonl", "key_value", "log_lines",
]
```

To:
```python
    "json", "jsonl", "key_value",
]
```

### Task 1.3: Remove `log_lines` from SYSTEM_PROMPT label list line

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — `SYSTEM_PROMPT` constant around lines 98-101

- [ ] **Step 1: Delete `log_lines` from the inline list**

Find:
```
Labels: plain, markdown, rst, latex, python, javascript, typescript, rust, go, \
java, c_cpp, objc, csharp, powershell, ruby, php, swift, kotlin, r, lua, \
graphql, sql, shell, css, yaml, toml, ini, dockerfile, makefile, html, xml, \
sgml, csv, tsv, pipe_table, fixed_width, json, jsonl, key_value, log_lines
```

Change the last line to end with `key_value` instead of `key_value, log_lines`:
```
Labels: plain, markdown, rst, latex, python, javascript, typescript, rust, go, \
java, c_cpp, objc, csharp, powershell, ruby, php, swift, kotlin, r, lua, \
graphql, sql, shell, css, yaml, toml, ini, dockerfile, makefile, html, xml, \
sgml, csv, tsv, pipe_table, fixed_width, json, jsonl, key_value
```

### Task 1.4: Remove `log_lines` from SYSTEM_PROMPT JSON template

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — JSON template at the end of `SYSTEM_PROMPT`

- [ ] **Step 1: Delete `"log_lines": 0` from the template**

Find the JSON template block that ends in `...,  "key_value": 0, "log_lines": 0}` and remove the `"log_lines": 0`, leaving the closing brace:

```python
{"plain": 0, "markdown": 0, "rst": 0, "latex": 0, "python": 0, "javascript": 0, \
"typescript": 0, "rust": 0, "go": 0, "java": 0, "c_cpp": 0, "objc": 0, \
"csharp": 0, "powershell": 0, "ruby": 0, "php": 0, "swift": 0, "kotlin": 0, \
"r": 0, "lua": 0, "graphql": 0, "sql": 0, "shell": 0, "css": 0, "yaml": 0, \
"toml": 0, "ini": 0, "dockerfile": 0, "makefile": 0, "html": 0, "xml": 0, \
"sgml": 0, "csv": 0, "tsv": 0, "pipe_table": 0, "fixed_width": 0, "json": 0, \
"jsonl": 0, "key_value": 0}"""
```

### Task 1.5: Run all annotate_detections tests and commit

- [ ] **Step 1: Run the file**

```bash
uv run pytest tests/test_annotate_detections.py -v
```

Expected: all green.

- [ ] **Step 2: Commit**

```bash
git add training/trainr/core/annotate_detections.py training/tests/test_annotate_detections.py
git commit -m "feat(detections): remove det_log_lines (sub_type-only now)

log_lines remains a ContentSubType. The detection label was redundant
with the sub_type column and will be replaced by a proper cross-cutting
det_log_content in subsequent commits. No trained model depends on
det_log_lines (detection head has never been trained)."
```

---

## Phase 2 — Introduce `SEMANTIC_LABELS` Constant

Creates a module-level constant documenting which detection labels are cross-cutting (NOT mirrored by a `ContentSubType`). This constant drives the routing test and future label-set invariants.

### Task 2.1: Write failing test for `SEMANTIC_LABELS` constant

**Files:**
- Modify: `training/tests/test_annotate_detections.py` — new `TestSemanticLabels` class

- [ ] **Step 1: Append this class after `TestDetectionLabels`**

```python
class TestSemanticLabels:
    """Detection labels that are NOT mirrored by a ContentSubType.

    These are cross-cutting phenomena (stack traces, diffs, embedded
    log content) that can appear inside any primary sub_type. They
    are the first detection-only labels in the classifier.
    """

    def test_constant_exists(self):
        from trainr.core.annotate_detections import SEMANTIC_LABELS

        assert isinstance(SEMANTIC_LABELS, frozenset)

    def test_semantic_labels_are_subset_of_detection_labels(self):
        from trainr.core.annotate_detections import (
            DETECTION_LABELS, SEMANTIC_LABELS,
        )

        for label in SEMANTIC_LABELS:
            assert label in DETECTION_LABELS, (
                f"Semantic label {label!r} declared but not in DETECTION_LABELS"
            )

    def test_semantic_labels_are_not_in_routing_table(self):
        """Semantic (detection-only) labels have no sub_type, so they
        must not appear as keys in ROUTING_TABLE."""
        from trainr.core.annotate_detections import (
            ROUTING_TABLE, SEMANTIC_LABELS,
        )

        for label in SEMANTIC_LABELS:
            assert label not in ROUTING_TABLE, (
                f"Semantic label {label!r} must not be in ROUTING_TABLE "
                f"(ROUTING_TABLE is keyed by sub_type)"
            )
```

- [ ] **Step 2: Run and verify the tests FAIL with ImportError**

```bash
uv run pytest tests/test_annotate_detections.py::TestSemanticLabels -v
```

Expected: failures due to missing `SEMANTIC_LABELS` import.

### Task 2.2: Add the `SEMANTIC_LABELS` constant

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — after `DETECTION_LABELS` definition

- [ ] **Step 1: Insert the constant after `DETECTION_LABELS`**

Insert this block immediately after the `DETECTION_LABELS = [...]` definition (around line 42):

```python
# ---------------------------------------------------------------------------
# Semantic (cross-cutting) detection labels — NOT mirrored by ContentSubType.
# These fire when the corresponding phenomenon is embedded anywhere in the
# text, regardless of the row's primary sub_type. Added in iter16 (2026-04-10).
# ---------------------------------------------------------------------------

SEMANTIC_LABELS: frozenset[str] = frozenset()
"""Detection labels that have no corresponding ContentSubType.

Empty until phases 3-5 append log_content, stack_trace, diff_patch.
"""
```

- [ ] **Step 2: Run and verify tests pass**

```bash
uv run pytest tests/test_annotate_detections.py::TestSemanticLabels -v
```

Expected: PASS (empty frozenset vacuously satisfies all three tests).

- [ ] **Step 3: Commit**

```bash
git add training/trainr/core/annotate_detections.py training/tests/test_annotate_detections.py
git commit -m "feat(detections): introduce SEMANTIC_LABELS frozenset

Documents which detection labels are NOT mirrored by a ContentSubType.
Empty in this commit — populated in subsequent phases with log_content,
stack_trace, and diff_patch."
```

---

## Phase 3 — Add `log_content` Label

Each new semantic label gets its own phase (own commit) so that if the 5k audit reveals a prompt bug in one label, reverting/iterating is surgical.

### Task 3.1: Write failing tests for log_content

**Files:**
- Modify: `training/tests/test_annotate_detections.py` — new `TestLogContentLabel` class

- [ ] **Step 1: Append this class**

```python
class TestLogContentLabel:
    """Regression-guard tests for the log_content semantic detection label."""

    def test_in_detection_labels(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        assert "log_content" in DETECTION_LABELS

    def test_in_semantic_labels(self):
        from trainr.core.annotate_detections import SEMANTIC_LABELS

        assert "log_content" in SEMANTIC_LABELS

    def test_in_system_prompt_label_list(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        label_list_line = next(
            line for line in SYSTEM_PROMPT.splitlines() if line.startswith("Labels:")
        )
        assert "log_content" in label_list_line

    def test_in_json_template(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert '"log_content": 0' in SYSTEM_PROMPT

    def test_definition_includes_uppercase_severity_rule(self):
        """Critical precision-tightening rule — if this is removed or
        reworded, the false-positive rate on lowercase 'error'/'info'
        in prose will explode."""
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert "UPPERCASE or BRACKETED" in SYSTEM_PROMPT

    def test_definition_includes_density_rule(self):
        """Density rule: two or more log-shaped lines in sequence."""
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert "TWO OR MORE consecutive lines" in SYSTEM_PROMPT

    def test_definition_includes_pure_trace_carveout(self):
        """Stack traces alone should fire stack_trace but NOT log_content."""
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert "PURE stack trace" in SYSTEM_PROMPT
```

- [ ] **Step 2: Update `test_labels_defined` count**

Bump from `== 39` to `== 40`:
```python
        # After phase 1: 39. After phase 3: log_content added → 40.
        # After phases 4-5: 2 more semantic labels → 42.
        assert len(DETECTION_LABELS) == 40
```

- [ ] **Step 3: Verify the tests FAIL**

```bash
uv run pytest tests/test_annotate_detections.py::TestLogContentLabel -v
```

Expected: all 7 new tests fail.

### Task 3.2: Add log_content to `DETECTION_LABELS` and `SEMANTIC_LABELS`

**Files:**
- Modify: `training/trainr/core/annotate_detections.py`

- [ ] **Step 1: Append to `DETECTION_LABELS`**

Change the last line of `DETECTION_LABELS` from:
```python
    "json", "jsonl", "key_value",
]
```

To:
```python
    "json", "jsonl", "key_value",
    # Added iter16 (2026-04-10): cross-cutting semantic detection labels
    "log_content",
]
```

- [ ] **Step 2: Update `SEMANTIC_LABELS` frozenset**

Change:
```python
SEMANTIC_LABELS: frozenset[str] = frozenset()
```

To:
```python
SEMANTIC_LABELS: frozenset[str] = frozenset({"log_content"})
```

### Task 3.3: Add log_content to SYSTEM_PROMPT label list line

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — `SYSTEM_PROMPT`

- [ ] **Step 1: Append to the inline list**

Change the label list line ending from:
```
sgml, csv, tsv, pipe_table, fixed_width, json, jsonl, key_value
```

To:
```
sgml, csv, tsv, pipe_table, fixed_width, json, jsonl, key_value, \
log_content
```

### Task 3.4: Add the log_content definition block to SYSTEM_PROMPT

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — end of `## Important distinctions` section in `SYSTEM_PROMPT`

- [ ] **Step 1: Append the definition**

Find the end of the `## Important distinctions` section (after the `graphql` definition, before `## Rules`). Append this definition block immediately before `## Rules`:

```
- "log_content" = embedded log output (distinct from sub_type=log_lines, \
which means the row IS primarily logs). Requires TWO OR MORE consecutive \
lines that each independently match a log-line schema. A log-line schema is \
ONE of:
  * `<timestamp> <severity-or-component> <message>` where timestamp is ISO \
8601 (`2024-01-15T10:23:45Z`), apache-style (`[15/Jan/2024:10:23:45 +0000]`), \
syslog-style (`Jan 15 10:23:45`), or unix epoch (ms/seconds)
  * A recognized named format: nginx/apache access log, syslog \
(`<pri>timestamp host proc[pid]: msg`), dockerd/container log, logfmt \
(`key=value key2=value2`) ONLY when co-occurring with a timestamp OR \
severity field
  * JSON log records: >=2 consecutive JSON objects each containing BOTH a \
timestamp field AND a `level`/`severity` field
Severity tokens must be UPPERCASE or BRACKETED (`INFO`, `[info]`, `WARN`, \
`WARNING`, `ERROR`, `DEBUG`, `TRACE`, `FATAL`, `CRITICAL`) — lowercase \
`error`/`info` in prose or code identifiers does NOT count.
Do NOT fire on: single-line error messages (even inside code fences or \
blockquotes), code that CALLS a logger (`log.info(...)`), sentences \
describing logging behavior, changelogs with leading dates (`2024-01-15 - \
fixed bug`), CSV/TSV with date columns, `ls -la` output, git commit logs. \
Lines inside quotation marks or markdown blockquotes do not contribute to \
the density count.
If the embedded output is a PURE stack trace (no surrounding non-trace log \
lines), fire "stack_trace": 1 but NOT "log_content": 1. If a stack trace \
appears inside otherwise-normal log output, fire BOTH.
```

### Task 3.5: Add log_content to JSON template

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — JSON template

- [ ] **Step 1: Append to the JSON example**

Change:
```python
"jsonl": 0, "key_value": 0}"""
```

To:
```python
"jsonl": 0, "key_value": 0, "log_content": 0}"""
```

### Task 3.6: Run tests and commit

- [ ] **Step 1: Run the full file**

```bash
uv run pytest tests/test_annotate_detections.py -v
```

Expected: all green.

- [ ] **Step 2: Commit**

```bash
git add training/trainr/core/annotate_detections.py training/tests/test_annotate_detections.py
git commit -m "feat(detections): add log_content semantic label

First detection-only label — fires on embedded log output anywhere in a
row, distinct from sub_type=log_lines which means the row IS primarily
logs. Strict density+schema definition with uppercase-severity rule to
keep false positives bounded at the ~0.1% target class prevalence."
```

---

## Phase 4 — Add `stack_trace` Label

### Task 4.1: Write failing tests for stack_trace

**Files:**
- Modify: `training/tests/test_annotate_detections.py` — new `TestStackTraceLabel` class

- [ ] **Step 1: Append this class**

```python
class TestStackTraceLabel:
    """Regression-guard tests for the stack_trace semantic detection label."""

    def test_in_detection_labels(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        assert "stack_trace" in DETECTION_LABELS

    def test_in_semantic_labels(self):
        from trainr.core.annotate_detections import SEMANTIC_LABELS

        assert "stack_trace" in SEMANTIC_LABELS

    def test_in_system_prompt_label_list(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        label_list_line = next(
            line for line in SYSTEM_PROMPT.splitlines() if line.startswith("Labels:")
        )
        assert "stack_trace" in label_list_line

    def test_in_json_template(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert '"stack_trace": 0' in SYSTEM_PROMPT

    def test_definition_includes_two_frame_rule(self):
        """Multi-frame requirement is the load-bearing anti-false-positive
        rule for single-line errors and prose mentioning 'at line X'."""
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert "TWO OR MORE" in SYSTEM_PROMPT
        assert "frames" in SYSTEM_PROMPT

    def test_definition_includes_dotnet_format(self):
        """The .NET format (`   at Namespace.Class.Method() in File.cs:line 42`)
        was flagged as highest-value addition by standard review."""
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert ".NET" in SYSTEM_PROMPT

    def test_definition_mentions_cofire_with_log_content(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert 'fire BOTH' in SYSTEM_PROMPT
```

- [ ] **Step 2: Update `test_labels_defined` count**

Bump from `== 40` to `== 41`.

- [ ] **Step 3: Verify tests FAIL**

```bash
uv run pytest tests/test_annotate_detections.py::TestStackTraceLabel -v
```

Expected: failures.

### Task 4.2: Add stack_trace to the label set

**Files:**
- Modify: `training/trainr/core/annotate_detections.py`

- [ ] **Step 1: Append to `DETECTION_LABELS`**

Change the semantic labels section:
```python
    "json", "jsonl", "key_value",
    # Added iter16 (2026-04-10): cross-cutting semantic detection labels
    "log_content",
]
```

To:
```python
    "json", "jsonl", "key_value",
    # Added iter16 (2026-04-10): cross-cutting semantic detection labels
    "log_content", "stack_trace",
]
```

- [ ] **Step 2: Update `SEMANTIC_LABELS`**

Change:
```python
SEMANTIC_LABELS: frozenset[str] = frozenset({"log_content"})
```

To:
```python
SEMANTIC_LABELS: frozenset[str] = frozenset({"log_content", "stack_trace"})
```

### Task 4.3: Add stack_trace to SYSTEM_PROMPT label list line

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — `SYSTEM_PROMPT`

- [ ] **Step 1: Append to the inline list**

Change:
```
sgml, csv, tsv, pipe_table, fixed_width, json, jsonl, key_value, \
log_content
```

To:
```
sgml, csv, tsv, pipe_table, fixed_width, json, jsonl, key_value, \
log_content, stack_trace
```

### Task 4.4: Add the stack_trace definition block

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — `SYSTEM_PROMPT` `## Important distinctions` section

- [ ] **Step 1: Append after the log_content definition**

Insert immediately after the `log_content` definition block:

```
- "stack_trace" = programmatic stack trace / traceback. Requires TWO OR \
MORE frames where each frame carries a file:line locator OR a \
package.Class.method locator (not just an `at` keyword). Strong signals \
by language:
  * Python: `Traceback (most recent call last):` header + `File "foo.py", \
line N, in func` frames + final `ErrorType: message` line
  * Java/JVM: `Exception in thread "..."` + \
`at com.pkg.Class.method(File.java:42)` + optional `Caused by:` + \
`... N more`
  * .NET: `   at Namespace.Class.Method() in File.cs:line 42` (leading \
whitespace is distinctive)
  * JS/Node: `Error: message` + `at fn (file:line:col)` / `at async fn`
  * Go: `goroutine N [running]:` + `main.foo(0x0)\n\tpath/file.go:42 +0x1a`
  * Rust: `thread 'main' panicked at` + numbered backtrace frames `0: ...`, \
`1: ...` with file:line
  * Ruby: `from file.rb:42:in 'method'` chain (one frame per line)
Truncation markers (`... N more frames`, `[truncated]`) do not disqualify \
a trace. Do NOT fire on: single-line error messages, the literal phrase \
"stack trace" in prose, tutorial pseudocode describing what a trace looks \
like, profiler output tables, or prose like "at line 5 it crashed, at \
line 6 it retried". If this trace appears inside log output, fire BOTH \
"stack_trace": 1 AND "log_content": 1.
```

### Task 4.5: Add stack_trace to JSON template

- [ ] **Step 1: Append to the JSON example**

Change:
```python
"jsonl": 0, "key_value": 0, "log_content": 0}"""
```

To:
```python
"jsonl": 0, "key_value": 0, "log_content": 0, "stack_trace": 0}"""
```

### Task 4.6: Run tests and commit

- [ ] **Step 1: Run**

```bash
uv run pytest tests/test_annotate_detections.py -v
```

Expected: all green.

- [ ] **Step 2: Commit**

```bash
git add training/trainr/core/annotate_detections.py training/tests/test_annotate_detections.py
git commit -m "feat(detections): add stack_trace semantic label

Multi-frame programmatic traceback detection across Python, Java/JVM,
.NET, JS/Node, Go, Rust, and Ruby. Requires two or more frames with
file:line or package.Class.method locators — single-line 'Error: foo'
messages and prose 'at line 5' do not qualify. Co-fires with log_content
when embedded in log output."
```

---

## Phase 5 — Add `diff_patch` Label

### Task 5.1: Write failing tests for diff_patch

**Files:**
- Modify: `training/tests/test_annotate_detections.py` — new `TestDiffPatchLabel` class

- [ ] **Step 1: Append this class**

```python
class TestDiffPatchLabel:
    """Regression-guard tests for the diff_patch semantic detection label."""

    def test_in_detection_labels(self):
        from trainr.core.annotate_detections import DETECTION_LABELS

        assert "diff_patch" in DETECTION_LABELS

    def test_in_semantic_labels(self):
        from trainr.core.annotate_detections import SEMANTIC_LABELS

        assert "diff_patch" in SEMANTIC_LABELS

    def test_in_system_prompt_label_list(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        label_list_line = next(
            line for line in SYSTEM_PROMPT.splitlines() if line.startswith("Labels:")
        )
        assert "diff_patch" in label_list_line

    def test_in_json_template(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert '"diff_patch": 0' in SYSTEM_PROMPT

    def test_definition_includes_required_markers(self):
        """The REQUIRED marker set is the load-bearing anti-markdown-list
        rule. If this is loosened, bullet lists will dominate false
        positives at 0.1% class prevalence."""
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert "REQUIRED" in SYSTEM_PROMPT
        assert "@@ -X,Y +A,B @@" in SYSTEM_PROMPT
        assert "diff --git" in SYSTEM_PROMPT

    def test_definition_includes_adjacent_file_markers_rule(self):
        """--- and +++ must be on adjacent lines so YAML frontmatter
        doesn't false-fire."""
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert "adjacent lines" in SYSTEM_PROMPT

    def test_definition_excludes_markdown_bullet_lists(self):
        from trainr.core.annotate_detections import SYSTEM_PROMPT

        assert "markdown bullet lists" in SYSTEM_PROMPT
```

- [ ] **Step 2: Update `test_labels_defined` count**

Bump from `== 41` to `== 42`.

- [ ] **Step 3: Verify tests FAIL**

```bash
uv run pytest tests/test_annotate_detections.py::TestDiffPatchLabel -v
```

Expected: failures.

### Task 5.2: Add diff_patch to the label set

**Files:**
- Modify: `training/trainr/core/annotate_detections.py`

- [ ] **Step 1: Append to `DETECTION_LABELS`**

Change:
```python
    "json", "jsonl", "key_value",
    # Added iter16 (2026-04-10): cross-cutting semantic detection labels
    "log_content", "stack_trace",
]
```

To:
```python
    "json", "jsonl", "key_value",
    # Added iter16 (2026-04-10): cross-cutting semantic detection labels
    "log_content", "stack_trace", "diff_patch",
]
```

- [ ] **Step 2: Update `SEMANTIC_LABELS`**

Change:
```python
SEMANTIC_LABELS: frozenset[str] = frozenset({"log_content", "stack_trace"})
```

To:
```python
SEMANTIC_LABELS: frozenset[str] = frozenset(
    {"log_content", "stack_trace", "diff_patch"}
)
```

### Task 5.3: Add diff_patch to SYSTEM_PROMPT label list line

- [ ] **Step 1: Append**

Change the label list ending from:
```
log_content, stack_trace
```

To:
```
log_content, stack_trace, diff_patch
```

### Task 5.4: Add the diff_patch definition block

**Files:**
- Modify: `training/trainr/core/annotate_detections.py` — after stack_trace definition

- [ ] **Step 1: Append after the stack_trace definition**

Insert immediately after the `stack_trace` definition:

```
- "diff_patch" = unified diff or git patch format. REQUIRED: at least one \
of {`@@ -X,Y +A,B @@` hunk header, `diff --git a/... b/...` header, paired \
`--- a/path` / `+++ b/path` file markers on adjacent lines}. Line prefixes \
alone are NOT sufficient. Additional signals: `+`/`-`/space line prefixes \
within a hunk, `index abc1234..def5678 100644` git metadata. The \
email-patch header `From abc1234 Mon Sep 17 00:00:00 2001` is a signal \
ONLY when co-occurring with `---`/`+++` or `@@` markers (otherwise a \
regular email header). CRITICAL anti-signals: markdown bullet lists using \
`-` or `+`, pro/con lists, code containing arithmetic `+`/`-`, isolated \
`+`/`-` lines without hunk context, YAML frontmatter `---` without an \
adjacent `+++`. Unified diffs only — context diffs (`*** file ***` \
separators) out of scope.
```

### Task 5.5: Add diff_patch to JSON template

- [ ] **Step 1: Append**

Change:
```python
"log_content": 0, "stack_trace": 0}"""
```

To:
```python
"log_content": 0, "stack_trace": 0, "diff_patch": 0}"""
```

### Task 5.6: Run tests and commit

- [ ] **Step 1: Run full test file**

```bash
uv run pytest tests/test_annotate_detections.py -v
```

Expected: all green. Label count is now 42.

- [ ] **Step 2: Commit**

```bash
git add training/trainr/core/annotate_detections.py training/tests/test_annotate_detections.py
git commit -m "feat(detections): add diff_patch semantic label

Unified diff / git patch detection with strict required-marker rule: at
least one of (@@ hunk header, diff --git header, adjacent --- / +++ pair)
must be present. Line prefixes alone are insufficient — prevents markdown
bullet lists from dominating false positives at 0.1% class prevalence."
```

---

## Phase 6 — Audit Sample Builder

### Task 6.1: Write smoke test for `build_audit_sample.py`

**Files:**
- Create: `training/tests/test_build_audit_sample.py`

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Verify the test FAILS (module does not exist yet)**

```bash
uv run pytest tests/test_build_audit_sample.py -v
```

Expected: ImportError on `trainr.core.build_audit_sample`.

### Task 6.2: Create `build_audit_sample.py`

**Files:**
- Create: `training/trainr/core/build_audit_sample.py`

- [ ] **Step 1: Write the module**

```python
"""Build the 5k audit input for semantic detection label validation.

Combines a stratified sample of the training corpus with targeted
injection of known-positive candidates for the 3 new semantic labels
(log_content, stack_trace, diff_patch). Injected rows are tagged with
an `audit_source` column so the downstream audit can:

1. Compute inter-annotator agreement on the `stratified` rows only
   (injected positives would inflate the stat).
2. Compute recall per new label on the `inject_*` rows.

Usage:
    uv run python -m trainr.core.build_audit_sample \
        --input data/curated/train/golden_train.parquet \
        --output data/audit/iter16_5k_input.parquet \
        --stratified-n 5000 \
        --injection-per-label 50
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

from trainr.core.annotate_detections import stratified_sample

# Regex patterns for targeted injection. Each pattern targets a label's
# high-precision surface features so that matches are very likely to be
# true positives for manual review.
INJECTION_PATTERNS: dict[str, list[str]] = {
    "stack_trace": [
        r"Traceback \(most recent call last\)",
        r"Exception in thread",
        r"goroutine \d+ \[",
        r"panicked at",
        r"\s+at [\w.$]+\(.*\.java:\d+\)",
        r"^\s+at \w+\.\w+\.\w+\(\) in .*\.cs:line \d+",  # .NET
    ],
    "diff_patch": [
        r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@",
        r"^diff --git a/",
    ],
    "log_content": [
        # Severity token + timestamp-shaped substring in same row.
        # Left broad — this is a generous superset; annotators judge.
        r"\b(INFO|WARN|ERROR|DEBUG|TRACE|FATAL)\b.*\d{4}-\d{2}-\d{2}",
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*\b(INFO|WARN|ERROR|DEBUG|TRACE|FATAL)\b",
        r"\[\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}",  # apache access log timestamp
    ],
}


def find_injection_candidates(
    corpus: pl.DataFrame,
    label: str,
    patterns: list[str],
    n: int,
    seed: int,
) -> pl.DataFrame:
    """Return up to `n` rows from `corpus` whose `text` matches any pattern.

    Matches are case-insensitive only if the regex uses inline flags.
    Uses polars' string.contains (regex) with OR composition across
    patterns. Samples uniformly from the match set.
    """
    # Build an OR regex
    combined = "|".join(f"(?:{p})" for p in patterns)
    matched = corpus.filter(pl.col("text").str.contains(combined))
    if len(matched) == 0:
        return matched.with_columns(pl.lit(f"inject_{label}").alias("audit_source"))
    if len(matched) > n:
        matched = matched.sample(n=n, seed=seed)
    return matched.with_columns(pl.lit(f"inject_{label}").alias("audit_source"))


def build_audit_sample(
    input_path: str,
    output_path: str,
    stratified_n: int = 5000,
    injection_per_label: int = 50,
    seed: int = 42,
) -> pl.DataFrame:
    """Build the audit input parquet.

    Args:
        input_path: Path to the corpus parquet (golden_train.parquet).
        output_path: Where to write the audit input parquet.
        stratified_n: Size of the stratified sample.
        injection_per_label: How many known-positive candidates to inject
            per semantic label (log_content, stack_trace, diff_patch).
        seed: RNG seed for sampling reproducibility.

    Returns:
        The final DataFrame with an `audit_source` column distinguishing
        'stratified' rows from 'inject_<label>' rows.
    """
    corpus = pl.read_parquet(input_path)
    print(
        f"  Corpus loaded: {len(corpus)} rows.",
        file=sys.stderr,
    )

    # 1. Stratified sample tagged as 'stratified'
    stratified = stratified_sample(corpus, n=stratified_n, seed=seed)
    stratified = stratified.with_columns(
        pl.lit("stratified").alias("audit_source"),
    )
    print(
        f"  Stratified sample: {len(stratified)} rows.",
        file=sys.stderr,
    )

    # 2. Per-label targeted injection
    injected_parts: list[pl.DataFrame] = []
    for label, patterns in INJECTION_PATTERNS.items():
        part = find_injection_candidates(
            corpus=corpus,
            label=label,
            patterns=patterns,
            n=injection_per_label,
            seed=seed,
        )
        print(
            f"  Injected {label}: {len(part)} rows.",
            file=sys.stderr,
        )
        injected_parts.append(part)

    # 3. Concatenate; schema alignment via how='diagonal_relaxed' in case
    # injected parts have slightly different column orders
    all_parts = [stratified] + injected_parts
    combined = pl.concat(all_parts, how="diagonal_relaxed")

    # Ensure output dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)
    print(
        f"  Wrote {len(combined)} rows to {output_path}.",
        file=sys.stderr,
    )
    return combined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Corpus parquet path")
    parser.add_argument("--output", required=True, help="Audit input parquet path")
    parser.add_argument("--stratified-n", type=int, default=5000)
    parser.add_argument("--injection-per-label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    build_audit_sample(
        input_path=args.input,
        output_path=args.output,
        stratified_n=args.stratified_n,
        injection_per_label=args.injection_per_label,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke tests**

```bash
uv run pytest tests/test_build_audit_sample.py -v
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add training/trainr/core/build_audit_sample.py training/tests/test_build_audit_sample.py
git commit -m "feat(audit): build_audit_sample.py — stratified + injected audit input

Composes the iter16 5k audit sample: stratified sample via the existing
stratified_sample() helper, plus targeted injection of known-positive
candidates for log_content, stack_trace, and diff_patch. Injected rows
are tagged with an audit_source column so the downstream audit can
exclude them from the agreement metric and measure per-label recall on
them separately."
```

---

## Phase 7 — Audit Metric + Report Generator

### Task 7.1: Write smoke test for `audit_semantic_labels.py`

**Files:**
- Create: `training/tests/test_audit_semantic_labels.py`

- [ ] **Step 1: Create the test file**

```python
"""Smoke tests for trainr.core.audit_semantic_labels."""

import tempfile
from pathlib import Path

import polars as pl


def _make_fake_annotated(
    audit_source_values: list[str],
    det_log_content: list[int],
    det_stack_trace: list[int],
    det_diff_patch: list[int],
) -> pl.DataFrame:
    """Build a minimal annotated-parquet fixture."""
    n = len(audit_source_values)
    return pl.DataFrame({
        "text": [f"row {i}" for i in range(n)],
        "sub_type": ["plain"] * n,
        "audit_source": audit_source_values,
        "det_log_content": det_log_content,
        "det_stack_trace": det_stack_trace,
        "det_diff_patch": det_diff_patch,
    })


def test_compute_recall_on_injected_positives_all_correct():
    from trainr.core.audit_semantic_labels import compute_recall_on_injected

    df = _make_fake_annotated(
        audit_source_values=[
            "inject_stack_trace", "inject_stack_trace",
            "inject_diff_patch",
            "inject_log_content", "inject_log_content",
            "stratified",
        ],
        det_log_content=[0, 0, 0, 1, 1, 0],
        det_stack_trace=[1, 1, 0, 0, 0, 0],
        det_diff_patch=[0, 0, 1, 0, 0, 0],
    )
    recall = compute_recall_on_injected(df)
    assert recall["stack_trace"] == 1.0
    assert recall["diff_patch"] == 1.0
    assert recall["log_content"] == 1.0


def test_compute_recall_on_injected_positives_partial():
    from trainr.core.audit_semantic_labels import compute_recall_on_injected

    df = _make_fake_annotated(
        audit_source_values=[
            "inject_stack_trace", "inject_stack_trace", "inject_stack_trace",
            "inject_stack_trace", "inject_stack_trace",
        ],
        det_log_content=[0, 0, 0, 0, 0],
        det_stack_trace=[1, 1, 1, 1, 0],  # 4/5 fired
        det_diff_patch=[0, 0, 0, 0, 0],
    )
    recall = compute_recall_on_injected(df)
    assert recall["stack_trace"] == 0.8


def test_agreement_excludes_injected_rows():
    """Inter-annotator agreement must be computed on stratified rows only."""
    from trainr.core.audit_semantic_labels import filter_for_agreement

    df = _make_fake_annotated(
        audit_source_values=[
            "stratified", "stratified", "inject_stack_trace",
        ],
        det_log_content=[0, 0, 1],
        det_stack_trace=[0, 0, 1],
        det_diff_patch=[0, 0, 0],
    )
    filtered = filter_for_agreement(df)
    assert len(filtered) == 2
    assert all(s == "stratified" for s in filtered["audit_source"].to_list())
```

- [ ] **Step 2: Verify tests fail**

```bash
uv run pytest tests/test_audit_semantic_labels.py -v
```

Expected: ImportError.

### Task 7.2: Create `audit_semantic_labels.py`

**Files:**
- Create: `training/trainr/core/audit_semantic_labels.py`

- [ ] **Step 1: Write the module**

```python
"""Audit report generator for the iter16 semantic label validation.

Reads 3 annotated parquets (one per annotator model), computes:

1. Inter-annotator agreement on stratified rows (ALL labels — catches
   regressions on existing labels from the SYSTEM_PROMPT length change).
2. Recall per new semantic label on injected positive rows.
3. Disagreement spot-check table for manual review.

Writes a markdown report gating the decision on the $400-600 full run.

Usage:
    uv run python -m trainr.core.audit_semantic_labels \
        --gemini data/audit/iter16_5k_gemini3flash.parquet \
        --sonnet data/audit/iter16_5k_sonnet.parquet \
        --gpt54mini data/audit/iter16_5k_gpt54mini.parquet \
        --output docs/accuracy_runs/2026-04-10-iteration-16-audit-report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from trainr.core.annotate_detections import DETECTION_LABELS, SEMANTIC_LABELS

STRATIFIED = "stratified"

# Pass criteria thresholds (from spec)
AGREEMENT_THRESHOLD = 0.995
RECALL_THRESHOLD = 0.90


def filter_for_agreement(df: pl.DataFrame) -> pl.DataFrame:
    """Return only stratified rows (exclude injected positives)."""
    return df.filter(pl.col("audit_source") == STRATIFIED)


def detection_columns(df: pl.DataFrame) -> list[str]:
    """Return det_* column names present in the DataFrame."""
    return [c for c in df.columns if c.startswith("det_")]


def compute_agreement_across_models(
    dfs: dict[str, pl.DataFrame],
) -> dict[str, float]:
    """Compute per-label mean agreement across models on aligned rows.

    Assumes all DataFrames have the same row ordering (which they will
    if produced from the same input by annotate_dataframe).

    Returns {label: agreement} where agreement is the fraction of rows
    on which the majority vote matches the modal vote, averaged across
    rows. Unanimous rows contribute 1.0; split rows contribute
    max(ones, zeros) / n_models.
    """
    model_names = list(dfs.keys())
    n_models = len(model_names)
    if n_models == 0:
        return {}

    first = dfs[model_names[0]]
    det_cols = detection_columns(first)
    n_rows = len(first)

    result: dict[str, float] = {}
    for col in det_cols:
        label = col[len("det_"):]
        votes_per_model = [dfs[m][col].to_list() for m in model_names]
        agreement_sum = 0.0
        for row_idx in range(n_rows):
            row_votes = [votes_per_model[m][row_idx] for m in range(n_models)]
            ones = sum(1 for v in row_votes if v == 1)
            zeros = n_models - ones
            agreement_sum += max(ones, zeros) / n_models
        result[label] = agreement_sum / n_rows if n_rows > 0 else 1.0
    return result


def compute_recall_on_injected(df: pl.DataFrame) -> dict[str, float]:
    """Per-semantic-label recall on injected positives.

    For each `inject_<label>` row, the label should have fired (det==1).
    Returns {label: recall_fraction}. Computed on a SINGLE annotated DF —
    call once per model if you want per-model recall.
    """
    result: dict[str, float] = {}
    for label in SEMANTIC_LABELS:
        injected = df.filter(pl.col("audit_source") == f"inject_{label}")
        if len(injected) == 0:
            result[label] = float("nan")
            continue
        col = f"det_{label}"
        if col not in df.columns:
            result[label] = 0.0
            continue
        fired = injected[col].sum()
        result[label] = fired / len(injected)
    return result


def compute_recall_majority(
    dfs: dict[str, pl.DataFrame],
) -> dict[str, float]:
    """Recall on injected positives where ≥2 of 3 models fire the label.

    This is the metric the pass criterion uses ("correctly fired by at
    least 2 of 3 annotators").
    """
    model_names = list(dfs.keys())
    if not model_names:
        return {label: float("nan") for label in SEMANTIC_LABELS}

    first = dfs[model_names[0]]
    result: dict[str, float] = {}
    for label in SEMANTIC_LABELS:
        col = f"det_{label}"
        injected_mask = first["audit_source"] == f"inject_{label}"
        n_inject = int(injected_mask.sum())
        if n_inject == 0:
            result[label] = float("nan")
            continue

        correct = 0
        votes_per_model = [
            dfs[m].filter(injected_mask)[col].to_list()
            for m in model_names
        ]
        for row_idx in range(n_inject):
            row_votes = [
                votes_per_model[m][row_idx] for m in range(len(model_names))
            ]
            if sum(row_votes) >= 2:
                correct += 1
        result[label] = correct / n_inject
    return result


def format_report(
    dfs: dict[str, pl.DataFrame],
    agreement: dict[str, float],
    recall: dict[str, float],
) -> str:
    """Build the markdown report with pass/fail verdicts."""
    lines = [
        "# Iter16 Semantic Label Audit Report",
        "",
        "**Date:** 2026-04-10",
        f"**Models:** {', '.join(dfs.keys())}",
        "",
        "## Pass Criteria",
        "",
        f"1. Inter-annotator agreement ≥{AGREEMENT_THRESHOLD} on stratified 5k "
        "for ALL labels.",
        f"2. Recall (majority ≥2 of 3) ≥{RECALL_THRESHOLD} on injected positives.",
        "3. Zero obvious rule violations in spot-check (manual).",
        "",
        "## Criterion 1: Inter-Annotator Agreement (stratified 5k)",
        "",
        "| Label | Agreement | Pass |",
        "|---|---:|:---:|",
    ]

    all_agree_pass = True
    for label in sorted(agreement.keys()):
        score = agreement[label]
        passed = score >= AGREEMENT_THRESHOLD
        if not passed:
            all_agree_pass = False
        lines.append(
            f"| {label} | {score:.4f} | {'✓' if passed else '✗'} |"
        )

    lines.extend([
        "",
        f"**Criterion 1 verdict: {'PASS' if all_agree_pass else 'FAIL'}**",
        "",
        "## Criterion 2: Recall on Injected Positives",
        "",
        "| Label | Recall (majority) | Pass |",
        "|---|---:|:---:|",
    ])

    all_recall_pass = True
    for label in sorted(SEMANTIC_LABELS):
        score = recall.get(label, float("nan"))
        passed = score >= RECALL_THRESHOLD
        if not passed:
            all_recall_pass = False
        lines.append(
            f"| {label} | {score:.4f} | {'✓' if passed else '✗'} |"
        )

    lines.extend([
        "",
        f"**Criterion 2 verdict: {'PASS' if all_recall_pass else 'FAIL'}**",
        "",
        "## Criterion 3: Spot-Check Rule Violations",
        "",
        "_Manual review required — scan disagreement rows and injected "
        "positives for obvious rule violations (e.g., log_content firing on "
        "lowercase error in prose)._",
        "",
        "## Overall Decision",
        "",
    ])

    if all_agree_pass and all_recall_pass:
        lines.append(
            "**Gate: PASS** (pending manual criterion 3). Proceed to "
            "full 90k annotation run."
        )
    else:
        lines.append(
            "**Gate: FAIL.** Iterate on label definitions; re-run audit."
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini", required=True, help="gemini-3-flash annotated parquet")
    parser.add_argument("--sonnet", required=True, help="sonnet-4.6 annotated parquet")
    parser.add_argument("--gpt54mini", required=True, help="gpt-5.4-mini annotated parquet")
    parser.add_argument("--output", required=True, help="Markdown report output path")
    args = parser.parse_args(argv)

    dfs = {
        "gemini3flash": pl.read_parquet(args.gemini),
        "sonnet": pl.read_parquet(args.sonnet),
        "gpt54mini": pl.read_parquet(args.gpt54mini),
    }

    # Filter to stratified rows for agreement
    stratified_dfs = {name: filter_for_agreement(df) for name, df in dfs.items()}
    print(
        f"  Agreement cohort: {len(next(iter(stratified_dfs.values())))} rows.",
        file=sys.stderr,
    )

    agreement = compute_agreement_across_models(stratified_dfs)
    recall = compute_recall_majority(dfs)

    report = format_report(dfs, agreement, recall)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"  Wrote report to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke tests**

```bash
uv run pytest tests/test_audit_semantic_labels.py -v
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add training/trainr/core/audit_semantic_labels.py training/tests/test_audit_semantic_labels.py
git commit -m "feat(audit): audit_semantic_labels.py — metrics + report generator

Computes per-label inter-annotator agreement on stratified rows
(injected positives excluded — they'd inflate the stat) and per-new-label
recall on injected positives using majority vote across 3 annotator
models. Emits a markdown report with pass/fail verdicts against the
two quantitative pass criteria from the iter16 spec."
```

---

## Phase 8 — Execute the 5k Audit

This phase is a one-shot execution plus manual review. It produces no new code — just runs the tools built in earlier phases.

### Task 8.1: Verify API keys and corpus are available

- [ ] **Step 1: Check env**

Run from `training/`:
```bash
uv run python -c "from trainr.shared.api import get_openrouter_api_key, get_anthropic_api_key; print('ok' if get_openrouter_api_key() and get_anthropic_api_key() else 'missing')"
```

Expected: `ok`. If `missing`, source the appropriate env file (e.g., `source ~/.openrouter.sh`).

- [ ] **Step 2: Verify corpus exists**

```bash
ls -lh data/curated/train/golden_train.parquet
```

Expected: file exists, ~hundreds of MB.

### Task 8.2: Build the audit input sample

- [ ] **Step 1: Run the builder**

```bash
mkdir -p data/audit
uv run python -m trainr.core.build_audit_sample \
  --input data/curated/train/golden_train.parquet \
  --output data/audit/iter16_5k_input.parquet \
  --stratified-n 5000 \
  --injection-per-label 50 \
  --seed 42
```

Expected stderr lines:
- `Corpus loaded: ~90626 rows.`
- `Stratified sample: 5000 rows.`
- `Injected stack_trace: <= 50 rows.`
- `Injected diff_patch: <= 50 rows.`
- `Injected log_content: <= 50 rows.`
- `Wrote ~5150 rows to data/audit/iter16_5k_input.parquet.`

- [ ] **Step 2: Sanity-check the output**

```bash
uv run python -c "
import polars as pl
df = pl.read_parquet('data/audit/iter16_5k_input.parquet')
print(f'total rows: {len(df)}')
print(df.group_by('audit_source').len().sort('audit_source'))
"
```

Expected: ~5000 stratified + up to 50 per inject_* source. If any injection count is 0, investigate the regex patterns before proceeding.

### Task 8.3: Run the 3-model annotation

- [ ] **Step 1: Run annotator with gemini-3-flash**

```bash
uv run python -m trainr.core.annotate_detections \
  --input data/audit/iter16_5k_input.parquet \
  --output data/audit/iter16_5k_gemini3flash.parquet \
  --backend openrouter \
  --model google/gemini-3-flash-preview \
  --concurrency 20
```

Expected wall clock: ~5-10 minutes. Cost: ~$5-8.

- [ ] **Step 2: Run annotator with sonnet-4.6**

```bash
uv run python -m trainr.core.annotate_detections \
  --input data/audit/iter16_5k_input.parquet \
  --output data/audit/iter16_5k_sonnet.parquet \
  --backend openrouter \
  --model anthropic/claude-sonnet-4-6 \
  --concurrency 20
```

Expected wall clock: ~5-10 minutes. Cost: ~$8-12.

- [ ] **Step 3: Run annotator with gpt-5.4-mini**

```bash
uv run python -m trainr.core.annotate_detections \
  --input data/audit/iter16_5k_input.parquet \
  --output data/audit/iter16_5k_gpt54mini.parquet \
  --backend openrouter \
  --model openai/gpt-5.4-mini \
  --concurrency 20
```

Expected wall clock: ~5-10 minutes. Cost: ~$5-8.

- [ ] **Step 4: Verify all 3 outputs exist**

```bash
ls -lh data/audit/iter16_5k_{gemini3flash,sonnet,gpt54mini}.parquet
```

Expected: 3 files present, roughly similar sizes.

### Task 8.4: Generate the audit report

- [ ] **Step 1: Run the report generator**

```bash
uv run python -m trainr.core.audit_semantic_labels \
  --gemini data/audit/iter16_5k_gemini3flash.parquet \
  --sonnet data/audit/iter16_5k_sonnet.parquet \
  --gpt54mini data/audit/iter16_5k_gpt54mini.parquet \
  --output docs/accuracy_runs/2026-04-10-iteration-16-audit-report.md
```

Expected: report written to the docs path.

- [ ] **Step 2: Read the report and capture verdict**

```bash
cat docs/accuracy_runs/2026-04-10-iteration-16-audit-report.md
```

Record whether each criterion passed:
- Criterion 1 (agreement ≥0.995 for all labels): __
- Criterion 2 (recall ≥0.90 for all 3 new labels): __

### Task 8.5: Manual spot-check (criterion 3)

- [ ] **Step 1: Sample disagreements for review**

```bash
uv run python -c "
import polars as pl
from trainr.core.annotate_detections import SEMANTIC_LABELS

gem = pl.read_parquet('data/audit/iter16_5k_gemini3flash.parquet')
son = pl.read_parquet('data/audit/iter16_5k_sonnet.parquet')
gpt = pl.read_parquet('data/audit/iter16_5k_gpt54mini.parquet')

for label in sorted(SEMANTIC_LABELS):
    col = f'det_{label}'
    # Rows where annotators disagree on a semantic label
    disagree_mask = (gem[col] != son[col]) | (son[col] != gpt[col]) | (gem[col] != gpt[col])
    n_dis = int(disagree_mask.sum())
    print(f'{label}: {n_dis} disagreements')
    if n_dis > 0 and n_dis <= 20:
        sample = gem.filter(disagree_mask).head(10)
        for i, row in enumerate(sample.iter_rows(named=True)):
            print(f'  [{i}] source={row[\"audit_source\"]} gem={gem.filter(disagree_mask)[col][i]} son={son.filter(disagree_mask)[col][i]} gpt={gpt.filter(disagree_mask)[col][i]}')
            print(f'      text: {row[\"text\"][:300]!r}')
            print()
"
```

- [ ] **Step 2: Review for rule violations**

For each disagreement, check whether any annotator violated a rule:
- Did `log_content` fire on lowercase `error`/`info` in prose? (uppercase rule)
- Did `log_content` fire on a pure traceback with no other log lines? (pure-trace carveout)
- Did `stack_trace` fire on a single `File "foo.py", line 42` line? (multi-frame rule)
- Did `diff_patch` fire on a markdown bullet list with no `@@`/`diff --git`/`---/+++`? (required-marker rule)

Record findings in a `spot_check_notes.md` scratch file under `data/audit/` (gitignored — not committed).

### Task 8.6: Gate decision

- [ ] **Step 1: Evaluate all three criteria**

- Criterion 1 agreement: PASS / FAIL
- Criterion 2 recall: PASS / FAIL
- Criterion 3 spot-check violations: PASS / FAIL

- [ ] **Step 2: Branch on outcome**

**If ALL PASS:** continue to Phase 9.

**If ANY FAIL:**
- Identify which label(s) and which rule(s) are causing the failure.
- Revise the relevant definition block in `SYSTEM_PROMPT` — tighten rules, not loosen.
- Re-run Task 8.3 (or just the failing model if obvious) and Task 8.4.
- Repeat until all pass. Budget ceiling: 2-3 iterations. If the labels are still struggling after the third pass, raise for redesign — some definitions may need restructuring, not re-wording.

---

## Phase 9 — Iteration Report + PR

### Task 9.1: Write the iteration 16 doc

**Files:**
- Create: `docs/accuracy_runs/2026-04-10-iteration-16.md`

- [ ] **Step 1: Write using iter15 as a structural template**

Sections to include:
- **Summary:** 3 new semantic labels added, det_log_lines removed, 5k audit passed.
- **Background:** link to the spec, restate the taxonomy decision.
- **Changes:** list of code and test changes per phase.
- **Validation:** agreement table, recall table, spot-check summary. Link to the audit report.
- **Architecture Notes:** first cross-cutting detection labels; implications for the retrain in the next branch.
- **Key Learnings:** what the audit revealed — were definitions tight enough on first try, or did we have to iterate?
- **Follow-Up:** the next branch (annotation + retrain), then the Rust port branch after that.
- **Cost Breakdown:** actual audit cost.

Use iter15 doc (`docs/accuracy_runs/2026-04-10-iteration-15.md`) as format reference.

### Task 9.2: Commit the iteration report + audit artifacts

- [ ] **Step 1: Commit**

```bash
git add docs/accuracy_runs/2026-04-10-iteration-16.md \
        docs/accuracy_runs/2026-04-10-iteration-16-audit-report.md
git commit -m "docs(iter16): iteration report + audit results

5k audit pass/fail results for the 3 new semantic detection labels.
Gates the downstream $400-600 full 90k annotation run."
```

### Task 9.3: Push and open the PR

- [ ] **Step 1: Push the branch**

```bash
git push -u origin three-head-part-deux
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "feat(detections): add 3 semantic labels, remove det_log_lines" --body "$(cat <<'EOF'
## Summary
- Add `log_content`, `stack_trace`, `diff_patch` as the first cross-cutting detection labels (detection-only, not mirrored by `ContentSubType`)
- Remove `det_log_lines` (redundant with `sub_type=log_lines`; detection head never trained so removal is free)
- Validated via 5k stratified audit + ~150 targeted positive injection across 3 annotator models

## Test plan
- [ ] `uv run pytest training/tests/test_annotate_detections.py -v` green
- [ ] `uv run pytest training/tests/test_build_audit_sample.py -v` green
- [ ] `uv run pytest training/tests/test_audit_semantic_labels.py -v` green
- [ ] 5k audit pass criteria met (see `docs/accuracy_runs/2026-04-10-iteration-16-audit-report.md`):
  - Inter-annotator agreement ≥0.995 on all labels
  - Recall ≥0.90 (majority of 3) on injected positives for all 3 new labels
  - Zero obvious rule violations in manual spot-check

## Spec
`docs/superpowers/specs/2026-04-10-semantic-detection-labels-design.md`

## Out of scope (next branches)
- Full 90k annotation run + retrain with detection head
- Rust port (copy artifacts, smoke test CLI) — workspace boundary
EOF
)"
```

- [ ] **Step 3: Return the PR URL for user review.**

---

## Self-Review Checklist (for the plan author, before starting execution)

- **Spec coverage:**
  - Remove `det_log_lines` → Phase 1 ✓
  - Add `log_content` / `stack_trace` / `diff_patch` → Phases 3, 4, 5 ✓
  - v2 label definitions matching spec → Tasks 3.4, 4.4, 5.4 ✓
  - Unit tests + regression-guards → tests throughout phases 1-5 ✓
  - 5k stratified + injection strategy → Phases 6-8 ✓
  - Pass criteria (agreement ≥0.995, recall ≥0.90, spot-check) → Task 8.6 ✓
  - Implementation order matches spec's "Implementation Order" section ✓
- **Placeholder scan:** no "TBD", no "add error handling," every code block contains real code.
- **Type consistency:** `SEMANTIC_LABELS` introduced in 2.2 referenced consistently in 3.2, 4.2, 5.2; `audit_source` column name consistent across builder (6.2) and auditor (7.2); `stratified`/`inject_<label>` tags consistent.
