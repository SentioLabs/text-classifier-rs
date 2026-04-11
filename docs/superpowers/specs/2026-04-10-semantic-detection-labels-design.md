# Semantic Detection Labels — log_content, stack_trace, diff_patch

**Date:** 2026-04-10
**Branch:** new branch off `main` post-PR#12-merge
**Workstream:** `training/` only (no `src/` changes — workspace boundary)
**Follows:** `docs/accuracy_runs/2026-04-10-iteration-15.md`

## Summary

Add 3 cross-cutting semantic detection labels — `log_content`, `stack_trace`, `diff_patch` — and remove `det_log_lines` from the detection label set. Validate with a 5k stratified audit plus targeted positive injection. This branch is a **prompt-only change**: no data pipeline runs, no model retrain, no Rust changes. It exists to lock in precise label definitions *before* the $400-600 full 90k annotation run in the next branch.

The design principle driving every decision here: **optimize for label definition quality upfront so the 5k audit passes clean and the downstream expensive run isn't wasted on prompt bugs.**

## Background

- **iter15 deferred these labels.** That iteration added 9 programming-language detection labels (`csharp`, `swift`, `kotlin`, `r`, `lua`, etc.) and expanded the corpus for language coverage, but explicitly postponed the semantic labels as "the original goal of this workstream, deferred from this iteration."
- **The detection head has never been trained.** `model_config.json` lacks a `detection_map` field; the Rust inference code in `tier2.rs` skips the head gracefully when absent. This means schema changes to `DETECTION_LABELS` are free — no production artifact depends on the current set.
- **`log_lines` has no definition.** The existing prompt lists `log_lines` in the label set but has zero positive signals, anti-signals, or disambiguation text. iter15 called this out: "currently fires inconsistently."

## Architectural Note: First Cross-Cutting Detection Labels

All 40 existing detection labels are 1:1 with `ContentSubType`. Every detection label so far has been "is this sub_type present anywhere?" The 3 new labels are the **first detection-only labels** — they have no corresponding sub_type.

This is correct by design: stack traces, unified diffs, and embedded log output are inherently cross-cutting phenomena. A stack trace is almost never a whole document; it's embedded in logs, bug reports, markdown docs, error emails. Same for diff patches (commit messages, PR descriptions, code review) and sample log output (tutorials, runbooks). Creating sub_types for them would be wrong — the primary sub_type of a bug report is still `markdown` or `plain`.

The detection head was designed from the start for cross-cutting signal. This branch is the first use of that capability as intended.

## Taxonomy Decision: Remove `det_log_lines`, Add `det_log_content`

- **`log_lines` remains a `ContentSubType`** (unchanged). Meaning: "this row IS primarily log lines" (e.g., a 10k-line nginx access log).
- **`det_log_lines` is REMOVED from `DETECTION_LABELS`**. The `sub_type` column already carries this signal; duplicating it as a detection label creates name-collision ambiguity.
- **`det_log_content` is ADDED as a detection label**. Meaning: "this text contains log content anywhere, possibly embedded in prose/markdown/code."

Removal is free because no trained model depends on `det_log_lines`. No annotated parquet with `det_*` columns exists yet (the 90k run is deferred to the next branch).

## Label Definitions (v2)

These are the exact strings to add to `SYSTEM_PROMPT` in `training/trainr/core/annotate_detections.py`, inserted into the `## Important distinctions` section.

```
- "log_content" = embedded log output (distinct from sub_type=log_lines, which
  means the row IS primarily logs). Requires TWO OR MORE consecutive lines that
  each independently match a log-line schema. A log-line schema is ONE of:
    * `<timestamp> <severity-or-component> <message>` where timestamp is
      ISO 8601 (`2024-01-15T10:23:45Z`), apache-style
      (`[15/Jan/2024:10:23:45 +0000]`), syslog-style (`Jan 15 10:23:45`),
      or unix epoch (ms/seconds)
    * A recognized named format: nginx/apache access log, syslog
      (`<pri>timestamp host proc[pid]: msg`), dockerd/container log,
      logfmt (`key=value key2=value2`) ONLY when co-occurring with a
      timestamp OR severity field
    * JSON log records: ≥2 consecutive JSON objects each containing BOTH a
      timestamp field AND a `level`/`severity` field
  Severity tokens must be UPPERCASE or BRACKETED (`INFO`, `[info]`, `WARN`,
  `WARNING`, `ERROR`, `DEBUG`, `TRACE`, `FATAL`, `CRITICAL`) — lowercase
  `error`/`info` in prose or code identifiers does NOT count.

  Do NOT fire on: single-line error messages (even inside code fences or
  blockquotes), code that CALLS a logger (`log.info(...)`), sentences
  describing logging behavior, changelogs with leading dates
  (`2024-01-15 - fixed bug`), CSV/TSV with date columns, `ls -la` output,
  git commit logs. Lines inside quotation marks or markdown blockquotes do
  not contribute to the density count.

  If the embedded output is a PURE stack trace (no surrounding non-trace log
  lines), fire "stack_trace": 1 but NOT "log_content": 1. If a stack trace
  appears inside otherwise-normal log output, fire BOTH.

- "stack_trace" = programmatic stack trace / traceback. Requires TWO OR MORE
  frames where each frame carries a file:line locator OR a
  package.Class.method locator (not just an `at` keyword). Strong signals
  by language:
  * Python: `Traceback (most recent call last):` header + `File "foo.py",
    line N, in func` frames + final `ErrorType: message` line
  * Java/JVM: `Exception in thread "..."` + `at com.pkg.Class.method(File.java:42)`
    + optional `Caused by:` + `... N more`
  * .NET: `   at Namespace.Class.Method() in File.cs:line 42` (leading
    whitespace is distinctive)
  * JS/Node: `Error: message` + `at fn (file:line:col)` / `at async fn`
  * Go: `goroutine N [running]:` + `main.foo(0x0)\n\tpath/file.go:42 +0x1a`
  * Rust: `thread 'main' panicked at` + numbered backtrace frames `0: ...`,
    `1: ...` with file:line
  * Ruby: `from file.rb:42:in 'method'` chain (one frame per line)
  Truncation markers (`... N more frames`, `[truncated]`) do not disqualify
  a trace. Do NOT fire on: single-line error messages, the literal phrase
  "stack trace" in prose, tutorial pseudocode describing what a trace looks
  like, profiler output tables, or prose like "at line 5 it crashed, at line
  6 it retried". If this trace appears inside log output, fire BOTH
  "stack_trace": 1 AND "log_content": 1.

- "diff_patch" = unified diff or git patch format. REQUIRED: at least one of
  {`@@ -X,Y +A,B @@` hunk header, `diff --git a/... b/...` header, paired
  `--- a/path` / `+++ b/path` file markers on adjacent lines}. Line prefixes
  alone are NOT sufficient. Additional signals: `+`/`-`/space line prefixes
  within a hunk, `index abc1234..def5678 100644` git metadata. The email-patch
  header `From abc1234 Mon Sep 17 00:00:00 2001` is a signal ONLY when
  co-occurring with `---`/`+++` or `@@` markers (otherwise a regular email
  header). CRITICAL anti-signals: markdown bullet lists using `-` or `+`,
  pro/con lists, code containing arithmetic `+`/`-`, isolated `+`/`-` lines
  without hunk context, YAML frontmatter `---` without an adjacent `+++`.
  Unified diffs only — context diffs (`*** file ***` separators) out of scope.
```

## Changes to `annotate_detections.py`

1. **`DETECTION_LABELS` list:** remove `"log_lines"`, append `"log_content"`, `"stack_trace"`, `"diff_patch"` (place at end, matching iter15's convention of appending new labels).
2. **`SYSTEM_PROMPT` label list line:** remove `log_lines`, append the 3 new labels in the inline list.
3. **`SYSTEM_PROMPT` `## Important distinctions` section:** append the 3 definition blocks above.
4. **`SYSTEM_PROMPT` JSON example template at the bottom:** remove `"log_lines": 0`, add `"log_content": 0, "stack_trace": 0, "diff_patch": 0`.

No other code changes. No routing table updates (routing is keyed by `sub_type`, unaffected).

## Testing

New file: `training/tests/test_annotate_detections.py` (pattern: iter15's `test_pull_real_data.py`).

Unit tests:

- `DETECTION_LABELS` contains `log_content`, `stack_trace`, `diff_patch` and does NOT contain `log_lines`.
- `parse_response` returns all 3 new labels with default 0.
- `parse_response` does NOT return a `log_lines` key.
- `parse_response` handles valid JSON with the new labels set to 1.

Regression-guard tests on `SYSTEM_PROMPT` text (catch prompt drift):

- Contains `"UPPERCASE or BRACKETED"` (log_content severity rule).
- Contains `"adjacent lines"` (diff_patch `---`/`+++` rule).
- Contains `".NET"` (stack_trace .NET format included).
- Contains `"PURE stack trace"` (log_content carveout for pure tracebacks).
- Does NOT contain `\"log_lines\":` (removed from JSON template).

These text-presence tests are ugly but have the same role as iter15's R-regex precision tests: they catch silent prompt regressions during refactoring.

## 5k Validation Audit

**Sample composition:**
- **5k stratified sample** via existing `stratified_sample(n=5000)` — matches iter15's methodology, directly comparable metrics.
- **+~150 targeted positive candidates** (~50 per new label), tagged with source metadata, **excluded from the inter-annotator agreement metric** (injected positives would inflate it), reviewed manually for precision.

**Injection queries (pre-filter the corpus, then sample):**
- `stack_trace`: regex `Traceback \(most recent call last\)`, `Exception in thread`, `goroutine \d+ \[`, `panicked at`, `\s+at [\w.]+\(.*\.java:`
- `diff_patch`: regex `^@@ -\d+,\d+ \+\d+,\d+ @@`, `^diff --git a/`
- `log_content`: text contains `\b(INFO|WARN|ERROR|DEBUG|TRACE|FATAL)\b` AND contains at least one timestamp-shaped substring. Generous superset — annotators judge.

**Annotator models:** `gemini-3-flash`, `sonnet-4.6`, `gpt-5.4-mini` (iter15 precedent).

**Pass criteria (ALL must hold — go/no-go on the $400-600 full run):**

1. **Inter-annotator agreement ≥0.995** on the stratified 5k for ALL labels, existing AND new. This catches prompt-length regression on the 40 existing labels as well as the 3 new ones.
2. **Recall ≥0.90 on targeted positives** — for each new label, ≥90% of injected candidate rows must be correctly fired by at least 2 of 3 annotators.
3. **Zero obvious rule violations** in spot-check. Examples of violations: `log_content` firing on lowercase `error` in prose (uppercase rule ignored); `stack_trace` firing on a single `File "foo.py", line 42` line (frame count rule ignored); `diff_patch` firing on a markdown bullet list (anti-signal ignored); `log_content` firing on a pure traceback (pure-trace carveout ignored).

**Cost estimate:** ~$20-25 total. Primary cost is the 3-model 5k audit; targeted injection adds negligible volume.

## Risks and Mitigations

- **Prompt length regression.** Adding 3 dense definition sections increases `SYSTEM_PROMPT` length by ~25%. Small models (nano/flash/mini) may behave differently with a longer prompt — existing label agreement could drift even without any definition changes to them. *Mitigation:* pass criterion #1 requires agreement on existing labels too, catching any regression.
- **.NET stack trace format over-interpretation.** The "leading whitespace is distinctive" hint may be read as "any indented `at Foo.Bar()` is a .NET frame." *Mitigation:* targeted injection includes known-good .NET traces; manual precision review catches overfire.
- **Pure-trace carveout for log_content is subtle.** "If the embedded output is a PURE stack trace... fire stack_trace but NOT log_content" is a conditional that small models may ignore. *Mitigation:* spot-check specifically for this pattern in the targeted stack_trace positives.
- **5k audit false-negative on precision.** At ~0.1% positive rate, even with targeted injection, precision measurement on negatives depends on spot-checks rather than ground truth. If the audit passes but precision is actually bad, the $400 full run will produce noisy data. *Mitigation:* explicit spot-check for rule violations in criterion #3; bias toward tightening over loosening on any ambiguity.

## Implementation Order

1. Edit `DETECTION_LABELS`, `SYSTEM_PROMPT`, and JSON template in `annotate_detections.py`
2. Write `test_annotate_detections.py` unit + regression-guard tests
3. Run unit tests (cheap, fast — should pass before any LLM calls)
4. Build injection candidate pool (grep the corpus with the injection queries; cap at 50 per label)
5. Stratified-sample 5k + concatenate injection candidates with a source-tag column
6. Run 3-model audit via `trainr data annotate-detections --routing --sample 5000 ...` plus a separate run on the injection pool
7. Compute agreement + recall metrics; run spot-check script for rule violations
8. **Gate:** if all 3 pass criteria hold → commit the `annotate_detections.py` changes, the new test file, the audit report (as a new doc under `docs/accuracy_runs/`), and this spec; open PR; merge. If not → iterate on definitions, re-audit.

## Non-Goals / Explicit YAGNI

Deliberately NOT in scope for this branch:

- Full 90k annotation run — next branch
- Retrain with detection head — next branch
- Rust port / `src/` changes — third branch (workspace boundary)
- Additional log formats: Windows Event Log, systemd, CloudWatch
- Additional stack trace formats: PHP, Erlang/Elixir
- Additional diff dialects: SVN, hg, context diffs
- Class weighting in detection head loss — belongs to the retrain branch
- Additional semantic labels beyond the 3 — YAGNI until we see how these perform

## Out-of-Scope Follow-Ups (for tracking)

- `textclf-1dd1.013xtr` — dedup pipeline non-idempotence (iter15 finding)
- Per-label F1 tracking in training metrics — belongs to retrain branch
- `pos_weight` in `BCEWithLogitsLoss` for 0.1%-prevalence labels — belongs to retrain branch
