# Session Handoff: iter17 — Phase 2 Complete, Paused Before Phase 3

**Branch:** `iter17-ab-regression-audit` (off `main`, 17 commits ahead)
**Last session ended:** 2026-04-11
**Next session starts at:** Phase 3 (annotation runs, ~$45 LLM budget)

---

## TL;DR

iter17 is an A/B regression audit gating iter18's ~$400-600 90k annotation run on proof that iter16's longer `SYSTEM_PROMPT` did not silently degrade any of the 40 pre-existing detection labels. The plan has 18 tasks across 6 phases. Phases 0-2 (11 code tasks) are complete, tested, and committed. Phases 3-5 (annotation runs + report + gate decision) remain and need real LLM spend + human judgment.

**To resume:** read this doc, verify the branch is checked out, then start Task 3.1 (git worktree setup) per the implementation plan at `docs/superpowers/plans/2026-04-10-iter17-ab-regression-audit.md`.

---

## What's Done (Phases 0-2, 11 code tasks)

### Phase 0 — Operational Fixes ✅

| Task | Commit | What |
|---|---|---|
| 0.1 | `7eafced` | Added `if __name__ == "__main__": main()` guard to `training/trainr/core/annotate_detections.py`. `python -m` invocation now works as a backup diagnostic path. |
| 0.2 | `497588c` | Tightened `INJECTION_PATTERNS["stack_trace"]` in `build_audit_sample.py`. Python requires Traceback+File adjacency; Java requires Exception-header+frame adjacency; Rust uses `(?m)^` line anchor to exclude `// error-pattern:` test directives. Polars-compatible (uses `[\s\S]{0,N}` instead of `(?s:.)`, `(?m)^` instead of lookbehind). Does NOT rebuild `iter16_5k_input.parquet` — reusing existing file is load-bearing for apples-to-apples A/B. |

### Phase 1 — Audit Helpers ✅

| Task | Commit(s) | What |
|---|---|---|
| 1.1 | `8f1ef7d`, `81adffc` | `compute_prevalence_per_label(dfs: dict[str, pl.DataFrame]) -> dict[str, float]` — majority-of-N fire rate per label on stratified rows. Filters internally (callers pass raw frames). Follow-up commit added schema-divergence coverage. |
| 1.2 | `99159b4`, `0d67183` | `load_annotator_parquets(paths: list[Path]) -> dict[str, pl.DataFrame]` with `EXPECTED_MODEL_SLUGS = frozenset({"gemini3flash", "sonnet", "gpt54mini"})` constant. Parses model slug from filename, asserts slug set equality, rejects wrong count / duplicates / unknown slugs. Follow-up commit added true-duplicate slug path coverage. |
| 1.3 | `e71e16c` | Byte-for-byte reproducibility smoke test for the iter16 audit report. Regenerates the committed report from archived parquets and fails loud on any drift. **Confirms Phase 1 made no regressions to existing audit output.** |

### Phase 2 — `compare_prompt_versions.py` Module ✅

| Task | Commit | What |
|---|---|---|
| 2.1 | `2b78826` | New module `training/trainr/core/compare_prompt_versions.py` with `LabelCategory`, `LabelVerdict`, `LabelRow`, `DeltaReport` dataclasses. `compare_prompt_versions()` function handles the happy-path case. `_compute_prev_ratio()` implements zero-handling: `0/0 → 1.0`, `0/>0 → inf`, `>0/0 → 0.0`. |
| 2.2 | `37383a8` | Dynamic `det_*` column introspection → `shared / iter15-only / iter16-only` categorization. Hard schema assertion between `after_frames` and `noise_floor_frames` (noise floor correctness depends on identical column sets). Partial-metric `LabelRow` entries for asymmetric labels. |
| 2.3 | `445751c` | `_compute_verdict()` pure helper combining hard agreement gate (`|Δagr| > 0.005`) with soft prevalence gate (`prev_ratio ∉ [0.5, 2.0]`). Verdicts can co-occur (`FAIL_AND_WARN`). Override eligibility is reported but NOT applied in the verdict — human review applies it per the Gate Decision Protocol. |
| 2.4 | `0846735` | `_assert_fingerprint_matches_input()` helper — hashes all non-`det_*` columns row-by-row against the input parquet. Fingerprint validation runs FIRST in `compare_prompt_versions()` body, before the schema assertion and before metric computation. Error messages name the diverging row index, column name, and source slug (`iter15/<slug>`, `iter16a/<slug>`, or `iter16b/<slug>`). |
| 2.5 | `739757f` | `format_delta_report(report) -> str` — pure markdown formatter rendering all 8 sections from the spec's §Report Schema (header, gate verdict summary, shared labels table, iter15-only section, iter16-only section, FAIL-agreement detail with override-eligibility hint, WARN-prevalence detail, noise floor table). Three formatting helpers: `_fmt_float` (4 dp), `_fmt_signed` (signed 4 dp), `_fmt_ratio` (3 dp). `None` renders as `—`. |
| 2.6 | `62cbd6d` | `trainr data compare-prompts` CLI wire-up. Module `main()` parses argv (argparse), expands globs via Python's `glob` module (NOT shell), calls `load_annotator_parquets` with slug validation, runs the comparison, writes the report, exits `2` on any FAIL-agreement (report is written BEFORE exit). Click wire-up in `training/trainr/commands/data.py`. |

### Test Coverage

**24 tests** in `training/tests/test_compare_prompt_versions.py`:

| Class | Count | Coverage |
|---|---:|---|
| `TestHappyPath` | 1 | All-shared, all-PASS baseline |
| `TestColumnCategorization` | 4 | iter15-only, iter16-only, mixed asymmetry, schema mismatch raise |
| `TestVerdictLogic` | 7 | PASS, FAIL_AGREEMENT, WARN_PREVALENCE (low + high), FAIL_AND_WARN co-occur, 0/0 no-warn, 0/>0 warn |
| `TestRowAlignmentFingerprint` | 4 | Passing, text-mutation-with-row-index, row-count-mismatch, missing-non-det-column |
| `TestFormatReport` | 3 | All sections present, FAIL count in summary, inf-ratio clean formatting |
| `TestMainCLI` | 3 | End-to-end from parquets, exits 2 on FAIL, raises on empty glob |
| Constants | 2 | `AGREEMENT_DELTA_THRESHOLD == 0.005`, `PREVALENCE_RATIO_LOW == 0.5`, `PREVALENCE_RATIO_HIGH == 2.0` |

Plus **14 tests** in `training/tests/test_audit_semantic_labels.py` (up from 4 pre-iter17):
- 5 `TestComputePrevalencePerLabel` (with schema-divergence coverage)
- 6 `TestLoadAnnotatorParquets` (with true-duplicate slug coverage)
- 1 `TestIter16ReportReproducibility` (byte-for-byte iter16 audit reproduction)
- 2 original/pre-existing tests

Plus **6 new tests** in `training/tests/test_build_audit_sample.py` under `TestStackTraceInjectionPatternTightening` (3 anti-cases, 3 positive cases).

**All tests green.** An end-to-end smoke test of the compare CLI against real iter16 parquets (running iter16 vs iter16 as both sides) produced a valid PASS report with all 42 shared labels and zero deltas — confirming the full code path works against real data.

### Architecture Summary

- **Workspace boundary respected**: Python-only changes under `training/`. Zero Rust code touched.
- **Polars-end-to-end**: no pandas introduced.
- **Dictionary-keyed-by-slug convention**: `dict[str, pl.DataFrame]` throughout, matching existing `compute_agreement_across_models`.
- **Pure functions where possible**: `_compute_verdict`, `_compute_prev_ratio`, `format_delta_report`, all formatting helpers.
- **Defensive guards at every boundary**: fingerprint before schema assert before metric compute; raise ValueError with specific row/column/slug context.

---

## What's Next (Phases 3-5, 7 tasks, ~$45 LLM spend)

### Phase 3 — Annotation Runs (~$45, real money)

Produces 9 parquet files in `training/data/audit/`. Spec concurrency cap: **≤ 2 annotation jobs in flight at any time, 40 in-flight LLM requests total**.

**Model ID lookup (do this first in the next session):**
```bash
grep -E 'google/gemini|anthropic/claude|openai/gpt' docs/accuracy_runs/2026-04-10-iteration-16.md
```

Expected mapping (verify against iter16 iteration doc):
- `gemini3flash` → `google/gemini-3-flash-preview`
- `sonnet` → `anthropic/claude-sonnet-4.6`
- `gpt54mini` → `openai/gpt-5.4-mini`

**Task 3.1 — Worktree setup.** Off commit `22bc292` (iter15 prompt state, Phase 0 test repairs applied):
```bash
git worktree add .worktrees/iter17-iter15-prompt 22bc292
cd .worktrees/iter17-iter15-prompt/training && uv sync && cd -
# Sanity: verify iter15 prompt state
grep -c 'log_content\|stack_trace\|diff_patch' .worktrees/iter17-iter15-prompt/training/trainr/core/annotate_detections.py  # expect 0
grep -c 'log_lines' .worktrees/iter17-iter15-prompt/training/trainr/core/annotate_detections.py  # expect >0
export REPO_ROOT="$(git rev-parse --show-toplevel)"
```

**Task 3.2 — iter15 side (3 runs, ~$15).** Run from worktree, one model at a time (concurrency cap):
```bash
cd .worktrees/iter17-iter15-prompt/training
for model_slug_pair in "gemini3flash google/gemini-3-flash-preview" "sonnet anthropic/claude-sonnet-4.6" "gpt54mini openai/gpt-5.4-mini"; do
    slug=${model_slug_pair% *}
    model=${model_slug_pair#* }
    uv run trainr data annotate-detections \
        --input "$REPO_ROOT/training/data/audit/iter16_5k_input.parquet" \
        --model "$model" \
        --backend openrouter \
        --output "$REPO_ROOT/training/data/audit/iter17_ab_iter15_${slug}.parquet"
done
```

Verify: 3 parquets, each 5065 rows, each has `det_log_lines` column, none has `det_log_content`.

**Task 3.3 — iter16a side (3 runs, ~$15).** From main repo. May run concurrently with Task 3.2 in a separate terminal (2 jobs in flight max across both sides). Same loop but change `iter15` → `iter16a` in output paths, run from `"$REPO_ROOT"/training`:
```bash
cd "$REPO_ROOT"/training
for model_slug_pair in "gemini3flash google/gemini-3-flash-preview" "sonnet anthropic/claude-sonnet-4.6" "gpt54mini openai/gpt-5.4-mini"; do
    slug=${model_slug_pair% *}
    model=${model_slug_pair#* }
    uv run trainr data annotate-detections \
        --input data/audit/iter16_5k_input.parquet \
        --model "$model" \
        --backend openrouter \
        --output "data/audit/iter17_ab_iter16a_${slug}.parquet"
done
```

Verify: 3 parquets, each 5065 rows, each has `det_log_content`, none has `det_log_lines`.

**Task 3.4 — iter16b side (3 runs, ~$15).** After iter15 and iter16a complete. Same as 3.3 but output `iter16b_{slug}.parquet`. This is the same-prompt noise floor companion.

**Sanity after Task 3.4:** iter16a vs iter16b should have slightly different fire patterns (not identical — that would signal a duplicate run or deterministic backend). Spot-check:
```bash
uv run --directory training python -c "
import polars as pl
a = pl.read_parquet('training/data/audit/iter17_ab_iter16a_gemini3flash.parquet')
b = pl.read_parquet('training/data/audit/iter17_ab_iter16b_gemini3flash.parquet')
same = (a['det_python'] == b['det_python']).sum()
print(f'det_python agreement: {same}/{len(a)}')
"
```
Expected: high but not 100% (typically >95% but <100% due to LLM nondeterminism).

### Phase 4 — Comparison + Report

**Task 4.1** — Run `trainr data compare-prompts`:
```bash
cd /home/bfirestone/devspace/personal/sentiolabs/text-classifier-rs
uv run --directory training trainr data compare-prompts \
    --before      "$(pwd)/training/data/audit/iter17_ab_iter15_*.parquet" \
    --after       "$(pwd)/training/data/audit/iter17_ab_iter16a_*.parquet" \
    --noise-floor "$(pwd)/training/data/audit/iter17_ab_iter16b_*.parquet" \
    --input       "$(pwd)/training/data/audit/iter16_5k_input.parquet" \
    --output      "$(pwd)/docs/accuracy_runs/2026-04-10-iter17-regression-report.md"
```
Exit 0 = PASS. Exit 2 = FAIL (at least one FAIL-agreement row; report still written).

Commit the report:
```bash
git add docs/accuracy_runs/2026-04-10-iter17-regression-report.md
git commit -m "docs(iter17): A/B regression audit report"
```

**Task 4.2** — Write iter17 iteration doc at `docs/accuracy_runs/2026-04-10-iteration-17.md` following the template at the bottom of the implementation plan (Task 4.2 section). Key sections to fill in:
- Gate decision: PASS / FAIL / PASS-with-override
- Per-row human review for each FAIL-agreement row (override eligibility: `|Δagr| ≤ 2 × noise_floor`)
- Per-row review for each WARN-prevalence row (semantic judgment on whether drift is legitimate tightening or silent bug)
- Key learnings

### Phase 5 — Gate Decision

**Task 5.1** — Final explicit gate decision commit + push:
```bash
# PASS case:
git commit --allow-empty -m "docs(iter17): gate decision — PASS"
# FAIL case (back to Task 3.3 with a revised prompt; each iteration is ~$30: iter16a + iter16b rerun):
git commit --allow-empty -m "docs(iter17): gate decision — FAIL"

git push
```

**FAIL path gotcha** — if the gate fails and a prompt iteration is needed, rerun **both** iter16a AND iter16b against the new prompt (~$30 per iteration). Reusing the cached iter16b from the old prompt would turn the noise floor into a prompt-drift measurement. iter15 parquets stay cached (the iter15 prompt state doesn't change).

---

## Gate Decision Protocol Reference

From the spec (`docs/superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md`):

**Hard gate (blocks iter18):** any shared label with `|Δagr| > 0.005` → FAIL-agreement. Overridable if `|Δagr| ≤ 2 × noise_floor` with documented reviewer sign-off.

**Soft gate (requires sign-off):** any shared label with `prev_ratio ∉ [0.5, 2.0]` → WARN-prevalence. Needs per-row human review in the iteration doc.

**Noise floor:** `|agr(iter16a) - agr(iter16b)|` per label. Informational floor for override eligibility; does NOT adjust the 0.005 threshold.

**iter15-only labels** (`det_log_lines` removed in iter16): reported for context but not gated.

**iter16-only labels** (`det_log_content`, `det_stack_trace`, `det_diff_patch` added in iter16): reported for context; cross-ref with iter16's own audit report.

---

## Gotchas and Notes for the Next Session

1. **Pyright diagnostic noise is expected.** The workspace-root Pyright can't resolve `polars` / `pytest` / `trainr.core.*` imports because they live in `training/.venv`. This cascades into false-positive "symbol not accessed" / "not defined" hints on otherwise-working code. Every Task in this session hit these warnings; all were verified false positives by running the actual tests under `uv run --directory training pytest ...`. Ignore unless you see something truly new.

2. **Test runner convention:** all `pytest` invocations go through `uv run --directory training pytest ...`. Running `pytest` directly from the repo root won't work — polars et al live in `training/.venv`.

3. **CLI runtime:** `trainr data <subcommand>` only works from `training/` or with `uv run --directory training trainr ...`.

4. **10 pre-existing test failures in unrelated modules.** `test_eval_onnx`, `test_pull_real_data`, `test_vote_labels`, `test_voting_pilot` have pre-existing failures that are NOT caused by iter17 changes. They were present before Task 0.1 and shouldn't gate iter17 landing.

5. **The archived iter16 parquets (`training/data/audit/iter16_5k_*.parquet`) must NOT be used as a noise floor reference.** They were produced before commit `c1ec175` (the CSV-log refinement), so comparing them against fresh iter16 runs measures prompt drift, not same-prompt variance. This was caught by SPEC_REVIEW during brainstorming and is why we use the fresh iter16a + iter16b approach.

6. **`iter16_5k_input.parquet` is immutable across iter17.** Do NOT rebuild it under any circumstances — doing so would invalidate the apples-to-apples A/B comparison. The injection-regex tightening in Task 0.2 was explicitly designed NOT to rebuild it (benefits are deferred to iter18+ audits).

7. **Concurrency discipline:** max 2 annotation jobs in flight, 40 in-flight LLM requests total. Running all 9 parquets in parallel would create rate-limit asymmetry and weaken the controlled-experiment claim.

8. **iter17 doesn't touch retraining or the Rust artifact.** Those are iter19 and iter20 respectively, each on their own branch per the spec's seam decomposition.

---

## Branch State at Session End

- **Branch:** `iter17-ab-regression-audit` (off `main`, 17 commits ahead)
- **Last commit:** `62cbd6d feat(cli): trainr data compare-prompts`
- **Uncommitted files:** this handoff doc (being committed in the same pause-and-push step)
- **Total tests in compare module:** 24 (all green)
- **Full audit suite:** 75 passed, 1 skipped (no regressions)
- **Annotation parquets for Phase 3:** 0 of 9 produced
- **Regression report:** not yet generated
- **Iteration doc:** not yet written
- **Gate decision:** not yet made

**Cost spent so far this branch:** $0 (code work only).
**Cost remaining to complete iter17:** ~$45 base + ~$30 per prompt iteration on FAIL.

---

## Plan and Spec References

- **Spec:** [`docs/superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md`](superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md)
- **Plan:** [`docs/superpowers/plans/2026-04-10-iter17-ab-regression-audit.md`](superpowers/plans/2026-04-10-iter17-ab-regression-audit.md)
- **Prior iteration:** [`docs/accuracy_runs/2026-04-10-iteration-16.md`](accuracy_runs/2026-04-10-iteration-16.md)

To resume implementation in a future session:

1. `git checkout iter17-ab-regression-audit`
2. Read this handoff doc
3. Confirm OpenRouter account is funded for ~$45
4. Verify exact model IDs via `grep -E 'google/|anthropic/|openai/' docs/accuracy_runs/2026-04-10-iteration-16.md`
5. Start with Task 3.1 per the plan
