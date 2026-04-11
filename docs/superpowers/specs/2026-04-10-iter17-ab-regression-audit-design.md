# iter17 — A/B Regression Audit Design

**Date:** 2026-04-10
**Branch target:** `iter17-ab-regression-audit` (off `three-head-part-deux`)
**Prior iteration:** [`docs/accuracy_runs/2026-04-10-iteration-16.md`](../../accuracy_runs/2026-04-10-iteration-16.md)
**Status:** Design — awaiting user review before plan.

## Purpose

Before committing ~$400-600 to iter18's 90k annotation run, prove that iter16's longer `SYSTEM_PROMPT` did not silently degrade any of the 40 pre-existing detection labels. iter16's own audit cannot answer this because it has no control group — it only measured absolute agreement under the iter16 prompt. iter17 runs the A/B comparison that iter16 explicitly deferred.

## Scope

iter17 is **A/B regression audit + operational follow-ups only.** It stops before the 90k run.

This follows iter16's "one branch = one seam" lesson: small, focused branches with cheap abort points. Natural seam decomposition from here:

- **iter17 (this):** A/B audit + operational follow-ups. ~$30. One gate outcome.
- **iter18:** 90k annotation run. ~$400-600. Own branch; output is a new parquet committed to DVC.
- **iter19:** Retrain with detection head + post-retrain 5k eval (primary model-quality gate). Own branch.
- **iter20:** Rust port (copy artifacts + smoke test). Own branch per CLAUDE.md workspace boundary.

Gate outcome model:
- **PASS** → iter18 unblocked. The 12 pre-existing "failing" labels in iter16's absolute audit are proven to be noise floors, not iter16 regressions.
- **FAIL** → iter17 iterates on prompt (likely: trim verbosity in the 3 new definition blocks, or reorder to reduce attention dilution on earlier labels), re-audits only the regressed labels, and re-runs the gate. No full 5k re-audit on iteration.

## Key Design Decisions

### Re-annotation strategy: full re-run both sides

Both the iter15 prompt and the iter16 prompt are annotated fresh on the same day, against the same 5065-row input (`training/data/audit/iter16_5k_input.parquet`). Cost ~$30 (vs ~$15 for "trust iter16 parquets as the after side").

**Rationale:**
- Quality and accuracy dominate cost. Past iterations have paid severely in time and LLM spend for chasing phantom data issues. Controlled experiments eliminate those before they start.
- True same-day controlled experiment: only variable is the prompt text. Provider state, model versions, request-time conditions are constant.
- Fresh runs unlock **noise floor data**: comparing the two iter16-prompt runs (fresh vs existing) gives per-label empirical within-prompt variance — useful context for interpreting any FAIL-agreement rows near the threshold.

### Worktree off commit `22bc292`, distinct filenames in main repo

The iter15 prompt state lives in a git worktree at `.worktrees/iter17-iter15-prompt/`, off commit `22bc292` (the last commit with iter15 `SYSTEM_PROMPT` state — Phase 0 iter16 test repairs already applied, tests green).

Both sides write their output parquets to the **main repo's** `training/data/audit/` directory via absolute path. Filenames disambiguate origin:

```
training/data/audit/iter17_ab_iter15_gemini3flash.parquet
training/data/audit/iter17_ab_iter15_sonnet46.parquet
training/data/audit/iter17_ab_iter15_gpt54mini.parquet
training/data/audit/iter17_ab_iter16_gemini3flash.parquet
training/data/audit/iter17_ab_iter16_sonnet46.parquet
training/data/audit/iter17_ab_iter16_gpt54mini.parquet
```

**Rationale:**
- Durable git audit trail: the iter15-side parquets are reproducible from commit `22bc292` + the documented CLI invocation.
- Parallelizable: both sides can run simultaneously since output paths don't collide.
- Single canonical location: analysis code uses one `glob()` on one directory; `git status` in the main repo makes the A/B additions visually obvious.
- Worktree needs its own `uv sync` for a working `trainr` CLI; one-time cost.

**Rejected alternatives:**
- *Checkout juggling in main repo* — serial-only, error-prone if the tree is touched mid-run.
- *File-paste (stash iter16 SYSTEM_PROMPT, paste iter15 version)* — no git audit trail of what actually ran; the "restore" step is a footgun if the run crashes mid-annotation.

### A/B comparison module: new `compare_prompt_versions.py`, with refactor pre-req

Option B from brainstorming: new module, not an extension of `audit_semantic_labels.py`.

**Rationale:**
- One module = one purpose. Delta logic stays quarantined from single-audit code.
- A/B audits will recur: any future prompt change big enough to worry about regressions will want this same tool. Building it properly in iter17 means iter18+ reuse it.
- The "refactor to expose primitives" pre-req is a feature, not a bug — it forces `audit_semantic_labels.py` to have a clean public surface, which is healthier than letting it grow ad hoc.
- Matches iter16's Key Learning #6: "two-stage review paid for itself because audit code was real modules with tests, not scripts." That discipline continues.

**Rejected alternatives:**
- *Extend `audit_semantic_labels.py`* — would grow from 288 → ~500 lines doing two jobs. Invites future bloat.
- *Ad-hoc script in `scripts/`* — no test discipline, no reuse, inconsistent with iter16's module treatment.

### Column asymmetry: tri-categorized report, shared-only gate, dynamic introspection

The A/B has two asymmetries:
1. **iter15-only:** `det_log_lines` (removed in iter16)
2. **iter16-only:** `det_log_content`, `det_stack_trace`, `det_diff_patch` (added in iter16)

The comparison module introspects column sets dynamically (reads both parquets, diffs `det_*` columns) and categorizes every label as `shared | iter15-only | iter16-only`. The gate evaluates only `shared` rows. The report includes all three categories with clear section headers.

**Rationale:**
- Dynamic introspection is robust to future schema drift — no hardcoded `DETECTION_LABELS` list.
- Defensive: the gate literally cannot evaluate a label that isn't `shared` because the column set drives categorization.
- Complete story in the report: the `det_log_lines` farewell agreement score is captured before it disappears from the codebase; the 3 new labels cross-reference to iter16's audit without re-litigating them.

**Explicitly rejected: renaming `det_log_lines` ↔ `det_log_content`.** Per iter16's architectural note, they are *not* semantically equivalent (`log_lines` = 1:1 with sub_type "IS primarily log lines"; `log_content` = cross-cutting "contains log output anywhere"). Treating them as equivalent would mask real signal and put noise on the gate.

### Gate model: hard agreement gate + soft prevalence warning + informational noise floor

Three signals per label, each with a different enforcement level:

| Signal | Threshold | Behavior |
|---|---|---|
| Agreement delta | `|Δagr| > 0.005` | **Hard FAIL-agreement** — blocks iter18 |
| Prevalence ratio | `prev_ratio ∉ [0.5, 2.0]` | **Soft WARN-prevalence** — requires human sign-off in iter17 iteration doc |
| Noise floor | per-label, from within-prompt variance | **Informational** — reported next to every row, used by human reviewer to override FAIL rows within noise |

**Rationale for each level:**

- **Agreement hard gate** matches iter16's documented threshold. Agreement has principled meaning (inter-annotator reliability), a stable threshold across iterations, and `0.005` is below the observable noise floor on most labels.

- **Prevalence soft warning, not hard gate.** A legitimate prompt tightening *should* drop prevalence on previously overfiring labels — that's the whole point of iter16's precision-over-recall approach. A hard prevalence gate would have blocked iter16 itself run backwards. But large shifts (>2x or <0.5x) still deserve human review to catch silent semantic drift. Human sign-off in the iteration doc is the right enforcement level.

- **Noise floor informational only.** A per-label noise floor from one within-prompt run is itself noisy. Using it to set per-label thresholds would bake meta-noise into the gate and make longitudinal reasoning hard. Instead, the gate threshold stays fixed at 0.005 and noise floor data enters the human review loop as context. If iter17's gate reports `FAIL-agreement on det_markdown, Δ=0.007` and the noise floor row says `det_markdown noise_floor=0.009`, the human reviewer overrides the gate and documents why in the iteration doc.

**Override protocol:** any FAIL-agreement row that the human reviewer wants to override must be documented in `docs/accuracy_runs/2026-04-10-iteration-17.md` with:
- Label name, Δagr, noise floor, reason for override
- Explicit statement that the reviewer accepts iter18 risk for this label

If any FAIL-agreement row is **not** overridden, iter18 is blocked.

### Operational follow-ups bundled into iter17

These are small but substantive, needed for iter17 tooling to be sound:

1. **`__main__` guard on `annotate_detections.py`.** Decision: **add the guard.** iter16 lost ~30 min diagnosing `python -m trainr.core.annotate_detections` silently exiting 0 because the module lacked a `__main__` guard. Adding `if __name__ == "__main__": main()` is cheap, restores a useful invocation path, and prevents future silent footguns. The click CLI (`trainr data annotate-detections`) remains the canonical invocation per iter16's actual usage; the guard just makes `python -m` work as a backup.

2. **`INJECTION_PATTERNS` tightening in `build_audit_sample.py`.** Specific fixes from iter16 §Follow-Up:
   - Python: require Traceback header + `File ` adjacency (not prose "at line 5")
   - Rust: exclude `// error-pattern:` prefix (test directives, not runtime traces)
   - Java: require frame adjacency to `Exception in thread` header

Regression tests per pattern: each tightened pattern has a test case with a previously-matching fixture that should now NOT match. **Does not rebuild `iter16_5k_input.parquet`** — reusing the existing file is load-bearing for apples-to-apples A/B, and changing it would invalidate the experiment. The tightening benefits iter18+ audits.

## Components

### Phase 1: Refactor `audit_semantic_labels.py` to expose primitives

Pre-req for the new compare module. Expose as public functions:

- `load_annotator_parquets(paths: list[Path]) -> list[pd.DataFrame]`
- `majority_vote(frames: list[pd.DataFrame], label: str) -> pd.Series`
- `compute_agreement_per_label(frames: list[pd.DataFrame], labels: list[str]) -> dict[str, float]`
- `compute_prevalence_per_label(frames: list[pd.DataFrame], labels: list[str]) -> dict[str, float]`

No behavior change. Existing iter16 audit report reproduces **byte-for-byte** after the refactor (smoke test enforces this).

### Phase 2: `compare_prompt_versions.py`

**Public API:**
```python
def compare_prompt_versions(
    before_parquets: list[Path],
    after_parquets: list[Path],
    input_parquet: Path,
    output_report: Path,
    reference_after_parquets: list[Path] | None = None,  # for noise floor
) -> DeltaReport:
    ...
```

**Click CLI:**
```
trainr data compare-prompts \
    --before 'training/data/audit/iter17_ab_iter15_*.parquet' \
    --after 'training/data/audit/iter17_ab_iter16_*.parquet' \
    --input training/data/audit/iter16_5k_input.parquet \
    --reference-after 'training/data/audit/iter16_5k_*.parquet' \
    --out docs/accuracy_runs/2026-04-10-iter17-regression-report.md
```

**Algorithm:**
1. Load all parquets into DataFrames. Validate each against the input parquet (row count, row order, sub_type column).
2. Discover `det_*` columns in each side's frames. Compute the union, intersection, and per-side uniques.
3. Categorize every `det_*` label: `shared` (in both) / `iter15-only` / `iter16-only`.
4. For each label, compute:
   - `iter15_agr` from iter15 frames (stratified rows only — exclude `audit_source=injected` rows using the existing `audit_source` column from `build_audit_sample.py`)
   - `iter16_agr` from iter16 frames (same row filter)
   - `iter15_prev`, `iter16_prev` — fire rate on stratified rows (majority-of-3)
   - `Δagr = iter16_agr - iter15_agr` (shared labels only)
   - `prev_ratio = iter16_prev / iter15_prev` (shared labels only, with zero-prevalence guard)
5. If `reference_after_parquets` provided, compute per-label noise floor: agreement delta between iter16-fresh parquets and iter16-existing parquets (`training/data/audit/iter16_5k_*.parquet`). Available for every label present in the iter16 prompt, including the 3 new semantic labels.
6. Compute verdict per label:
   - `shared` + `|Δagr| > 0.005` → `FAIL-agreement`
   - `shared` + `prev_ratio ∉ [0.5, 2.0]` → `WARN-prevalence` (can co-occur with `FAIL-agreement`)
   - `shared` + neither → `PASS`
   - `iter15-only` / `iter16-only` → no verdict (categorical)
7. Emit markdown report (section structure below).
8. Return `DeltaReport` object for programmatic access by tests.

**Schema-drift guards:** raise `ValueError` with explanatory message if parquets have different row counts, if `sub_type` column is missing, if `audit_source` column is missing, if any parquet is empty, or if both sides have zero shared labels.

**NaN-aware aggregation:** follow iter16 Phase 7 patterns. Missing data never silently resolves to PASS.

### Phase 3: Worktree setup + parallel annotation runs

**Setup:**
```bash
git worktree add .worktrees/iter17-iter15-prompt 22bc292
cd .worktrees/iter17-iter15-prompt/training
uv sync
# sanity: verify SYSTEM_PROMPT lacks the 3 new labels
grep -c 'log_content' trainr/core/annotate_detections.py  # expect 0
grep -c 'stack_trace' trainr/core/annotate_detections.py  # expect 0
grep -c 'diff_patch' trainr/core/annotate_detections.py   # expect 0
grep -c 'log_lines' trainr/core/annotate_detections.py    # expect >0
```

Model IDs here are placeholders for the actual OpenRouter model strings used in iter16 (`google/gemini-3-flash`, `anthropic/claude-sonnet-4.6`, `openai/gpt-5.4-mini`); the plan will substitute the literal values.

**Run iter15 side (from worktree). `$REPO_ROOT` is the absolute path to the main repo:**
```bash
for model_slug in gemini3flash sonnet46 gpt54mini; do
    uv run trainr data annotate-detections \
        --input "$REPO_ROOT/training/data/audit/iter16_5k_input.parquet" \
        --model "<openrouter-id-for-$model_slug>" \
        --output "$REPO_ROOT/training/data/audit/iter17_ab_iter15_${model_slug}.parquet" &
done
wait
```

**Run iter16 side (from main repo, parallel):**
```bash
for model_slug in gemini3flash sonnet46 gpt54mini; do
    uv run trainr data annotate-detections \
        --input training/data/audit/iter16_5k_input.parquet \
        --model "<openrouter-id-for-$model_slug>" \
        --output "training/data/audit/iter17_ab_iter16_${model_slug}.parquet" &
done
wait
```

Both sides run concurrently; each side's 3 models are also parallel. Total wall clock ≈ max single-model time.

### Phase 4: Comparison + report

Run the `trainr data compare-prompts` CLI as shown above. Output: `docs/accuracy_runs/2026-04-10-iter17-regression-report.md`.

## Data Flow

```
iter16_5k_input.parquet (unchanged, apples-to-apples anchor)
        │
        ├─► .worktrees/iter17-iter15-prompt/@22bc292 ─► iter17_ab_iter15_{gemini3flash,sonnet46,gpt54mini}.parquet
        │
        └─► main repo @HEAD                          ─► iter17_ab_iter16_{gemini3flash,sonnet46,gpt54mini}.parquet
                                                          │
                                                          │   existing iter16_5k_{...}.parquet (noise floor reference)
                                                          │
                                                          ▼
                                             compare_prompt_versions
                                                          │
                                                          ▼
                                       2026-04-10-iter17-regression-report.md
                                                          │
                                                          ▼
                                              iter17 gate decision
                                                          │
                                              ┌───────────┴───────────┐
                                              ▼                       ▼
                                        PASS → iter18           FAIL → prompt iteration
                                                                      + targeted re-audit
```

## Report Schema

**File:** `docs/accuracy_runs/2026-04-10-iter17-regression-report.md`

**Structure:**

```markdown
# iter17 A/B Regression Audit Report

## Gate verdict
**PASS** / **FAIL** / **PASS with human override**

Summary: N shared labels, M FAIL-agreement, P WARN-prevalence, Q overrides.

## Shared labels (gate applies)

| label | iter15_agr | iter16_agr | Δagr | iter15_prev | iter16_prev | prev_ratio | noise_floor | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| det_python | 0.9987 | 0.9985 | -0.0002 | 0.031 | 0.030 | 0.97 | 0.0012 | PASS |
| ... | | | | | | | | |

## iter15-only labels (removed in iter16, context only)
| label | iter15_agr | iter15_prev | note |
|---|---:|---:|---|
| det_log_lines | 0.9945 | 0.009 | Replaced by det_log_content in iter16 |

## iter16-only labels (new in iter16, cross-ref iter16 audit)
| label | iter16_agr | iter16_prev | iter16_audit_ref |
|---|---:|---:|---|
| det_log_content | 0.9986 | 0.011 | Passed 0.995 threshold in iter16 audit |
| det_stack_trace | 0.9997 | 0.004 | Passed 0.995 threshold in iter16 audit |
| det_diff_patch | 0.9999 | 0.002 | Passed 0.995 threshold in iter16 audit |

## FAIL-agreement rows (if any)
Detailed per-row breakdown with noise floor context for human review.

## WARN-prevalence rows (if any)
Detailed per-row breakdown requiring human sign-off.

## Noise floor table (informational)
Per-label within-prompt variance from iter16-fresh vs iter16-existing runs.
```

## Test Plan

### Unit tests for `compare_prompt_versions.py`

Fixture DataFrames covering:

- **Happy path (all shared):** 3 labels, 2 frames per side, no asymmetry, all PASS
- **iter15-only label:** 1 label only in iter15 frames → categorized, no verdict
- **iter16-only label:** 1 label only in iter16 frames → categorized, no verdict
- **Mixed asymmetry:** 1 shared + 1 iter15-only + 1 iter16-only
- **Hard-fail row:** Δagr = 0.006 → FAIL-agreement
- **Hard-fail at exact threshold:** Δagr = 0.005 → PASS (strict `>`)
- **Soft-warn row:** prev_ratio = 0.4 → WARN-prevalence
- **Both fail+warn:** same row has both
- **NaN handling:** one frame has NaN in a label column → doesn't silently PASS
- **Schema drift row count mismatch:** raises ValueError with clear message
- **Schema drift missing sub_type:** raises ValueError
- **Empty parquet:** raises ValueError
- **Zero shared labels:** raises ValueError
- **Dynamic column introspection:** correctly identifies shared set when frames have different column sets
- **Zero-prevalence prev_ratio:** handled gracefully (e.g., `inf` or sentinel, not crash)
- **Noise floor with reference parquets:** computed correctly
- **Noise floor without reference parquets:** field omitted, not crash

### Regression tests for tightened `INJECTION_PATTERNS`

Each tightened pattern has a test case with a fixture that **previously matched and should now NOT match**:

- Python: prose containing "at line 42" without Traceback header → no match
- Rust: `// error-pattern:thread 'main' panicked at` test directive → no match
- Java: `at com.foo.Bar.method(Bar.java:15)` without `Exception in thread` header above → no match

Plus positive test cases that should still match (regression guard on the tightening not going too far).

### Refactor smoke test for `audit_semantic_labels.py`

- Reproduce iter16's audit report byte-for-byte from existing parquets after the refactor. If the output differs from the committed iter16 audit report, the refactor broke something.

## Out of Scope

Deferred to later iterations:

- **iter18:** 90k annotation run, retraining with detection head, class weighting for low-prevalence semantic labels, per-label F1 tracking in training metrics.
- **iter19:** Post-retrain 5k eval (category + sub_type regression, per-detection-label F1).
- **iter20:** Rust artifact refresh, Rust smoke test, CI pass.
- **Not blocking any branch:**
  - Rebuilding `iter16_5k_input.parquet` with tightened injection patterns (would invalidate the A/B).
  - Addressing pre-existing `plain` / `markdown` agreement floor beyond what the A/B naturally clarifies.
  - `textclf-1dd1.013xtr` dedup pipeline non-idempotence (iter15 finding, still open).

## Cost

| Item | Cost |
|---|---:|
| iter15 side: 5065 rows × 3 models | ~$15 |
| iter16 side (fresh): 5065 rows × 3 models | ~$15 |
| **Total annotation** | **~$30** |
| Tooling, review, worktree setup | $0 (time only) |

## Estimated Commit Plan

~10-12 commits, matching iter16's shape and TDD + two-stage-review discipline.

- **P0a:** `feat(detections): add __main__ guard to annotate_detections.py`
- **P0b:** `fix(audit): tighten INJECTION_PATTERNS for python/rust/java stack traces` + regression tests
- **P1:** `refactor(audit): expose primitives from audit_semantic_labels.py` + byte-for-byte smoke test
- **P2a:** `feat(audit): compare_prompt_versions.py skeleton + happy-path test`
- **P2b:** `feat(audit): column categorization + tri-report structure`
- **P2c:** `feat(audit): hard agreement gate + soft prevalence gate + verdict logic`
- **P2d:** `feat(audit): noise floor reporting`
- **P2e:** `feat(cli): trainr data compare-prompts` + integration test
- **P3:** worktree setup + parallel annotation runs (parquets only, no code commits)
- **P4:** `docs(iter17): regression report + iteration doc`
- **P5:** gate decision commit (PASS/FAIL/override)

Two-stage review per code phase: spec-compliance review + code-quality review, per iter16 precedent.

## Gate Decision Protocol

At end of iter17, before closing the branch:

1. Read the iter17 regression report.
2. Enumerate all `FAIL-agreement` rows.
3. For each, compare `|Δagr|` against `noise_floor`:
   - `|Δagr| ≤ 2 × noise_floor` → eligible for override. Document in iteration doc with reasoning.
   - `|Δagr| > 2 × noise_floor` → real regression. Block iter18, iterate on prompt, re-audit regressed labels.
4. Enumerate all `WARN-prevalence` rows. For each, review manually and document sign-off or objection in iteration doc.
5. Final gate decision committed as `docs(iter17): gate decision — {PASS|FAIL}` with a one-paragraph justification.

If the gate PASSes, iter18 branch is unblocked. If it FAILs without eligible overrides, iter17 iterates.
