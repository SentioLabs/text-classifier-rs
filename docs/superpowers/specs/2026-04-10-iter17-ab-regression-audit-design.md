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

- **iter17 (this):** A/B audit + operational follow-ups. ~$45. One gate outcome.
- **iter18:** 90k annotation run. ~$400-600. Own branch; output is a new parquet committed to DVC.
- **iter19:** Retrain with detection head + post-retrain 5k eval (primary model-quality gate). Own branch.
- **iter20:** Rust port (copy artifacts + smoke test). Own branch per CLAUDE.md workspace boundary.

Gate outcome model:
- **PASS** → iter18 unblocked. The 12 pre-existing "failing" labels in iter16's absolute audit are shown to be pre-existing floors (stable across prompt versions), not iter16-induced regressions.
- **FAIL** → iter17 iterates on prompt (likely: trim verbosity in the 3 new definition blocks, or reorder to reduce attention dilution on earlier labels) and re-runs **both** iter16 sides. Each iteration costs ~$30 (iter16a **and** iter16b rerun on all 5065 rows × 3 models each — the annotator emits all `det_*` columns in a single global-`SYSTEM_PROMPT` call, so partial label re-annotation is not possible, **and** the noise floor must come from a pair of runs on the *new* prompt, not the original iter16b). The iter15-side parquets are cached and reused across iterations. Only the *report* filters to the regressed labels for readability.

> **Why both iter16 sides must rerun on every prompt iteration:** noise floor is defined as `|agr(iter16a) - agr(iter16b)|` where both sides use the *same* prompt. If a prompt iteration reruns only iter16a against a cached iter16b (on the old prompt), the resulting "noise floor" measures prompt drift between the two iter16 revisions instead of same-prompt LLM variance — the exact failure mode Codex caught for the original archived-parquet plan. The iteration cost is therefore $30 not $15.

## Key Design Decisions

### Re-annotation strategy: iter15 ×1 + iter16 ×2 (~$45 total)

The A/B runs three fresh annotation passes on the same day against the same 5065-row input (`training/data/audit/iter16_5k_input.parquet`):

1. **iter15 prompt × 3 models × 1 run** (~$15) — the "before" side.
2. **iter16 prompt × 3 models × run A** (~$15) — the "after" side.
3. **iter16 prompt × 3 models × run B** (~$15) — independent same-prompt run for empirical **noise floor** (within-prompt LLM nondeterminism).

**Rationale:**
- Quality and accuracy dominate cost. Past iterations have paid severely in time and LLM spend for chasing phantom data issues. Controlled experiments eliminate those before they start.
- True same-day controlled experiment: only variable between runs 1 and 2 is the prompt text. Provider state, model versions, request-time conditions are held constant.
- Two fresh iter16 runs (2 and 3) give real per-label same-prompt variance — the only defensible noise floor for the gate.
- iter15's noise floor is assumed equal to iter16's: within-prompt variance is primarily LLM nondeterminism (temperature, sampling, provider-side routing), not prompt-sensitive, so one measurement on iter16 is a defensible proxy for both sides. Running iter15 twice would cost another $15 for marginal rigor gain.

**Archived iter16 parquets (`training/data/audit/iter16_5k_*.parquet`) are NOT used as a noise-floor reference.** They were produced before the `c1ec175` CSV-log refinement, so comparing them against fresh iter16 runs would measure *prompt drift*, not same-prompt variance — a subtle but gate-invalidating mistake. The archived parquets stay on disk for historical reference only and do not feed the compare module.

### Worktree off commit `22bc292`, distinct filenames in main repo

The iter15 prompt state lives in a git worktree at `.worktrees/iter17-iter15-prompt/`, off commit `22bc292` (the last commit with iter15 `SYSTEM_PROMPT` state — Phase 0 iter16 test repairs already applied, tests green).

Both sides write their output parquets to the **main repo's** `training/data/audit/` directory via absolute path. Filenames disambiguate origin:

```
training/data/audit/iter17_ab_iter15_gemini3flash.parquet
training/data/audit/iter17_ab_iter15_sonnet.parquet
training/data/audit/iter17_ab_iter15_gpt54mini.parquet
training/data/audit/iter17_ab_iter16a_gemini3flash.parquet
training/data/audit/iter17_ab_iter16a_sonnet.parquet
training/data/audit/iter17_ab_iter16a_gpt54mini.parquet
training/data/audit/iter17_ab_iter16b_gemini3flash.parquet
training/data/audit/iter17_ab_iter16b_sonnet.parquet
training/data/audit/iter17_ab_iter16b_gpt54mini.parquet
```

Nine parquets total: three runs (`iter15`, `iter16a`, `iter16b`) × three models. `iter16a` is the primary "after" side that enters the A/B gate; `iter16b` is the noise-floor companion compared against `iter16a`.

**Rationale:**
- Durable git audit trail: the iter15-side parquets are reproducible from commit `22bc292` + the documented CLI invocation.
- No output-path collisions: all nine parquets have distinct filenames, so the concurrency cap (see Phase 3) is the only constraint on scheduling — not filesystem contention.
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

### Gate model: hard agreement gate, soft prevalence warning, noise-floor-bounded override

Three signals per label, each with a defined role in the gate decision:

| Signal | Threshold | Behavior |
|---|---|---|
| Agreement delta | `|Δagr| > 0.005` | **Hard FAIL-agreement** — blocks iter18 unless overridden per protocol below |
| Prevalence ratio | `prev_ratio ∉ [0.5, 2.0]` | **Soft WARN-prevalence** — requires human sign-off in iter17 iteration doc |
| Noise floor | per-label, from iter16a vs iter16b | **Bounds override eligibility** for FAIL-agreement rows (see protocol) |

**Rationale for each level:**

- **Agreement hard gate** matches iter16's documented threshold. Agreement has principled meaning (inter-annotator reliability), a stable threshold across iterations, and `0.005` is below the observable noise floor on most labels.

- **Prevalence soft warning, not hard gate.** A legitimate prompt tightening *should* drop prevalence on previously overfiring labels — that's the whole point of iter16's precision-over-recall approach. A hard prevalence gate would have blocked iter16 itself run backwards. But large shifts (>2x or <0.5x) still deserve human review to catch silent semantic drift. Human sign-off in the iteration doc is the right enforcement level.

- **Noise floor bounds override eligibility, does not adjust the gate threshold.** The 0.005 agreement gate stays fixed across iterations for longitudinal consistency — noise floor never moves the pass/fail line itself. But when a FAIL-agreement row needs human review, the noise floor tells the reviewer whether the delta is plausibly within same-prompt variance (and therefore overridable) or clearly outside it (real regression). Using a 2× multiplier makes the override criterion testable and deterministic rather than freeform judgment.

**Override protocol:** a FAIL-agreement row may be overridden by the human reviewer if and only if `|Δagr| ≤ 2 × noise_floor` for that label. Override must be documented in `docs/accuracy_runs/2026-04-10-iteration-17.md` with:
- Label name, Δagr, noise floor, `|Δagr| / noise_floor` ratio
- Explicit statement that the reviewer accepts iter18 risk for this label

Any FAIL-agreement row with `|Δagr| > 2 × noise_floor` is a real regression and **cannot be overridden** — iter18 is blocked until iter17 iterates on the prompt and the rerun passes the gate.

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

Pre-req for the new compare module. Expose as public functions (all types use `polars`, matching the existing audit stack — no `pandas`):

- `load_annotator_parquets(paths: list[Path]) -> list[pl.DataFrame]`
- `majority_vote(frames: list[pl.DataFrame], label: str) -> pl.Series`
- `compute_agreement_per_label(frames: list[pl.DataFrame], labels: list[str]) -> dict[str, float]`
- `compute_prevalence_per_label(frames: list[pl.DataFrame], labels: list[str]) -> dict[str, float]`

No behavior change. Existing iter16 audit report reproduces **byte-for-byte** after the refactor (smoke test enforces this).

### Phase 2: `compare_prompt_versions.py`

**Public API:**
```python
def compare_prompt_versions(
    before_parquets: list[Path],
    after_parquets: list[Path],
    noise_floor_parquets: list[Path],
    input_parquet: Path,
    output_report: Path,
) -> DeltaReport:
    ...
```

`before_parquets` = iter15 × 3 models. `after_parquets` = iter16a × 3 models (the A/B after side). `noise_floor_parquets` = iter16b × 3 models (same-prompt companion for computing per-label noise floor against iter16a). All three lists are required — there is no fallback to archived parquets.

**Click CLI:**
```
trainr data compare-prompts \
    --before        'training/data/audit/iter17_ab_iter15_*.parquet' \
    --after         'training/data/audit/iter17_ab_iter16a_*.parquet' \
    --noise-floor   'training/data/audit/iter17_ab_iter16b_*.parquet' \
    --input         training/data/audit/iter16_5k_input.parquet \
    --output        docs/accuracy_runs/2026-04-10-iter17-regression-report.md
```

Globs are expanded **inside the CLI** via Python's `glob` module, not by the shell. Quoted glob arguments work regardless of caller environment, and the CLI asserts that each glob resolves to exactly 3 parquets (one per model) — mismatch raises a clear error before any compute.

**Algorithm:**
1. Expand each glob via Python's `glob` module. For each glob, **parse the model slug out of each filename** (regex `iter17_ab_<side>_(?P<slug>[a-z0-9]+)\.parquet`) and assert:
   - Exactly 3 parquets matched.
   - The slug set equals the expected set `{gemini3flash, sonnet, gpt54mini}` (or whatever canonical slugs the plan locks in; the set is a module-level constant).
   - No duplicates (implied by set equality but named explicitly in the error message on violation).

   This catches the "three files in the right glob but one of them is a duplicate or misnamed" failure mode that a naive count check cannot. It also produces a deterministic model→parquet mapping for downstream majority-voting.

2. Load all 10 parquets (3 before + 3 after + 3 noise-floor + 1 input) via `polars`.
3. **Row-alignment validation (fingerprint-based).** The current `iter16_5k_input.parquet` has 5065 rows but only 5059 unique `text` values, so `text`-only equality is insufficient. Instead, build a fingerprint per row by hashing the concatenation of all non-`det_*` columns from the input parquet (e.g., `text`, `sub_type`, `source`, `audit_source`, any other passthrough columns). For each annotation parquet, compute the same fingerprint and assert it equals the input's fingerprint column-for-column, row-for-row. Any mismatch raises `ValueError` with the first differing row index and the offending columns named.
4. Discover `det_*` columns in each frame. Compute the union, intersection, and per-side uniques across the `before` vs `after` frames. **Then hard-assert that `after` and `noise_floor` have identical `det_*` column sets** — if they differ, raise `ValueError` naming the symmetric difference. The noise floor arm's correctness depends on this; we do not silently tolerate schema mismatch on the input to the override criterion.
5. Categorize every `det_*` label: `shared` (in both `before` and `after`) / `iter15-only` / `iter16-only`.
6. For each label, compute (stratified rows only — filter to `audit_source == "stratified"` using the column written by `build_audit_sample.py`; injected rows have values of the form `inject_<label>` and must be excluded from agreement and prevalence metrics):
   - `iter15_agr` from `before` frames (inter-annotator agreement, majority-of-3)
   - `iter16_agr` from `after` frames
   - `iter15_prev`, `iter16_prev` — fire rate (majority-of-3)
   - `Δagr = iter16_agr - iter15_agr` (shared labels only)
   - `prev_ratio = iter16_prev / iter15_prev` (shared labels only, with explicit zero-handling rules below)

**Zero-prevalence rules for `prev_ratio`:**
| `iter15_prev` | `iter16_prev` | `prev_ratio` | WARN-prevalence? |
|---|---|---|---|
| 0 | 0 | `1.0` | no — label is silent on both sides, no drift |
| 0 | >0 | `inf` | yes — new-firing label, semantic drift |
| >0 | 0 | `0.0` | yes — newly silent label, semantic drift |
| >0 | >0 | `iter16_prev / iter15_prev` | yes iff `∉ [0.5, 2.0]` |

The `0/0 → 1.0` rule is load-bearing: without it, a shared label that never fires on the stratified sample (common at ~0.001 prevalence) would produce a sentinel `prev_ratio` that trips WARN-prevalence for no semantic reason.
7. **Noise floor.** For each label present in both `after` and `noise_floor` frame sets, compute `noise_floor[label] = |agr(after) - agr(noise_floor)|`. This is real same-prompt variance because both sets use the iter16 prompt at commit HEAD.
8. Compute verdict per label:
   - `shared` + `|Δagr| > 0.005` → `FAIL-agreement`
   - `shared` + `prev_ratio ∉ [0.5, 2.0]` → `WARN-prevalence` (can co-occur with `FAIL-agreement`)
   - `shared` + neither → `PASS`
   - `iter15-only` / `iter16-only` → no verdict (categorical, context only)
9. Emit markdown report (section structure below).
10. Return `DeltaReport` object for programmatic access by tests.

**Schema-drift guards:** raise `ValueError` with explanatory message if any glob doesn't resolve to exactly 3 parquets, if a glob's model-slug set differs from the expected canonical set (naming the missing/extra slugs), if parquets have different row counts from the input, if fingerprint rows mismatch (name the first diverging row index and columns), if `sub_type` column is missing, if `audit_source` column is missing, if any parquet is empty, if `before` and `after` have zero shared labels, or if `after` and `noise_floor` have differing `det_*` column sets (naming the symmetric difference).

**NaN-aware aggregation:** follow iter16 Phase 7 patterns. Missing data never silently resolves to PASS.

### Phase 3: Worktree setup + annotation runs with concurrency cap

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

**Concurrency discipline.** Each `trainr data annotate-detections` call defaults to `--concurrency 20` (20 in-flight LLM requests per job). Running all 9 jobs in parallel would mean ~180 in-flight requests plus retries, which introduces provider rate-limit asymmetry across runs and weakens the "only variable is prompt text" claim. The spec-level constraint is therefore:

> **At most 2 annotation jobs running concurrently at any time, with unchanged per-job concurrency. Hard cap of 40 in-flight LLM requests.**

The plan picks the exact scheduling, but one workable schedule is "iter15 side and iter16a side in parallel (2 jobs at a time, 3 rounds to finish 6 jobs), then iter16b side serially (3 rounds of 1 job each)." Wall clock ≈ 6× single-model time — deliberately slower than the original sketch because quality dominates speed.

**Run iter15 side (from worktree, one model at a time). `$REPO_ROOT` is the absolute path to the main repo:**
```bash
for model_slug in gemini3flash sonnet gpt54mini; do
    uv run trainr data annotate-detections \
        --input "$REPO_ROOT/training/data/audit/iter16_5k_input.parquet" \
        --model "<openrouter-id-for-$model_slug>" \
        --output "$REPO_ROOT/training/data/audit/iter17_ab_iter15_${model_slug}.parquet"
done
```

**Run iter16a side (from main repo, may run concurrently with iter15 side — max 2 in flight):**
```bash
for model_slug in gemini3flash sonnet gpt54mini; do
    uv run trainr data annotate-detections \
        --input training/data/audit/iter16_5k_input.parquet \
        --model "<openrouter-id-for-$model_slug>" \
        --output "training/data/audit/iter17_ab_iter16a_${model_slug}.parquet"
done
```

**Run iter16b side (from main repo, after iter15 and iter16a complete, one job at a time):**
```bash
for model_slug in gemini3flash sonnet gpt54mini; do
    uv run trainr data annotate-detections \
        --input training/data/audit/iter16_5k_input.parquet \
        --model "<openrouter-id-for-$model_slug>" \
        --output "training/data/audit/iter17_ab_iter16b_${model_slug}.parquet"
done
```

### Phase 4: Comparison + report

Run the `trainr data compare-prompts` CLI as shown above. Output: `docs/accuracy_runs/2026-04-10-iter17-regression-report.md`.

## Data Flow

```
iter16_5k_input.parquet (unchanged, apples-to-apples anchor)
        │
        ├─► .worktrees/iter17-iter15-prompt/@22bc292 ─► iter17_ab_iter15_{gemini3flash,sonnet,gpt54mini}.parquet
        │
        ├─► main repo @HEAD (run A, iter16 prompt)   ─► iter17_ab_iter16a_{gemini3flash,sonnet,gpt54mini}.parquet
        │
        └─► main repo @HEAD (run B, iter16 prompt)   ─► iter17_ab_iter16b_{gemini3flash,sonnet,gpt54mini}.parquet
                                                          │
                                                          ▼
                                             compare_prompt_versions
                                          (before=iter15, after=iter16a,
                                           noise_floor=iter16b)
                                                          │
                                                          ▼
                                       2026-04-10-iter17-regression-report.md
                                                          │
                                                          ▼
                                              iter17 gate decision
                                                          │
                                              ┌───────────┴───────────┐
                                              ▼                       ▼
                                        PASS → iter18           FAIL → iter17 iterates
                                                                      (fresh iter16a + iter16b
                                                                       pair per iteration, ~$30;
                                                                       only iter15 cached)
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

## Noise floor table
Per-label same-prompt variance from iter16a vs iter16b runs. Used to bound override eligibility for FAIL-agreement rows (see Gate Decision Protocol).
```

## Test Plan

### Unit tests for `compare_prompt_versions.py`

Fixture DataFrames covering:

- **Happy path (all shared):** 3 labels, 3 frames per side (before/after/noise-floor), no asymmetry, all PASS
- **iter15-only label:** 1 label only in `before` frames → categorized, no verdict, no noise floor
- **iter16-only label:** 1 label only in `after`/`noise-floor` frames → categorized, no verdict, noise floor populated
- **Mixed asymmetry:** 1 shared + 1 iter15-only + 1 iter16-only
- **Hard-fail row (overridable):** Δagr = 0.006, noise_floor = 0.004 → FAIL-agreement, `|Δagr| ≤ 2 × noise_floor` → eligible
- **Hard-fail row (not overridable):** Δagr = 0.020, noise_floor = 0.004 → FAIL-agreement, `|Δagr| > 2 × noise_floor` → blocks
- **Hard-fail at exact threshold:** Δagr = 0.005 → PASS (strict `>`)
- **Soft-warn row:** prev_ratio = 0.4 → WARN-prevalence
- **Both fail+warn:** same row has both verdicts
- **NaN handling:** one frame has NaN in a label column → doesn't silently PASS
- **Fingerprint mismatch:** annotation parquet diverges from input on a non-`det_*` column → raises ValueError naming row index and columns
- **Schema drift row count mismatch:** raises ValueError with clear message
- **Schema drift missing `sub_type`:** raises ValueError
- **Schema drift missing `audit_source`:** raises ValueError
- **Empty parquet:** raises ValueError
- **Zero shared labels:** raises ValueError
- **Glob resolves to wrong parquet count:** < 3 or > 3 → raises ValueError before any compute
- **Glob resolves to 3 parquets but wrong slug set:** e.g. `{gemini3flash, sonnet, sonnet}` (duplicate) or `{gemini3flash, sonnet, claude35}` (unknown slug) → raises ValueError naming the symmetric difference against the canonical set
- **`after` and `noise_floor` column set mismatch:** e.g. `noise_floor` missing `det_python` → raises ValueError naming the diff; the error comes before any noise floor computation
- **`prev_ratio` zero/zero:** `iter15_prev=0, iter16_prev=0` → `prev_ratio = 1.0`, no WARN-prevalence (label silent on both sides)
- **`prev_ratio` zero/nonzero:** `iter15_prev=0, iter16_prev=0.03` → `prev_ratio = inf`, WARN-prevalence fires
- **`prev_ratio` nonzero/zero:** `iter15_prev=0.03, iter16_prev=0` → `prev_ratio = 0.0`, WARN-prevalence fires
- **`audit_source` filtering:** fixture containing both `stratified` and `inject_det_python` rows → metrics computed only over `stratified` subset
- **Dynamic column introspection:** correctly identifies shared set when `before` and `after` have different column sets
- **Zero-prevalence prev_ratio:** handled gracefully (e.g., `inf` or sentinel, not crash)
- **Noise floor computation:** `|agr(after) - agr(noise_floor)|` matches hand-computed fixture for a multi-label case

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
| iter16a side (A/B after): 5065 rows × 3 models | ~$15 |
| iter16b side (noise floor companion): 5065 rows × 3 models | ~$15 |
| **Total annotation** | **~$45** |
| Tooling, review, worktree setup | $0 (time only) |
| Each FAIL-path prompt iteration (if needed) | +~$30 (iter16a **and** iter16b rerun; only iter15 cached — see FAIL-path note in §Scope) |

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
- **P3:** worktree setup + annotation runs (iter15, iter16a, iter16b with concurrency cap; parquets only, no code commits)
- **P4:** `docs(iter17): regression report + iteration doc`
- **P5:** gate decision commit (PASS/FAIL/override)

Two-stage review per code phase: spec-compliance review + code-quality review, per iter16 precedent.

## Gate Decision Protocol

At end of iter17, before closing the branch:

1. Read the iter17 regression report.
2. Enumerate all `FAIL-agreement` rows.
3. For each, compare `|Δagr|` against `noise_floor` (both reported in the shared-labels table):
   - `|Δagr| ≤ 2 × noise_floor` → eligible for override. Document in iteration doc with label name, Δagr, noise floor, `|Δagr| / noise_floor` ratio, and explicit reviewer sign-off accepting iter18 risk.
   - `|Δagr| > 2 × noise_floor` → real regression, not overridable. Block iter18, iterate on iter16 prompt, run a **fresh pair** of iter16 annotation passes (iter16a + iter16b, ~$30 total; only iter15 parquets cached and reused), and re-run `trainr data compare-prompts` with the new iter16a/iter16b parquets against the cached iter15 parquets. Reusing the original iter16b would turn noise floor back into prompt-drift measurement.
4. Enumerate all `WARN-prevalence` rows. For each, review the label's before/after prevalence and semantic definition, and document sign-off or objection in the iteration doc.
5. Final gate decision committed as `docs(iter17): gate decision — {PASS|FAIL}` with a one-paragraph justification.

If the gate PASSes (all FAIL-agreement rows either have no FAIL rows at all, or all FAIL rows are within-noise and documented as overrides), iter18 branch is unblocked. Otherwise iter17 iterates until the gate PASSes.
