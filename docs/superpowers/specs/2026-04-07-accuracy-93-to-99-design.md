# Accuracy Improvement Plan: 93% → 97% → 99%

**Date**: 2026-04-07
**Branch**: `add-multilabel-classification`
**Starting point**: Iteration 11 — 93.0% clear, 40.2% boundary

## Problem Statement

After 11 iterations of accuracy work, clear eval is at 93.0% but each iteration has been whack-a-mole — multiple variables change simultaneously, artifacts go stale, and it's impossible to attribute improvements or regressions. The path to 97% and 99% requires a systematic approach with controlled experiments and trustworthy measurement.

## Diagnostic Summary

### Current Error Breakdown (4,071 clear eval samples, 285 errors)

| Error Pattern | Count | % of Errors |
|---------------|-------|-------------|
| structured → prose | 155 | 54.4% |
| structured → code | 45 | 15.8% |
| code → prose | 28 | 9.8% |
| code → structured | 19 | 6.7% |
| prose → code | 13 | 4.6% |
| code → skip | 12 | 4.2% |
| prose → structured | 12 | 4.2% |
| structured → skip | 1 | 0.4% |

### Root Causes Identified

1. **JSONL sub-type**: 4.5% accuracy (134/156 misclassified as prose in stale eval). Only 750 training samples out of 121K (0.6%).
2. **pipe_table sub-type**: 6.0% accuracy (101/116 misclassified as prose). Only 518 training samples. Many eval samples are markdown-with-tables, not pure tables — possible label issue.
3. **Python/Rust featurizer gap**: ~5.7pp accuracy difference between featurizers. Training uses Python, production uses Rust. Cross-iteration comparisons are unreliable.
4. **Stale artifacts**: The `slice_report.clear.json` (87.7%) does not match the current model (93.0%). Eval artifacts are from a previous model.
5. **Missing features in canonical data**: `json_lines_ratio` and `section_header_ratio` exist in `golden_featurized.parquet` and the trained model but not in `golden.parquet`.

### Key Structural Facts

- Model: 40-feature feedforward NN with dual heads (category + sub-type), ONNX export
- Training data: 121,811 rows, 0 synthetic, 29 sub-types across 3 categories
- Eval: 4,071 clear samples, 3,156 boundary samples (JSONL format)
- Features that SHOULD discriminate JSONL from prose: `key_value_ratio` (0.97 vs 0.17), `delimiter_consistency` (0.41 vs 0.02) — large gaps, yet model fails

---

## Plan Structure

Three gated phases. Each phase must hit its exit criteria before the next begins.

| Phase | Goal | Exit Criteria |
|-------|------|---------------|
| **Phase 0** | Trustworthy measurement | Featurizer gap < 0.5pp; all artifacts in sync; manifest tracks drift |
| **Phase 1** | 93% → 97% | Clear eval >= 97.0%, no category regresses > 1pp |
| **Phase 2** | 97% → 99% | Clear eval >= 99.0% |

**Critical rule**: One variable per experiment. Never change data AND features AND architecture in the same iteration. The manifest enforces this.

---

## Phase 0: Infrastructure (C+)

### 0a. Python/Rust Featurizer Alignment

**Problem**: Training/eval uses Python featurizer, production uses Rust. They produce different values for the same text (~5.7pp accuracy impact).

**Approach**:
- Run both featurizers on a 500-sample corpus (sampled from clear.jsonl)
- Diff all 40 features per sample, identify divergences
- Fix the Python featurizer to match Rust (Rust is production source of truth)
- Validate: max per-feature mean absolute error < 0.01

**Why Python matches Rust**: Rust runs in production. Training data must reflect production behavior.

### 0b. Eval Contract

**Problem**: Stale artifacts — current slice report doesn't match current model.

**Approach**:
- Every training run produces a versioned output directory: `training/output/<run-id>/`
  - `model.onnx`, `model_config.json`, `metrics.json`
  - `eval_predictions.clear.jsonl`, `eval_predictions.boundary.jsonl`
  - `slice_report.clear.json`, `slice_report.boundary.json`
  - `training_manifest.json`
- `trainr pipeline train` runs eval automatically after training — no separate manual step
- Symlink `training/output/latest` → most recent run

### 0c. Manifest & Drift Detection

**Problem**: No way to detect when dataset, features, or eval set changed since the model was trained.

**Approach**:
- `training_manifest.json` written at train time:
  ```json
  {
    "run_id": "run-2026-04-07-001",
    "dataset_sha256": "abc...",
    "dataset_rows": 98328,
    "featurizer_version": "2.0",
    "feature_count": 40,
    "feature_names": ["line_length_cv", "..."],
    "eval_clear_sha256": "def...",
    "eval_boundary_sha256": "ghi...",
    "model_sha256": "jkl...",
    "timestamp": "2026-04-07T10:00:00Z"
  }
  ```
- `trainr eval verify` computes current SHAs, compares against manifest, warns on any mismatch
- Featurizer version bumped manually when feature extraction logic changes

### Phase 0 Exit Criteria

1. Python/Rust featurizer gap < 0.5pp on clear eval
2. Fresh eval with aligned featurizer produces the true baseline accuracy
3. Manifest written, `trainr eval verify` passes
4. This baseline number becomes the starting point for Phase 1

---

## Phase 1: 93% → 97%

**Error budget**: 285 errors → <= 122 errors. Must fix >= 163 errors.

### 1a. Re-baseline (Measurement Only)

After Phase 0, re-run eval with aligned featurizer and record:
- True starting accuracy (may differ from current 93.0%)
- Per-sample error breakdown by sub-type and source
- Fresh confusion matrix

**No changes to data or model.** This is the controlled starting point.

### 1b. Fix JSONL and pipe_table (Target: +3-4pp)

**Why first**: JSONL (134 errors) + pipe_table (101 errors) = up to 235 structured→prose errors. Even if the 93% model fixed some, these are the dominant failure mode.

**Approach — data quality only, no feature/architecture changes**:

1. **Audit pipe_table labels**: Many eval samples are markdown documents with embedded tables. Reclassify samples where table content is < 30% of text as `prose/markdown` instead of `structured/pipe_table`.

2. **Augment JSONL training data**: 750 JSONL rows in 121K (0.6%) is severe underrepresentation. Add real JSONL from the-stack-v2. Target: >= 2,000 JSONL training samples.

3. **Augment pipe_table training data**: 518 rows. Add real pure pipe tables (not markdown-with-tables). Target: >= 1,500 pipe_table training samples.

**Exit criteria**: JSONL accuracy >= 80%, pipe_table >= 70%, overall clear eval >= baseline + 2pp.

### 1c. Fix Remaining Structured Errors (Target: → 97%)

After 1b, examine fresh error data. Expected remaining targets:
- TSV → code (29 errors in stale data)
- RST → code (29 errors)
- key_value → code (19 errors)
- INI → code (18 errors)

**Approach depends on 1b results**, but likely:
- Label audit for ambiguous samples (TSV that looks like tab-indented code)
- Training data augmentation for underrepresented sub-types
- Feature gap check: verify discriminative features compute correctly after 0a alignment

**Exit criteria**: Clear eval >= 97.0%, no category regresses > 1pp from 1a baseline.

---

## Phase 2: 97% → 99%

**Error budget**: ~122 errors → <= 41 errors. Must fix >= 81 errors.

### 2a. Error Taxonomy at 97%

Classify every remaining error into three buckets:
- **Mislabeled**: Eval label is wrong (fix the label)
- **Underrepresented**: Model hasn't seen enough examples (fix with data)
- **Genuinely hard**: 40 features don't capture the distinction (fix with features or architecture)

Use `audit_model_errors.py` LLM voting on the ~122 remaining errors. Small enough for manual review of LLM judgments.

**Exit criteria**: Every error is bucketed. Bucket distribution determines 2b strategy.

### 2b. Strategy by Bucket

**If mostly mislabeled (> 40%)**:
- Fix eval labels via manual review (~122 errors is feasible in an hour)
- Re-audit corresponding training labels
- Retrain, re-eval

**If mostly underrepresented (> 40%)**:
- Targeted data augmentation for specific sub-types/patterns
- Source real data from HuggingFace for rare sub-types

**If mostly genuinely hard (> 40%)**:
- New features targeting specific confusion patterns (one at a time)
- Architecture changes: deeper network, attention, or ensemble (one at a time)
- Character n-gram or token-level features if structural features hit a ceiling

### 2c. The 99% Decision Point

If stuck at 98-98.5%, there's a strategic question:

> Can a 40-feature feedforward NN reach 99% on this task, or does the last 1% require richer input representation?

Options at that point:
- Add character n-gram or token-level features (lightweight, +20-50 features)
- Switch to a small transformer on raw text (different architecture)
- Accept 98.5% as the ceiling for this approach

This is a deliberate decision point, not something to fall into by accident.

---

## Deferred Work

- **Full pipeline orchestrator** (`trainr pipeline run`): tracked as `textclf-1dd1.05bymn`
- **Boundary eval improvement**: separate workstream after clear eval reaches 97%
- **New sub-type eval samples**: c_cpp and objc have no boundary test cases yet

## Principles

1. **One variable per experiment**: Change data OR features OR architecture, never multiple.
2. **Measure before fixing**: Every phase starts with a measurement step.
3. **Manifest everything**: Every training run is traceable back to its inputs.
4. **Python matches Rust**: Production featurizer is source of truth.
5. **Labels before model**: Most "model errors" are actually label errors at high accuracy.
