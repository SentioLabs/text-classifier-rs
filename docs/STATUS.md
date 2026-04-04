# Text Classifier Status — Session Summary

**Date**: 2026-03-29
**Branch**: `feat/hierarchical-taxonomy` (PR #4)
**Tests**: 120 passing (with `--features onnx-model`)

---

## What Was Built

Three phases of work completed in a single session:

### Phase 1: Hierarchical Taxonomy + Tier 1 Rules
- **New type system**: `TextCategory` (Prose, Code, Structured, Artifact, Skip) + `ContentSubType` (33 variants)
- **18 features**: 10 original + 8 new (delimiter_consistency, json_brace_depth, key_value_ratio, xml_tag_ratio, log_line_ratio, comment_ratio, numeric_field_ratio, repetitive_structure_score)
- **Two-pass Tier 1**: category detection (skip→structured→code→artifact→prose) then sub-type refinement
- **Per-type confidence thresholds**: Prose 0.65, Code 0.70, Structured 0.60, Artifact 0.75
- Backward-compatible `TextType` alias, deprecated `text_type` accessor

### Phase 2: Training Pipeline
- **`training/generate.py`**: Synthetic data generation via Claude API (Sonnet 4) with batching, plus perturbation-based augmentation
- **`training/train.py`**: PyTorch dual-head NN (18 features → category + sub-type) with ONNX export
- **`classify validate`**: Accuracy, precision/recall/F1, confusion matrix, tier breakdown, `--verbose` flag for misclassification details
- **`classify label-corpus --with-features`**: Combined label + feature export
- **Makefile targets**: `training-setup`, `generate-data`, `generate-fixtures`, `train`, `validate`, `test-model`, `test-model-ambiguous`, `update-model`, `build-onnx`, `train-pipeline`

### Phase 3: ONNX Model Integration
- Replaced fasttext with embedded ONNX model (`include_bytes!`)
- Feature flag: `--features onnx-model`
- `Classifier::new()` auto-loads embedded model — zero-config
- Removed `--model` CLI flag, removed `Train` command
- Python bindings simplified: `Classifier()` with no args

---

## Current Accuracy

### Fixture Test Set (36 clean samples)

```
Overall accuracy:  86.1%
Tier 1 (rules):    32
Tier 2 (model):    4

Category     Precision  Recall  F1     N
artifact     1.00       1.00    1.00   3
code         0.82       0.88    0.85   16
prose        0.83       1.00    0.91   5
structured   0.90       0.75    0.82   12
```

### Ambiguous Test Set (102 boundary-case samples)

```
Overall accuracy:  45.1%
Tier 1 (rules):    86
Tier 2 (model):    16

Category     Precision  Recall  F1     N
artifact     0.31       0.24    0.27   17
code         0.76       0.47    0.58   34
prose        0.81       0.38    0.52   34
structured   0.65       0.76    0.70   17
```

### Model Training Metrics (on validation split of 3602 samples)

```
Category accuracy:    95.0%
Sub-type accuracy:    83.5%
Best val loss:        0.318
Epochs trained:       100
```

---

## What's Working Well

1. **Artifact detection** — 100% on clean samples after excluding Unicode decorative characters from `symbol_ratio`
2. **Structured data detection** — CSV, TSV, JSON, log lines all correctly identified via delimiter_consistency and json_brace_depth features
3. **Code detection** — Python, Rust, JavaScript, SQL, Shell all correctly identified via indentation + symbol patterns
4. **Prose detection** — Clean prose samples at 100% recall
5. **Config file detection** — TOML with comments now correctly classified as Code (was Structured before)
6. **Training pipeline** — End-to-end: `make train-pipeline` generates data → trains → embeds → validates
7. **ONNX integration** — Model loads from embedded bytes, zero-config deployment
8. **Validation tooling** — `--verbose` flag shows exactly why each misclassification happened

---

## What's Not Working Well

### Problem 1: Tier 1 is too confident on ambiguous inputs
**Impact**: 86 of 102 ambiguous samples decided by Tier 1, only 16 reach the model
**Root cause**: Per-type thresholds (0.60-0.75) are too low — Tier 1 accepts even when it shouldn't
**Evidence**: Many misclassifications have confidence 0.70-0.90 from Tier 1 on boundary cases
**Fix needed**: Either raise thresholds or add "competing category" logic that lowers confidence when multiple categories score similarly

### Problem 2: Prose → Artifact (9 misses on ambiguous set)
**Impact**: Technical prose with structured formatting (meeting notes, specs, reports with lists) classified as artifact
**Root cause**: The model was trained on clean prose samples; it's never seen "structured-looking prose"
**Evidence**: All 9 misses go through Tier 2 (model) with high confidence (0.93-1.00) for artifact
**Fix needed**: Add structured-prose examples to training data (reports, meeting notes, specs, API docs with lists/tables)

### Problem 3: Code → Structured (6 misses on ambiguous set)
**Impact**: Config files (Kubernetes YAML, docker-compose, Terraform, systemd units) classified as Structured instead of Code
**Root cause**: `key_value_ratio` trigger in `try_structured()` fires before `try_code()` catches them
**Evidence**: Configs with indentation + key-value patterns hit the structured detector first
**Fix needed**: Tighten the `try_structured` guards — if `leading_whitespace_ratio > 0.3` AND `key_value_ratio > 0.5`, it's almost certainly a config file (Code), not structured data

### Problem 4: Artifact → Code (1-2 misses)
**Impact**: OCR'd documents with special characters misclassified as code
**Root cause**: Even after the Unicode symbol fix, some OCR artifacts have ASCII symbols that push `symbol_ratio` high
**Fix needed**: Add an artifact-specific signal — perhaps `short_line_ratio` combined with `line_uniqueness` should override symbol_ratio for artifact detection

### Problem 5: Model trained on Tier 1 labels
**Impact**: Model inherits Tier 1's mistakes — it can't be better than its teacher
**Root cause**: `label-corpus` generates labels from Tier 1 rules; model learns those labels
**Fix needed**: Manually curate a golden test set with human-verified labels, especially for boundary cases. Use this for evaluation AND as training data.

---

## Training Data

| Dataset | Rows | Source |
|---------|------|--------|
| `fixtures.csv` | 36 | Feature extraction from `tests/fixtures/` |
| `synthetic.csv` | 3110 | Claude Sonnet 4 API (200 samples/type × 18 types, batched) |
| `perturbations.csv` | 456 | Gaussian noise on fixture features |
| **`combined.csv`** | **3602** | All of the above |
| `test_set.jsonl` | 36 | Fixture text + ground-truth labels |
| `ambiguous_test_set.jsonl` | 102 | Claude-generated boundary cases |

---

## Recommended Next Steps (Priority Order)

### 1. Add competing-category confidence reduction
When Tier 1's top two candidate categories are close in score, reduce the returned confidence so the model gets to decide. This is the highest-leverage change — it immediately doubles the number of samples the model sees.

### 2. Create a golden evaluation set
Manually label 50-100 samples across all categories, especially boundary cases. This breaks the circular dependency where the model is evaluated against the same labels it was trained on.

### 3. Add structured-prose training data
Generate or collect examples of: meeting notes, API documentation, technical specs, reports with tables, README files with code blocks. Label these as "prose" and add to training data.

### 4. Tighten Tier 1 structured/code boundary
The `try_structured` function needs stronger guards for config-file patterns. Specifically: if a sample has both `key_value_ratio > 0.5` AND `leading_whitespace_ratio > 0.2`, it should be deferred to `try_code` rather than claimed by `try_structured`.

### 5. Scale training data
Current: 3602 samples. Target: 10K+. Increase `--samples-per-type` to 500, add more sub-type categories to the synthetic generation list, and include ambiguous boundary cases in the training set.

### 6. Hyperparameter tuning
- Try deeper network (add a 3rd hidden layer)
- Experiment with category loss weight (currently 1.0 category + 0.3 sub-type)
- Try learning rate scheduling
- Increase dropout for better generalization on small dataset

---

## Key Commands

```bash
# Build & test
cargo test --features onnx-model          # 120 tests
cargo build --release --features onnx-model

# Training pipeline
make training-setup                        # Set up Python env
make generate-data                         # Generate all training data (needs ANTHROPIC_API_KEY)
make train                                 # Train model + ONNX export
make update-model                          # Copy model to src/ for embedding
make train-pipeline                        # All of the above + validate

# Validation
make test-model                            # Validate against fixture test set
make test-model-ambiguous                  # Validate against ambiguous test set
make generate-ambiguous                    # Regenerate ambiguous test set (needs API key)
classify validate --input file.jsonl --verbose  # Show misclassification details
classify validate --input file.jsonl --json     # JSON output for scripting

# Working directory
cd .worktrees/feat-hierarchical-taxonomy   # Worktree for this branch
```

---

## Architecture

```
classify(text)
  │
  ├─ Short-circuit: <5 words → Skip
  │
  ├─ extract_features(text) → FeatureVector (18 f32 fields)
  │
  ├─ Tier 1: classify_tier1(features)
  │    ├─ Pass 1: try_structured → try_code → try_artifact → try_prose → fallback
  │    └─ Pass 2: refine_sub_type(category, features)
  │
  ├─ If confidence >= per-type threshold → return Tier 1 result
  │
  └─ Tier 2: model.classify(features)  [onnx-model feature]
       ├─ Z-score standardize features
       ├─ ONNX session.run() → category_logits + sub_type_logits
       ├─ Softmax + argmax → TextCategory + ContentSubType
       └─ return Classification { category, sub_type, confidence, tier: Model }
```
