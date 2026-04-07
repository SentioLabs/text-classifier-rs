# Iteration 8: Label Cleanup, Data Quality & Model Retraining

**Date:** 2026-04-05 to 2026-04-06
**Branch:** `add-multilabel-classification`
**Tags:** `dataset-2026-04-05.r3` through `dataset-2026-04-06.r2`

## Overview

Full data quality overhaul of the 115K golden training dataset. Dropped synthetic data, pulled fresh real data, validated labels via multi-model tiered voting, re-annotated with per-model routing, refreshed eval sets, fixed taxonomy mismatches, and retrained the Tier 2 ONNX model.

## Dataset Evolution

| Stage | Rows | Notes |
|-------|------|-------|
| Start (pre-cleanup) | 115,206 | ~9.6% mislabeled from synthetic + Magika errors |
| Drop synthetic | ~69,000 | Removed all source="unknown" rows |
| Pull real data | 123,050 | Fresh data from The Stack V1 with format validation |
| Phase 4: Voting | 112,151 | 111,223 kept, 928 relabeled, 10,899 dropped |
| Fix OCR csv | 123,050→112,151 | Dropped 1,362 OCR garbage, added 1,500 clean CSV |
| Fix fixed_width | 111,791 | Dropped 927 mislabeled, added 567 from Stack |
| Fix category labels | 111,791 | yaml/toml/ini: code→structured (9,577 rows) |

**Final dataset:** 111,791 rows, 28 sub_types, 38 features, 29 det_* columns.

## Category Distribution (Final)

| Category | Rows | % |
|----------|------|---|
| Prose | 49,480 | 44.3% |
| Code | 36,542 | 32.7% |
| Structured | 25,769 | 23.0% |

## Model Results

### Validation Metrics
| Metric | Value |
|--------|-------|
| Best val loss | 0.2035 |
| Best epoch | 90 (of 105, early stopped) |
| Category accuracy | 98.2% |
| Sub-type accuracy | 92.5% |
| Detection F1 | 0.784 |
| Detection precision | 0.931 |
| Detection recall | 0.677 |

### Eval Set Results

| Eval Set | Samples | Accuracy | Prose F1 | Code F1 | Structured F1 |
|----------|---------|----------|----------|---------|---------------|
| clear.jsonl | 4,071 | **93.4%** | 0.86 | 0.97 | 0.92 |
| boundary.jsonl | 3,156 | **59.6%** | 0.48 | 0.65 | 0.66 |

### Clear Eval Confusion Matrix
```
                prose    code    structured
prose            554      18          15
code              18    1891          28
structured       126      62        1359
```

## Key Decisions & Fixes

### 1. Dropped Synthetic Data
The ~46K synthetic rows had sub_types assigned from generation prompts rather than validated against output. Not worth the label noise.

### 2. Multi-Model Tiered Voting
- Tier 1: cheap models (nano, flash-lite, gem3-flash) confirm/reject existing labels
- Tier 2: premium models escalate disagreements
- 91.2% Tier 1 agreement on pilot (5K sample), well above 85% go/no-go threshold
- Switched Sonnet→Gem3-Flash mid-run: identical TP rates, 30x cheaper

### 3. Model Routing Optimization
Original plan used claude-sonnet-4-6 ($3/$15 per 1M tokens) for go/html/shell/jsonl/markdown. Benchmark comparison showed Gem3-Flash ($0.10/$0.40) had identical true positive rates. Also replaced Haiku with Gem3-Flash for json/makefile/python. Final routing is 100% OpenRouter for voting and annotation.

### 4. Taxonomy Alignment
Discovered yaml/toml/ini were labeled as `code` in training but `structured` in Rust types.rs. This mismatch caused 72% of the structured↔code confusion. Fixing it improved clear eval from 90.9%→93.4% and structured F1 from 0.85→0.92.

### 5. Eval Set Refresh
Old eval sets were 71-100% synthetic with noisy labels. Replaced with:
- **clear.jsonl:** 4,071 rows from The Stack, LogHub, wikitext, pile. LLM-confirmed labels.
- **boundary.jsonl:** 3,156 rows from voting-dropped texts. Real data where multiple models disagreed, LLM-relabeled. Intentionally hard.

### 6. Checkpoint Bug Fix
The `flushed_up_to` linear pointer stalled when concurrent workers completed out of order (OpenRouter phase finished but Anthropic indices were scattered below the pointer). Replaced with set-based tracking.

## Cost Breakdown

| Phase | Estimated | Actual | Notes |
|-------|-----------|--------|-------|
| Voting Tier 1 (Sonnet) | $72 | ~$250 | Token estimates were 2-3x low; Sonnet is expensive |
| Voting Tier 1 (Gem3-Flash restart) | $3 | ~$3 | After routing fix |
| Voting Tier 2 | $5-10 | ~$2 | Only 1,020 escalations |
| Detection annotation | $25-36 | ~$30 | All OpenRouter |
| Eval annotation | ~$1 | ~$1 | 15K rows via Gem3-Flash |
| **Total** | **~$100** | **~$286** | Sonnet routing was the main overrun |

### Lessons Learned on Cost
- Token estimates must use the distribution (P90/P95), not the mean — large texts at premium prices dominate
- Always benchmark cheaper models before committing to premium — Gem3-Flash matched Sonnet on all tested types
- The voting pipeline should save det_* columns to avoid redundant annotation passes (~$30 wasted)

## Remaining Confusion Analysis (Clear Eval)

### Code → Structured (46 errors)
- ini→structured: still the largest single confusion, but halved from 67→~30 after category fix
- Residual: css→json, shell→ini (structural similarity)

### Structured → Code (62 errors)
- key_value→ini/toml: structurally near-identical formats
- jsonl→html/javascript: feature-based model can't distinguish JSONL from code

### Prose → Structured/Code (33 errors)
- plain→structured/code: edge cases (code comments, structured-looking prose)

## Possible Next Steps to Improve Accuracy

### 1. Feature Engineering for Remaining Confusions

**key_value vs ini** (biggest remaining confusion):
- Add a `section_header_ratio` feature detecting `[section]` patterns — INI files have them, key_value does not
- Add an `equals_sign_assignment_ratio` vs `colon_assignment_ratio` — INI uses `=`, many key_value formats use `:`

**jsonl detection** (21 errors):
- Add a `json_lines_ratio` feature: count lines that parse as valid JSON / total lines. JSONL would score ~1.0, code would score ~0.0
- This is expensive to compute but highly discriminative

### 2. Sub-type Aware Category Inference

Currently the model has separate category and sub_type heads. If the sub_type head predicts `ini` with high confidence, the category should be `structured` regardless of what the category head says. Implementing a **post-hoc category override** from sub_type→category mapping would eliminate all cases where the sub_type is correct but the category is wrong.

### 3. Detection Head Improvement (Recall: 67.7%)

The detection head has high precision (93.1%) but low recall (67.7%). This means it's conservative — it misses detections but rarely produces false positives. To improve:
- **Lower the detection threshold** from 0.5 to 0.3 — trade some precision for recall
- **Class-weighted detection loss** — upweight rare sub_types (sgml, pipe_table, fixed_width) during training
- **Use the det_* columns as additional input features** — the LLM annotations capture semantic understanding that structural features miss

### 4. Training Data Balancing

Current distribution is heavily skewed: `plain` has 43K rows (39%) while `pipe_table` has 507 (0.5%). Options:
- **Downsample** plain to ~10K to reduce bias
- **Upsample** rare types via augmentation (text truncation, line shuffling, noise injection)
- **Focal loss** instead of cross-entropy — automatically downweights easy/common examples

### 5. Ensemble with Detection Head

Instead of using Tier 2 as a standalone classifier, combine Tier 1 (rule-based) + Tier 2 (ONNX model) + detection head (multi-label) in an ensemble:
- Tier 1 provides high-confidence decisions
- When Tier 1 is uncertain, use both the category/sub_type heads AND the detection head
- If `det_ini=1` and `det_key_value=0`, that resolves the ambiguity directly

### 6. Boundary-Focused Training

The boundary eval (59.6%) shows the model struggles on genuinely ambiguous texts. Options:
- **Hard example mining** — identify training samples near decision boundaries and oversample them
- **Contrastive learning** — train with pairs of similar-but-different types (ini vs key_value, yaml vs toml) to sharpen boundaries
- **Curriculum learning** — train on easy examples first, gradually introduce harder boundary cases

### 7. JSONL-Specific Improvements

JSONL is the weakest type (6 rows in clear eval, 21 errors in prior eval). The model can't distinguish JSONL from code using structural features alone because:
- JSONL looks like JSON (braces, quotes, colons)
- JSON is `structured`, but the same characters appear in JavaScript/HTML
- A `json_lines_ratio` feature (% of lines that are valid JSON objects) would be nearly perfect for this

### 8. Larger Model Architecture

Current: 38→256→64→32→heads. Options:
- **Wider layers** (38→512→256→128) — more capacity for subtle distinctions
- **Residual connections** — help gradient flow for deeper networks
- **Attention over features** — let the model learn which features matter for which sub_types

### Priority Ranking

| # | Improvement | Expected Impact | Effort |
|---|------------|----------------|--------|
| 1 | Sub-type→category override | +1-2% clear acc | Low (inference-only change) |
| 2 | section_header + json_lines features | +1-2% clear acc | Medium (feature eng + retrain) |
| 3 | Downsample plain + focal loss | +0.5-1% overall | Low (training config) |
| 4 | Detection threshold tuning | +5-10% det recall | Low (config change) |
| 5 | Ensemble Tier1+Tier2+detection | +2-3% boundary acc | Medium (inference logic) |
| 6 | Boundary-focused training | +3-5% boundary acc | High (data + training changes) |
| 7 | Larger architecture | +1-2% overall | Medium (arch change + retrain) |
