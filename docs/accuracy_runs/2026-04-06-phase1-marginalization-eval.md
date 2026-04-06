# Phase 1 Validation: Marginalization + Threshold=0.3

**Date:** 2026-04-06
**Branch:** `add-multilabel-classification`
**Model:** `src/model.onnx` (iteration 8 weights, unchanged)
**Inference change:** Category prediction via sub-type probability marginalization (instead of category head directly)

## Summary

Phase 1 tested whether marginalizing category probabilities from the sub-type head would improve accuracy over the direct category head, using the same ONNX model weights from iteration 8. **Result: regression, not improvement.** Marginalization decreases accuracy on both eval sets. The detection threshold change (0.5 to 0.3) could not be evaluated because the current model config lacks a `detection_map`.

## Eval Results

### Clear Eval (`clear.jsonl`, 4,071 samples)

| Metric | Iter 8 Baseline | Cat Head (current) | Marginalization | Delta (marg vs baseline) |
|--------|-----------------|-------------------|-----------------|--------------------------|
| Overall accuracy | **93.4%** | 88.1% | 86.9% | -6.5pp |
| Prose F1 | 0.86 | 0.79 | 0.74 | -0.12 |
| Code F1 | 0.97 | 0.95 | 0.94 | -0.03 |
| Structured F1 | 0.92 | 0.87 | 0.84 | -0.08 |

### Boundary Eval (`boundary.jsonl`, 3,156 samples)

| Metric | Iter 8 Baseline | Cat Head (current) | Marginalization | Delta (marg vs baseline) |
|--------|-----------------|-------------------|-----------------|--------------------------|
| Overall accuracy | **59.6%** | 60.2% | 41.0% | -18.6pp |
| Prose F1 | 0.48 | — | 0.53 | +0.05 |
| Code F1 | 0.65 | — | 0.42 | -0.23 |
| Structured F1 | 0.66 | — | 0.41 | -0.25 |

### Per-Sub_Type Accuracy (Clear Eval)

| Sub-type | Cat Head | Marginalization | Delta |
|----------|----------|-----------------|-------|
| ini | 86.8% (112/129) | 83.7% (108/129) | -3.1pp |
| key_value | 87.3% (117/134) | 84.3% (113/134) | -3.0pp |
| jsonl | 10.9% (17/156) | 9.0% (14/156) | -1.9pp |
| plain | 99.3% (149/150) | 100.0% (150/150) | +0.7pp |

### Detection Metrics

Not evaluated. The current `model_config.json` (both `src/` and `training/models/`) does not contain a `detection_map`, so the detection head outputs are not decoded. Detection recall/precision comparison against the iteration 8 baseline (67.7% / 93.1%) is blocked on adding a `detection_map` to the model config.

## Confusion Matrix (Clear Eval, Marginalization)

```
                prose    code    structured    skip
prose             538      41           8       0
code               33    1859          38       7
structured        296     107        1142       2
```

### Key Misclassification Pattern

296 structured samples were predicted as prose via marginalization. Breakdown by sub_type:

| Sub-type | Misclassified as prose |
|----------|-----------------------|
| jsonl | 126 |
| pipe_table | 103 |
| fixed_width | 26 |
| log_lines | 19 |
| csv | 16 |
| yaml | 4 |
| json | 1 |
| toml | 1 |

## Analysis

### Why marginalization hurts

The model was trained with a dedicated category head loss. The category head is well-calibrated for its 3-class task. Marginalizing over 30 sub-types introduces noise because:

1. **Probability dispersion** — Prose has 4 sub-types (plain, markdown, rst, latex) that collectively accumulate background probability even when the true class is structured.
2. **The `unknown` sub-type maps to `skip`** — Any probability mass on `unknown` (index 27) goes to a category that doesn't exist in the eval data, effectively wasting probability budget.
3. **Sub-type head wasn't trained for marginalization** — The sub-type head was optimized to distinguish between 30 sub-types, not to produce well-calibrated per-category probability sums.

### Gap between current cat-head (88.1%) and iteration 8 baseline (93.4%)

The 5.3pp gap between the current category head accuracy (88.1%) and the iteration 8 reported accuracy (93.4%) suggests either:
- The Python `featurize.py` produces different feature values than the Rust feature extractor used during iteration 8 eval
- The iteration 8 eval used a different code path (possibly the Rust ONNX integration directly)

This discrepancy is outside the scope of this validation but should be investigated.

## Conclusion

**Phase 1 is NOT validated.** Marginalization from the sub-type head degrades accuracy compared to both the category head and the iteration 8 baseline. The model weights were not trained with marginalization in mind.

### Recommendations

1. **Do not ship marginalization with current weights.** Use the category head directly for category prediction.
2. **Retrain with marginalization-aware loss** if marginalization is desired — e.g., train only the sub-type head and derive category from marginalization, with an auxiliary category-level loss computed on the marginalized probabilities.
3. **Add `detection_map` to model config** to enable detection threshold evaluation.
4. **Investigate the Python vs Rust feature extractor gap** (88.1% vs 93.4%) before further eval work.
