# Phase 2: Retrain with 40 Features, Focal Loss, and Downsampled Plain

**Date:** 2026-04-06
**Branch:** `add-multilabel-classification`
**Model:** Retrained with 40 features (added `section_header_ratio`, `json_lines_ratio`), focal loss (gamma=2.0), and plain sub-type capped at 10,000 samples.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Features | 40 (was 38) |
| Focal loss gamma | 2.0 |
| Detection threshold | 0.3 |
| Max per sub-type | 10,000 (plain: 57,260 -> 10,000) |
| Training samples | 48,396 (after dedup from 88,884 downsampled from 136,144) |
| Validation samples | 12,100 |
| Epochs | 93 (early stopped at patience=15, best at epoch 78) |
| Best val loss | 0.279 |
| Val category accuracy | 94.9% |
| Val sub-type accuracy | 85.5% |

## Eval Results

### Clear Eval (`clear.jsonl`, 4,071 samples)

| Metric | Iter 8 Baseline | Phase 1 (Cat Head) | Phase 2 (This Run) | Delta (Phase 2 vs Phase 1) |
|--------|-----------------|-------------------|--------------------|-----------------------------|
| Overall accuracy | **93.4%** | 88.1% | 87.7% | -0.4pp |
| Prose F1 | 0.86 | 0.79 | 0.76 | -0.03 |
| Code F1 | 0.97 | 0.95 | 0.95 | 0.00 |
| Structured F1 | 0.92 | 0.87 | 0.84 | -0.03 |

### Boundary Eval (`boundary.jsonl`, 3,156 samples)

| Metric | Iter 8 Baseline | Phase 1 (Cat Head) | Phase 2 (This Run) | Delta (Phase 2 vs Phase 1) |
|--------|-----------------|-------------------|--------------------|-----------------------------|
| Overall accuracy | **59.6%** | 60.2% | 40.7% | -19.5pp |
| Prose F1 | 0.48 | -- | 0.49 | -- |
| Code F1 | 0.65 | -- | 0.41 | -- |
| Structured F1 | 0.66 | -- | 0.46 | -- |

### Per-Sub_Type Accuracy (Clear Eval, Sorted Ascending)

| Sub-type | Phase 2 Accuracy | Count |
|----------|-----------------|-------|
| jsonl | 4.5% | 156 |
| pipe_table | 6.0% | 116 |
| sgml | 50.0% | 2 |
| tsv | 77.7% | 148 |
| fixed_width | 79.4% | 107 |
| rst | 79.6% | 147 |
| key_value | 84.3% | 134 |
| ini | 84.5% | 129 |
| log_lines | 85.6% | 160 |
| shell | 89.9% | 148 |
| latex | 93.1% | 145 |
| csv | 94.6% | 149 |
| xml | 94.6% | 149 |
| markdown | 95.2% | 145 |
| python | 95.3% | 150 |
| yaml | 96.6% | 149 |
| sql | 97.3% | 146 |
| css | 97.3% | 150 |
| dockerfile | 97.3% | 150 |
| json | 97.3% | 150 |
| makefile | 97.3% | 150 |
| toml | 98.0% | 149 |
| go | 98.0% | 150 |
| typescript | 98.6% | 146 |
| javascript | 98.7% | 150 |
| html | 99.3% | 148 |
| rust | 99.3% | 148 |
| plain | 99.3% | 150 |
| java | 100.0% | 150 |

### Confusion Matrix (Clear Eval)

```
                prose    code    structured    skip
prose             539      40           8       0
code               20    1881          31       5
structured        275     116        1150       6
```

### Detection Metrics

Not evaluated. Training data does not contain detection labels (`det_*` columns absent from `golden_raw.parquet`). Detection head was not trained with labeled data.

## Analysis

### Key Findings

1. **Phase 2 did NOT improve over Phase 1.** Clear eval accuracy dropped slightly (87.7% vs 88.1%), and boundary eval dropped significantly (40.7% vs 60.2%).

2. **Downsampling plain hurt boundary performance.** Reducing plain from 57K to 10K samples likely removed many boundary-adjacent examples that the model needed to learn the prose/structured boundary.

3. **jsonl and pipe_table remain the weakest sub-types.** JSONL content often looks like prose (especially when values are natural language text), causing 86% of JSONL samples to be misclassified as prose. Pipe tables suffer a similar issue.

4. **The Python featurizer vs Rust feature extractor gap persists.** The iteration 8 baseline of 93.4% was measured using the Rust ONNX integration, not the Python featurizer. The gap between Python eval (87.7%) and iteration 8 Rust eval (93.4%) suggests feature extraction differences between the two implementations.

5. **Focal loss effect unclear.** With gamma=2.0, the model trained well (val_cat_acc=94.9%) but eval accuracy didn't improve, suggesting the issue is in the eval pipeline or data mismatch, not training dynamics.

### Recommendations

1. **Revert to iteration 8 model weights** for production use until the Python/Rust featurizer gap is investigated.
2. **Do not downsample plain below 20K** without boundary-aware stratification.
3. **Investigate Python vs Rust feature extraction** to understand the 5.7pp accuracy gap on clear eval.
4. **Consider JSONL-specific detection** rather than category-level classification for JSONL content.
