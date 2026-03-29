# Design: 99% Classification Accuracy

**Date**: 2026-03-29
**Branch**: `feat/hierarchical-taxonomy`
**Goal**: Achieve 99% classification accuracy on a golden evaluation set for an open-source text classifier handling arbitrary input.

---

## Context

The text classifier categorizes input into 4 categories: **Prose**, **Code**, **Structured**, **Artifact** (plus **Skip** for empty/trivial input). The current two-tier architecture uses rule-based Tier 1 as the primary classifier with an ONNX model (Tier 2) as fallback.

### Current Performance

| Test Set | Accuracy | Tier 1 | Tier 2 |
|----------|----------|--------|--------|
| Fixtures (36 clean samples) | 86.1% | 32 | 4 |
| Ambiguous (102 boundary cases) | 45.1% | 86 | 16 |
| Model validation split (3602 samples) | 95.0% | — | — |

### Root Cause

The model (95% accuracy) significantly outperforms the rules (86.1% / 45.1%), but Tier 1 claims most samples before the model can classify them. Tier 1 accepts at confidence thresholds of 0.60-0.75, which is too low for ambiguous inputs. The model is also trained on Tier 1's labels, creating a circular dependency that caps its potential.

---

## Architecture: Model-Primary Classification

Flip the hierarchy. The model classifies everything; Tier 1 becomes a fast-path short-circuit for trivially obvious cases only.

### New Classification Flow

```
classify(text)
  |
  +-- Short-circuit: empty / <5 words --> Skip (confidence 1.0)
  |
  +-- extract_features(text) --> FeatureVector (18 features)
  |
  +-- Tier 1: High-confidence short-circuits only (threshold >= 0.95)
  |    +-- Valid JSON (brace_depth > 0.05 + parseable structure) --> Structured, 0.98
  |    +-- HTML document (xml_tag_ratio > 0.5 + has <html> tag) --> Code, 0.98
  |    +-- Pure CSV/TSV (delimiter_consistency > 0.9 + >= 5 lines) --> Structured, 0.98
  |    +-- Pure prose (sentence_punct > 0.06 + alpha > 0.85 + no code signals) --> Prose, 0.95
  |    +-- All others --> fall through to model
  |
  +-- Tier 2: Model classifies everything else
       +-- Z-score standardize features
       +-- ONNX inference --> category_logits + sub_type_logits
       +-- Softmax --> probabilities
       +-- Return top category with confidence
```

### Key Changes

1. **Tier 1 threshold raised to 0.95** -- only fires on trivially obvious cases
2. **Tier 1 rules simplified** -- ~4-5 short-circuit rules replacing ~20 rule paths with mutual exclusion guards
3. **Model becomes primary classifier** -- handles 80-90% of inputs (currently ~15%)
4. **No competing-category logic needed** -- model handles ambiguity natively via softmax

### What Stays the Same

- Feature extraction (18 features)
- ONNX inference pipeline
- CLI interface
- Python bindings

### Tier 1 Short-Circuit Criteria

Each short-circuit must be:
- Based on a near-deterministic signal (not a fuzzy threshold)
- Testable with a simple assertion
- Defensible (e.g., "if this text has `<html>` and 50%+ XML tags, it's code, period")

---

## Golden Evaluation Set

Two separate JSONL files measuring distinct aspects of accuracy.

### Clear Set: 4,000 samples

- 1,000 per category (Prose, Code, Structured, Artifact)
- Unambiguous samples that any reasonable classifier should get right
- Purpose: "don't regress" safety net -- should stay at 99%+

### Boundary Set: 6,000 samples

- 6 category pairs: Prose/Code, Prose/Structured, Prose/Artifact, Code/Structured, Code/Artifact, Structured/Artifact
- 1,000 per pair (500 labeled each direction)
- Deliberately ambiguous samples at category boundaries
- Purpose: measure improvement on the hard cases

### Generation

- **Model**: GPT-5.4 (different model family from training data to avoid shared biases)
- **Format**: `{"text": "...", "expected_category": "prose", "boundary_pair": null}` (clear) or `{"text": "...", "expected_category": "code", "boundary_pair": "code_structured"}` (boundary)
- **Variety**: domain/topic/length rotation across batches
- **Independence**: labels come from the generation prompt, not from running the classifier
- **Version-controlled**: committed to `eval/` directory (Git LFS if needed)

### Success Metrics

| Set | Target |
|-----|--------|
| Clear (4K) | >= 99% accuracy |
| Boundary (6K) | >= 99% accuracy |
| Combined (10K) | >= 99% accuracy |

### Validation Command

```bash
classify validate --input eval/clear.jsonl              # per-category P/R/F1
classify validate --input eval/boundary.jsonl            # per-pair accuracy
classify validate --input eval/clear.jsonl eval/boundary.jsonl  # combined report
```

---

## Training Data Pipeline

### Target: 50K+ Deduplicated Samples

| Source | Samples | Purpose |
|--------|---------|---------|
| LLM-generated clear | 20,000 (5K/category) | Core category learning |
| LLM-generated boundary | 24,000 (4K/pair x 6 pairs) | Decision boundary learning |
| Real-world corpus | 5,000+ | Ground truth from actual text |
| Perturbations | ~1,000 | Feature-space augmentation |
| **Total** | **~50,000** | |

### Generation Strategy

- **Model**: Claude Sonnet (cost-effective at scale, consistent with existing pipeline)
- **Variety seeding**: Each batch gets a different domain/topic seed (e.g., "generate Python code about astronomy", "generate YAML config for a Kubernetes deployment")
- **Sub-type rotation**: Cycle through all 33 ContentSubType variants systematically
- **Length/complexity variation**: Explicitly request "3 lines", "20 lines", "100+ lines" in different batches
- **Temperature**: 0.9-1.0 for higher diversity
- **Labels from prompt**: breaks the circular dependency with Tier 1

### Overgenerate then dedup: ~60K raw --> ~50K clean

### FAISS Two-Layer Deduplication

#### Layer 1: Feature-Space Dedup
- Extract 18 structural features for all generated samples via `classify features`
- Build FAISS `IndexFlatL2` (exact L2 distance -- brute force is fast at 60K x 18 dims)
- For each sample, query k-nearest neighbors
- Remove samples where nearest neighbor distance < threshold (tunable)
- Catches: texts that the classifier literally cannot distinguish

#### Layer 2: Text-Level Semantic Dedup
- Encode each text with `sentence-transformers/all-MiniLM-L6-v2` (384 dims, fast)
- Build FAISS `IndexFlatIP` (cosine similarity via inner product on normalized vectors)
- Remove samples with cosine similarity > 0.9 to any existing sample
- Catches: paraphrases and near-duplicates that might map to slightly different feature vectors

#### Pipeline

```
generate 60K samples (Claude Sonnet API)
  --> extract features (classify features)
  --> FAISS Layer 1: feature-space dedup (remove structural duplicates)
  --> FAISS Layer 2: embedding dedup (remove semantic duplicates)
  --> ~50K unique, diverse samples
```

#### Implementation

- New file: `training/dedup.py`
- Dependencies: `faiss-cpu`, `sentence-transformers`
- Makefile targets: `dedup`, integrated into `generate-golden-train`

---

## Model Architecture Upgrade

### Current
```
18 --> 64 --> 32 --> category_head (5)
                 --> sub_type_head (41)
Dropout: 0.2
```

### Proposed
```
18 --> 128 --> 64 --> 32 --> category_head (5)
                         --> sub_type_head (41)
Dropout: 0.3
```

### Training Configuration Changes

| Parameter | Current | Proposed |
|-----------|---------|----------|
| Hidden layers | 2 (64, 32) | 3 (128, 64, 32) |
| Dropout | 0.2 | 0.3 |
| Epochs | 100 | 200 |
| Early stopping patience | 10 | 15 |
| LR scheduling | None | Reduce on plateau |
| Training samples | 3,602 | ~50,000 |
| Loss weights | 1.0 cat + 0.3 sub | 1.0 cat + 0.3 sub (unchanged) |

### Same ONNX Export Pipeline

- Opset 17
- Input: features (batch x 18)
- Outputs: category_logits (batch x 5), sub_type_logits (batch x 41)
- Embedded via `include_bytes!`

---

## Implementation Phases

### Phase 1: Golden Eval Set
**Deliverable**: `eval/clear.jsonl` (4K) + `eval/boundary.jsonl` (6K) + baseline measurements

- Extend `generate.py` with `eval-clear` and `eval-boundary` generation modes
- Use GPT-5.4 API for generation
- Generate 10K samples with variety seeding
- Run current classifier against both sets, record baseline numbers
- Version-control the eval files

### Phase 2: Training Data Generation + FAISS Dedup
**Deliverable**: `training/data/golden_train.csv` (~50K deduplicated samples)

- Extend `generate.py` with `golden-train` mode using Claude Sonnet
- Variety seeding: sub-type x domain x length matrix
- Generate ~60K raw samples
- Implement `training/dedup.py` with two-layer FAISS dedup
- Target: ~50K unique samples after dedup
- Makefile targets: `generate-golden-eval`, `generate-golden-train`, `dedup`

### Phase 3: Model-Primary Architecture
**Deliverable**: Refactored `tier1.rs` + updated `lib.rs` classification flow

- Simplify `tier1.rs` from ~20 rule paths to ~5 high-confidence short-circuits
- Raise Tier 1 acceptance threshold to 0.95
- Update `lib.rs` classify flow
- All 120 existing tests must still pass
- Measure against golden eval set

### Phase 4: Model Upgrade + Retrain
**Deliverable**: New `model.onnx` trained on 50K samples with deeper architecture

- Update `training/train.py` with new architecture (18 --> 128 --> 64 --> 32)
- Dropout 0.3, LR scheduling, 200 epochs
- Train on golden training data
- Validate against golden eval set
- Target: clear >= 99%, boundary >= 99%

### Phase 5: Iterate to 99%
**Deliverable**: Classifier meeting 99% combined accuracy

- Analyze misclassifications with `--verbose`
- Targeted fixes: add training data for failure patterns, tune short-circuits, experiment with hyperparameters
- Each iteration: regenerate problem areas --> dedup --> retrain --> measure
- Stop when combined accuracy >= 99% on golden eval set

### Phase Dependencies

```
Phase 1 --> Phase 2 --> Phase 4
               \
Phase 1 --> Phase 3 --> Phase 4 --> Phase 5
```

Phases 2 and 3 can run in parallel.

---

## Risks and Mitigations

### LLM-generated eval may not reflect real-world distribution
- **Mitigation**: After hitting 99% on golden set, validate against real-world text (GitHub scrapes, Wikipedia, Common Crawl). GPT-5.4 for eval + Claude Sonnet for training reduces shared bias.

### FAISS dedup thresholds may be too aggressive or lenient
- **Mitigation**: Start conservative (remove only very close duplicates), measure feature-space coverage, tighten if needed. Log removal counts at each threshold.

### 50K samples may not be enough
- **Mitigation**: Phase 5 is iterative. Pipeline scales -- just generate more batches if we plateau.

### Model inference speed regression
- **Mitigation**: The deeper model (18 --> 128 --> 64 --> 32) is still tiny for ONNX on 18 input features. Benchmark before/after.

---

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Model-primary (B) | Model already outperforms rules (95% vs 86%). Scales with data, not manual rule-writing. |
| Eval set size | 10K (4K clear + 6K boundary) | Statistically meaningful, separate sets for independent measurement |
| Training set size | 50K (overgenerate 60K, dedup to 50K) | Enough capacity for boundary learning; pipeline scales if needed |
| Eval generation model | GPT-5.4 | Different model family from training to avoid shared biases |
| Training generation model | Claude Sonnet | Consistent with existing pipeline, cost-effective at scale |
| Dedup strategy | FAISS two-layer (feature-space + semantic) | Fast, catches both structural and semantic duplicates |
| Model architecture | 18-->128-->64-->32 + dropout 0.3 | More capacity for boundary cases, regularization for larger dataset |
| Eval distribution | Uniform 1K/category + 1K/pair | Equal statistical power per category and boundary |
