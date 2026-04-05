# Iteration 6: Dual-LLM Label Audit + Conditional Architecture Upgrade

**Date:** 2026-04-04
**Epic:** `feathierar-16wj.039e26`
**Target:** 97%+ accuracy on clear eval (currently 93.2%, 271 errors / 3,998 samples)
**Approach:** Strictly sequential — label audit first, architecture changes only if needed

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sequencing | Labels first, arch conditional | Label corrections change ground truth; arch experiments on dirty labels are wasted runs |
| Vote system | 4-way: model + label + Haiku 4.5 + GPT-5.4-mini | Independent failure modes: model generalizes from features, LLMs reason semantically |
| Majority threshold | 3/4 auto-corrects, 2-2 → manual review | Clear majority wins; genuine ties need human judgment |
| LLM routing | Both LLMs via OpenRouter | Single API key, single client, simpler code |
| Training audit scope | Eval errors + training errors for weak sub-types only | Targeted audit of unknown, json, jsonl, plain, ini, csv — balances cost vs coverage |
| Architecture changes | Single bundle, conditional on <97% | Low risk with clean labels; BatchNorm compensates for reduced dropout |
| Hyperparameter sweep | CLI flags only, no orchestration | Enables manual sweep via shell loop if needed later |
| Eval/output format | Keep JSONL | Small files, human-debuggable, streaming-write friendly |
| Data file naming | No version suffixes, DVC tracks versions | Overwrite `clear.jsonl`, `golden_raw.parquet` in place; DVC hashes capture history |
| Backward compat | Keep existing --backend/--model flags | New --dual-llm flag adds 4-way mode without breaking single-LLM usage |

## Phase 1: Dual-LLM Label Audit

### 1a. Extend `audit_model_errors.py` for dual-LLM voting

**File:** `training/audit_model_errors.py`

Add `--dual-llm` flag that calls both LLMs via OpenRouter concurrently:

```
# Single-LLM mode (backward compat):
uv run training/audit_model_errors.py \
    --predictions training/output/eval_predictions.clear.jsonl \
    --output training/output/model_error_votes.jsonl \
    --backend openrouter --model openai/gpt-5.4-mini --concurrency 20

# Dual-LLM mode (new):
uv run training/audit_model_errors.py \
    --predictions training/output/eval_predictions.clear.jsonl \
    --output training/output/model_error_votes.jsonl \
    --ties-output training/output/manual_review.jsonl \
    --dual-llm --concurrency 20
```

**Changes:**
- Add `--dual-llm` flag (mutually exclusive with `--backend`/`--model`)
- Add `--ties-output` path for 2-2 ties (default: `manual_review.jsonl` alongside `--output`)
- Create single OpenRouter client (`openai.AsyncOpenAI` with OpenRouter base URL)
- For each sample, fire both LLM calls concurrently under shared semaphore
- Default models: `anthropic/claude-haiku-4-5` and `openai/gpt-5.4-mini`
- Add `--haiku-model` and `--gpt-model` overrides

**4-way vote logic:**
```python
voters = {
    "current_label": sample["expected_category"],
    "model": sample["predicted_category"],
    "haiku": haiku_result,
    "gpt": gpt_result,
}
# Count votes per category
counts = Counter(voters.values())
winner, winner_count = counts.most_common(1)[0]

if winner_count >= 3:
    verdict = "KEEP_LABEL" if winner == voters["current_label"] else "FIX_LABEL"
elif winner_count == 2:
    # Check if it's a 2-2 split
    if len(counts) == 2:
        verdict = "TIE"  # → manual_review.jsonl
    else:
        # 2-1-1 split: plurality wins
        verdict = "KEEP_LABEL" if winner == voters["current_label"] else "FIX_LABEL"
```

**Output JSONL fields (superset of current):**
```json
{
    "index": 42,
    "current_category": "prose",
    "predicted_category": "structured",
    "haiku_category": "structured",
    "gpt_category": "structured",
    "vote_counts": {"structured": 3, "prose": 1},
    "winner": "structured",
    "sub_type": "plain",
    "verdict": "CORRECT",
    "magika_category": "structured",
    "text_preview": "Name: John Doe\\nAge: 42\\n..."
}
```

The `verdict` and `magika_category` fields maintain compatibility with `apply_corrections.py`.

### 1b. Run audit on eval errors (271 samples)

```bash
source ~/.a.sh

# Generate predictions if not already present
uv run training/eval_onnx.py \
    --model src/model.onnx --config src/model_config.json \
    --eval training/data/eval/clear.jsonl \
    --predictions training/output/eval_predictions.clear.jsonl

# Dual-LLM audit
uv run training/audit_model_errors.py \
    --predictions training/output/eval_predictions.clear.jsonl \
    --output training/output/model_error_votes.jsonl \
    --ties-output training/output/manual_review.jsonl \
    --dual-llm --concurrency 20
```

**Expected output:** ~271 samples audited, split into:
- `model_error_votes.jsonl` — auto-corrections (3/4 majority)
- `manual_review.jsonl` — 2-2 ties for human review

### 1c. Manual review of ties

User reviews `manual_review.jsonl`, edits verdicts to `CORRECT` or `MODEL_WRONG`, then merges back:
```bash
# After manual review, concatenate corrections:
cat training/output/model_error_votes.jsonl training/output/manual_review.jsonl \
    > training/output/all_votes.jsonl
```

### 1d. Apply corrections to eval set

```bash
uv run training/apply_corrections.py \
    --votes training/output/all_votes.jsonl \
    --input training/data/eval/clear.jsonl \
    --output training/data/eval/clear.jsonl \
    --no-remap-subtypes
```

### 1e. Extend `eval_onnx.py` for Parquet input

**File:** `training/eval_onnx.py`

Add ability to read Parquet files alongside JSONL, so we can run model predictions on the training data without format conversion.

**Changes:**
- Detect file extension (`.jsonl` vs `.parquet`) in `load_eval_samples()`
- For Parquet: use `polars.read_parquet()`, convert rows to same dict format
- Add `--filter-subtypes` flag to `audit_model_errors.py` to only audit errors for specific sub-types

### 1f. Targeted training data audit

Run model on training data, filter to weak sub-types, audit disagreements:

```bash
# Generate predictions on training data (Parquet input)
uv run training/eval_onnx.py \
    --model src/model.onnx --config src/model_config.json \
    --eval training/data/curated/train/golden_raw.parquet \
    --predictions training/output/eval_predictions.train.jsonl

# Audit only weak sub-types
uv run training/audit_model_errors.py \
    --predictions training/output/eval_predictions.train.jsonl \
    --output training/output/train_error_votes.jsonl \
    --ties-output training/output/train_manual_review.jsonl \
    --dual-llm --concurrency 20 \
    --filter-subtypes unknown,json,jsonl,plain,ini,csv
```

### 1g. Apply training corrections

```bash
uv run training/apply_corrections.py \
    --votes training/output/train_error_votes.jsonl \
    --input training/data/curated/train/golden_raw.parquet \
    --output training/data/curated/train/golden_raw.parquet \
    --no-remap-subtypes
```

### 1h. DVC snapshot

```bash
source ~/.b2.sh
dvc add training/data/eval/clear.jsonl training/data/curated/train/golden_raw.parquet
dvc push
git add training/data/*.dvc
git commit -m "chore(data): DVC snapshot after iteration 6 label audit"
```

## Phase 2: Retrain + Evaluate

Full pipeline on corrected data:

```bash
# Featurize
uv run training/featurize.py \
    --input training/data/curated/train/golden_raw.parquet \
    --output training/data/curated/train/golden_featurized.parquet

# Dedup
uv run training/dedup.py \
    --input training/data/curated/train/golden_featurized.parquet \
    --output training/data/curated/train/golden_train.parquet

# Train
uv run training/train.py \
    --data training/data/curated/train/golden_train.parquet \
    --output training/output/

# Evaluate
uv run training/eval_onnx.py \
    --model training/output/model.onnx \
    --config training/output/model_config.json \
    --eval training/data/eval/clear.jsonl \
    --eval training/data/eval/boundary.jsonl
```

**Decision gate:** If clear eval accuracy >= 97%, skip Phase 3. Copy model artifacts to `src/` and ship.

## Phase 3: Architecture Upgrade (Conditional — only if <97%)

### 3a. Modify `TextClassifier` in `training/train.py`

**Current architecture:**
```python
nn.Linear(n_features, 128) → ReLU → Dropout(0.3)
nn.Linear(128, 64)         → ReLU → Dropout(0.3)
nn.Linear(64, 32)          → ReLU → Dropout(0.3)
```

**New architecture:**
```python
nn.Linear(n_features, 256) → BatchNorm1d(256) → ReLU → Dropout(0.15)
nn.Linear(256, 64)         → BatchNorm1d(64)  → ReLU → Dropout(0.15)
nn.Linear(64, 32)          → BatchNorm1d(32)  → ReLU → Dropout(0.15)
```

**Additional changes:**
- Sub-type loss weight: `0.3 → 0.5`
- LR warmup: `LinearLR` for first 10 epochs, chained with existing `ReduceLROnPlateau` via `SequentialLR`

### 3b. Add CLI flags to `train.py`

```
--dropout FLOAT       Dropout rate (default: 0.15)
--hidden-dim INT      First hidden layer width (default: 256)
--sub-type-weight FLOAT  Sub-type loss weight (default: 0.5)
--no-batchnorm        Disable BatchNorm layers
--warmup-epochs INT   LR warmup epochs (default: 10, 0 to disable)
```

These flags enable future manual sweeps without code changes.

### 3c. Retrain + evaluate with new architecture

```bash
uv run training/train.py \
    --data training/data/curated/train/golden_train.parquet \
    --output training/output/ \
    --dropout 0.15 --hidden-dim 256 --sub-type-weight 0.5 --warmup-epochs 10
```

### 3d. Update Rust model

```bash
cp training/output/model.onnx src/model.onnx
cp training/output/model_config.json src/model_config.json
cargo build --features onnx-model
cargo test --features onnx-model
```

No Rust code changes needed — ONNX runtime handles BatchNorm ops transparently.

## Files Modified

### Modified
| File | Changes |
|------|---------|
| `training/audit_model_errors.py` | Add `--dual-llm`, `--ties-output`, `--filter-subtypes`, `--haiku-model`, `--gpt-model`; 4-way vote logic; dual OpenRouter client |
| `training/eval_onnx.py` | Add Parquet input support (detect file extension, polars reader) |
| `training/train.py` | Add BatchNorm, CLI flags (`--dropout`, `--hidden-dim`, `--sub-type-weight`, `--no-batchnorm`, `--warmup-epochs`), LR warmup |
| `src/model.onnx` | Retrained model (38 features, potentially new architecture) |
| `src/model_config.json` | Updated feature means/stds for retrained model |

### Data (DVC-tracked, overwritten in place)
| File | Changes |
|------|---------|
| `training/data/eval/clear.jsonl` | Label corrections from 4-way audit |
| `training/data/curated/train/golden_raw.parquet` | Targeted label corrections for weak sub-types |

### New output artifacts (gitignored, ephemeral)
| File | Purpose |
|------|---------|
| `training/output/model_error_votes.jsonl` | Auto-corrections from 4-way vote |
| `training/output/manual_review.jsonl` | 2-2 ties for human review |
| `training/output/eval_predictions.train.jsonl` | Model predictions on training data |
| `training/output/train_error_votes.jsonl` | Training data corrections |

## Success Criteria

- Clear eval accuracy >= 97% (errors reduced from 271 to ~120 or fewer)
- Boundary eval accuracy improved from 85.3% baseline
- No regression on any per-category F1 score
- All 2-2 ties manually reviewed and resolved
- DVC snapshot of corrected data pushed to B2

## Risks

| Risk | Mitigation |
|------|------------|
| LLM votes may be noisy for genuinely ambiguous samples | 4-way voting + manual review for ties filters noise |
| Training data audit may surface thousands of disagreements | `--filter-subtypes` scopes to known weak areas; can adjust concurrency/cost |
| Architecture bundle may not help if labels are the bottleneck | Conditional gate — skip if audit alone reaches 97% |
| BatchNorm changes ONNX graph structure | ort handles BatchNorm ops; verified by `cargo test --features onnx-model` |

## Forward Compatibility: Multi-Label Features Head

Epic `feathierar-16wj.03j58p` adds a 3rd output head (sigmoid, binary features like `has_code`, `has_tables`) immediately after Iteration 6. The architecture changes here are designed to be additive:

| Iteration 6 change | Multi-label interaction | Status |
|---|---|---|
| Wider shared backbone (→256→64→32) | Multi-label head attaches to same 32-dim embedding. Wider backbone carries more signal — beneficial. | Compatible |
| CLI flags (`--dropout`, `--hidden-dim`) | Multi-label adds `--features-weight`, `--num-feature-labels`. Naming convention: shared-backbone params vs head-specific params. | Compatible |
| Sub-type loss weight flag | Total loss becomes `cat + W_sub * sub + W_feat * features`. Weight flags are already head-specific. | Compatible |
| LR warmup + SequentialLR | Operates on total loss including future BCE term. | Compatible |
| ONNX export (2 output tensors) | Multi-label adds 3rd tensor. `output_names` list extends to `["category_logits", "sub_type_logits", "feature_logits"]`. | Additive change |
| Rust `tier2.rs` | Multi-label extracts `outputs[2]`. Iteration 6 doesn't touch Rust inference code. | No conflict |

**No defensive abstractions needed** — the multi-label changes are purely additive to the foundation laid here.

## Related Issues

- `feathierar-16wj.04udib` — Consolidate training scripts into unified CLI tool (P4 backlog)
- `feathierar-16wj.03j58p` — Multi-label classification: add features output head (next after Iteration 6)
