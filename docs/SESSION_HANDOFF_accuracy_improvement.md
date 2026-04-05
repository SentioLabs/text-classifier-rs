# Session Handoff: Model Accuracy Improvement (84% → 90.1%)

**Date**: 2026-04-03
**Branch**: `feat/hierarchical-taxonomy`
**Working directory**: `.worktrees/feat-hierarchical-taxonomy/training/`
**Arc epic**: `textclassi-0hba.09paif` (closed — all 6 child tasks complete)

---

## What Was Done

### 1. Class-Weighted Loss (train.py)

Added inverse-frequency class weighting to `CrossEntropyLoss` so the model penalizes minority class (artifact, structured) misclassifications more heavily:

```python
class_counts = np.bincount(y_cat_train, minlength=NUM_CATEGORIES)
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * len(class_weights)
# ... passed to nn.CrossEntropyLoss(weight=weight_tensor)
```

### 2. Category Downsampling (split_dataset.py)

Added `--max-per-category` flag to `split_dataset.py`. The Taskfile uses `--max-per-category 50000` to cap any single category from dominating training.

### 3. Four Content-Level Features (featurize.py)

Added 4 new features targeting the artifact detection problem. Model now has **22 features** (18 structural + 4 content-level):

| Feature | What It Measures | Why It Helps |
|---------|-----------------|--------------|
| `dictionary_word_ratio` | Fraction of tokens found in a 227K English word dictionary (`data/wordlist.txt`) | Prose: ~0.85-0.95. OCR garbage: ~0.1-0.3. Directly separates artifact from prose. |
| `encoding_error_ratio` | Fraction of U+FFFD replacement chars + mojibake sequences (Ã©, â€", etc.) | PDF dumps/OCR: high. Everything else: ~0. Very high signal for artifact. |
| `repeated_ngram_ratio` | Fraction of 3-word n-gram types appearing more than once | PDF dumps with repeated headers/footers: high. Prose: low. |
| `sentence_coherence_score` | Fraction of lines starting uppercase + ending with .!? | Prose: ~0.7-0.9. OCR garbage: ~0.1. Code/tables: low. |

### 4. Real Data Sourcing (sample_real_data.py)

Created `sample_real_data.py` to source real-world text from HuggingFace datasets. **Target: 138K samples. Achieved: 83K samples** (see HF issues below).

### 5. Taskfile + FEATURE_COLUMNS Updates

- Added `train:sample-real-v2` task
- Updated `train:generate` to concat `real_samples.jsonl` + `real_samples_v2.jsonl` + `synthetic_samples.jsonl`
- Updated `train:split` to use `--max-per-category 50000`
- Updated `FEATURE_COLUMNS` from 18 → 22 features
- Updated `TextClassifier` default `n_features` from 18 → 22

### 6. Retrain and Evaluate

Full pipeline: combine 142K raw → split (132K training, 10K eval) → featurize (22 features) → dedup (93K after 29% semantic dedup) → train (154 epochs, early stop at 139) → eval.

---

## Current Results

### Eval: clear.jsonl (4,000 samples)

| Metric | Baseline | Now | Delta |
|--------|----------|-----|-------|
| **Overall accuracy** | 84.0% | **90.1%** | +6.1% |
| Prose recall | 93% | **95%** | +2% |
| Code recall | 97% | **97%** | — |
| Structured recall | 89% | **93%** | +4% |
| **Artifact recall** | 57% | **75%** | **+18%** |

### Confusion Matrix (clear eval)

```
                   prose    code  structured  artifact  skip
prose                948      34           2        16     0
code                  10     969          13         8     0
structured            31      18         933        18     0
artifact              77      85          85       753     0
```

### Key Observation: Artifact Is Still the Bottleneck

Artifact has 75% recall but the confusion matrix shows **247 artifact samples misclassified**:
- 77 as prose (31%)
- 85 as code (34%)
- 85 as structured (34%)

This is the primary drag on overall accuracy. Fixing artifact recall from 75% → 95% alone would push overall accuracy from 90.1% to ~95%.

### Eval: boundary.jsonl (6,000 samples)

Overall: 53.3% (up from 52.8% — boundary is inherently hard, these are ambiguous samples).

### Validation Metrics (from training)

```json
{
  "best_val_loss": 0.2559,
  "best_epoch": 139,
  "total_epochs": 154,
  "val_category_accuracy": 0.9706,
  "val_sub_type_accuracy": 0.8300
}
```

The 97% validation accuracy vs 90% eval accuracy gap suggests the eval set has harder/more diverse samples than the training distribution — especially for artifact.

---

## Training Data Distribution

### Combined raw data (142K → 132K after eval split)

| Category | v1 Real | v1 Synthetic | v2 Real | Total | After Eval Split |
|----------|---------|-------------|---------|-------|-----------------|
| code | 25,200 | 13,862 | 0 | 39,062 | ~36,562 |
| prose | 4,200 | 4,411 | 18,986 | 27,597 | ~25,097 |
| artifact | 0 | 4,411 | 33,000 | 37,411 | ~34,910 |
| structured | 0 | 7,319 | 31,000 | 38,319 | ~35,820 |

After semantic dedup: **93,657 total training samples**.

### v2 Real Data Sub-type Breakdown (82,986 samples achieved)

| Sub-type | Count | Source |
|----------|-------|--------|
| boilerplate | 10,000 | Synthetic templates (HF The Stack failed) |
| ocr_garbage | 15,000 | Programmatic generation |
| skip | 8,000 | Programmatic generation |
| plain (prose) | 18,986 | HF: arxiv_abstracts (10K), wikipedia_full (~10K from section extraction) + generated_prose (9K, only partial yielded) |
| csv | 10,511 | Programmatic generation |
| tsv | 4,489 | Programmatic generation |
| log_lines | 6,000 | Programmatic generation |
| ini | 3,392 | Programmatic generation |
| key_value | 3,193 | Programmatic generation |
| xml | 3,415 | Programmatic generation |

**Note**: Zero `pdf_dump` samples in v2 (both finepdfs and scientific_papers HF sources failed — see below). All 33K artifact samples are synthetic OCR garbage, boilerplate, and skip. This is a major gap.

---

## HuggingFace Dataset Issues (Critical for Next Session)

### Sources That FAILED

| Source | Dataset | Error | Root Cause | Fix |
|--------|---------|-------|------------|-----|
| `sample_finepdfs` | `HuggingFaceFW/finepdfs_100BT` | `CastError: column names don't match` | Different parquet shards have different schemas (some have `ocr_quality_scores`, `fw_edu_scores`, etc., others don't). Using `data_files="data/000_00000.parquet"` for a single shard also failed because the dataset metadata defines a narrower schema than the actual files contain. | **Fix**: Use `datasets.load_dataset(..., trust_remote_code=True)` or explicitly set `features=None` to let each shard define its own schema. Alternatively, load individual parquet files directly with `pd.read_parquet()` or `polars.read_parquet()` from the HF URL. |
| `sample_scientific_papers` | `armanc/scientific_papers` | `Dataset scripts are no longer supported` | This dataset uses a legacy loading script (`scientific_papers.py`) that the `datasets` library no longer supports. | **Fix**: Use `ccdv/arxiv-summarization` instead (same data, different upload format). Or use `arxiv-community/arxiv_dataset`. Or download the parquet files directly. |
| `sample_boilerplate` | `bigcode/the-stack` | `directory at .../data/license doesn't contain any data files` | The Stack v1 organizes by programming language subdirectories under `data/`. There is no `data/license` subdirectory. License files would be under various language directories. | **Fix**: Use `bigcode/the-stack-dedup` (now un-gated per user). Filter by `path` field containing LICENSE/NOTICE/COPYING. Don't use `data_dir` parameter — stream the full dataset and filter client-side, OR use a specific language subset like `data_dir="data/text"`. |
| `sample_wikipedia_paragraphs` | `agentlans/wikipedia-paragraphs` | Silently yielded 0 samples | The dataset loaded but the `paragraph` field may not exist or the text didn't pass the 50-10000 char filter. | **Fix**: Investigate the actual schema of this dataset. Check field names (`text`, `paragraph`, `content`). |
| `sample_wikipedia_full` | `wikimedia/wikipedia` | `ParquetConfig.__init__() got an unexpected keyword argument 'language'` | The `wikimedia/wikipedia` dataset changed its config API. The `language` parameter is no longer valid. | **Fix**: Use `name="20231101.en"` instead of `language="20231101.en"` as the config name. Check the dataset card for current loading instructions. |
| `sample_stack_configs` | `bigcode/the-stack` | Same as boilerplate — hung trying to stream, never produced samples | The Stack requires specifying a `data_dir` for the language subset, but config files (.json, .yaml, .toml) span multiple language dirs. Without `data_dir`, it tries to load the entire 6TB dataset. | **Fix**: Use `bigcode/the-stack-dedup` with specific language data dirs like `data_dir="data/json"`, `data_dir="data/yaml"`, `data_dir="data/toml"`. These are actual subdirectories in The Stack's organization. |

### Sources That WORKED

| Source | Dataset | Samples | Notes |
|--------|---------|---------|-------|
| `sample_arxiv_abstracts` | `gfissore/arxiv-abstracts-2021` | ~10,000 | Clean, reliable. Good prose source. |
| `sample_wikipedia_full` | `wikimedia/wikipedia` | ~10,000 (with section extraction) | Worked after the `language` parameter failed — fell through to produce ~10K section-level samples from a cached/partial load. Actually unclear if this truly produced 10K or if it was the prose generators making up the difference. |

### Sources That Are Programmatic (Always Work)

| Source | Count | Quality |
|--------|-------|---------|
| `generate_skip_samples` | 8,000 | Good — random whitespace, empty strings, fragments |
| `generate_ocr_garbage` | 15,000 | **Mediocre** — 6 methods of garbling but all synthetic. Real OCR garbage from actual PDFs would be much more diverse. |
| `generate_csv_samples` | 15,000 | Good — varied schemas, delimiters, data types |
| `generate_log_samples` | 6,000 | Good — syslog, access, nginx, JSON formats |
| `generate_kv_samples` | 10,000 | Good — INI, properties, XML configs |
| `generate_prose_variants` | 9,000 | **Mediocre** — template-based letters/emails/dialogues. Very repetitive patterns. |
| `_generate_synthetic_boilerplate` | 10,000 | **Mediocre** — only 4 license templates (MIT, Apache, BSD, GPL) with author/year variation |

---

## Why Artifact Recall Is Still at 75% (Not 95%)

### Root Cause Analysis

1. **No real PDF dump data in training**: The v2 data has zero `pdf_dump` samples because both finepdfs and scientific_papers HF sources failed. All artifact training data is synthetic (OCR garbage generators, license templates, skip fragments). The eval set likely contains real-world artifact patterns the model has never seen.

2. **Synthetic OCR garbage doesn't match real OCR noise**: The `generate_ocr_garbage()` function uses 6 methods (random Unicode, broken encoding, mixed scripts, repeated garble, number-heavy garble, symbol-heavy garble). Real OCR output has very different patterns — broken word boundaries, partial character recognition, layout artifacts from multi-column pages, header/footer repetition.

3. **Artifact ↔ prose confusion (77 samples)**: PDF dumps of well-formatted documents look like prose. The `dictionary_word_ratio` helps but isn't enough when the PDF text is actually readable English with just some formatting noise (broken line breaks, hyphenated words, stray page numbers).

4. **Artifact ↔ code confusion (85 samples)**: Some PDF dumps contain code snippets or technical content with special characters, making them look like code to the feature extractors.

5. **Artifact ↔ structured confusion (85 samples)**: PDF dumps of tables, forms, and structured documents have delimiter patterns and key-value patterns that look like structured data.

### What Would Fix It

1. **Get real PDF dump data into training**: Fix the finepdfs and/or scientific_papers HF sources. Even 5-10K real PDF extractions would dramatically improve artifact recall because the model would learn the actual feature distributions of real PDF text.

2. **Get real OCR garbage**: Use `rubentito/OCR-IDL` or `allenai/cord19` with proper schema handling. The OCR-Quality dataset (`Aslan-mingye/OCR-Quality`) is small (1K) but high-quality human-annotated.

3. **Add PDF-specific features**: Consider features like `page_number_density` (fraction of lines that are just numbers — page numbers), `hyphenated_line_break_ratio` (lines ending with `-` followed by a word continuation), `header_footer_repetition` (repeated short lines appearing at regular intervals).

4. **More diverse boilerplate**: Beyond 4 license templates, include: contributor agreements, codes of conduct, changelog entries, auto-generated API docs, README boilerplate, GitHub issue/PR templates.

---

## Remaining HF Datasets to Try

These datasets from the original plan were never successfully sourced:

### Artifact (highest priority)
- **`HuggingFaceFW/finepdfs_100BT`** — Fix: load parquet files directly, or use `trust_remote_code=True`, or try the `CC-MAIN-*` configs instead of default
- **`ccdv/arxiv-summarization`** — Replacement for `armanc/scientific_papers`. Has `article` field with full paper text
- **`rubentito/OCR-IDL`** — 19M pages of industry documents with OCR. May need auth.
- **`allenai/cord19`** — COVID-19 papers. Filter by encoding error density for artifact samples
- **`Aslan-mingye/OCR-Quality`** — Small (1K) but human-annotated OCR quality scores

### Prose
- **`agentlans/wikipedia-paragraphs`** — Debug the field name issue (may be `text` not `paragraph`)
- **`wikimedia/wikipedia`** — Fix: use `name="20231101.en"` as config name instead of `language=` kwarg
- **`dennlinger/wiki-paragraphs`** — Section-level Wikipedia text with headers. Good for markdown sub-type.
- Book corpora, news datasets

### Structured
- **`bigcode/the-stack-dedup`** — Now un-gated. Use `data_dir="data/json"`, `data_dir="data/yaml"`, `data_dir="data/toml"` for config files. Use `data_dir="data/csv"` for tabular data if it exists.

---

## File Inventory

### Modified Files (committed + pushed)
- `training/train.py` — class weights, 22 FEATURE_COLUMNS, n_features=22
- `training/test_train.py` — updated for 22 features, added class weight tests
- `training/split_dataset.py` — `--max-per-category` flag
- `training/test_split_dataset.py` — downsampling tests
- `training/featurize.py` — 4 new content-level features, 22-feature registry
- `training/test_featurize.py` — tests for all 4 new features
- `training/Taskfile.yml` — `train:sample-real-v2`, updated `train:generate` and `train:split`
- `training/dedup.py` — NaN text handling fix
- `training/data/wordlist.txt` — 227K English words for `dictionary_word_ratio`

### New Files (committed + pushed)
- `training/sample_real_data.py` — HF dataset sampler + programmatic generators
- `training/test_sample_real_data.py` — 36 tests for generators

### Generated Data (not committed, in `.worktrees/feat-hierarchical-taxonomy/training/data/`)
- `real_samples_v2.jsonl` — 83K samples (artifact + prose + structured)
- `raw_samples.jsonl` — 142K combined samples
- `golden_raw.csv` — 132K training split
- `golden_featurized.csv` — 132K with 22 features
- `golden_train.csv` — 93K after dedup

### Model Output (not committed, in `.worktrees/feat-hierarchical-taxonomy/training/output/`)
- `model.onnx` — trained model (22 features input)
- `model_config.json` — feature names, means, stds, label maps
- `metrics.json` — training metrics

---

## Recommended Next Steps (Priority Order)

### 1. Fix HF PDF Data Sources (~+10-15% artifact recall expected)

This is the single highest-impact change. The model has never seen real PDF dump text.

```python
# Option A: Load finepdfs parquet directly
import polars as pl
df = pl.read_parquet("hf://datasets/HuggingFaceFW/finepdfs_100BT/data/000_00000.parquet",
                     columns=["text"])  # only read the text column

# Option B: Use arxiv summarization instead of scientific_papers
ds = datasets.load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
for row in ds:
    text = row["article"]  # full paper text with PDF extraction noise
```

### 2. Fix The Stack Dedup Access for Configs + Boilerplate

The user has un-gated `bigcode/the-stack-dedup`. Use it for:
- Real config files (JSON, YAML, TOML) → structured training data
- Real LICENSE/NOTICE files → artifact boilerplate training data

```python
# Configs by language subdirectory
ds = datasets.load_dataset("bigcode/the-stack-dedup", data_dir="data/json",
                           split="train", streaming=True)
```

### 3. Add Artifact-Specific Features

The current 4 content features help but more could be added:
- `page_number_density` — lines that are just 1-3 digit numbers (page numbers in PDFs)
- `broken_word_ratio` — words split by line breaks with hyphens
- `mixed_formatting_score` — mix of font indicators, spacing inconsistencies
- `short_repeated_line_ratio` — lines ≤20 chars that repeat (headers/footers)

### 4. Increase Training Data Volume

Current: 93K after dedup. Target: 150-200K. The model has 14K parameters so ~140K-280K is the productive range. More diverse real data (not more synthetic data) is what's needed.

### 5. Experiment with Model Architecture

The current architecture (22→128→64→32→5 with dropout 0.3) may benefit from:
- Wider layers (22→256→128→64→5)
- Lower dropout (0.2) since we now have more data
- Learning rate warmup or cosine annealing

---

## How to Run the Pipeline

```bash
cd .worktrees/feat-hierarchical-taxonomy/training

# 1. Generate data (fix sample_real_data.py first)
task train:sample-real-v2

# 2. Combine all sources
task train:generate

# 3. Split with downsampling
task train:split

# 4. Featurize (22 features)
task train:featurize

# 5. Dedup
KMP_DUPLICATE_LIB_OK=TRUE task train:dedup

# 6. Train
task train:train

# 7. Evaluate
task train:eval-onnx
```

Or run the full pipeline: `task train:pipeline`

---

## Environment Notes

- Python 3.12 via `uv`
- All training scripts use PEP 723 inline deps (managed by `uv run`)
- `KMP_DUPLICATE_LIB_OK=TRUE` required for dedup (OpenMP conflict between torch and faiss)
- HuggingFace token: set `HF_TOKEN` env var for gated datasets
- The `dedup.py` has a fix for NaN text values (line 117: converts non-string values to str before encoding)
