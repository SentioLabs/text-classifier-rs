# CSV to Parquet Migration + Pandas to Polars Standardization

**Date:** 2026-04-04
**Status:** Approved

## Motivation

The curated training CSVs (`golden_raw.csv`, `golden_featurized.csv`, `golden_train.csv`) contain multiline text fields that are fragile in CSV format — quoting/escaping edge cases cause parser mismatches and silent corruption. At 300-400 MB each, they are also inefficient to store and transfer via DVC.

Parquet eliminates the multiline problem entirely (binary columnar format, no escaping), provides built-in compression (2-5x smaller), preserves type information (float64 stays float64), and enables column-selective reads.

Separately, the training pipeline uses a mix of pandas and polars. Standardizing on polars for all parquet I/O removes pandas as a dependency from `train.py` and `dedup.py`, reducing install weight and unifying the dataframe API across the pipeline.

## Scope

### In scope

| Script | Current state | Change |
|--------|--------------|--------|
| `split_dataset.py` | stdlib `csv`, no deps | Add `polars` dep, write `golden_raw.parquet` |
| `generate.py` | stdlib `csv` (golden-train mode) | Add `polars` (lazy import), write `golden_raw.parquet` |
| `featurize.py` | polars + CSV | Swap `read_csv`/`write_csv` to `read_parquet`/`write_parquet` |
| `dedup.py` | pandas + CSV | Migrate pandas to polars, read/write parquet |
| `train.py` | pandas + CSV | Migrate pandas to polars, read parquet |
| `Taskfile.yml` | References `.csv` paths | Update to `.parquet` |
| `test_split_dataset.py` | Verifies CSV output | Verify parquet output via polars |
| `test_featurize.py` | Writes/reads CSV fixtures | Write/read parquet fixtures |
| `test_dedup.py` | pandas CSV fixtures | polars parquet fixtures |
| `test_train.py` | pandas CSV fixtures | polars parquet fixtures |
| `test_generate.py` | Verifies CSV output | Verify parquet output (golden-train mode) |

### Deferred (future session)

- `apply_corrections.py` — multi-format dispatch, will add `.parquet` support later
- `correct_labels.py` — same pattern
- `audit_labels_vote.py` — same pattern

### Out of scope (no change)

- JSONL files (eval sets, source data) — record-oriented, streamed in Rust
- `data/manual/` CSVs (`fixtures.csv`, `perturbations.csv`) — small, human-editable, git-tracked
- `generate.py` non-golden modes (fixtures, synthetic, perturb, combined) — stay CSV
- Rust code — never reads these files

## Script-by-script design

### 1. `split_dataset.py`

**Dependencies:** `[]` becomes `["polars"]`

**Change:** Replace the `csv.DictWriter` block (~lines 247-259) that writes the training set:

```python
# Before
with open(train_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[...])
    writer.writeheader()
    for s in training:
        writer.writerow({...})

# After
pl.DataFrame([
    {
        "text": s.get("text", ""),
        "category": s.get("expected_category", ""),
        "sub_type": s.get("sub_type", ""),
        "source": s.get("source", "unknown"),
        "model": s.get("model", "unknown"),
    }
    for s in training
]).write_parquet(train_path)
```

Update `--train-output` default from `golden_raw.csv` to `golden_raw.parquet`. Update docstring.

### 2. `generate.py` (golden-train mode only)

**Dependencies:** Add `polars` (lazy import — only golden-train mode needs it).

**Change:** Replace `csv.DictWriter` block at ~lines 1085-1090:

```python
# Before
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=golden_columns)
    writer.writeheader()
    writer.writerows(all_rows)

# After
import polars as pl
pl.DataFrame(all_rows).select(golden_columns).write_parquet(output_path)
```

Update output path from `golden_raw.csv` to `golden_raw.parquet`.

### 3. `featurize.py`

**Dependencies:** No change (already uses polars).

**Change:** Swap format methods:

```python
# Before
df = pl.read_csv(str(input_path))
result.write_csv(str(output_path))

# After
df = pl.read_parquet(str(input_path))
result.write_parquet(str(output_path))
```

Update defaults from `.csv` to `.parquet`. Update docstring.

### 4. `dedup.py` (pandas to polars)

**Dependencies:** `["faiss-cpu", "sentence-transformers", "pandas", "numpy"]` becomes `["faiss-cpu", "sentence-transformers", "polars", "numpy"]`

**Key migrations:**

| pandas | polars |
|--------|--------|
| `pd.read_csv(path)` | `pl.read_parquet(path)` |
| `df[cols].values.astype(np.float32)` | `df.select(cols).to_numpy().astype(np.float32)` |
| `df.select_dtypes(include=[np.number])` | `import polars.selectors as cs; df.select(cs.numeric())` |
| `df.loc[keep].reset_index(drop=True)` | `df.filter(pl.Series(keep))` |
| `df[text_col].tolist()` | `df.get_column(text_col).to_list()` |
| `df.copy()` | `df.clone()` |
| `df.to_csv(path, index=False)` | `df.write_parquet(path)` |

Update arg defaults and docstrings from `.csv` to `.parquet`.

### 5. `train.py` (pandas to polars)

**Dependencies:** `["torch", "pandas", "numpy", ...]` becomes `["torch", "polars", "numpy", ...]`

**Key migrations in `load_and_prepare_data()`:**

| pandas | polars |
|--------|--------|
| `pd.read_csv(path)` | `pl.read_parquet(path)` |
| `df[list(cols)].values.astype(np.float32)` | `df.select(list(cols)).to_numpy().astype(np.float32)` |
| `df["category"].map(CATEGORY_MAP).values` | `df.get_column("category").replace_strict(CATEGORY_MAP).to_numpy()` |
| `df["sub_type"].fillna("unknown")` | `df.with_columns(pl.col("sub_type").fill_null("unknown"))` |
| `df["sub_type"].unique()` | `df.get_column("sub_type").unique().sort().to_list()` |
| `df["source"].fillna(...).astype(str).values` | `df.get_column("source").fill_null("unknown_source").cast(pl.Utf8).to_numpy()` |
| `"source" in df.columns` | `"source" in df.columns` (same API) |

Rename parameter `csv_path` to `data_path` in `load_and_prepare_data()`. Update docstrings.

### 6. `Taskfile.yml`

Update four task definitions:

```yaml
# train:split
golden_raw.csv -> golden_raw.parquet

# train:featurize
golden_raw.csv -> golden_raw.parquet
golden_featurized.csv -> golden_featurized.parquet

# train:dedup
golden_featurized.csv -> golden_featurized.parquet
golden_train.csv -> golden_train.parquet

# train:train
golden_train.csv -> golden_train.parquet
```

### 7. Tests

Each test file needs two categories of changes:

**Fixture creation:** Tests that write CSV fixtures for input need to write parquet instead. For tests using `csv.writer` or `csv.DictWriter`, switch to `pl.DataFrame(...).write_parquet()`. For tests using `pd.DataFrame.to_csv()`, switch to `pl.DataFrame(...).write_parquet()`.

**Output verification:** Tests that read output CSVs via `csv.DictReader` or `pd.read_csv` switch to `pl.read_parquet()`.

**Test file extensions:** Update all `tmp_path / "*.csv"` references to `tmp_path / "*.parquet"` for files in the golden pipeline. Leave `.csv` references for non-golden files (fixtures, perturbations, combined).

## Dependency summary

| Script | Removed | Added |
|--------|---------|-------|
| `split_dataset.py` | — | `polars` |
| `generate.py` | — | `polars` (lazy) |
| `featurize.py` | — | — |
| `dedup.py` | `pandas` | `polars` |
| `train.py` | `pandas` | `polars` |

## File format details

- **Compression:** Use polars default (zstd). No need to configure explicitly.
- **Schema:** Polars infers types from the DataFrame. String columns stay `Utf8`, feature columns stay `Float32`/`Float64`.
- **No row index:** Parquet has no equivalent of CSV's implicit row order issues — polars preserves insertion order.

## Risk and rollback

- **DVC tracking:** The `.dvc` files will update on the next `dvc add` after regenerating data. Old CSV data in remote storage remains accessible via `dvc checkout` with the old `.dvc` hash.
- **Deferred scripts:** `apply_corrections.py`, `correct_labels.py`, and `audit_labels_vote.py` will fail if pointed at `.parquet` files until they are migrated. This is intentional — they dispatch on file extension and will need `.parquet` support added in a future session.
