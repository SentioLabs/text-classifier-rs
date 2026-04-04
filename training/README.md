# Training Pipeline

The training pipeline generates labeled feature data from text samples and trains a dual-head neural network that classifies text by category (prose, code, structured) and sub-type. The trained model is exported to ONNX format for use by the Rust classifier's Tier 2 path.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [DVC](https://dvc.org/) with S3 support: `uv tool install 'dvc[s3]'`
- Rust toolchain (the `classify` binary must be built before data generation)
- B2 credentials for dataset storage:
  - `AWS_ACCESS_KEY_ID` environment variable
  - `AWS_SECRET_ACCESS_KEY` environment variable
- Optional: `ANTHROPIC_API_KEY` environment variable for synthetic data generation

## Quick Start

```bash
make training-setup     # create venv and install dependencies
cd training && dvc pull  # restore DVC-tracked datasets from remote storage
make train-pipeline     # generate data + train model
```

To include synthetic data generated via the Claude API:

```bash
export ANTHROPIC_API_KEY=sk-...
make train-pipeline
```

## Directory Structure

```text
training/
  data/
    source/
      real/              # Real samples from The Stack + HuggingFace
      synthetic/         # LLM-generated samples
      merged/            # Combined raw_samples.jsonl
      pilot/             # Pilot validation samples (regenerable, gitignored)
    curated/
      train/             # golden_raw.csv -> golden_featurized.csv -> golden_train.csv
    eval/                # Frozen eval sets (clear.jsonl, boundary.jsonl)
    archive/             # Historical timestamped snapshots
    manual/              # Hand-maintained inputs (Git-tracked)
  .dvc/                  # DVC configuration
  generate.py            # Data generation script
  train.py               # PyTorch training + ONNX export
  featurize.py           # Feature extraction
  sample_real_data.py    # Real-data sampling from The Stack
  eval_onnx.py           # ONNX model evaluation
  pyproject.toml         # Python dependencies
  Taskfile.yml           # Task runner definitions
  models/                # Model artifacts (gitignored except config)
    model.onnx
    model_config.json
    metrics.json
```

## Dataset Management

Training datasets are tracked with [DVC](https://dvc.org/) and stored in a Backblaze B2 bucket. Git tracks only the `.dvc` pointer files; the actual data is pulled on demand.

### Restore datasets

After cloning or switching branches, pull all DVC-tracked data:

```bash
cd training && dvc pull
```

### Update a dataset

When you add or modify samples in a tracked directory, push the changes to remote storage:

```bash
dvc add data/source/real
git add data/source/real.dvc
git commit -m "Refresh real samples"
dvc push
```

### Release tagging

Tag a dataset release so it can be restored later:

```bash
git tag training-data/vYYYY-MM-DD-rN
```

### Restore a historical snapshot

Check out a previous tagged release and pull the corresponding data:

```bash
git checkout <tag> -- training/
cd training && dvc pull
```

## Data Generation

The real-data sampler exposes a quota plan so category coverage can be reviewed before spending time on networked downloads:

```bash
cd training
task train:sample-real-v2-plan   # print source quotas and category balance
task train:sample-real-v2        # execute the sampling plan
```

The `sample_real_data.py` plan distinguishes:

- Real structured config files from The Stack dedup
- Real prose and code samples from HuggingFace datasets

Run `generate.py` with one of four modes via `--mode`:

| Mode | What it does | API key required |
|------|-------------|-----------------|
| `fixtures` | Extracts features from `tests/fixtures/` text files using the `classify` CLI | No |
| `synthetic` | Generates text samples via the Claude API, then extracts features | Yes |
| `perturb` | Adds Gaussian noise to fixture features to create boundary cases | No |
| `all` | Runs all three modes and combines results | Optional |

Each mode writes a CSV to `training/data/`.

```bash
# Fixtures only (fast, no API key)
make generate-fixtures

# All modes
make generate-data
```

### CLI options

```bash
cd training && uv run python generate.py \
    --mode all \
    --output data/ \
    --samples-per-type 50 \
    --api-key "$ANTHROPIC_API_KEY"
```

## Training

The training script (`train.py`) reads a curated CSV and trains a dual-head feedforward network:

- **Shared layers**: Linear(18, 64) - ReLU - Dropout(0.2) - Linear(64, 32) - ReLU - Dropout(0.2)
- **Category head**: Linear(32, 3) -- predicts prose, code, or structured
- **Sub-type head**: Linear(32, N) -- predicts a finer label (e.g., python, csv, markdown)

Slice-aware refinements are available behind explicit flags:

- `--group-val-by-source`: when the CSV includes a `source` column with multiple source groups, validation is split by disjoint source groups to reduce source leakage.

Default hyperparameters:

| Parameter | Default |
|-----------|---------|
| Epochs | 200 (with early stopping) |
| Batch size | 64 |
| Learning rate | 0.001 (Adam) |
| Early stopping patience | 15 epochs |
| Validation split | 20% (stratified) |

```bash
make train

# Or with custom options:
cd training && uv run python train.py \
    --data data/curated/train/golden_train.csv \
    --output models/ \
    --group-val-by-source \
    --epochs 200 \
    --patience 15
```

The `task train:train` command uses row-level stratified validation. Add `--group-val-by-source` when your training CSV has multiple meaningful source groups.

## Validation

After training, validate the model against a labeled JSONL test set using the Rust CLI:

```bash
# Text output
classify validate --input test.jsonl

# JSON output
classify validate --input test.jsonl --json
```

You can generate labeled data with `classify label-corpus --with-features` and use that as validation input.

## Output Files

| File | Contents |
|------|----------|
| `model.onnx` | Trained ONNX model (opset 17) with two outputs: `category_logits` and `sub_type_logits` |
| `model_config.json` | Feature names, Z-score normalization stats (mean/std), category map, and sub-type map |
| `metrics.json` | Training results: best validation loss, best epoch, total epochs, category accuracy, sub-type accuracy |

The Rust classifier loads `model.onnx` and `model_config.json` at runtime to normalize input features and decode predictions.
