# Training Pipeline

The training pipeline generates labeled feature data from text samples and trains a dual-head neural network that classifies text by category (prose, code, structured, artifact, skip) and sub-type. The trained model is exported to ONNX format for use by the Rust classifier's Tier 2 path.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Rust toolchain (the `classify` binary must be built before data generation)
- Optional: `ANTHROPIC_API_KEY` environment variable for synthetic data generation

## Quick Start

```bash
make training-setup     # create venv and install dependencies
make train-pipeline     # generate data + train model (fixtures only, no API key needed)
```

To include synthetic data generated via the Claude API:

```bash
export ANTHROPIC_API_KEY=sk-...
make train-pipeline
```

## Data Generation

Run `generate.py` with one of four modes via `--mode`:

| Mode | What it does | API key required |
|------|-------------|-----------------|
| `fixtures` | Extracts features from `tests/fixtures/` text files using the `classify` CLI | No |
| `synthetic` | Generates text samples via the Claude API, then extracts features | Yes |
| `perturb` | Adds Gaussian noise to fixture features to create boundary cases | No |
| `all` | Runs all three modes and combines results into `combined.csv` | Optional |

Each mode writes a CSV to `training/data/`. The `all` mode produces a merged `combined.csv` used as training input.

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

The training script (`train.py`) reads a combined CSV and trains a dual-head feedforward network:

- **Shared layers**: Linear(18, 64) - ReLU - Dropout(0.2) - Linear(64, 32) - ReLU - Dropout(0.2)
- **Category head**: Linear(32, 5) -- predicts prose, code, structured, artifact, or skip
- **Sub-type head**: Linear(32, N) -- predicts a finer label (e.g., python, csv, markdown)

Default hyperparameters:

| Parameter | Default |
|-----------|---------|
| Epochs | 100 (with early stopping) |
| Batch size | 64 |
| Learning rate | 0.001 (Adam) |
| Early stopping patience | 10 epochs |
| Validation split | 20% (stratified) |

```bash
make train

# Or with custom options:
cd training && uv run python train.py \
    --data data/combined.csv \
    --output models/ \
    --epochs 200 \
    --patience 15
```

## Validation

After training, validate the model against a labeled JSONL test set using the Rust CLI:

```bash
# Text output
classify validate --input test.jsonl

# JSON output
classify validate --input test.jsonl --json
```

You can generate labeled data with `classify label-corpus --with-features` and use that as validation input.

## Directory Structure

```text
training/
  generate.py          # Data generation script (4 modes)
  train.py             # PyTorch training + ONNX export
  pyproject.toml       # Python dependencies
  test_generate.py     # Tests for generate.py
  test_train.py        # Tests for train.py
  data/                # Generated CSV files (gitignored)
    fixtures.csv
    synthetic.csv
    perturbations.csv
    combined.csv
  models/              # Model artifacts (gitignored except config)
    model.onnx
    model_config.json
    metrics.json
```

## Output Files

| File | Contents |
|------|----------|
| `model.onnx` | Trained ONNX model (opset 17) with two outputs: `category_logits` and `sub_type_logits` |
| `model_config.json` | Feature names, Z-score normalization stats (mean/std), category map, and sub-type map |
| `metrics.json` | Training results: best validation loss, best epoch, total epochs, category accuracy, sub-type accuracy |

The Rust classifier loads `model.onnx` and `model_config.json` at runtime to normalize input features and decode predictions.
