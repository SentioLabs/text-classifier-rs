# text-classifier

Rust library + CLI that classifies English text fields for translation eligibility.

## Quick Start

```bash
# Build the CLI
cargo build --release

# Classify text from stdin
echo "This is a paragraph of prose." | ./target/release/classify

# Classify a JSONL file
./target/release/classify file input.jsonl --output classified.jsonl

# Filter for translation pipeline
./target/release/classify filter export.jsonl.gz \
  --translatable translatable.jsonl.gz \
  --skipped skipped.jsonl.gz \
  --text-fields bodytext,summarytext,title
```

## Architecture

Two-tier hybrid classification:
- **Tier 1** — Structural features (line length CV, character entropy, sentence punctuation rate, etc.)
- **Tier 2** — Optional fasttext model for ambiguous cases

See `docs/plans/2026-02-20-text-classifier-design.md` for full design.

## Building

```bash
# CLI only
cargo build --release

# With fasttext model support
cargo build --release --features model

# Python extension
uv venv && uv pip install maturin
source .venv/bin/activate
maturin develop --release

# Run tests
cargo test
```

## Python Usage

```python
from text_classifier import Classifier

clf = Classifier()  # or Classifier(model_path="model.bin")
result = clf.classify("Some text here")
print(f"{result.text_type} (confidence={result.confidence:.2f})")
```

## Categories

| Category | Description | Translatable |
|----------|-------------|:---:|
| `translatable` | Human-readable prose | Yes |
| `code` | Source code, scripts, markup | No |
| `tabular` | Tables, CSVs, spreadsheets | No |
| `pdf_dump` | OCR garbage, PDF artifacts | No |
| `skip` | Too short or ambiguous | No |
