# text-classifier-rs

[![CI](https://github.com/SentioLabs/text-classifier-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SentioLabs/text-classifier-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Classify text fields by structural type. Text is categorized as **prose**, **code**, **tabular**, **pdf_dump**, or **skip** — useful for filtering large JSONL exports before downstream processing (translation, indexing, analysis, etc.).

Available as a **Rust CLI**, a **Rust library crate**, and a **pip-installable Python package**.

## How It Works

Two-tier hybrid classification:

1. **Tier 1 (structural)** — 10 statistical features (line-length CV, character entropy, symbol ratio, etc.) fed into a rule-based classifier. Handles ~90% of inputs with high confidence.
2. **Tier 2 (model)** — Optional fasttext model catches ambiguous cases where Tier 1 confidence falls below 0.7.

| Category | Description |
|----------|-------------|
| `prose` | Human-readable natural language |
| `code` | Source code, scripts, config, markup |
| `tabular` | Tables, CSVs, TSV, spreadsheet data |
| `pdf_dump` | OCR garbage, PDF extraction artifacts |
| `skip` | Too short or ambiguous to classify |

## Installation

### CLI (from source)

```bash
cargo install --path .
# or
make install
```

### Python

```bash
# uv (recommended)
uv add git+https://github.com/SentioLabs/text-classifier-rs --tag v0.2.0

# pip
pip install git+https://github.com/SentioLabs/text-classifier-rs@v0.2.0

# or build from source:
make python-setup && make python-build
```

> **Note:** Installing from git requires a Rust toolchain — maturin compiles the extension locally. Pin to a release tag for reproducible builds.

### As a Rust dependency

```toml
[dependencies]
text-classifier = { git = "https://github.com/SentioLabs/text-classifier-rs" }
```

## Usage

### CLI

```bash
# Classify text from stdin
echo "This is a paragraph of English prose." | classify

# Classify a JSONL file (supports .gz)
classify file input.jsonl -o classified.jsonl

# Filter by category
classify filter export.jsonl.gz \
  --prose prose.jsonl \
  --skipped skipped.jsonl \
  --text-fields bodytext,summarytext,title

# Dump raw feature vectors as CSV (useful for analysis)
classify features input.jsonl -o features.csv

# Generate training labels from Tier 1
classify label-corpus input.jsonl -o labeled.jsonl
```

### Python

```python
from text_classifier import Classifier

clf = Classifier()
result = clf.classify("Some text to classify")
print(f"{result.text_type} (confidence={result.confidence:.2f})")

# With a fasttext model for Tier 2
clf = Classifier(model_path="model.bin")
```

### Rust

```rust
use text_classifier::{Classifier, TextType};

let clf = Classifier::new();
let result = clf.classify("Hello world, this is a test paragraph.");

if result.text_type == TextType::Prose {
    println!("Human-readable prose detected");
}

// Batch classification (parallel via rayon)
let results = clf.classify_batch(&["text one", "text two"]);
```

## Development

```bash
make help          # show all available commands
make build         # debug build
make test          # run all tests
make lint          # format check + clippy
make check         # lint + test (CI equivalent)
make review        # full review with pass/fail summary
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor guide.

## Architecture

```
text → Classifier::classify()
       ├─ short-circuit if < 5 words → Skip
       ├─ extract_features() → 10 f32 structural signals
       ├─ classify_tier1() → rule-based, priority-ordered
       │   confidence ≥ 0.7? → return
       └─ ModelClassifier::classify() → Tier 2 (optional fasttext)
```

### Structural Features

| Feature | What it measures | Key discriminator |
|---------|-----------------|-------------------|
| `line_length_cv` | Coefficient of variation of line lengths | Prose (high) vs tables (low) |
| `char_entropy` | Shannon entropy of character distribution | English (~4.0) vs garbled (>5.0) |
| `leading_whitespace_ratio` | Lines starting with >2 spaces | Code indentation |
| `tab_density` | Tab chars as fraction of total | TSV / spreadsheet data |
| `sentence_punctuation_rate` | Sentence-ending punct per word | Prose (~0.04-0.08) vs code (~0) |
| `paragraph_break_rate` | Double-newline frequency | Prose paragraph structure |
| `alpha_ratio` | Alphanumeric + space fraction | Prose (>0.75) vs code (lower) |
| `line_uniqueness` | Unique lines / total lines | Data dumps have many repeats |
| `short_line_ratio` | Lines with 1-14 chars | OCR / PDF dump signal |
| `symbol_ratio` | Non-alpha, excluding common punct | Code (>0.15) vs prose (<0.05) |

## License

[MIT](LICENSE)
