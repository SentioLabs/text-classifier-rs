# text-classifier-rs

[![CI](https://github.com/SentioLabs/text-classifier-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/SentioLabs/text-classifier-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Classify text fields by structural type. Text is categorized as **prose**, **code**, or **structured** — useful for filtering large JSONL exports before downstream processing (translation, indexing, analysis, etc.).

Available as a **Rust CLI**, a **Rust library crate**, and a **pip-installable Python package**.

## How It Works

Two-tier hybrid classification with an embedded ONNX model:

1. **Model-first** — When compiled with `--features onnx-model`, a 41-feature neural network classifies all inputs. The model outputs a probability distribution over 31 sub-types (e.g. `python`, `markdown`, `csv`), marginalized into 3 categories.
2. **Tier 1 fallback** — Without the model, rule-based structural short-circuits handle high-confidence cases (pure prose, tabular data, markup).

| Category | Description | Example sub-types |
|----------|-------------|-------------------|
| `prose` | Human-readable natural language | plain, markdown, rst, latex |
| `code` | Source code, scripts, config, markup | python, rust, shell, html, sql |
| `structured` | Tables, data formats, config files | csv, json, yaml, log_lines, key_value |

The model also produces a **sub-type probability distribution** — useful as a ranking tensor in search systems like Vespa.

## Installation

### CLI (from source)

```bash
cargo install --path .
# or with the embedded ONNX model:
cargo install --path . --features onnx-model
```

### Python

```bash
# uv (recommended)
uv add git+https://github.com/SentioLabs/text-classifier-rs --tag v0.3.0

# pip
pip install git+https://github.com/SentioLabs/text-classifier-rs@v0.3.0

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

# Filter sub-type scores to significant entries only
classify --min-score 0.01 file input.jsonl -o classified.jsonl

# Filter by category
classify filter export.jsonl.gz \
  --prose prose.jsonl \
  --skipped non_prose.jsonl \
  --text-fields bodytext,summarytext,title

# Dump raw feature vectors as CSV (all 41 features)
classify features input.jsonl -o features.csv

# Generate training labels from Tier 1
classify label-corpus input.jsonl -o labeled.jsonl
```

Output from `classify file` includes the full classification and sub-type scores:

```json
{
  "_classification": {
    "category": "code",
    "sub_type": "python",
    "confidence": 0.97,
    "tier": "model",
    "sub_type_scores": {
      "code": [{"sub_type": "python", "score": 0.89}, {"sub_type": "shell", "score": 0.05}],
      "prose": [{"sub_type": "plain", "score": 0.03}],
      "structured": [{"sub_type": "ini", "score": 0.01}]
    }
  }
}
```

Use `--min-score 0.01` to drop noise entries (31 sub-types → 2-5 significant ones).

### Python

```python
from text_classifier import Classifier

clf = Classifier()
result = clf.classify("Some text to classify")
print(f"{result.category} (confidence={result.confidence:.2f})")

# With score filtering for production use
clf = Classifier(min_score=0.01)
result = clf.classify("import sys; print(sys.argv)")
print(result.sub_type_scores)  # JSON string with significant sub-types only
```

### Rust

```rust
use text_classifier::{Classifier, TextCategory};

// Model-first classification (requires --features onnx-model)
let clf = Classifier::with_embedded_model()
    .min_score(0.01);  // only keep scores >= 1%

let result = clf.classify("Hello world, this is a test paragraph.");
assert_eq!(result.category, TextCategory::Prose);

// Sub-type probability distribution available on result.sub_type_scores
// Detection head scores available on result.detections

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
       ├─ empty? → prose, confidence 0.0
       ├─ too short (< 5 words)? → prose, confidence 0.5
       ├─ extract_features() → 41 structural signals
       ├─ ONNX model loaded?
       │   └─ yes → model inference → category + sub-type + scores
       └─ no → classify_tier1() → rule-based fallback
```

### Structural Features (41)

The feature vector includes signals for line structure, character composition, syntax patterns, and content type indicators. Key discriminators:

| Feature | What it measures | Key discriminator |
|---------|-----------------|-------------------|
| `line_length_cv` | Coefficient of variation of line lengths | Prose (high) vs tables (low) |
| `char_entropy` | Shannon entropy of character distribution | Natural language (~4.0) vs noise (>5.0) |
| `symbol_ratio` | Non-alpha, excluding common punct | Code (>0.15) vs prose (<0.05) |
| `sentence_punctuation_rate` | Sentence-ending punct per word | Prose (~0.04-0.08) vs code (~0) |
| `json_lines_ratio` | Lines matching `{...}` pattern | JSONL detection |
| `shebang_present` | Text starts with `#!` | Shell/script detection |
| `dictionary_word_ratio` | Known English words fraction | Prose vs code/structured |
| `inline_markup_count` | Bold, italic, code, link patterns | Markdown detection |

See `src/features.rs` for the full list of 41 features.

## License

[MIT](LICENSE)
