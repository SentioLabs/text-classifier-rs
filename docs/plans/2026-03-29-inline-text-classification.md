# Inline Text Classification

Add the ability to classify text directly from the CLI as a positional argument and via a module-level convenience function in Python bindings.

## Motivation

Currently the CLI requires either piping text through stdin or passing a JSONL file via subcommands. For quick one-off classification, users want:

```bash
classify "Hello world, this is prose."
```

Similarly, Python users must instantiate a `Classifier` object before classifying. A module-level `classify()` function provides a more ergonomic API for simple use cases.

## CLI Design

### Root command changes (`src/main.rs`)

Add an optional positional argument and `--json` flag to the root `Cli` struct:

```rust
#[derive(Parser)]
#[command(name = "classify", about = "Classify text by structural type")]
struct Cli {
    /// Text to classify (if omitted, reads from stdin)
    text: Option<String>,

    /// Output as JSON instead of human-friendly format
    #[arg(long)]
    json: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}
```

### Dispatch priority

1. If a **subcommand** is present, route to subcommand handler (unchanged)
2. If **positional text** is present, classify it and output result
3. Otherwise, read from stdin (existing behavior, now also respects `--json`)

### Output formats

**Human-friendly** (default):
```
prose (confidence: 0.95, tier: structural)
```

With sub_type present:
```
code/config (confidence: 0.85, tier: structural)
```

**JSON** (`--json` flag):
```json
{"category":"prose","sub_type":null,"confidence":0.95,"reason":"high alpha ratio...","tier":"structural"}
```

The `--json` flag also applies to stdin mode, giving users control over output format in both paths.

## Python Bindings Design

### Module-level convenience function (`src/python.rs`)

```rust
#[pyfunction]
fn classify(text: &str) -> PyClassification {
    let classifier = RustClassifier::new();
    classifier.classify(text).into()
}
```

Registered in the module alongside existing classes:
```rust
m.add_function(wrap_pyfunction!(classify, m)?)?;
```

### Usage

```python
# New convenience function
from text_classifier import classify
result = classify("some text")
print(result.category)  # "prose"

# Existing class-based API (unchanged)
from text_classifier import Classifier
c = Classifier()
result = c.classify("some text")
```

**Note:** The Python convenience function uses the full Classifier (Tier 1 + Tier 2), unlike the Rust `text_classifier::classify()` which is Tier 1 only. This is intentional: Python users expect the best available result by default.

## Files Changed

| File | Change |
|------|--------|
| `src/main.rs` | Add `text: Option<String>` and `json: bool` to `Cli` struct, update dispatch logic, add `classify_inline()` helper |
| `src/python.rs` | Add `#[pyfunction] fn classify()`, register in module |

## Test Plan

- CLI: `classify "prose text"` outputs human-friendly format
- CLI: `classify --json "prose text"` outputs valid JSON
- CLI: stdin path still works (`echo "text" | classify`)
- CLI: subcommands unaffected (`classify file ...` routes correctly)
- Python: `from text_classifier import classify; classify("text")` returns Classification
- Python: existing `Classifier().classify()` unchanged

## Scope

This is a small, additive change. No new types, no architectural changes. Estimated 2 files modified, ~40 lines of new code.
