# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Rust library + CLI that classifies text fields by structural type. Text is categorized as `prose`, `code`, `tabular`, `pdf_dump`, or `skip`. Useful for filtering large JSONL exports before downstream processing.

## Build & Test Commands

The project has a self-documenting `Makefile` — run `make help` for all targets.

```bash
make check                         # lint + test (preferred for pre-commit)
make review                        # full status report (fmt, clippy, tests)
make lint                          # fmt-check + clippy only
```

### Raw Cargo Commands

```bash
cargo build                        # dev build (Tier 1 only)
cargo build --release              # release build
cargo build --features model       # with fasttext Tier 2 support
cargo test                         # all tests
cargo test test_name               # single test by name
cargo test --test test_tier1       # single test file
cargo fmt --check                  # check formatting
cargo clippy --all-targets -- -D warnings  # lint
```

### Python Extension

```bash
uv venv && uv pip install maturin
source .venv/bin/activate
maturin develop --release          # builds with --features python
```

## Architecture

**Two-tier hybrid classification** — Tier 1 decides most cases; Tier 2 handles ambiguous ones.

### Classification Flow

`Classifier::classify(text)` → short-circuit if <5 words → `extract_features()` → `classify_tier1()` → if confidence ≥ 0.7, return → else `ModelClassifier::classify()` (Tier 2)

### Module Roles

- **`types.rs`** — Core types: `TextType` (5 categories), `Classification`, `FeatureVector` (10 f32 fields), `Tier`
- **`features.rs`** — Extracts 10 structural features from text (samples first 10k chars). Each feature has a specific discriminative purpose documented in the function comments (e.g., `line_length_cv` distinguishes prose from tables, `symbol_ratio` catches code)
- **`tier1.rs`** — Rule-based classifier using feature thresholds. Priority-ordered: tabular → code → pdf_dump → prose → fallback. Code detection has 4 sub-paths (indented, flat, config-like, dense/minified). Returns low-confidence fallback (0.5) when ambiguous to trigger Tier 2
- **`tier2.rs`** — Optional fasttext model wrapper behind `--features model` cargo feature flag. Falls back to simple heuristic when no model loaded
- **`python.rs`** — PyO3 bindings behind `--features python`. Exposes `Classifier` and `Classification` classes
- **`main.rs`** — CLI with subcommands: `file` (classify JSONL), `filter` (split prose/skipped), `features` (dump CSV), `label-corpus` (generate training labels). Supports `.gz` input via flate2

### Key Design Decisions

- Confidence threshold of **0.7** gates Tier 1 acceptance (`tier1::MIN_CONFIDENCE`) and the Tier 1→2 handoff (`lib.rs:70`)
- Feature extraction samples first **10k characters** for performance on large documents
- The `filter` subcommand auto-passes short fields (<50 chars) as prose and lets uncertain classifications (confidence < `min_confidence`) through to the prose output
- `symbol_ratio` deliberately excludes common punctuation (`. , ; : ! ? - ' "`) to avoid false positives on prose

## Test Fixtures

Tests use text files in `tests/fixtures/` organized by category (`prose/`, `code/`, `tabular/`, `pdf_dump/`). Add new fixtures there when testing new edge cases. The `read_fixture()` helper is defined locally in each test file.

## Rust Edition

Uses **edition 2024** — `floor_char_boundary` is stable without feature flags.

## CLI Binary

The binary is named `classify`. After `cargo build --release` it's at `target/release/classify`. Subcommands: `file`, `filter`, `features`, `label-corpus`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR to `main`: fmt check → clippy → tests.

## Session Completion

See **[AGENTS.md § Landing the Plane](AGENTS.md#landing-the-plane-session-completion)** for the mandatory session completion workflow.
