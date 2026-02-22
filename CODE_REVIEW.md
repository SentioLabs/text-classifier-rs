# Code Review — text-classifier-rs

**Date:** 2026-02-21
**Scope:** Full review of all Rust source code (`src/`, `tests/`)
**Method:** 5 parallel review agents (CLAUDE.md compliance, bug scan, comment accuracy, architecture/design, correctness/edge cases), findings deduplicated and scored for confidence ≥ 80/100.
**Status:** All actionable findings resolved. See resolution notes below.

---

## Summary

13 issues found across 7 source files. 3 high-severity, 8 medium-severity, 2 low-severity.
**Resolved:** 11 fixed, 1 won't-fix (M4), 1 already fixed by H2 (M5).

The codebase is well-structured and idiomatic Rust. No panics, crashes, integer overflows, or memory safety issues were found. All confidence values are correctly bounded to `[0.0, 1.0]`. The issues fall into three themes:

1. **Classification blind spots** — single-line inputs, shallow configs, tab-indented code, and flat SQL hit edge cases in the rule system
2. **Unicode/CJK assumptions** — byte-vs-char confusion and whitespace-delimited word counting break on non-ASCII text
3. **API and documentation gaps** — dead features, divergent public APIs, and misleading comments

---

## High Severity

### H1. Single-line text misclassified as Tabular with confidence 1.0

**Files:** `src/features.rs:67-84`, `src/tier1.rs:48-68`

`compute_line_length_cv()` returns `0.0` for any single-line input (early return at `lines.len() < 2`). The `try_tabular` rule fires when `line_length_cv < 0.15`, which `0.0` trivially satisfies. Combined with any input lacking sentence punctuation and having low symbols, the confidence formula yields `0.7 + 0.3 * (1.0 - 0.0/0.15) = 1.0`.

This affects:
- Single-line SQL: `SELECT id, name, email FROM users WHERE active = 1` → Tabular (1.0)
- Single-line imports: `import os; import sys; from typing import Dict, List` → Tabular (1.0)
- Hard-wrapped prose with uniform line lengths (legal docs, formatted paragraphs)
- Bullet lists with similar-length items

**Root cause:** Zero CV from one data point is not evidence of uniform tabular structure — it is absence of evidence. The tabular rule needs a minimum line count guard (e.g., `lines.len() >= 3`).

**Resolution:** Added `line_count: usize` field to `FeatureVector`. `try_tabular` now requires `f.line_count >= 3`.

---

### H2. Shallow config files misclassified as PdfDump

**Files:** `src/tier1.rs:139-157`, `src/features.rs:191-201`

`compute_short_line_ratio()` counts lines with trimmed length in `1..=14` characters. Config files (YAML, TOML, INI) consist predominantly of short `key: value` pairs that fall within this range (e.g., `debug: true` = 11 chars, `port: 8080` = 10 chars). This drives `short_line_ratio` above 0.5 or 0.8, triggering `try_pdf_dump`.

```yaml
name: test
version: 1.0
config:
  debug: true
  host: localhost
  port: 8080
```

Result: `PdfDump` with confidence 0.96 and reason "short lines (ratio=0.89), garbled content".

The existing YAML test fixture (`tests/fixtures/code/yaml_config.txt`) avoids this because it is a deeply nested Kubernetes manifest with long indented lines. Simple shallow configs fail.

**Root cause:** Short lines alone are insufficient to distinguish OCR garbage from legitimate short-line formats. The rule needs an additional discriminator (e.g., checking for structural patterns like `key: value` or requiring low `alpha_ratio`).

**Resolution:** Added `alpha_ratio < 0.75` guard to the first `try_pdf_dump` branch. Config files (high alpha) are excluded; OCR garbage (low alpha) still detected. Comment rewritten to match.

---

### H3. `label-corpus` subcommand uses model for its own training labels

**Files:** `src/main.rs:371-413`

The subcommand is documented as "Generate labels from Tier 1 for model training" but calls `classifier.classify(text)` on line 391, where `classifier` is the `Classifier` struct initialized from the `--model` CLI argument. If the user supplies `--model path/to/model.bin`, ambiguous texts (Tier 1 confidence < 0.7) will be labeled by the model itself.

```bash
classify --model current_model.bin label-corpus --input corpus.jsonl --output labels.jsonl
```

This creates circular/contaminated training data: the model's own predictions become its training labels, reinforcing biases rather than correcting them.

**Fix:** `label-corpus` should use the free function `text_classifier::classify()` (Tier 1 only) or force `Classifier::new()` regardless of the `--model` flag.

**Resolution:** `label_corpus` now calls `text_classifier::classify()` instead of `classifier.classify()`.

---

## Medium Severity

### M1. CJK/Unicode text: byte-length denominator vs char-count numerator

**File:** `src/features.rs:29`

```rust
let total_chars = sample.len().max(1) as f32;  // .len() returns BYTES, not chars
```

This value is passed as denominator to `compute_alpha_ratio`, `compute_symbol_ratio`, and `compute_tab_density`, but their numerators count Unicode code points via `.chars().filter(...).count()`. For multi-byte text:

| Text | Chars | Bytes | alpha_ratio (actual) | alpha_ratio (correct) |
|------|-------|-------|---------------------|-----------------------|
| 100 CJK chars | 100 | 300 | 0.33 | 1.0 |
| 100 Arabic chars | 100 | 200 | 0.50 | 1.0 |

CJK prose gets `alpha_ratio ≈ 0.33`, failing both `try_prose` (needs > 0.70) and `fallback_classification` (needs > 0.55), resulting in `Skip`.

**Fix:** `let total_chars = sample.chars().count().max(1) as f32;`

**Resolution:** Fixed.

---

### M2. CJK text always classified as Skip due to `is_too_short`

**Files:** `src/tier1.rs:44-46`, `src/lib.rs:17,57`

```rust
pub fn is_too_short(text: &str) -> bool {
    text.split_whitespace().count() < 5
}
```

Chinese, Japanese, and Thai text does not use ASCII spaces as word separators. A 10,000-character Chinese essay is a single whitespace-delimited token. `split_whitespace().count()` returns 1, which is less than 5, so the text is unconditionally classified as `Skip` with confidence 1.0.

**Fix:** Add a complementary character-count check: `text.split_whitespace().count() < 5 && text.chars().count() < 20` (or similar).

**Resolution:** Fixed. `is_too_short` now requires both conditions (`< 5` words AND `< 20` chars).

---

### M3. Tab-indented code invisible to leading_whitespace_ratio

**File:** `src/features.rs:115-125`

```rust
let leading: usize = line.chars().take_while(|c| c.is_whitespace()).count();
leading > 2
```

A tab character is a single character, so one-tab indentation has `leading = 1`, which fails the `> 2` threshold. This means:
- 1-tab lines: `leading = 1` — NOT counted
- 2-tab lines: `leading = 2` — NOT counted
- 3+ spaces: counted

Affected languages: Go (tabs), Makefiles (tabs), many shell scripts, 2-space JavaScript/Ruby/Elixir. All four code detection paths in `try_code` that require `leading_whitespace_ratio > 0.10` or higher fail for tab-indented files. Only Path D (dense symbols, `symbol_ratio > 0.12`) remains available.

**Resolution:** Fixed. Tabs count as 4 columns in `compute_leading_whitespace_ratio`.

---

### M4. Flat/unindented SQL and scripts fall through to Skip

**File:** `src/tier1.rs:78-80`

The "flat code" path requires `leading_whitespace_ratio > 0.10`. Truly flat SQL — all keywords left-aligned — has `leading_whitespace_ratio = 0.0`. The "dense" path requires `symbol_ratio > 0.12`, but moderate SQL (parentheses, underscores, equals) has `symbol_ratio ≈ 0.04-0.08`.

```sql
SELECT id, name, email FROM users WHERE status = 'active' AND age > 18
GROUP BY id ORDER BY name ASC LIMIT 100
```

No code path fires. Falls to `fallback_classification()` → `Skip` at confidence 0.5.

**Resolution:** Won't fix. SQL's low symbol density (most SQL "symbols" like `,`, `'`, `=` are excluded from `symbol_ratio`) makes it genuinely ambiguous for a structural classifier. The 0.5-confidence fallback correctly routes to Tier 2.

---

### M5. `try_pdf_dump` comment claims disjunction that doesn't exist

**File:** `src/tier1.rs:140-143`

Comment says: "moderate short-line ratio with either high symbols **or** low line uniqueness."

Code checks: `f.short_line_ratio > 0.5 && f.line_uniqueness < 0.5`

The "high symbols" branch does not exist. A reader relying on the comment would believe `symbol_ratio` plays a role in PDF dump detection; it does not.

**Resolution:** Already fixed in H2 — comment was rewritten to match the actual code.

---

### M6. `char_entropy` and `paragraph_break_rate` computed but never used

**Files:** `src/features.rs:88-111` (char_entropy), `src/features.rs:159-162` (paragraph_break_rate)

Both features are computed on every `extract_features()` call, serialized to CSV by the `features` subcommand, and exposed via Python bindings — but neither is referenced in any classification rule in `tier1.rs` or `tier2.rs`.

`char_entropy` uses a `HashMap<char, u32>` allocation and full-text iteration — it is the most expensive single computation in `extract_features()` and contributes nothing to classification outcomes. The doc comments describe discriminative ranges (e.g., "English prose ~4.0-4.5, code ~3.0-3.5") that create a false expectation the classifier uses them.

Additionally, `compute_line_uniqueness()` accepts a `_n_lines: usize` parameter that is explicitly unused (underscore prefix), indicating an API design inconsistency.

**Resolution:** Fixed. Doc comments updated to clarify these features are not used in Tier 1 rules. Unused `_n_lines` parameter removed from `compute_line_uniqueness`.

---

### M7. `classify()` free function vs `Classifier::classify()` behavioral divergence

**Files:** `src/lib.rs:16-28`, `src/lib.rs:56-76`

The free function `classify(text)` always returns the raw Tier 1 result, including low-confidence (0.5) fallback results. It never falls through to Tier 2. `Classifier::classify()` checks `tier1_result.confidence >= 0.7` and invokes Tier 2 for anything below.

When a model is loaded, the divergence is material: `classify()` returns the Tier 1 fallback at confidence 0.5, while `Classifier::classify()` returns the model's prediction. A library consumer using the simpler `classify()` API with `--features model` gets Tier 1 only, with no indication that model-augmented classification requires `Classifier`.

The `classify_tier1` doc comment (tier1.rs:7-9) also says callers "should fall through to Tier 2" on low confidence, but `classify()` does not honor this contract.

**Resolution:** Fixed. `classify_tier1` doc comment reworded to describe the low-confidence signal without implying a caller obligation.

---

### M8. Hardcoded `0.7` in lib.rs instead of referencing `tier1::MIN_CONFIDENCE`

**Files:** `src/lib.rs:70`, `src/tier1.rs:4`

CLAUDE.md states: "Confidence threshold of 0.7 gates Tier 1 acceptance (`tier1::MIN_CONFIDENCE`) and the Tier 1→2 handoff (`lib.rs:70`)."

In practice, `MIN_CONFIDENCE` is private (`const`, not `pub`) in `tier1.rs`, so `lib.rs` hardcodes the literal `0.7` independently. If one is ever changed without changing the other, the Tier 1 acceptance threshold and the Tier 1→2 handoff threshold will silently diverge.

**Fix:** Make the constant `pub(crate)` and reference it in `lib.rs`:
```rust
// tier1.rs
pub(crate) const MIN_CONFIDENCE: f32 = 0.7;

// lib.rs
if tier1_result.confidence >= tier1::MIN_CONFIDENCE {
```

**Resolution:** Fixed.

---

## Low Severity

### L1. Missing text field causes silent pass-through

**File:** `src/main.rs:207-216` (classify_file), `src/main.rs:390-407` (label_corpus)

When `text_field` does not exist in a document, `classify_file` writes the document to output without any `_classification` field (silent pass-through). `label_corpus` silently drops the document entirely. No counter, warning, or log message is emitted. On a large JSONL file with a mistyped field name, every document silently passes through unclassified.

**Resolution:** Fixed. Added `missing_field` counter and warning to both `classify_file` and `label_corpus`.

---

### L2. Filter auto-prose for short fields not counted in summary statistics

**File:** `src/main.rs:261-268, 278-280`

Fields shorter than 50 bytes are auto-classified as prose and the loop `continue`s before reaching `category_counts`. The filter summary's "Categories:" breakdown undercounts prose. Additionally, the 50-byte threshold uses `.len()` (bytes) not character count, creating inconsistency with `is_too_short` (word count) for CJK text.

**Resolution:** Fixed. Auto-prose fields now increment `category_counts`. Comment updated to say "bytes" instead of "chars".

---

## Informational Notes

These were found but scored below the confidence threshold for inclusion as issues:

- **Undocumented stub subcommands** (`train`, `validate`): Live CLI subcommands that always `exit(1)`. CLAUDE.md's subcommand list is incomplete.
- **`alpha_ratio` comment says "> 0.75"** but the actual prose threshold is `> 0.70` (5pp discrepancy).
- **`symbol_ratio` doc omits `\r`** from the excluded punctuation list (code excludes 14 chars, comment lists 13).
- **CRLF paragraph breaks**: `paragraph_break_rate` uses `"\n\n"` which never matches `"\r\n\r\n"`. Latent only — feature is unused in rules.
- **Case-sensitive `.gz` detection**: `.GZ` files treated as plaintext, producing a misleading JSON parse error.
- **CLAUDE.md code sub-path order**: Documents "indented, flat, config-like, dense" but actual priority is indented → config_like → dense → flat.
- **`min_confidence` parameter inoperative without model**: Only produces different behavior when Tier 2 model returns intermediate confidence values.
