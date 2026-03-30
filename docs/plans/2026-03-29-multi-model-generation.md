# Design: Multi-Model Synthetic Data Generation

**Date**: 2026-03-29
**Branch**: `feat/hierarchical-taxonomy`
**Parent Epic**: `textclassi-0hba.01577p` (99% Classification Accuracy)
**Supersedes**: T1 (generate_eval.py), T2, T4 golden-train mode, T7, T9

---

## Context

The 99% accuracy epic established the classifier infrastructure:
- Model-primary Tier 1 architecture (T5)
- Deeper model 18->128->64->32 (T6)
- FAISS two-layer dedup pipeline (T3)
- Eval JSONL schema validation (T0)
- Makefile targets (T8)

What remains is generating high-quality synthetic data and training. The original plan used GPT-5.4 alone for eval and Claude Sonnet alone for training. This creates **distribution collapse** -- the classifier learns model-specific fingerprints rather than structural category signals.

This design replaces the single-model generation with a multi-model pipeline using OpenRouter to access 15+ models across all major families.

---

## Core Principle: Model Diversity

Distribution collapse is the primary risk. Every LLM has house-style tendencies -- token choices, indentation preferences, comment styles, prose cadence. If you over-index on one model, the classifier learns those fingerprints, not the actual category signal.

Rules:
- **5-7 distinct models per sub-type** minimum
- **15% cap** per model per sub-type
- Deliberately vary model families: OpenAI, Anthropic, Google, Mistral, xAI, DeepSeek, Meta, Cohere
- Use reasoning-mode variants (DeepSeek R1, GPT-5 thinking) as distinct generation modes

---

## OpenRouter Generation Script

**New file**: `training/generate_openrouter.py`

**Replaces**: `training/generate_eval.py` and `training/generate.py` golden-train mode.

### API Design

OpenRouter provides an OpenAI-compatible API. Single API key, single endpoint, model specified per request:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.6",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.2,
)
```

### Model Roster

#### Primary Models (~65% of dataset)

| Model ID | Strengths |
|---|---|
| `anthropic/claude-sonnet-4.6` | Code (all), Markdown, LaTeX, structured data |
| `openai/gpt-5` | Prose, instruction-dense code, complex structured |
| `openai/gpt-5.4` | Code, long-context, Go/Rust/TypeScript |
| `qwen/qwen3-235b-a22b` | Code, Config, Markup, JSON, multilingual |
| `deepseek/deepseek-chat-v3-0324` | Code, SQL, JSON/JSONL, LogLines |
| `mistralai/mistral-large-2411` | Prose (all), Markdown, RST, LaTeX |
| `meta-llama/llama-3.3-70b-instruct` | Prose, Shell, general variety |

#### Secondary Models (~25% of dataset)

| Model ID | Strengths |
|---|---|
| `x-ai/grok-3-beta` | Prose, Markdown |
| `deepseek/deepseek-r1` | Reasoning-mode prose |
| `google/gemini-2.0-flash-001` | Tabular, HTML, XML |
| `cohere/command-r-plus-08-2024` | LogLines, KeyValue, structured |
| `mistralai/codestral-2501` | All Code sub-types |
| `google/gemma-3-27b-it` | Prose variation |
| `qwen/qwen3-30b-a3b` | Code, Config, Structured |

#### Edge Case Models (~10%, Skip/Artifact only)

| Model ID | Use For |
|---|---|
| `meta-llama/llama-3.1-8b-instruct` | TooShort, Ambiguous |
| `microsoft/phi-4` | Short/degenerate samples |
| `openai/gpt-5.4-nano` | Bulk Skip sub-types |

### Per Sub-Type Configuration

Each of the 33 sub-types has a config specifying:
- Which models to use and their relative weights
- 5+ prompt templates with content domain rotation
- Temperature range (0.7-1.0 for structured, 0.8-1.5 for prose/code)
- Length buckets: short (3-10 lines), medium (20-50 lines), long (100+ lines)
- Adversarial prompting strategy (Artifact/Skip sub-types)

### Sample Allocation (60K total)

```
Prose (Plain, Markdown, RST, LaTeX)                  ~10,000 samples
Code (9 languages + 5 config + 3 markup)             ~24,000 samples
Structured (Tabular + Data, 8 sub-types)             ~14,000 samples
Artifact (PdfDump, OcrGarbage, Boilerplate)          ~6,000 samples
Skip (TooShort, Empty, Ambiguous)                    ~6,000 samples
```

### Output Format (JSONL with inline provenance)

```json
{
  "text": "<generated content>",
  "expected_category": "code",
  "sub_type": "python",
  "boundary_pair": null,
  "model": "anthropic/claude-sonnet-4.6",
  "temperature": 1.2,
  "prompt_template": "template_b",
  "content_domain": "web",
  "length_bucket": "medium",
  "reasoning_mode": false
}
```

Provenance fields are inline for atomic records, streaming, and filtering. The classifier's validate command ignores fields it doesn't recognize.

### CLI

```bash
# Pilot batch (500 samples, full model mix)
python training/generate_openrouter.py \
  --output data/pilot_samples.jsonl \
  --total-samples 500 --pilot

# Full generation
python training/generate_openrouter.py \
  --output data/raw_samples.jsonl \
  --total-samples 60000

# Dry run (prints generation plan)
python training/generate_openrouter.py \
  --output data/raw_samples.jsonl \
  --total-samples 60000 --dry-run
```

### Features

- **Model rotation**: weighted random selection per sub-type, 15% cap enforced
- **Temperature variation**: 3+ values per sub-type spread across the range
- **Prompt template rotation**: 5+ templates per sub-type
- **Domain x length matrix**: systematic coverage
- **Resume support**: reads existing output file, skips completed sub-type/count combos
- **Progress reporting**: count, model distribution, diversity stats every 500 samples
- **Adversarial prompting**: special templates for Artifact/Skip per the diversity plan

---

## Pilot Batch Validation

Before the full 60K run, generate a 500-sample pilot with the full model mix.

**New file**: `training/validate_pilot.py`

```bash
python training/validate_pilot.py --input data/pilot_samples.jsonl
```

### Reports

- Per sub-type: sample count, model distribution, temperature spread, template coverage
- Diversity checklist pass/fail per sub-type (7 items from diversity plan):
  1. At least 5 distinct models contributed
  2. No single model exceeds 15% of samples
  3. At least 3 temperature values represented
  4. At least 4 distinct prompt templates used
  5. Short, medium, and long length samples all present
  6. Content domain is varied
  7. At least one reasoning-mode model for Prose sub-types
- Runs classifier against pilot, prints confusion matrix
- Flags "too easy" (>99% accuracy) and "too hard" (<50%) sub-types
- Prints 50 random stratified samples to terminal for quick manual audit

### Pilot Workflow

1. `task train:pilot` -- generate + validate
2. Review diversity report and manual audit samples
3. Adjust prompt templates / model weights if needed
4. Repeat until satisfied
5. Run full generation

---

## Dataset Split

**New file**: `training/split_dataset.py`

After generating 60K raw samples, split into eval (10K) and training (50K).

```bash
python training/split_dataset.py \
  --input data/raw_samples.jsonl \
  --eval-output eval/clear.jsonl \
  --eval-boundary-output eval/boundary.jsonl \
  --train-output training/data/golden_raw.csv \
  --eval-per-category 1000 \
  --eval-per-pair 1000
```

### Split Strategy

- For each category: randomly sample 1K for eval clear set, stratified by model to maintain diversity
- For boundary pairs: sample 1K per pair (500 each direction) for eval boundary set
- Everything else goes to training CSV
- Deterministic split (seeded random) for reproducibility
- Post-split diversity verification: no single model >15% in any sub-type slice of either set

---

## Eval Schema Update

Update `training/eval_schema.py` (existing):
- `validate_sample()` continues requiring `text`, `expected_category`, `boundary_pair`
- Extra provenance fields accepted and ignored by validation
- New `validate_provenance(sample: dict) -> bool` checks provenance fields exist and are valid
- New `diversity_report(path: str)` prints per-sub-type model distribution, temperature spread, template coverage

---

## Taskfile Integration

**New file**: `training/Taskfile.yml`

All training pipeline commands move from Makefile to Taskfile with `train:` namespace.

```yaml
version: '3'

tasks:
  train:pilot:
    desc: Generate 500 pilot samples and validate diversity
    cmds:
      - python generate_openrouter.py --output data/pilot_samples.jsonl --total-samples 500 --pilot
      - python validate_pilot.py --input data/pilot_samples.jsonl

  train:generate:
    desc: Generate 60K samples via OpenRouter (full model mix)
    cmds:
      - python generate_openrouter.py --output data/raw_samples.jsonl --total-samples 60000

  train:split:
    desc: Split raw samples into 10K eval + 50K training
    cmds:
      - python split_dataset.py --input data/raw_samples.jsonl --eval-output ../eval/clear.jsonl --eval-boundary-output ../eval/boundary.jsonl --train-output data/golden_raw.csv

  train:dedup:
    desc: FAISS dedup on training split
    cmds:
      - python dedup.py --input data/golden_raw.csv --output data/golden_train.csv

  train:train:
    desc: Train model on deduped golden data
    cmds:
      - python train.py --input data/golden_train.csv --output output/

  train:update-model:
    desc: Copy trained model to src/ for embedding
    cmds:
      - cp output/model.onnx ../src/model.onnx
      - cp output/model_config.json ../src/model_config.json

  train:validate:
    desc: Validate against both golden eval sets
    cmds:
      - cd .. && cargo build --release --features onnx-model
      - cd .. && ./target/release/classify validate --input eval/clear.jsonl --json
      - cd .. && ./target/release/classify validate --input eval/boundary.jsonl --json

  train:pipeline:
    desc: Full pipeline -- generate -> split -> dedup -> train -> embed -> validate
    cmds:
      - task train:generate
      - task train:split
      - task train:dedup
      - task train:train
      - task train:update-model
      - task train:validate
```

**Makefile cleanup**: Remove all training-related targets (generate-data, generate-fixtures, generate-ambiguous, train, validate, test-model, test-model-ambiguous, update-model, build-onnx, train-pipeline, generate-golden-eval, generate-golden-train, dedup, golden-pipeline, validate-golden, validate-golden-clear, validate-golden-boundary). Keep only Rust build/test/lint targets.

---

## Prompt Variation Strategy

For each sub-type, maintain 5+ distinct prompt templates. Example for Python:

```
Template A: "Write a short Python script that {task}. No comments, no docstrings."
Template B: "Write a well-documented Python function for {task} with type hints and a docstring."
Template C: "Write a Python class for {task} following PEP 8 conventions."
Template D: "Show a Python one-liner that {task}."
Template E: "Write a Python unit test for a function that {task}."
```

Templates are defined per sub-type in the generation script. Rotated uniformly across models to prevent template-model correlations.

### Adversarial Prompting (Artifact/Skip)

**OcrGarbage / PdfDump**: "Simulate the raw text output of a PDF parser on a poorly scanned document with approximately 40% OCR error rate..."

**Boilerplate**: "Generate a dense block of legal/cookie/GDPR boilerplate text as it would appear extracted from a website footer..."

**TooShort**: "Generate a text fragment that is between 3 and 8 tokens long and would be completely unclassifiable in isolation."

**OcrGarbage supplement**: Programmatically degrade real text with random character substitution (~15-40% of chars), word boundary splitting, ligature corruption, Unicode replacement character injection.

---

## Implementation Phases

### Phase 1: OpenRouter Generation Script + Pilot
- Create `training/generate_openrouter.py` with full model roster and sub-type configs
- Create `training/validate_pilot.py` for diversity auditing
- Create `training/Taskfile.yml` with `train:` namespace
- Clean up Makefile (remove training targets)
- Update `training/eval_schema.py` for provenance fields
- Run pilot, validate diversity

### Phase 2: Full Generation + Split
- Generate 60K samples via OpenRouter
- Create `training/split_dataset.py`
- Split into 10K eval + 50K training
- Run FAISS dedup on training split
- Record baseline accuracy against eval sets

### Phase 3: Retrain + Validate
- Train upgraded model on 50K golden data
- Embed model, run test suite
- Validate against golden eval sets
- Compare to baseline

### Phase 4: Iterate to 99%
- Analyze misclassifications with --verbose
- Targeted fixes: more training data for failure patterns, prompt template adjustments
- Each iteration: regenerate problem areas -> dedup -> retrain -> measure
- Stop when combined accuracy >= 99%

### Phase Dependencies

```
Phase 1 -> Phase 2 -> Phase 3 -> Phase 4
```

Strictly sequential -- each phase depends on the output of the previous.

---

## Relationship to First Epic

### Completed work that stays (from textclassi-0hba.01577p)

| Task | What it built |
|------|---------------|
| T0 | `training/eval_schema.py` -- JSONL schema validation (gets minor update) |
| T3 | `training/dedup.py` -- FAISS two-layer dedup pipeline |
| T5 | Model-primary Tier 1 architecture (src/tier1.rs, src/lib.rs) |
| T6 | Model architecture upgrade 18->128->64->32 (training/train.py) |

### Superseded work

| Task | What it built | Replaced by |
|------|---------------|-------------|
| T1 | `training/generate_eval.py` (GPT-5.4 only) | `training/generate_openrouter.py` |
| T4 | golden-train mode in `generate.py` (Claude Sonnet only) | `training/generate_openrouter.py` |
| T8 | Makefile training targets | `training/Taskfile.yml` |

### Open tasks from first epic to close

- T2 (Generate Golden Eval Set) -- superseded by Phase 2
- T7 (Retrain on Golden Data) -- superseded by Phase 3
- T9 (Iterate to 99%) -- superseded by Phase 4

---

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API gateway | OpenRouter | Single key, single endpoint, access to 15+ model families |
| Generation approach | Single pipeline, split after | Same diversity in eval and training; simpler than separate runs |
| Provenance storage | Inline in JSONL | Atomic records, streaming-friendly, no sync bugs with sidecars |
| Model diversity target | 5-7 per sub-type, 15% cap | Prevents distribution collapse and model fingerprinting |
| Pilot validation | 500 samples before full run | Catches prompt template leaks, length skew, fingerprinting before API spend |
| Build tool | Taskfile (go-task) | Modern YAML-based, colon namespace (train:pilot), cleaner than Makefile for training pipeline |
| Taskfile location | training/Taskfile.yml | Co-located with training scripts, Makefile stays for Rust targets only |
