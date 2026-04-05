# Synthetic Eval Data Generation Plan
## Text Classifier: 50k Sample Dataset via OpenRouter

> **Priority: Quality and diversity over cost.** The goal is a dataset that tests the classifier
> against the full realistic distribution of each sub-type — not the cheapest samples that
> technically match the label. Use frontier models as primary contributors everywhere quality
> matters. Small/cheap models are reserved for Skip and Artifact sub-types where degraded or
> degenerate content is the point.

---

### Classifier Target

The classifier detects **5 categories** and **33 sub-types**:

| Category | Sub-types |
|---|---|
| Prose | Plain, Markdown, Rst, Latex |
| Code | Python, JavaScript, TypeScript, Rust, Go, Java, Sql, Shell, Css |
| Code > Config | Yaml, Toml, Ini, Dockerfile, Makefile |
| Code > Markup | Html, Xml, Sgml |
| Structured > Tabular | Csv, Tsv, PipeTable, FixedWidth |
| Structured > Data | Json, Jsonl, KeyValue, LogLines |
| Artifact | PdfDump, OcrGarbage, Boilerplate |
| Skip | TooShort, Empty, Ambiguous |
| Fallback | Unknown |

---

### Core Principle: Model Diversity

**Distribution collapse is the primary risk.** If you over-index on one model, the classifier
learns that model's stylistic fingerprints, not the actual category signal. Every model has
house-style tendencies — token choices, indentation preferences, comment styles, prose cadence.
Broad model coverage neutralizes all of these.

- Use **5–7 distinct models per sub-type** where possible
- Cap any single model at **~15% of samples for a given sub-type**
- Deliberately vary model families: open-weight, OpenAI, Anthropic, Mistral, Google, xAI, DeepSeek
- Use **reasoning mode** variants (DeepSeek R1, GPT-5 with thinking) as distinct generation modes —
  they produce structurally different text even for the same content

---

### Models to Use (OpenRouter)

#### Primary Frontier Models — Target ~65% of the dataset

These should be the workhorses. They produce the highest-quality, most realistic samples and
span the major closed-source style families. Do not treat them as "expensive extras."

| Model ID | Strengths |
|---|---|
| `anthropic/claude-sonnet-4.6` | Code (all languages), Markdown, LaTeX, structured data — precise format compliance, distinct Anthropic voice |
| `openai/gpt-5` | Prose, instruction-dense code, complex structured formats — strong OpenAI house style |
| `openai/gpt-5.4` | Code and long-context generation — unified Codex+GPT line, excellent on Go/Rust/TypeScript |
| `qwen/qwen3-235b-a22b` | Code, Config, Markup, JSON, non-Western prose variation — best model for multilingual diversity |
| `deepseek/deepseek-chat-v3-0324` | Code, SQL, JSON/JSONL, LogLines — strongest model for structured/data sub-types |
| `mistralai/mistral-large-2411` | Prose (all), Markdown, RST, LaTeX — distinct European style family, meaningfully different from US frontier |
| `meta-llama/llama-3.3-70b-instruct` | Prose, Shell, general variety — open-weight generation style is genuinely distinct from closed frontier |

#### Secondary Models — Target ~25% of the dataset

Each adds a distinct stylistic fingerprint or specialization. Spread these across sub-types
to widen the distribution.

| Model ID | Strengths |
|---|---|
| `x-ai/grok-3-beta` | Prose and Markdown — xAI style is noticeably different from OpenAI/Anthropic, adds real variance |
| `deepseek/deepseek-r1` | Reasoning-mode prose — generates structurally distinct text vs non-reasoning models; use for Plain/Markdown |
| `openai/gpt-5` (thinking mode) | Prefix prompts with `"Think carefully about this."` — produces a distinct generation mode from standard completions |
| `google/gemini-2.0-flash-001` | Tabular formats, HTML, XML — Google house style is distinct; strong on structured content |
| `cohere/command-r-plus-08-2024` | LogLines, KeyValue, structured data — enterprise RAG-style output patterns |
| `mistralai/codestral-2501` | All Code sub-types — dedicated code model, generation patterns differ from general-purpose frontier |
| `google/gemma-3-27b-it` | Prose variation — Google open-weight, stylistically distinct from Gemini |
| `qwen/qwen3-30b-a3b` | Code, Config, Structured — MoE model with different generation characteristics than dense Qwen3 235B |

#### Edge Case Models — Use only for Skip and Artifact sub-types (~10%)

For `TooShort`, `Empty`, `OcrGarbage`, `PdfDump` — the *point* is degraded or degenerate
content. Frontier models over-refine these and resist producing convincing garbage.

| Model ID | Use For |
|---|---|
| `meta-llama/llama-3.1-8b-instruct` | TooShort, Ambiguous edge cases |
| `microsoft/phi-4` | Short/degenerate samples |
| `openai/gpt-5.4-nano` | Bulk Skip sub-types at speed |

---

### Sample Allocation (50k total, ~1,515 per sub-type at balanced distribution)

```
Prose (Plain, Markdown, RST, LaTeX)                  ~8,000 samples
  Primary: Sonnet 4.6, GPT-5, Mistral Large, Llama 3.3
  Secondary: Grok-3, DeepSeek R1 (reasoning mode), Gemma 3
  Notes:
    - LaTeX → Sonnet 4.6 + Mistral Large + Qwen3 235B (best LaTeX compliance)
    - Use GPT-5 thinking mode for a slice of Plain prose
    - RST → Mistral Large + Llama 3.3 (strong on technical prose formats)

Code (9 languages + 5 config + 3 markup)             ~20,000 samples
  Primary: Sonnet 4.6, GPT-5.4, Deepseek V3, Qwen3 235B
  Secondary: Codestral, Llama 3.3, Grok-3
  Notes:
    - Go, Rust, TypeScript → GPT-5.4 + Sonnet 4.6 (best quality for these languages)
    - SQL → Deepseek V3 primary
    - Python, JavaScript → all primary models, widest spread
    - HTML/XML/SGML → Qwen3 235B + Gemini Flash + GPT-5.4
    - Dockerfile, Makefile → Sonnet 4.6 + Llama 3.3 + Qwen3

Structured (Tabular + Data, 8 sub-types)              ~12,000 samples
  Primary: Deepseek V3, Qwen3 235B, Sonnet 4.6, GPT-5
  Secondary: Gemini Flash, Command-R+, Codestral
  Notes:
    - JSON/JSONL → Deepseek V3 + Sonnet 4.6 + GPT-5 (all excellent)
    - CSV/TSV → Gemini Flash + Qwen3 + GPT-5.4 (good at clean tabular output)
    - LogLines → Command-R+ + Llama 3.3 + Deepseek V3
    - KeyValue → Qwen3 + Sonnet 4.6 + GPT-5

Artifact (PdfDump, OcrGarbage, Boilerplate)           ~5,000 samples
  Primary: Llama 3.3, Gemini Flash + adversarial prompting
  Secondary: Phi-4 for degenerate samples
  Note: See adversarial prompting section — also supplement with
        programmatically degraded real text for OcrGarbage

Skip (TooShort, Empty, Ambiguous)                     ~5,000 samples
  Models: Llama 8B, Phi-4, GPT-5.4-nano + truncation
```

---

### Generation Parameters

- **Temperature range: `0.7`–`1.5`** — use the wider range for Prose and Code to maximize
  stylistic diversity; narrower (`0.7`–`1.0`) for Structured where format correctness matters
- Run each sub-type prompt at **at least 3 temperature values** spread across the range
- **Vary prompt phrasing across batches** — use 5+ prompt templates per sub-type; prompt-template
  artifacts are a real failure mode
- **Vary content domain** per sub-type: e.g. Python samples should span data science, web, systems,
  scripting, tests, CLI tools — not just one domain
- **Vary length** intentionally: short (50–200 tokens), medium (200–600), long (600–1500) within
  each sub-type — include edge-length samples near sub-type boundaries

---

### Prompt Variation Strategy

For each sub-type, maintain a library of distinct prompt templates. Example for Python:

```
Template A: "Write a short Python script that [task]. No comments, no docstrings."
Template B: "Write a well-documented Python function for [task] with type hints and a docstring."
Template C: "Write a Python class for [task] following PEP 8 conventions."
Template D: "Show a Python one-liner that [task]."
Template E: "Write a Python unit test for a function that [task]."
```

Rotate templates uniformly across models to prevent template-model correlations.

---

### Adversarial Prompting for Hard Sub-types

Frontier models resist generating garbage/degenerate content by default. Use these strategies:

**OcrGarbage / PdfDump:**
```
Simulate the raw text output of a PDF parser on a poorly scanned document
with approximately 40% OCR error rate. Include garbled characters, broken
word boundaries, misread symbols, and random newlines mid-word. Output only
the simulated OCR text, no explanation.
```

**Boilerplate:**
```
Generate a dense block of legal/cookie/GDPR boilerplate text as it would
appear extracted from a website footer or ToS page. Make it repetitive,
run-on, and devoid of meaningful structure.
```

**TooShort:**
```
Generate a text fragment that is between 3 and 8 tokens long and would be
completely unclassifiable in isolation. Output only the fragment.
```

**Empty / Ambiguous:**
```
Generate a text sample containing only whitespace, punctuation, or Unicode
symbols with no discernible language, code, or structured content.
```

> **OcrGarbage supplement:** Pure synthetic OCR garbage often lacks authentic noise
> characteristics. Supplement with real text that has been programmatically degraded:
> random character substitution (~15–40% of chars), word boundary splitting, ligature
> corruption, and Unicode replacement character injection. Apply to a mix of real prose,
> code, and structured data snippets.

---

### Validation Workflow

Before committing to the full 50k run:

1. Generate a **500-sample pilot batch** (~15 per sub-type), using the full model mix
2. Manually audit **50 samples** — at least one per sub-type, a few randoms
3. Run your classifier and inspect the **confusion matrix**
4. Red flags to look for:
   - Sub-types hitting 99%+ accuracy — samples may be too clean/obvious
   - Systematic misclassification along model lines — fingerprinting
   - Prompt templates leaking into generated content (e.g. "Here is a Python script:")
   - Length distribution skew — most samples clustering in one length bucket
5. Adjust prompt templates, model weights, and temperature ranges before scaling

---

### Diversity Checklist per Sub-type

Before sign-off on any sub-type's sample batch, verify:

- [ ] At least 5 distinct models contributed
- [ ] No single model exceeds 15% of samples
- [ ] At least 3 temperature values represented
- [ ] At least 4 distinct prompt templates used
- [ ] Short, medium, and long length samples all present
- [ ] Content domain is varied (not just one topic/application)
- [ ] At least one reasoning-mode model included (DeepSeek R1 or GPT-5 thinking) for Prose sub-types

---

### Output Format

Each generated sample should be stored with full provenance metadata:

```json
{
  "text": "<generated content>",
  "category": "Code",
  "sub_type": "Python",
  "model": "anthropic/claude-sonnet-4.6",
  "temperature": 1.2,
  "prompt_template": "template_b",
  "prompt_version": "v1",
  "content_domain": "web",
  "length_bucket": "medium",
  "reasoning_mode": false,
  "synthetic": true
}
```

Store as **JSONL**, one record per line, partitioned by category for easy loading.
The `prompt_template` and `content_domain` fields are critical for post-hoc diversity audits.
