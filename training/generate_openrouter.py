#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["openai", "tqdm"]
# ///
"""Multi-model OpenRouter generation script for synthetic training data.

Uses the OpenRouter API (OpenAI-compatible) to generate diverse synthetic text
samples across 15+ LLM models with diversity controls including weighted model
selection, temperature variation, prompt rotation, and domain rotation.

Usage:
    python training/generate_openrouter.py --output /tmp/out.jsonl --dry-run
    python training/generate_openrouter.py --output /tmp/out.jsonl --pilot
    python training/generate_openrouter.py --output /tmp/out.jsonl --total-samples 60000
"""

import argparse
from collections import defaultdict
import concurrent.futures
import threading
import itertools
import json
import math
import os
import random
import sys
import time

from tqdm import tqdm

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

# Ensure the training directory is importable for eval_schema
sys.path.insert(0, os.path.dirname(__file__))

from eval_schema import validate_sample

# ---------------------------------------------------------------------------
# Model rosters
# ---------------------------------------------------------------------------

PRIMARY_MODELS: list[str] = [
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5",
    "openai/gpt-5.4",
    "qwen/qwen3-235b-a22b",
    "deepseek/deepseek-v3.2",
    "mistralai/mistral-large-2512",
    "meta-llama/llama-3.3-70b-instruct",
]

SECONDARY_MODELS: list[str] = [
    "x-ai/grok-4-fast",
    "deepseek/deepseek-r1-0528",
    "google/gemini-2.5-flash",
    "cohere/command-a",
    "mistralai/codestral-2508",
    "google/gemma-3-27b-it",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-coder",
    "meta-llama/llama-4-maverick",
]

EDGE_CASE_MODELS: list[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "microsoft/phi-4",
    "openai/gpt-5.4-nano",
]

ALL_MODELS: list[str] = PRIMARY_MODELS + SECONDARY_MODELS + EDGE_CASE_MODELS

# ---------------------------------------------------------------------------
# Domain seeds (50+)
# ---------------------------------------------------------------------------

DOMAIN_SEEDS: list[str] = [
    # Sciences & academia
    "astronomy", "biology", "chemistry", "climate science", "genetics",
    "marine biology", "mathematics", "meteorology", "neuroscience",
    "nuclear physics", "oceanography", "pharmacology", "physics",
    # Medicine & health
    "healthcare", "medical imaging", "clinical trials", "epidemiology",
    # Software & tech verticals
    "web development", "mobile apps", "microservices", "REST APIs",
    "database design", "frontend frameworks", "backend systems",
    "distributed systems", "cloud infrastructure", "serverless",
    "machine learning", "data engineering", "data pipelines",
    "search engines", "recommendation systems", "computer vision",
    # DevOps & infrastructure
    "devops", "CI/CD pipelines", "container orchestration",
    "monitoring and alerting", "DNS and networking", "load balancing",
    "infrastructure as code", "site reliability engineering",
    # Security
    "cybersecurity", "authentication systems", "encryption",
    "penetration testing", "compliance auditing",
    # Business & enterprise
    "finance", "accounting", "payroll systems", "inventory management",
    "CRM systems", "ERP integration", "invoicing", "HR management",
    "insurance", "real estate", "supply chain",
    # Consumer & lifestyle
    "e-commerce", "social media", "fitness tracking", "recipe management",
    "travel planning", "personal finance", "streaming media",
    "gaming", "music production", "photography",
    # Data formats & reporting
    "log analysis", "CSV report generation", "API response formatting",
    "database migrations", "ETL pipelines", "data warehousing",
    # Creative & content
    "journalism", "technical writing", "fiction writing",
    "screenwriting", "game design documentation", "API documentation",
    # Legal & compliance
    "GDPR compliance", "HIPAA regulations", "terms of service",
    "privacy policies", "audit logging", "legal contracts",
    # Education
    "education", "online courses", "exam systems", "grading platforms",
    # Industry & manufacturing
    "manufacturing", "robotics", "automotive", "aviation",
    "agriculture", "energy", "telecommunications",
    # Misc
    "cryptocurrency", "urban planning", "logistics",
    "environmental science", "fashion", "food science",
    "linguistics", "marketing", "philosophy", "political science",
    "psychology", "sociology", "sports", "veterinary medicine",
    "electronics", "nanotechnology", "military", "government",
]

# ---------------------------------------------------------------------------
# Length buckets: (min_lines, max_lines)
# ---------------------------------------------------------------------------

LENGTH_BUCKETS: dict[str, tuple[int, int]] = {
    "short": (3, 10),
    "medium": (20, 50),
    "long": (100, 200),
}

# ---------------------------------------------------------------------------
# Boundary pairs: (category_a, category_b)
# ---------------------------------------------------------------------------

BOUNDARY_PAIRS: list[tuple[str, str]] = [
    ("prose", "code"),
    ("prose", "structured"),
    ("prose", "artifact"),
    ("code", "structured"),
    ("code", "artifact"),
    ("structured", "artifact"),
]

# ---------------------------------------------------------------------------
# Helper: build weighted model lists for sub-types
# ---------------------------------------------------------------------------

def _weighted_models(
    primary: list[str],
    secondary: list[str] | None = None,
    primary_share: float = 0.70,
) -> list[tuple[str, float]]:
    """Build a weighted model list with <=15% cap per model.

    Primary models share primary_share of the total weight (default 70%).
    Secondary models share the remainder (default 30%).
    This allows cost optimization by putting cheaper models in primary.
    """
    models: list[tuple[str, float]] = []
    n_primary = len(primary)
    n_secondary = len(secondary) if secondary else 0

    per_primary = primary_share / n_primary if n_primary else 0
    per_secondary = (1.0 - primary_share) / n_secondary if n_secondary else 0

    for m in primary:
        models.append((m, per_primary))
    if secondary:
        for m in secondary:
            models.append((m, per_secondary))

    # Normalise (should already sum to 1.0 but be safe)
    total_weight = sum(w for _, w in models)
    if total_weight > 0:
        models = [(m, w / total_weight) for m, w in models]
    return models


# ---------------------------------------------------------------------------
# Prompt templates per sub-type category
# ---------------------------------------------------------------------------

_PROSE_TEMPLATES = [
    "Write a {task} about {domain}. Use natural paragraphs with varied sentence lengths.",
    "Compose a short essay on {domain}. Include an introduction, body, and conclusion.",
    "Draft a {task} reviewing recent developments in {domain}.",
    "Write a detailed explanation of a concept in {domain} for a general audience.",
    "Create a {task} that discusses the future of {domain} with concrete examples.",
    "Author a persuasive piece arguing for innovation in {domain}.",
]

_CODE_TEMPLATES = [
    "Write a {task} in the appropriate language for {domain}. Include comments.",
    "Generate a well-structured {task} related to {domain} with error handling.",
    "Create a {task} that demonstrates best practices for {domain}.",
    "Write a {task} implementing a common algorithm or utility for {domain}.",
    "Draft a {task} with type annotations and documentation for {domain}.",
    "Generate a {task} that processes data related to {domain}.",
]

_STRUCTURED_TEMPLATES = [
    "Generate a {task} containing {domain} data with realistic field names and values.",
    "Create a {task} dataset about {domain} with at least 10 records.",
    "Produce a {task} file with {domain} metrics and measurements.",
    "Write a {task} with hierarchical {domain} data including nested fields.",
    "Generate a {task} log or data dump related to {domain} operations.",
    "Create a realistic {task} export of {domain} records.",
]

_ARTIFACT_TEMPLATES = [
    "Generate a {task} that looks like {domain} document extraction noise.",
    "Create a {task} resembling OCR output from a {domain} document with errors.",
    "Produce a {task} that mimics boilerplate headers/footers from a {domain} PDF.",
    "Write a {task} that simulates garbled text extraction from a {domain} scan.",
    "Generate a {task} with mixed encoding artifacts from a {domain} source.",
    "Create a {task} that resembles a corrupted {domain} document dump.",
]

_ADVERSARIAL_ARTIFACT_TEMPLATES = [
    "Generate text that appears to be {domain} content but is actually OCR garbage with random character substitutions.",
    "Create a {task} that mixes valid {domain} text with PDF extraction artifacts like page numbers, headers, and broken words.",
    "Produce a {task} that looks like a {domain} document but has boilerplate legal disclaimers and cookie notices mixed in.",
    "Write text that simulates a badly-extracted {domain} PDF where tables are mangled into random character sequences.",
    "Generate a {task} with {domain} content corrupted by encoding issues — mojibake, replacement characters, and broken Unicode.",
]

# ---------------------------------------------------------------------------
# Sub-type task descriptions (used in {task} placeholder)
# ---------------------------------------------------------------------------

_SUB_TYPE_TASKS: dict[str, str] = {
    # Prose
    "plain": "plain text article",
    "markdown": "Markdown document",
    "rst": "reStructuredText document",
    "latex": "LaTeX document",
    # Code - Languages
    "python": "Python script",
    "javascript": "JavaScript module",
    "typescript": "TypeScript module",
    "rust": "Rust program",
    "go": "Go program",
    "java": "Java class",
    "sql": "SQL query or schema",
    "shell": "shell script",
    "css": "CSS stylesheet",
    # Code - Config
    "yaml": "YAML configuration file",
    "toml": "TOML configuration file",
    "ini": "INI configuration file",
    "dockerfile": "Dockerfile",
    "makefile": "Makefile",
    # Code - Markup
    "html": "HTML page",
    "xml": "XML document",
    "sgml": "SGML document",
    # Structured - Tabular
    "csv": "CSV data file",
    "tsv": "TSV data file",
    "pipe_table": "pipe-delimited table",
    "fixed_width": "fixed-width formatted table",
    # Structured - Data
    "json": "JSON data file",
    "jsonl": "JSONL data file",
    "key_value": "key-value configuration dump",
    "log_lines": "structured log file",
    # Artifact
    "pdf_dump": "PDF text extraction dump",
    "ocr_garbage": "OCR text output",
    "boilerplate": "boilerplate template text",
    # Artifact - Skip (adversarial too-short/empty-like content)
    "skip": "near-empty or too-short text fragment",
}

# Category -> template list mapping
_CATEGORY_TEMPLATES: dict[str, list[str]] = {
    "prose": _PROSE_TEMPLATES,
    "code": _CODE_TEMPLATES,
    "structured": _STRUCTURED_TEMPLATES,
    "artifact": _ARTIFACT_TEMPLATES,
}

# ---------------------------------------------------------------------------
# SUB_TYPE_CONFIG: 33 sub-types with weighted models, prompts, temp ranges
# ---------------------------------------------------------------------------

def _build_sub_type_config() -> dict[str, dict]:
    """Build the SUB_TYPE_CONFIG dict for all 33 sub-types."""
    # Category -> sub-types mapping (mirrors generate.py GOLDEN_SUB_TYPES)
    category_sub_types = {
        "prose": ["plain", "markdown", "rst", "latex"],
        "code": [
            "python", "javascript", "typescript", "rust", "go", "java",
            "sql", "shell", "css",
            "yaml", "toml", "ini", "dockerfile", "makefile",
            "html", "xml", "sgml",
        ],
        "structured": [
            "csv", "tsv", "pipe_table", "fixed_width",
            "json", "jsonl", "key_value", "log_lines",
        ],
        "artifact": ["pdf_dump", "ocr_garbage", "boilerplate", "skip"],
    }

    # Model pools per category: cheap models (90%), frontier accent (10%)
    # Cheap: DeepSeek, Llama, Qwen, Codestral, Gemma, etc. (~$0.10-$1.50/M tokens)
    # Frontier: Claude, GPT-5/5.4 (~$3-15/M tokens) — kept at 10% for diversity
    _code_models = _weighted_models(
        ["deepseek/deepseek-v3.2", "meta-llama/llama-3.3-70b-instruct",
         "qwen/qwen3-235b-a22b", "mistralai/codestral-2508", "google/gemini-2.5-flash",
         "qwen/qwen3-coder", "meta-llama/llama-4-maverick"],
        ["anthropic/claude-sonnet-4.6", "openai/gpt-5.4"],
        primary_share=0.90,
    )
    _prose_models = _weighted_models(
        ["meta-llama/llama-3.3-70b-instruct", "google/gemma-3-27b-it",
         "qwen/qwen3-235b-a22b", "cohere/command-a", "mistralai/mistral-large-2512",
         "deepseek/deepseek-r1-0528", "meta-llama/llama-4-maverick"],
        ["anthropic/claude-sonnet-4.6", "openai/gpt-5"],
        primary_share=0.90,
    )
    _structured_models = _weighted_models(
        ["deepseek/deepseek-v3.2", "qwen/qwen3-235b-a22b",
         "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.5-flash",
         "mistralai/codestral-2508", "qwen/qwen3-coder", "cohere/command-a"],
        ["anthropic/claude-sonnet-4.6", "openai/gpt-5.4"],
        primary_share=0.90,
    )
    _artifact_models = _weighted_models(
        ["meta-llama/llama-3.1-8b-instruct", "microsoft/phi-4",
         "openai/gpt-5.4-nano", "deepseek/deepseek-v3.2",
         "meta-llama/llama-3.3-70b-instruct", "google/gemma-3-27b-it",
         "qwen/qwen3-30b-a3b"],
        ["anthropic/claude-sonnet-4.6", "openai/gpt-5"],
        primary_share=0.90,
    )

    model_pools = {
        "prose": _prose_models,
        "code": _code_models,
        "structured": _structured_models,
        "artifact": _artifact_models,
    }

    temp_ranges = {
        "prose": (0.6, 1.2),
        "code": (0.3, 0.9),
        "structured": (0.2, 0.7),
        "artifact": (0.7, 1.4),
    }

    config: dict[str, dict] = {}
    for category, sub_types in category_sub_types.items():
        templates = _CATEGORY_TEMPLATES[category]
        # For artifact, also include adversarial templates
        if category == "artifact":
            templates = templates + _ADVERSARIAL_ARTIFACT_TEMPLATES

        for st in sub_types:
            task = _SUB_TYPE_TASKS[st]
            # Build concrete prompt templates with {task} partially filled
            # but keep {domain} as placeholder
            prompt_list = [t.replace("{task}", task) for t in templates]
            config[st] = {
                "category": category,
                "models": model_pools[category],
                "prompt_templates": prompt_list,
                "temperature_range": temp_ranges[category],
                "domains": list(DOMAIN_SEEDS),
            }

    return config


SUB_TYPE_CONFIG: dict[str, dict] = _build_sub_type_config()


# ---------------------------------------------------------------------------
# OpenRouter client helper
# ---------------------------------------------------------------------------

def _create_client():
    """Create an OpenAI-compatible client for OpenRouter."""
    if openai is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# API call with exponential backoff
# ---------------------------------------------------------------------------

# Global stats for tracking API call outcomes
_api_stats: dict[str, dict[str, int]] = {}


def _record_stat(model: str, outcome: str) -> None:
    """Record an API call outcome for end-of-run summary."""
    if model not in _api_stats:
        _api_stats[model] = {"success": 0, "empty_response": 0, "parse_error": 0, "api_error": 0, "raw_wrapped": 0}
    _api_stats[model][outcome] = _api_stats[model].get(outcome, 0) + 1


def _parse_json_response(content: str) -> list | None:
    """Try multiple strategies to extract a JSON array from a model response.

    Returns a list of dicts on success, or None if no JSON array can be found
    (indicating the model returned raw content that should be wrapped as-is).
    """
    # Strategy 1: Direct parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "text" in result:
            return [result]
        return result  # let caller handle non-list
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON array in the response (models often add prose around it)
    start = content.find("[")
    if start != -1:
        # Find the matching closing bracket
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "[":
                depth += 1
            elif content[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(content[start:i + 1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break

    # Strategy 3: Try to parse with JSONDecoder (handles trailing content)
    try:
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(content)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "text" in result:
            return [result]
    except (json.JSONDecodeError, ValueError):
        pass

    # No JSON found — return None to signal raw content wrapping
    return None


def print_api_stats() -> None:
    """Print a summary of API call outcomes per model."""
    if not _api_stats:
        return
    print("\n" + "=" * 72)
    print("  API CALL STATS (per model)")
    print("=" * 72)
    for model in sorted(_api_stats):
        s = _api_stats[model]
        total = sum(s.values())
        ok = s.get("success", 0) + s.get("raw_wrapped", 0)
        success_pct = (ok / total * 100) if total else 0
        parts = [f"success={s.get('success', 0)}"]
        if s.get("raw_wrapped"):
            parts.append(f"raw_wrapped={s['raw_wrapped']}")
        if s.get("empty_response"):
            parts.append(f"empty={s['empty_response']}")
        if s.get("parse_error"):
            parts.append(f"parse_err={s['parse_error']}")
        if s.get("api_error"):
            parts.append(f"api_err={s['api_error']}")
        print(f"  {model}: {', '.join(parts)} ({success_pct:.0f}% yield)")
    print("=" * 72)


def _call_api_with_retry(
    client,
    model: str,
    messages: list[dict],
    temperature: float,
    max_retries: int = 3,
) -> list[dict]:
    """Call the chat API with exponential backoff on errors.

    Returns parsed list of sample dicts, or empty list on failure.
    Logs errors to stderr and tracks stats per model.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body={
                    "provider": {
                        "data_collection": "deny",
                        "allow_fallbacks": True,
                        "ignore": ["azure"],
                        "sort": "price",
                    },
                },
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [
                    line for i, line in enumerate(lines)
                    if not (i == 0 or (i == len(lines) - 1 and line.strip() == "```"))
                ]
                content = "\n".join(lines)

            if not content:
                _record_stat(model, "empty_response")
                print(f"  [WARN] {model}: empty response content", file=sys.stderr)
                return []

            samples = _parse_json_response(content)
            if samples is None:
                # Last resort: model returned raw content instead of JSON —
                # wrap the entire response as a single sample
                _record_stat(model, "raw_wrapped")
                return [{"text": content}]

            if not isinstance(samples, list):
                _record_stat(model, "parse_error")
                print(
                    f"  [WARN] {model}: response is {type(samples).__name__}, not list. "
                    f"First 100 chars: {content[:100]}",
                    file=sys.stderr,
                )
                return []

            _record_stat(model, "success")
            return samples

        except json.JSONDecodeError as e:
            _record_stat(model, "parse_error")
            snippet = content[:200] if 'content' in dir() else '(no content)'
            print(
                f"  [WARN] {model}: JSON parse error on attempt {attempt + 1}/{max_retries}: {e}. "
                f"Response: {snippet}",
                file=sys.stderr,
            )
            last_error = e

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 0.5  # 0.5s, 1s, 2s
                print(
                    f"  [WARN] {model}: API error on attempt {attempt + 1}/{max_retries}: {e}. "
                    f"Retrying in {wait:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
            continue

    _record_stat(model, "api_error")
    print(
        f"  [ERROR] {model}: all {max_retries} retries exhausted. Last error: {last_error}",
        file=sys.stderr,
    )
    return []


# ---------------------------------------------------------------------------
# Weighted model selection
# ---------------------------------------------------------------------------

def _select_model(models: list[tuple[str, float]]) -> str:
    """Select a model from a weighted list."""
    model_ids = [m for m, _ in models]
    weights = [w for _, w in models]
    return random.choices(model_ids, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def generate_samples(
    sub_type: str,
    count: int,
    config: dict,
    client,
) -> list[dict]:
    """Generate clear samples for a single sub-type.

    Uses weighted model selection, temperature variation, prompt rotation,
    domain rotation, and length bucket cycling. Returns list of dicts with
    all provenance fields.
    """
    results: list[dict] = []
    category = config["category"]
    models = config["models"]
    templates = config["prompt_templates"]
    temp_lo, temp_hi = config["temperature_range"]
    bucket_names = list(LENGTH_BUCKETS.keys())

    domain_cycle = itertools.cycle(DOMAIN_SEEDS)
    template_cycle = itertools.cycle(templates)
    bucket_cycle = itertools.cycle(bucket_names)

    pbar = tqdm(total=count, desc=f"  {sub_type}", unit="sample", leave=False)
    collected = 0
    empty_streak = 0
    while collected < count:
        model_id = _select_model(models)
        domain = next(domain_cycle)
        template = next(template_cycle)
        bucket = next(bucket_cycle)
        lo, hi = LENGTH_BUCKETS[bucket]
        temperature = round(random.uniform(temp_lo, temp_hi), 2)

        batch_size = min(10, count - collected)
        prompt = template.format(domain=domain, task=_SUB_TYPE_TASKS.get(sub_type, sub_type))
        prompt += f"\nGenerate {batch_size} distinct examples, each {lo}-{hi} lines long."
        prompt += f"\nReturn a JSON array of {batch_size} objects, each with a single 'text' field."

        messages = [
            {"role": "system", "content": "You are a dataset generation assistant. Return only valid JSON arrays."},
            {"role": "user", "content": prompt},
        ]

        pbar.set_postfix(model=model_id.split("/")[-1], domain=domain[:12])
        raw_samples = _call_api_with_retry(client, model_id, messages, temperature)

        if not raw_samples:
            empty_streak += 1
            if empty_streak >= 3:
                break
            continue
        empty_streak = 0

        for raw in raw_samples:
            if collected >= count:
                break
            # Handle both {"text": "..."} dicts and bare strings
            if isinstance(raw, str):
                text = raw
            elif isinstance(raw, dict):
                text = raw.get("text", "")
            else:
                continue
            if not isinstance(text, str) or not text.strip():
                continue

            sample = {
                "text": text,
                "expected_category": category,
                "sub_type": sub_type,
                "boundary_pair": None,
                "model": model_id,
                "temperature": temperature,
                "prompt_template": template,
                "content_domain": domain,
                "length_bucket": bucket,
                "reasoning_mode": "deepseek" in model_id.lower() and "r1" in model_id.lower(),
            }

            if validate_sample(sample):
                results.append(sample)
                collected += 1
                pbar.update(1)

    pbar.close()
    return results


def generate_boundary_samples(
    pair: tuple[str, str],
    count: int,
    client,
) -> list[dict]:
    """Generate boundary-ambiguous samples for a category pair.

    Generates count/2 samples per direction (500 per direction when count=1000).
    Uses boundary-specific prompts with domain and length rotation.
    """
    cat_a, cat_b = pair
    boundary_label = f"{cat_a}_{cat_b}"
    per_direction = math.ceil(count / 2)
    results: list[dict] = []
    bucket_names = list(LENGTH_BUCKETS.keys())
    domain_cycle = itertools.cycle(DOMAIN_SEEDS)
    bucket_cycle = itertools.cycle(bucket_names)

    pbar = tqdm(total=count, desc=f"  {cat_a}/{cat_b}", unit="sample", leave=False)
    for label in [cat_a, cat_b]:
        other = cat_b if label == cat_a else cat_a
        collected = 0
        empty_streak = 0
        while collected < per_direction:
            domain = next(domain_cycle)
            bucket = next(bucket_cycle)
            lo, hi = LENGTH_BUCKETS[bucket]
            temperature = round(random.uniform(0.5, 1.2), 2)

            # Pick a model from primary pool
            model_id = random.choice(PRIMARY_MODELS)

            batch_size = min(10, per_direction - collected)
            prompt = (
                f"Generate {batch_size} distinct text samples about {domain} that are ambiguous between "
                f"'{cat_a}' and '{cat_b}', but should be labeled as '{label}'.\n"
                f"Each text should be {lo}-{hi} lines long.\n"
                f"They should have characteristics of both {cat_a} and {cat_b}, "
                f"but on balance belong to '{label}' rather than '{other}'.\n\n"
                f"Return a JSON array of {batch_size} objects, each with a single 'text' field."
            )

            messages = [
                {"role": "system", "content": "You are a dataset generation assistant. Return only valid JSON arrays."},
                {"role": "user", "content": prompt},
            ]

            pbar.set_postfix(label=label, model=model_id.split("/")[-1])
            raw_samples = _call_api_with_retry(client, model_id, messages, temperature)

            if not raw_samples:
                empty_streak += 1
                if empty_streak >= 3:
                    break
                continue
            empty_streak = 0

            for raw in raw_samples:
                if collected >= per_direction:
                    break
                if isinstance(raw, str):
                    text = raw
                elif isinstance(raw, dict):
                    text = raw.get("text", "")
                else:
                    continue
                if not isinstance(text, str) or not text.strip():
                    continue

                sample = {
                    "text": text,
                    "expected_category": label,
                    "sub_type": None,
                    "boundary_pair": boundary_label,
                    "model": model_id,
                    "temperature": temperature,
                    "prompt_template": "boundary",
                    "content_domain": domain,
                    "length_bucket": bucket,
                    "reasoning_mode": "deepseek" in model_id.lower() and "r1" in model_id.lower(),
                }

                if validate_sample(sample):
                    results.append(sample)
                    collected += 1
                    pbar.update(1)

    pbar.close()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data via OpenRouter multi-model API.",
    )
    parser.add_argument(
        "--output",
        default="training/data/openrouter.jsonl",
        help="Output JSONL file path (default: training/data/openrouter.jsonl)",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=60000,
        help="Total number of samples to generate (default: 60000)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Pilot mode: generate ~500 samples (~15/sub-type)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print generation plan without making API calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing output file",
    )
    return parser


def _print_plan(total_samples: int, pilot: bool) -> None:
    """Print generation plan summary."""
    effective = 500 if pilot else total_samples
    n_sub_types = len(SUB_TYPE_CONFIG)
    per_sub_type = max(1, effective // (n_sub_types + len(BOUNDARY_PAIRS)))
    boundary_budget = max(1, effective - per_sub_type * n_sub_types)

    print("=" * 60)
    print("DRY RUN — Generation Plan")
    print("=" * 60)
    print(f"Total samples: {effective}")
    print(f"Sub-types: {n_sub_types}")
    print(f"Samples per sub-type: ~{per_sub_type}")
    print(f"Boundary budget: ~{boundary_budget}")
    print(f"Boundary pairs: {len(BOUNDARY_PAIRS)}")
    print(f"Domain seeds: {len(DOMAIN_SEEDS)}")
    print(f"Length buckets: {list(LENGTH_BUCKETS.keys())}")
    print()

    print("Sub-type breakdown:")
    for category in ["prose", "code", "structured", "artifact"]:
        sub_types = [st for st, c in SUB_TYPE_CONFIG.items() if c["category"] == category]
        models = SUB_TYPE_CONFIG[sub_types[0]]["models"]
        model_names = [m for m, _ in models]
        print(f"\n  {category} ({len(sub_types)} sub-types):")
        for st in sub_types:
            print(f"    - {st}: ~{per_sub_type} samples")
        print(f"    models: {', '.join(model_names)}")

    print()
    print("Boundary pairs:")
    for pair in BOUNDARY_PAIRS:
        print(f"  - {pair[0]} <-> {pair[1]}")

    print()
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the generation script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    total = 2000 if args.pilot else args.total_samples

    if args.dry_run:
        _print_plan(total, args.pilot)
        return

    # Real generation
    client = _create_client()

    # Calculate budget
    n_sub_types = len(SUB_TYPE_CONFIG)
    boundary_total = len(BOUNDARY_PAIRS) * 1000  # 500 per direction
    clear_total = total - min(boundary_total, total // 3)
    per_sub_type = max(1, clear_total // n_sub_types)

    # Resume support — count existing samples per sub-type and boundary pair
    existing_per_sub_type: dict[str, int] = defaultdict(int)
    existing_boundary: int = 0
    existing_count = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        existing_count += 1
                        if sample.get("boundary_pair"):
                            existing_boundary += 1
                        else:
                            st = sample.get("sub_type", "unknown")
                            existing_per_sub_type[st] += 1
                    except json.JSONDecodeError:
                        pass
        print(f"Resuming from {existing_count} existing samples ({existing_count - existing_boundary} clear, {existing_boundary} boundary)")
        completed = [st for st, c in existing_per_sub_type.items() if c >= per_sub_type]
        incomplete = {st: per_sub_type - c for st, c in existing_per_sub_type.items() if c < per_sub_type}
        missing = [st for st in SUB_TYPE_CONFIG if st not in existing_per_sub_type]
        if completed:
            print(f"  Skipping {len(completed)} completed sub-types: {', '.join(completed[:5])}{'...' if len(completed) > 5 else ''}")
        if incomplete:
            print(f"  Resuming {len(incomplete)} incomplete sub-types: {', '.join(f'{st}({n} remaining)' for st, n in incomplete.items())}")
        if missing:
            print(f"  Starting {len(missing)} new sub-types: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")

    new_samples = 0
    max_workers = 4  # concurrent API calls

    # Open output file for incremental writing (append mode for resume)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    _output_lock = threading.Lock()
    _output_file = open(args.output, "a")

    def _flush_samples(samples: list[dict]) -> int:
        """Write samples to output file immediately (thread-safe)."""
        with _output_lock:
            for sample in samples:
                _output_file.write(json.dumps(sample) + "\n")
            _output_file.flush()
        return len(samples)

    # Build list of sub-types that still need generation
    sub_type_items = []
    for st, cfg in SUB_TYPE_CONFIG.items():
        existing = existing_per_sub_type.get(st, 0)
        remaining = per_sub_type - existing
        if remaining > 0:
            sub_type_items.append((st, remaining, cfg))

    if sub_type_items:
        print(f"\nGenerating clear samples ({len(sub_type_items)} sub-types remaining, {max_workers} workers):")
        pbar = tqdm(total=len(sub_type_items), desc="Sub-types", unit="type")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_samples, st, count, cfg, client): st
                for st, count, cfg in sub_type_items
            }
            for future in concurrent.futures.as_completed(futures):
                samples = future.result()
                new_samples += _flush_samples(samples)
                pbar.update(1)
        pbar.close()
    else:
        print("\nAll clear sub-types already complete, skipping.")

    # Generate boundary samples (parallel)
    clear_done = sum(existing_per_sub_type.values()) + new_samples
    boundary_budget = max(0, total - clear_done)
    boundary_per_pair = max(1, boundary_budget // len(BOUNDARY_PAIRS)) if boundary_budget > 0 else 0
    if existing_boundary > 0:
        already_per_pair = existing_boundary // len(BOUNDARY_PAIRS)
        boundary_per_pair = max(0, boundary_per_pair - already_per_pair)

    if boundary_per_pair > 0:
        print(f"\nGenerating boundary samples ({boundary_per_pair}/pair, {len(BOUNDARY_PAIRS)} pairs, {max_workers} workers):")
        pbar = tqdm(total=len(BOUNDARY_PAIRS), desc="Pairs", unit="pair")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_boundary_samples, pair, boundary_per_pair, client): pair
                for pair in BOUNDARY_PAIRS
            }
            for future in concurrent.futures.as_completed(futures):
                samples = future.result()
                new_samples += _flush_samples(samples)
                pbar.update(1)
        pbar.close()
    else:
        print("\nAll boundary samples already complete, skipping.")

    _output_file.close()

    print(f"Wrote {new_samples} new samples to {args.output}")
    print(f"Total: {existing_count + new_samples}")

    # Print API call stats summary
    print_api_stats()


if __name__ == "__main__":
    main()
