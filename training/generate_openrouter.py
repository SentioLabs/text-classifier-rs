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
    "astronomy", "finance", "healthcare", "devops", "gaming",
    "machine learning", "cybersecurity", "education", "agriculture",
    "automotive", "aviation", "biology", "chemistry", "climate science",
    "cryptocurrency", "data engineering", "e-commerce", "electronics",
    "energy", "environmental science", "fashion", "food science",
    "genetics", "geography", "government", "insurance", "journalism",
    "law", "linguistics", "logistics", "manufacturing", "marine biology",
    "marketing", "mathematics", "meteorology", "military", "music",
    "nanotechnology", "neuroscience", "nuclear physics", "oceanography",
    "pharmacology", "philosophy", "photography", "political science",
    "psychology", "real estate", "robotics", "sociology", "sports",
    "telecommunications", "urban planning", "veterinary medicine",
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

def _weighted_models(primary: list[str], secondary: list[str] | None = None) -> list[tuple[str, float]]:
    """Build a weighted model list with <=15% cap per model.

    Primary models get equal share; secondary models (if given) get smaller
    shares. Total weight normalises so no single model exceeds 15%.
    """
    models: list[tuple[str, float]] = []
    n_primary = len(primary)
    n_secondary = len(secondary) if secondary else 0
    total = n_primary + n_secondary

    # Equal weighting across all models to respect 15% cap
    for m in primary:
        models.append((m, 1.0))
    if secondary:
        for m in secondary:
            models.append((m, 1.0))

    # Normalise and verify cap
    total_weight = sum(w for _, w in models)
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

    # Model pools per category (5-7 models, weighted, no single model > 15%)
    _code_models = _weighted_models(
        ["anthropic/claude-sonnet-4.6", "openai/gpt-5.4", "deepseek/deepseek-v3.2",
         "mistralai/codestral-2508"],
        ["meta-llama/llama-3.3-70b-instruct", "qwen/qwen3-235b-a22b", "google/gemini-2.5-flash"],
    )
    _prose_models = _weighted_models(
        ["anthropic/claude-sonnet-4.6", "openai/gpt-5", "openai/gpt-5.4",
         "mistralai/mistral-large-2512"],
        ["x-ai/grok-4-fast", "cohere/command-a", "google/gemma-3-27b-it"],
    )
    _structured_models = _weighted_models(
        ["openai/gpt-5.4", "deepseek/deepseek-v3.2", "qwen/qwen3-235b-a22b",
         "anthropic/claude-sonnet-4.6"],
        ["google/gemini-2.5-flash", "meta-llama/llama-3.3-70b-instruct", "mistralai/mistral-large-2512"],
    )
    _artifact_models = _weighted_models(
        ["openai/gpt-5", "anthropic/claude-sonnet-4.6", "deepseek/deepseek-r1-0528",
         "openai/gpt-5.4"],
        ["microsoft/phi-4", "meta-llama/llama-3.1-8b-instruct", "openai/gpt-5.4-nano"],
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
        _api_stats[model] = {"success": 0, "empty_response": 0, "parse_error": 0, "api_error": 0}
    _api_stats[model][outcome] = _api_stats[model].get(outcome, 0) + 1


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
        success_pct = (s["success"] / total * 100) if total else 0
        parts = [f"success={s['success']}"]
        if s["empty_response"]:
            parts.append(f"empty={s['empty_response']}")
        if s["parse_error"]:
            parts.append(f"parse_err={s['parse_error']}")
        if s["api_error"]:
            parts.append(f"api_err={s['api_error']}")
        print(f"  {model}: {', '.join(parts)} ({success_pct:.0f}% success rate)")
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

            samples = json.loads(content)
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
            # Show first 200 chars of what we tried to parse
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

        prompt = template.format(domain=domain, task=_SUB_TYPE_TASKS.get(sub_type, sub_type))
        prompt += f"\nGenerate text that is {lo}-{hi} lines long."
        prompt += f"\nReturn a JSON array of objects with a single 'text' field."

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
            text = raw.get("text", "")
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

            prompt = (
                f"Generate text about {domain} that is ambiguous between "
                f"'{cat_a}' and '{cat_b}', but should be labeled as '{label}'.\n"
                f"The text should be {lo}-{hi} lines long.\n"
                f"It should have characteristics of both {cat_a} and {cat_b}, "
                f"but on balance belongs to '{label}' rather than '{other}'.\n\n"
                f"Return a JSON array of objects with a single 'text' field."
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
                text = raw.get("text", "")
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

    total = 500 if args.pilot else args.total_samples

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

    # Resume support
    existing_count = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"Resuming from {existing_count} existing samples")

    all_samples: list[dict] = []
    generated = existing_count

    # Generate clear samples per sub-type
    print(f"\nGenerating clear samples ({per_sub_type}/sub-type, {len(SUB_TYPE_CONFIG)} sub-types):")
    for sub_type, config in tqdm(SUB_TYPE_CONFIG.items(), desc="Sub-types", unit="type"):
        samples = generate_samples(sub_type, per_sub_type, config, client)
        all_samples.extend(samples)
        generated += len(samples)

    # Generate boundary samples
    boundary_per_pair = max(1, (total - len(all_samples) - existing_count) // len(BOUNDARY_PAIRS))
    print(f"\nGenerating boundary samples ({boundary_per_pair}/pair, {len(BOUNDARY_PAIRS)} pairs):")
    for pair in tqdm(BOUNDARY_PAIRS, desc="Pairs", unit="pair"):
        samples = generate_boundary_samples(pair, boundary_per_pair, client)
        all_samples.extend(samples)
        generated += len(samples)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mode = "a" if args.resume and os.path.exists(args.output) else "w"
    with open(args.output, mode) as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Wrote {len(all_samples)} samples to {args.output}")
    print(f"Total: {generated}")

    # Print API call stats summary
    print_api_stats()


if __name__ == "__main__":
    main()
