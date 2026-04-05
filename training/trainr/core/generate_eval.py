#!/usr/bin/env python3
"""Generate golden eval set using the OpenAI GPT-5.4 API.

Produces clear and boundary samples with variety seeding across domains
and length buckets. Writes eval/clear.jsonl and eval/boundary.jsonl.
"""

import argparse
import json
import math
import os
import sys

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

# ── Shared constants (verbatim from T0: Foundation) ──────────────────

VALID_CATEGORIES = {"prose", "code", "structured", "artifact"}

VALID_BOUNDARY_PAIRS = {
    "prose_code",
    "prose_structured",
    "prose_artifact",
    "code_structured",
    "code_artifact",
    "structured_artifact",
}

# ── Domain seeds (~50 topics) ────────────────────────────────────────

DOMAIN_SEEDS = [
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

# ── Length buckets ───────────────────────────────────────────────────

LENGTH_BUCKETS = {
    "short": (3, 10),
    "medium": (20, 50),
    "long": (100, 200),
}

# ── Batch size for API calls ────────────────────────────────────────

BATCH_SIZE = 20


# ── Shared validation ───────────────────────────────────────────────

def validate_sample(sample: dict) -> bool:
    """Validate a sample dict according to the shared contract.

    A valid sample has:
      - non-empty "text" string
      - "label" in VALID_CATEGORIES
      - "kind" in {"clear", "boundary"}
      - if kind == "boundary": "boundary_pair" in VALID_BOUNDARY_PAIRS
        and label must be one of the two categories in the pair
    """
    if not isinstance(sample, dict):
        return False

    text = sample.get("text")
    if not isinstance(text, str) or not text.strip():
        return False

    label = sample.get("label")
    if label not in VALID_CATEGORIES:
        return False

    kind = sample.get("kind")
    if kind not in ("clear", "boundary"):
        return False

    if kind == "boundary":
        pair = sample.get("boundary_pair")
        if pair not in VALID_BOUNDARY_PAIRS:
            return False
        pair_categories = set(pair.split("_"))
        if label not in pair_categories:
            return False

    return True


# ── Client helper ────────────────────────────────────────────────────

def _create_client():
    """Create an OpenAI client using the OPENAI_API_KEY env var."""
    if openai is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return openai.OpenAI(api_key=api_key)


# ── Prompt builders ──────────────────────────────────────────────────

def _clear_prompt(category: str, domain: str, length_bucket: str, count: int) -> str:
    lo, hi = LENGTH_BUCKETS[length_bucket]
    return (
        f"Generate {count} unambiguous examples of '{category}' text about {domain}.\n"
        f"Each example should be {lo}-{hi} lines long.\n\n"
        f"Category definitions:\n"
        f"- prose: Natural language paragraphs — articles, essays, reviews, comments.\n"
        f"- code: Source code, scripts, config-as-code (Dockerfiles, Makefiles, k8s YAML).\n"
        f"- structured: Tabular or structured data — CSV, TSV, JSON data arrays, XML data.\n"
        f"- artifact: Non-content artifacts — headers, footers, boilerplate, legal disclaimers, OCR noise.\n\n"
        f"Return a JSON array of objects, each with:\n"
        f'  - "text": the generated text (string)\n'
        f'  - "label": "{category}"\n'
        f'  - "kind": "clear"\n\n'
        f"Return ONLY the JSON array, no markdown fences or extra text."
    )


def _boundary_prompt(pair: str, label: str, domain: str, length_bucket: str, count: int) -> str:
    cats = pair.split("_")
    lo, hi = LENGTH_BUCKETS[length_bucket]
    other = cats[1] if label == cats[0] else cats[0]
    return (
        f"Generate {count} text samples about {domain} that are genuinely ambiguous "
        f"between '{cats[0]}' and '{cats[1]}', but should be labeled as '{label}'.\n"
        f"Each example should be {lo}-{hi} lines long.\n\n"
        f"The text should have characteristics of both {cats[0]} and {cats[1]}, "
        f"but on balance belongs to '{label}' rather than '{other}'.\n\n"
        f"For example, a YAML Kubernetes config is ambiguous between code and structured, "
        f"but label it as code because it's infrastructure-as-code.\n\n"
        f"Return a JSON array of objects, each with:\n"
        f'  - "text": the generated text (string)\n'
        f'  - "label": "{label}"\n'
        f'  - "kind": "boundary"\n'
        f'  - "boundary_pair": "{pair}"\n\n'
        f"Return ONLY the JSON array, no markdown fences or extra text."
    )


# ── API call helpers ─────────────────────────────────────────────────

def _call_api(client, model: str, prompt: str) -> list[dict]:
    """Call the OpenAI chat API and parse the JSON array response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a dataset generation assistant. Return only valid JSON arrays."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )
    content = response.choices[0].message.content
    # Strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last fence lines
        lines = [l for i, l in enumerate(lines) if not (i == 0 or (i == len(lines) - 1 and l.strip() == "```"))]
        content = "\n".join(lines)
    try:
        samples = json.loads(content)
    except json.JSONDecodeError:
        return []
    if not isinstance(samples, list):
        return []
    return samples


# ── Core generation functions ────────────────────────────────────────

def generate_clear_samples(
    category: str,
    count: int,
    domain_seeds: list[str],
    client=None,
    model: str = "gpt-5.4",
) -> list[dict]:
    """Generate unambiguous samples for a single category.

    Rotates through domain seeds and length buckets. Uses batches of
    BATCH_SIZE samples per API call. Validates each sample before accepting.
    """
    if client is None:
        client = _create_client()

    results: list[dict] = []
    bucket_names = list(LENGTH_BUCKETS.keys())
    seed_idx = 0
    bucket_idx = 0
    collected = 0

    while collected < count:
        remaining = count - collected
        batch = min(BATCH_SIZE, remaining)
        domain = domain_seeds[seed_idx % len(domain_seeds)]
        bucket = bucket_names[bucket_idx % len(bucket_names)]

        prompt = _clear_prompt(category, domain, bucket, batch)
        samples = _call_api(client, model, prompt)

        for s in samples:
            if collected >= count:
                break
            if validate_sample(s) and s.get("label") == category and s.get("kind") == "clear":
                results.append(s)
                collected += 1

        seed_idx += 1
        bucket_idx += 1

        # Progress reporting
        if collected > 0 and collected % 100 == 0:
            print(f"  [{category}] {collected}/{count} samples collected")

    return results


def generate_boundary_samples(
    pair: str,
    count: int,
    domain_seeds: list[str],
    client=None,
    model: str = "gpt-5.4",
) -> list[dict]:
    """Generate ambiguous boundary samples for a category pair.

    Generates count/2 samples labeled for each side of the pair (500 per direction
    when count=1000). Rotates through domain seeds and length buckets.
    """
    if client is None:
        client = _create_client()

    cats = pair.split("_")
    per_direction = math.ceil(count / 2)
    results: list[dict] = []
    bucket_names = list(LENGTH_BUCKETS.keys())

    for label in cats:
        seed_idx = 0
        bucket_idx = 0
        collected = 0

        while collected < per_direction:
            remaining = per_direction - collected
            batch = min(BATCH_SIZE, remaining)
            domain = domain_seeds[seed_idx % len(domain_seeds)]
            bucket = bucket_names[bucket_idx % len(bucket_names)]

            prompt = _boundary_prompt(pair, label, domain, bucket, batch)
            samples = _call_api(client, model, prompt)

            for s in samples:
                if collected >= per_direction:
                    break
                if validate_sample(s) and s.get("kind") == "boundary" and s.get("boundary_pair") == pair:
                    results.append(s)
                    collected += 1

            seed_idx += 1
            bucket_idx += 1

            # Progress reporting
            total = len(results)
            if total > 0 and total % 100 == 0:
                print(f"  [{pair}/{label}] {collected}/{per_direction} samples collected")

    return results


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate golden eval set using the OpenAI GPT-5.4 API.",
    )
    parser.add_argument(
        "--mode",
        choices=["clear", "boundary", "all"],
        required=True,
        help="What to generate: clear samples, boundary samples, or both.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write eval JSONL files.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=1000,
        help="Number of clear samples per category (default: 1000).",
    )
    parser.add_argument(
        "--samples-per-pair",
        type=int,
        default=1000,
        help="Number of boundary samples per pair (default: 1000).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="OpenAI model to use (default: gpt-5.4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without calling the API.",
    )
    return parser.parse_args(argv)


def _write_jsonl(path: str, samples: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def run(args: argparse.Namespace) -> None:
    """Execute the generation pipeline based on parsed args."""
    if args.dry_run:
        _run_dry(args)
        return

    client = _create_client()

    if args.mode in ("clear", "all"):
        clear_samples: list[dict] = []
        for category in sorted(VALID_CATEGORIES):
            print(f"Generating {args.samples_per_category} clear samples for '{category}'...")
            samples = generate_clear_samples(
                category=category,
                count=args.samples_per_category,
                domain_seeds=DOMAIN_SEEDS,
                client=client,
                model=args.model,
            )
            clear_samples.extend(samples)
            print(f"  Done: {len(samples)} samples for '{category}'")
        out_path = os.path.join(args.output_dir, "clear.jsonl")
        _write_jsonl(out_path, clear_samples)
        print(f"Wrote {len(clear_samples)} clear samples to {out_path}")

    if args.mode in ("boundary", "all"):
        boundary_samples: list[dict] = []
        for pair in sorted(VALID_BOUNDARY_PAIRS):
            print(f"Generating {args.samples_per_pair} boundary samples for '{pair}'...")
            samples = generate_boundary_samples(
                pair=pair,
                count=args.samples_per_pair,
                domain_seeds=DOMAIN_SEEDS,
                client=client,
                model=args.model,
            )
            boundary_samples.extend(samples)
            print(f"  Done: {len(samples)} samples for '{pair}'")
        out_path = os.path.join(args.output_dir, "boundary.jsonl")
        _write_jsonl(out_path, boundary_samples)
        print(f"Wrote {len(boundary_samples)} boundary samples to {out_path}")


def _run_dry(args: argparse.Namespace) -> None:
    """Print a summary of what would be generated without calling the API."""
    print("=== DRY RUN ===")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print()

    if args.mode in ("clear", "all"):
        total = args.samples_per_category * len(VALID_CATEGORIES)
        batches_per_cat = math.ceil(args.samples_per_category / BATCH_SIZE)
        print(f"Clear samples:")
        for category in sorted(VALID_CATEGORIES):
            print(f"  - {category}: {args.samples_per_category} samples ({batches_per_cat} API calls)")
        print(f"  Total clear: {total}")
        print()

    if args.mode in ("boundary", "all"):
        total = args.samples_per_pair * len(VALID_BOUNDARY_PAIRS)
        per_direction = math.ceil(args.samples_per_pair / 2)
        batches_per_dir = math.ceil(per_direction / BATCH_SIZE)
        print(f"Boundary samples:")
        for pair in sorted(VALID_BOUNDARY_PAIRS):
            cats = pair.split("_")
            print(
                f"  - {pair}: {args.samples_per_pair} samples "
                f"({per_direction} per direction, {batches_per_dir} API calls per direction)"
            )
        print(f"  Total boundary: {total}")
        print()

    print(f"Domain seeds: {len(DOMAIN_SEEDS)} topics")
    print(f"Length buckets: {', '.join(LENGTH_BUCKETS.keys())}")
    print("=== END DRY RUN ===")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


