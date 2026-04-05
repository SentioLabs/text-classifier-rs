#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets"]
# ///
"""Sample real code/text from bigcode/the-stack (v1) for training data.

Downloads and filters real code samples from The Stack v1 dataset for 14
language sub-types, outputting the same JSONL provenance format as the
OpenRouter generator.

Usage:
    python training/sample_the_stack.py --dry-run
    python training/sample_the_stack.py --output data/source/real/real_samples.jsonl
    python training/sample_the_stack.py --languages python,rust --per-language 500

Requires: pip install datasets
"""

import argparse
import json
import re
import sys

try:
    import datasets
except ImportError:
    datasets = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Language -> (category, sub_type) mapping
# ---------------------------------------------------------------------------

STACK_LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    "python": ("code", "python"),
    "javascript": ("code", "javascript"),
    "typescript": ("code", "typescript"),
    "rust": ("code", "rust"),
    "go": ("code", "go"),
    "java": ("code", "java"),
    "sql": ("code", "sql"),
    "shell": ("code", "shell"),
    "css": ("code", "css"),
    "dockerfile": ("code", "dockerfile"),
    "makefile": ("code", "makefile"),
    "html": ("code", "html"),
    "markdown": ("prose", "markdown"),
    "tex": ("prose", "latex"),
}

# Patterns that indicate auto-generated content (checked against first line)
_AUTO_GENERATED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^//\s*Code generated", re.IGNORECASE),
    re.compile(r"^#\s*AUTO[- ]GENERATED", re.IGNORECASE),
    re.compile(r"^/\*\s*AUTO\s+GENERATED", re.IGNORECASE),
    re.compile(r"^#\s*This file is auto-generated", re.IGNORECASE),
    re.compile(r"^//\s*DO NOT EDIT", re.IGNORECASE),
    re.compile(r"^#\s*DO NOT EDIT", re.IGNORECASE),
    re.compile(r"^/\*\s*DO NOT EDIT", re.IGNORECASE),
    re.compile(r"^<!--\s*AUTO[- ]GENERATED", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_sample(content: str) -> bool:
    """Return True if the sample passes all quality filters."""
    lines = content.split("\n")
    non_empty = [line for line in lines if line.strip()]

    # Reject if fewer than 5 non-empty lines
    if len(non_empty) < 5:
        return False

    # Reject if more than 500 total lines
    if len(lines) > 500:
        return False

    # Reject auto-generated content (check first non-empty line)
    if non_empty:
        first_line = non_empty[0].strip()
        for pattern in _AUTO_GENERATED_PATTERNS:
            if pattern.match(first_line):
                return False

    # Reject binary content (>10% non-printable characters)
    if content:
        non_printable = sum(
            1 for c in content
            if not c.isprintable() and c not in ("\n", "\r", "\t")
        )
        if non_printable / len(content) > 0.10:
            return False

    return True


# ---------------------------------------------------------------------------
# Length bucketing
# ---------------------------------------------------------------------------


def compute_length_bucket(content: str) -> str:
    """Classify content length into short/medium/long by non-empty line count."""
    non_empty = sum(1 for line in content.split("\n") if line.strip())
    if non_empty < 10:
        return "short"
    elif non_empty <= 50:
        return "medium"
    else:
        return "long"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_language(
    data_dir: str,
    category: str,
    sub_type: str,
    count: int,
    seed: int = 42,
) -> list[dict]:
    """Sample and filter real code from The Stack for one language."""
    if datasets is None:
        raise ImportError("The 'datasets' package is required: pip install datasets")

    ds = datasets.load_dataset(
        "bigcode/the-stack",
        data_dir=f"data/{data_dir}",
        streaming=True,
        split="train",
    )

    results: list[dict] = []
    seen = 0

    for item in ds.shuffle(seed=seed, buffer_size=10000):
        content = item["content"]
        if not filter_sample(content):
            continue

        results.append({
            "text": content,
            "expected_category": category,
            "sub_type": sub_type,
            "boundary_pair": None,
            "model": "real/the-stack-v1",
            "temperature": None,
            "prompt_template": None,
            "content_domain": data_dir,
            "length_bucket": compute_length_bucket(content),
            "reasoning_mode": False,
        })

        seen += 1
        if seen % 100 == 0:
            print(f"  {data_dir}: {seen}/{count} samples collected")

        if seen >= count:
            break

    return results


def sample_all(
    output_path: str,
    per_language: int = 2100,
    seed: int = 42,
    languages: list[str] | None = None,
) -> None:
    """Sample from all configured languages and write JSONL output."""
    if languages is None:
        languages = list(STACK_LANGUAGE_MAP.keys())

    total = 0
    with open(output_path, "a") as f:
        for lang in languages:
            category, sub_type = STACK_LANGUAGE_MAP[lang]
            print(f"Sampling {lang} ({category}/{sub_type})...")
            samples = sample_language(lang, category, sub_type, per_language, seed)
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
            count = len(samples)
            total += count
            print(f"  {lang}: {count} samples written")

    print(f"\nSummary:")
    print(f"  Total samples: {total}")
    print(f"  Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_languages(
    languages_str: str | None,
    language_map: dict[str, tuple[str, str]],
) -> list[str]:
    """Parse comma-separated language filter, or return all languages."""
    if languages_str is None:
        return list(language_map.keys())

    selected = [lang.strip() for lang in languages_str.split(",")]
    for lang in selected:
        if lang not in language_map:
            raise ValueError(
                f"Unknown language: {lang}. "
                f"Valid: {', '.join(sorted(language_map.keys()))}"
            )
    return selected


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Sample real code from bigcode/the-stack (v1)"
    )
    parser.add_argument(
        "--output",
        default="data/source/real/real_samples.jsonl",
        help="Output JSONL file path (default: data/source/real/real_samples.jsonl)",
    )
    parser.add_argument(
        "--per-language",
        type=int,
        default=1800,
        help="Samples per language (default: 2100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without downloading",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated list of languages to sample (default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    languages = _parse_languages(args.languages, STACK_LANGUAGE_MAP)

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no data will be downloaded")
        print("=" * 60)
        print(f"\n{len(languages)} languages to sample:")
        for lang in languages:
            category, sub_type = STACK_LANGUAGE_MAP[lang]
            print(f"  {lang:15s} -> {category}/{sub_type}")
        print(f"\nPer language: {args.per_language}")
        print(f"Estimated total: {len(languages) * args.per_language}")
        print(f"Seed: {args.seed}")
        print(f"Output: {args.output}")
        return

    sample_all(
        output_path=args.output,
        per_language=args.per_language,
        seed=args.seed,
        languages=languages,
    )


if __name__ == "__main__":
    main()
