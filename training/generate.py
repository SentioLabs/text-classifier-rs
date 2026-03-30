#!/usr/bin/env python3
"""Generate synthetic training data for the text classifier.

Modes:
  fixtures  — Extract features from test fixture files.
  synthetic — Generate text via Claude API and extract features.
  perturb   — Generate boundary cases by perturbing fixture features.
  all       — Run all modes and combine results.
"""

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFY_BIN = str(PROJECT_ROOT / "target" / "release" / "classify")
DEFAULT_FIXTURES_DIR = str(PROJECT_ROOT / "tests" / "fixtures")

# Maps fixture directory names to training categories
DIRECTORY_TO_CATEGORY = {
    "prose": "prose",
    "code": "code",
    "tabular": "structured",
    "pdf_dump": "artifact",
}

# Shared constant (verbatim from T0: Foundation)
VALID_CATEGORIES = {"prose", "code", "structured", "artifact"}

# All 33 ContentSubType variants grouped by category (mirrors types.rs)
GOLDEN_SUB_TYPES = {
    "prose": [
        "plain", "markdown", "rst", "latex",
    ],
    "code": [
        # Languages
        "python", "javascript", "typescript", "rust", "go", "java", "sql", "shell", "css",
        # Config
        "yaml", "toml", "ini", "dockerfile", "makefile",
        # Markup
        "html", "xml", "sgml",
    ],
    "structured": [
        # Tabular
        "csv", "tsv", "pipe_table", "fixed_width",
        # Data
        "json", "jsonl", "key_value", "log_lines",
    ],
    "artifact": [
        "pdf_dump", "ocr_garbage", "boilerplate",
    ],
}

# 50+ domain seeds for variety
GOLDEN_DOMAIN_SEEDS = [
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

# Length buckets: (min_lines, max_lines)
GOLDEN_LENGTH_BUCKETS = {
    "short": (3, 10),
    "medium": (20, 50),
    "long": (100, 200),
}

# Boundary pairs for golden training data (reuses existing AMBIGUOUS_PAIRS pattern)
GOLDEN_BOUNDARY_PAIRS = [
    {
        "cat_a": "code",
        "cat_b": "structured",
        "label": "code",
        "examples": "- TOML/INI config files (key=value but are config code)\n"
                    "- YAML with data-like content\n"
                    "- .env files with connection strings",
    },
    {
        "cat_a": "code",
        "cat_b": "prose",
        "label": "code",
        "examples": "- Python with extensive docstrings\n"
                    "- Shell scripts with long comment blocks\n"
                    "- SQL with detailed inline comments",
    },
    {
        "cat_a": "prose",
        "cat_b": "code",
        "label": "prose",
        "examples": "- Technical documentation about code\n"
                    "- API reference with code-like terms\n"
                    "- README files describing functions",
    },
    {
        "cat_a": "prose",
        "cat_b": "structured",
        "label": "prose",
        "examples": "- Markdown with tables and lists\n"
                    "- Technical writing with key-value descriptions\n"
                    "- Reports with tabular data embedded in text",
    },
    {
        "cat_a": "structured",
        "cat_b": "code",
        "label": "structured",
        "examples": "- JSON with code-like field names\n"
                    "- CSV with URL columns and special characters\n"
                    "- Log files with structured + freeform fields",
    },
    {
        "cat_a": "artifact",
        "cat_b": "structured",
        "label": "artifact",
        "examples": "- OCR'd tables with garbled text\n"
                    "- PDF-extracted invoices with broken formatting\n"
                    "- Scanned forms with partial structure",
    },
]

# (category, sub_type) pairs for synthetic generation
SYNTHETIC_TYPE_PAIRS = [
    ("prose", "plain"),
    ("prose", "markdown"),
    ("prose", "latex"),
    ("code", "python"),
    ("code", "javascript"),
    ("code", "rust"),
    ("code", "sql"),
    ("code", "shell"),
    ("code", "yaml"),
    ("code", "html"),
    ("structured", "csv"),
    ("structured", "tsv"),
    ("structured", "json"),
    ("structured", "jsonl"),
    ("structured", "key_value"),
    ("structured", "log_lines"),
    ("artifact", "pdf_dump"),
    ("artifact", "boilerplate"),
]

# Feature columns output by the classify CLI (excluding line_count)
FEATURE_COLUMNS = [
    "line_length_cv",
    "char_entropy",
    "leading_whitespace_ratio",
    "tab_density",
    "sentence_punctuation_rate",
    "paragraph_break_rate",
    "alpha_ratio",
    "line_uniqueness",
    "short_line_ratio",
    "symbol_ratio",
    "delimiter_consistency",
    "json_brace_depth",
    "key_value_ratio",
    "xml_tag_ratio",
    "log_line_ratio",
    "comment_ratio",
    "numeric_field_ratio",
    "repetitive_structure_score",
]

# All output columns: features + metadata
OUTPUT_COLUMNS = FEATURE_COLUMNS + ["category", "sub_type", "line_count"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_json_array(text: str) -> list:
    """Extract a JSON array from text that may be wrapped in markdown fences."""
    # Try direct parse first
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try to find the first [ ... ] in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Last resort: extract strings one at a time using json.JSONDecoder
    # This handles cases where the array is valid up to a point then breaks
    if start != -1:
        decoder = json.JSONDecoder()
        items = []
        pos = start + 1  # skip the [
        content = text[:end + 1] if end != -1 else text
        while pos < len(content):
            # Skip whitespace and commas
            while pos < len(content) and content[pos] in " \t\n\r,":
                pos += 1
            if pos >= len(content) or content[pos] == "]":
                break
            try:
                item, next_pos = decoder.raw_decode(content, pos)
                if isinstance(item, str):
                    items.append(item)
                pos = next_pos
            except json.JSONDecodeError:
                # Skip to next quote to try the next string
                next_quote = content.find('"', pos + 1)
                if next_quote == -1:
                    break
                pos = next_quote
        if items:
            return items

    raise ValueError(f"Could not extract JSON array from response ({len(text)} chars)")


def map_directory_to_category(dirname: str) -> str:
    """Map a fixture directory name to a training category."""
    return DIRECTORY_TO_CATEGORY.get(dirname, dirname)


def derive_sub_type(filename: str) -> str:
    """Derive a sub_type label from a fixture filename."""
    return Path(filename).stem


def extract_features_via_cli(text: str, classify_bin: str) -> dict:
    """Extract features from a text string using the classify CLI.

    Creates a temporary JSONL file, runs ``classify features``, and parses
    the resulting CSV back into a dict.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.jsonl")
        output_path = os.path.join(tmpdir, "output.csv")

        with open(input_path, "w") as f:
            json.dump({"text": text}, f)
            f.write("\n")

        result = subprocess.run(
            [
                classify_bin,
                "features",
                input_path,
                "--text-field",
                "text",
                "--output",
                output_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"classify features failed (exit {result.returncode}): {result.stderr}"
            )

        with open(output_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)

        # Convert numeric strings to float/int
        features = {}
        for key, val in row.items():
            if key == "line_count":
                features[key] = int(val)
            else:
                features[key] = float(val)
        return features


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate training data for the text classifier."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "fixtures", "synthetic", "perturb", "test-set", "ambiguous-test-set", "golden-train"],
        default="all",
        help="Generation mode (default: all)",
    )
    parser.add_argument(
        "--output",
        default="training/data/",
        help="Output directory (default: training/data/)",
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=50,
        help="Number of synthetic samples per (category, sub_type) pair (default: 50)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (default: $ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Claude model for synthetic generation (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be generated without calling the API (golden-train mode only).",
    )
    return parser


# ---------------------------------------------------------------------------
# Mode: fixtures
# ---------------------------------------------------------------------------


def run_fixtures_mode(
    fixtures_dir: str,
    output_dir: str,
    classify_bin: str = DEFAULT_CLASSIFY_BIN,
) -> str:
    """Extract features from test fixture files and write to CSV.

    Returns the path to the output CSV.
    """
    fixtures_path = Path(fixtures_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fixtures.csv")

    # Collect all fixture files first for progress bar
    all_files = []
    for category_dir in sorted(fixtures_path.iterdir()):
        if not category_dir.is_dir():
            continue
        category = map_directory_to_category(category_dir.name)
        for txt_file in sorted(category_dir.glob("*.txt")):
            all_files.append((category, txt_file))

    rows = []
    for category, txt_file in tqdm(all_files, desc="Extracting fixtures", unit="file"):
        sub_type = derive_sub_type(txt_file.name)
        text = txt_file.read_text(errors="replace")
        try:
            features = extract_features_via_cli(text, classify_bin)
        except RuntimeError as e:
            tqdm.write(f"Warning: skipping {txt_file}: {e}")
            continue
        row = {col: features.get(col, 0.0) for col in FEATURE_COLUMNS}
        row["category"] = category
        row["sub_type"] = sub_type
        row["line_count"] = features.get("line_count", 0)
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Fixtures: wrote {len(rows)} rows to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Mode: synthetic
# ---------------------------------------------------------------------------


PROMPT_TEMPLATE = """\
Generate {n} diverse examples of {sub_type} text content (category: {category}).

Each example should be a realistic text sample (5-40 lines) that a classifier \
would identify as "{category}" with sub-type "{sub_type}".

Vary the style, complexity, and content.

You MUST respond with ONLY a valid JSON array of strings. No markdown fences, \
no explanation, no preamble. The first character of your response must be [ and \
the last character must be ]."""


def run_synthetic_mode(
    output_dir: str,
    classify_bin: str = DEFAULT_CLASSIFY_BIN,
    api_key: str | None = None,
    samples_per_type: int = 50,
    model: str = "claude-sonnet-4-20250514",
) -> str | None:
    """Generate synthetic text via Claude API and extract features.

    Returns the path to the output CSV, or None if no API key is available.
    """
    if not api_key:
        print(
            "Warning: ANTHROPIC_API_KEY not set, skipping synthetic generation.",
            file=sys.stderr,
        )
        return None

    try:
        import anthropic
    except ImportError:
        print(
            "Warning: anthropic package not installed, skipping synthetic generation.",
            file=sys.stderr,
        )
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic.csv")
    client = anthropic.Anthropic(api_key=api_key)

    BATCH_SIZE = 50  # Max samples Claude can fit in one response

    rows = []
    type_bar = tqdm(SYNTHETIC_TYPE_PAIRS, desc="Synthetic types", unit="type")
    for category, sub_type in type_bar:
        type_bar.set_postfix_str(f"{category}/{sub_type}")
        n_batches = max(1, (samples_per_type + BATCH_SIZE - 1) // BATCH_SIZE)
        type_extracted = 0

        for batch_i in range(n_batches):
            batch_n = min(BATCH_SIZE, samples_per_type - batch_i * BATCH_SIZE)
            if batch_n <= 0:
                break
            prompt = PROMPT_TEMPLATE.format(
                n=batch_n, category=category, sub_type=sub_type
            )
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "["},
                    ],
                )
                response_text = "[" + message.content[0].text
                samples = extract_json_array(response_text)
            except Exception as e:
                tqdm.write(
                    f"Warning: failed to generate {category}/{sub_type} batch {batch_i+1}: {e}"
                )
                continue

            for sample in samples:
                if not isinstance(sample, str) or not sample.strip():
                    continue
                try:
                    features = extract_features_via_cli(sample, classify_bin)
                except RuntimeError as e:
                    tqdm.write(f"Warning: feature extraction failed: {e}")
                    continue
                row = {col: features.get(col, 0.0) for col in FEATURE_COLUMNS}
                row["category"] = category
                row["sub_type"] = sub_type
                row["line_count"] = features.get("line_count", 0)
                rows.append(row)
                type_extracted += 1

        tqdm.write(f"  {category}/{sub_type}: {type_extracted}/{samples_per_type} samples extracted")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Synthetic: wrote {len(rows)} rows to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Mode: perturb
# ---------------------------------------------------------------------------


def run_perturb_mode(
    output_dir: str,
    num_perturbations: int = 12,
    noise_std: float = 0.12,
) -> str:
    """Generate perturbation-based hard examples from fixtures.csv.

    Adds Gaussian noise to feature values to create boundary cases.
    Returns the path to the output CSV.
    """
    fixtures_path = os.path.join(output_dir, "fixtures.csv")
    output_path = os.path.join(output_dir, "perturbations.csv")

    with open(fixtures_path) as f:
        reader = csv.DictReader(f)
        fixture_rows = list(reader)

    perturbed_rows = []
    for row in tqdm(fixture_rows, desc="Perturbing fixtures", unit="row"):
        n = random.randint(10, 15)
        for _ in range(n):
            new_row = {}
            for col in OUTPUT_COLUMNS:
                if col in ("category", "sub_type"):
                    new_row[col] = row[col]
                elif col == "line_count":
                    # Perturb line_count but keep it as a non-negative integer
                    val = int(row[col])
                    noise = random.gauss(0, max(1, val * 0.1))
                    new_row[col] = max(0, round(val + noise))
                else:
                    val = float(row[col])
                    noise = random.gauss(0, noise_std)
                    new_row[col] = max(0.0, round(val + noise, 4))
            perturbed_rows.append(new_row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(perturbed_rows)

    print(f"Perturbations: wrote {len(perturbed_rows)} rows to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Mode: all
# ---------------------------------------------------------------------------


def combine_csvs(csv_paths: list[str], output_dir: str) -> str:
    """Combine multiple CSV files into a single combined.csv.

    Returns the path to the combined CSV.
    """
    output_path = os.path.join(output_dir, "combined.csv")
    all_rows = []
    for path in csv_paths:
        if path is None or not Path(path).exists():
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            all_rows.extend(reader)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Combined: wrote {len(all_rows)} rows to {output_path}")
    return output_path


def run_all_mode(
    fixtures_dir: str,
    output_dir: str,
    classify_bin: str,
    api_key: str | None,
    samples_per_type: int,
    model: str = "claude-sonnet-4-20250514",
) -> None:
    """Run all modes and combine results."""
    csv_paths = []

    fixtures_path = run_fixtures_mode(fixtures_dir, output_dir, classify_bin)
    csv_paths.append(fixtures_path)

    synthetic_path = run_synthetic_mode(
        output_dir, classify_bin, api_key, samples_per_type, model
    )
    if synthetic_path:
        csv_paths.append(synthetic_path)

    perturb_path = run_perturb_mode(output_dir)
    csv_paths.append(perturb_path)

    run_test_set_mode(fixtures_dir, output_dir)

    combined_path = combine_csvs(csv_paths, output_dir)
    print(f"\nSummary:")
    for path in csv_paths + [combined_path]:
        if path and Path(path).exists():
            with open(path) as f:
                count = sum(1 for _ in csv.DictReader(f))
            print(f"  {Path(path).name}: {count} rows")


# ---------------------------------------------------------------------------
# Mode: test-set
# ---------------------------------------------------------------------------


def run_test_set_mode(fixtures_dir: str, output_dir: str) -> str:
    """Generate a labeled JSONL test set from fixture files.

    Writes text + ground-truth labels derived from directory names.
    This is for use with ``classify validate --input test_set.jsonl``.

    Returns the path to the output JSONL file.
    """
    fixtures_path = Path(fixtures_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_set.jsonl")

    count = 0
    with open(output_path, "w") as f:
        for category_dir in sorted(fixtures_path.iterdir()):
            if not category_dir.is_dir():
                continue
            category = map_directory_to_category(category_dir.name)
            for txt_file in sorted(category_dir.glob("*.txt")):
                text = txt_file.read_text(errors="replace")
                sub_type = derive_sub_type(txt_file.name)
                record = {
                    "text": text,
                    "label": category,
                    "sub_type": sub_type,
                }
                json.dump(record, f)
                f.write("\n")
                count += 1

    print(f"Test set: wrote {count} samples to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Mode: ambiguous-test-set
# ---------------------------------------------------------------------------

AMBIGUOUS_PROMPT_TEMPLATE = """\
Generate {n} text samples that are deliberately AMBIGUOUS between the categories \
"{cat_a}" and "{cat_b}".

Each sample should be realistic text (5-30 lines) that could plausibly be classified \
as either category. The ground truth label is "{label}" but the text should have \
strong features of both categories.

Examples of ambiguity between these categories:
{examples}

Vary the style and content. Each sample should be ambiguous in a DIFFERENT way.

You MUST respond with ONLY a valid JSON array of strings. No markdown fences, \
no explanation. First character must be [ and last must be ]."""

AMBIGUOUS_PAIRS = [
    {
        "cat_a": "code",
        "cat_b": "structured",
        "label": "code",
        "examples": "- TOML/INI config files (key=value but are config code)\n"
                    "- YAML with data-like content\n"
                    "- .env files with connection strings",
    },
    {
        "cat_a": "code",
        "cat_b": "prose",
        "label": "code",
        "examples": "- Python with extensive docstrings\n"
                    "- Shell scripts with long comment blocks\n"
                    "- SQL with detailed inline comments",
    },
    {
        "cat_a": "prose",
        "cat_b": "code",
        "label": "prose",
        "examples": "- Technical documentation about code\n"
                    "- API reference with code-like terms\n"
                    "- README files describing functions",
    },
    {
        "cat_a": "prose",
        "cat_b": "structured",
        "label": "prose",
        "examples": "- Markdown with tables and lists\n"
                    "- Technical writing with key-value descriptions\n"
                    "- Reports with tabular data embedded in text",
    },
    {
        "cat_a": "structured",
        "cat_b": "code",
        "label": "structured",
        "examples": "- JSON with code-like field names\n"
                    "- CSV with URL columns and special characters\n"
                    "- Log files with structured + freeform fields",
    },
    {
        "cat_a": "artifact",
        "cat_b": "structured",
        "label": "artifact",
        "examples": "- OCR'd tables with garbled text\n"
                    "- PDF-extracted invoices with broken formatting\n"
                    "- Scanned forms with partial structure",
    },
]


def run_ambiguous_test_set_mode(
    output_dir: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    samples_per_pair: int = 17,
) -> str:
    """Generate ambiguous test samples via Claude API.

    Creates samples that sit on the boundary between two categories.
    Returns path to the output JSONL file.
    """
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY required for ambiguous test set generation.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package required.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ambiguous_test_set.jsonl")
    client = anthropic.Anthropic(api_key=api_key)

    count = 0
    with open(output_path, "w") as f:
        for pair in tqdm(AMBIGUOUS_PAIRS, desc="Generating ambiguous pairs", unit="pair"):
            prompt = AMBIGUOUS_PROMPT_TEMPLATE.format(
                n=samples_per_pair,
                cat_a=pair["cat_a"],
                cat_b=pair["cat_b"],
                label=pair["label"],
                examples=pair["examples"],
            )
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "["},
                    ],
                )
                response_text = "[" + message.content[0].text
                samples = extract_json_array(response_text)
            except Exception as e:
                tqdm.write(
                    f"Warning: failed to generate {pair['cat_a']}/{pair['cat_b']}: {e}"
                )
                continue

            for sample in samples:
                if not isinstance(sample, str) or not sample.strip():
                    continue
                record = {
                    "text": sample,
                    "label": pair["label"],
                    "ambiguous_with": pair["cat_a"] if pair["label"] == pair["cat_b"] else pair["cat_b"],
                }
                json.dump(record, f)
                f.write("\n")
                count += 1

            tqdm.write(f"  {pair['cat_a']} vs {pair['cat_b']} (label={pair['label']}): {len(samples)} samples")

    print(f"Ambiguous test set: wrote {count} samples to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Mode: golden-train
# ---------------------------------------------------------------------------


GOLDEN_CLEAR_PROMPT_TEMPLATE = """\
Generate {n} diverse examples of {sub_type} text content (category: {category}) \
about {domain}.

Each example should be a realistic text sample ({lo}-{hi} lines) that a classifier \
would unambiguously identify as "{category}" with sub-type "{sub_type}".

Vary the style, complexity, and content within the domain of {domain}.

You MUST respond with ONLY a valid JSON array of strings. No markdown fences, \
no explanation, no preamble. The first character of your response must be [ and \
the last character must be ]."""


GOLDEN_BOUNDARY_PROMPT_TEMPLATE = """\
Generate {n} text samples about {domain} that are genuinely AMBIGUOUS between \
the categories "{cat_a}" and "{cat_b}".

Each sample should be realistic text ({lo}-{hi} lines) that could plausibly be \
classified as either category. The ground truth label is "{label}" but the text \
should have strong features of both categories.

Examples of ambiguity between these categories:
{examples}

Vary the style and content. Each sample should be ambiguous in a DIFFERENT way.

You MUST respond with ONLY a valid JSON array of strings. No markdown fences, \
no explanation. First character must be [ and last must be ]."""


GOLDEN_BATCH_SIZE = 20


def generate_golden_clear(
    category: str,
    sub_types: list[str],
    count: int,
    domain_seeds: list[str],
    length_buckets: dict[str, tuple[int, int]],
    client=None,
    model: str = "claude-sonnet-4-20250514",
) -> list[dict]:
    """Generate unambiguous training samples for a category.

    For each sub-type under the category, generates ``count / len(sub_types)``
    samples. Rotates domain seeds (each batch of 20 gets a different domain)
    and cycles through length buckets.

    Returns a list of dicts with keys: text, category, sub_type, source.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    results: list[dict] = []
    bucket_names = list(length_buckets.keys())
    per_sub_type = max(1, count // len(sub_types))

    for sub_type in sub_types:
        seed_cycle = itertools.cycle(domain_seeds)
        bucket_cycle = itertools.cycle(bucket_names)
        collected = 0

        while collected < per_sub_type:
            remaining = per_sub_type - collected
            batch_n = min(GOLDEN_BATCH_SIZE, remaining)
            domain = next(seed_cycle)
            bucket = next(bucket_cycle)
            lo, hi = length_buckets[bucket]

            prompt = GOLDEN_CLEAR_PROMPT_TEMPLATE.format(
                n=batch_n,
                category=category,
                sub_type=sub_type,
                domain=domain,
                lo=lo,
                hi=hi,
            )
            try:
                message = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    temperature=0.95,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "["},
                    ],
                )
                response_text = "[" + message.content[0].text
                samples = extract_json_array(response_text)
            except Exception as e:
                tqdm.write(
                    f"Warning: failed to generate {category}/{sub_type}: {e}"
                )
                break

            for sample in samples:
                if collected >= per_sub_type:
                    break
                if not isinstance(sample, str) or not sample.strip():
                    continue
                results.append({
                    "text": sample,
                    "category": category,
                    "sub_type": sub_type,
                    "source": "golden_clear",
                })
                collected += 1

    return results


def generate_golden_boundary(
    pair: dict,
    count: int,
    domain_seeds: list[str],
    length_buckets: dict[str, tuple[int, int]],
    client=None,
    model: str = "claude-sonnet-4-20250514",
) -> list[dict]:
    """Generate boundary training samples for a category pair.

    Labels come from the prompt, not the classifier. Rotates domain seeds
    and length buckets for variety.

    Returns a list of dicts with keys: text, category, sub_type, source.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    results: list[dict] = []
    bucket_names = list(length_buckets.keys())
    seed_cycle = itertools.cycle(domain_seeds)
    bucket_cycle = itertools.cycle(bucket_names)
    collected = 0

    while collected < count:
        remaining = count - collected
        batch_n = min(GOLDEN_BATCH_SIZE, remaining)
        domain = next(seed_cycle)
        bucket = next(bucket_cycle)
        lo, hi = length_buckets[bucket]

        prompt = GOLDEN_BOUNDARY_PROMPT_TEMPLATE.format(
            n=batch_n,
            cat_a=pair["cat_a"],
            cat_b=pair["cat_b"],
            label=pair["label"],
            examples=pair["examples"],
            domain=domain,
            lo=lo,
            hi=hi,
        )
        try:
            message = client.messages.create(
                model=model,
                max_tokens=8192,
                temperature=0.95,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "["},
                ],
            )
            response_text = "[" + message.content[0].text
            samples = extract_json_array(response_text)
        except Exception as e:
            tqdm.write(
                f"Warning: failed to generate boundary {pair['cat_a']}/{pair['cat_b']}: {e}"
            )
            break

        for sample in samples:
            if collected >= count:
                break
            if not isinstance(sample, str) or not sample.strip():
                continue
            results.append({
                "text": sample,
                "category": pair["label"],
                "sub_type": f"boundary_{pair['cat_a']}_{pair['cat_b']}",
                "source": "golden_boundary",
            })
            collected += 1

    return results


def run_golden_train_mode(
    output_dir: str,
    samples_per_type: int = 200,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    dry_run: bool = False,
) -> str | None:
    """Run golden-train generation mode.

    Generates clear samples (per sub-type) and boundary samples (per pair),
    writing results to ``golden_raw.csv``.

    Args:
        output_dir: Directory to write the output CSV.
        samples_per_type: Number of samples per sub-type for clear generation.
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model: Claude model to use.
        dry_run: If True, print summary without calling the API.

    Returns:
        Path to the output CSV, or None if dry run.
    """
    if dry_run:
        _run_golden_train_dry(output_dir, samples_per_type)
        return None

    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY required for golden-train generation.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package required.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "golden_raw.csv")
    client = anthropic.Anthropic(api_key=api_key)

    all_rows: list[dict] = []

    # Generate clear samples
    for category in sorted(VALID_CATEGORIES):
        sub_types = GOLDEN_SUB_TYPES[category]
        print(f"Generating {samples_per_type * len(sub_types)} clear samples for '{category}' "
              f"({samples_per_type} x {len(sub_types)} sub-types)...")
        samples = generate_golden_clear(
            category=category,
            sub_types=sub_types,
            count=samples_per_type * len(sub_types),
            domain_seeds=GOLDEN_DOMAIN_SEEDS,
            length_buckets=GOLDEN_LENGTH_BUCKETS,
            client=client,
            model=model,
        )
        all_rows.extend(samples)
        print(f"  Done: {len(samples)} clear samples for '{category}'")

    # Generate boundary samples
    boundary_per_pair = 4000
    for pair in GOLDEN_BOUNDARY_PAIRS:
        pair_label = f"{pair['cat_a']}_vs_{pair['cat_b']}"
        print(f"Generating {boundary_per_pair} boundary samples for '{pair_label}'...")
        samples = generate_golden_boundary(
            pair=pair,
            count=boundary_per_pair,
            domain_seeds=GOLDEN_DOMAIN_SEEDS,
            length_buckets=GOLDEN_LENGTH_BUCKETS,
            client=client,
            model=model,
        )
        all_rows.extend(samples)
        print(f"  Done: {len(samples)} boundary samples for '{pair_label}'")

    # Write CSV
    golden_columns = ["text", "category", "sub_type", "source"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=golden_columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Golden train: wrote {len(all_rows)} rows to {output_path}")
    return output_path


def _run_golden_train_dry(output_dir: str, samples_per_type: int) -> None:
    """Print a summary of what golden-train would generate."""
    print("=== DRY RUN (golden-train) ===")
    print(f"Output directory: {output_dir}")
    print()

    print("Clear samples:")
    total_clear = 0
    for category in sorted(VALID_CATEGORIES):
        sub_types = GOLDEN_SUB_TYPES[category]
        cat_total = samples_per_type * len(sub_types)
        total_clear += cat_total
        batches = math.ceil(cat_total / GOLDEN_BATCH_SIZE)
        print(f"  - {category}: {len(sub_types)} sub-types x {samples_per_type} = "
              f"{cat_total} samples ({batches} API calls)")
        for st in sub_types:
            per_st = max(1, cat_total // len(sub_types))
            print(f"      {st}: {per_st} samples")
    print(f"  Total clear: {total_clear}")
    print()

    print("Boundary samples:")
    boundary_per_pair = 4000
    total_boundary = boundary_per_pair * len(GOLDEN_BOUNDARY_PAIRS)
    for pair in GOLDEN_BOUNDARY_PAIRS:
        batches = math.ceil(boundary_per_pair / GOLDEN_BATCH_SIZE)
        print(f"  - {pair['cat_a']} vs {pair['cat_b']} (label={pair['label']}): "
              f"{boundary_per_pair} samples ({batches} API calls)")
    print(f"  Total boundary: {total_boundary}")
    print()

    print(f"Grand total: {total_clear + total_boundary} samples")
    print(f"Domain seeds: {len(GOLDEN_DOMAIN_SEEDS)} topics")
    print(f"Length buckets: {', '.join(GOLDEN_LENGTH_BUCKETS.keys())}")
    print("=== END DRY RUN ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    classify_bin = DEFAULT_CLASSIFY_BIN
    if not Path(classify_bin).exists():
        print("Building classify binary...", file=sys.stderr)
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error: cargo build failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        if not Path(classify_bin).exists():
            print(f"Error: classify binary not found at {classify_bin} after build.", file=sys.stderr)
            sys.exit(1)

    fixtures_dir = DEFAULT_FIXTURES_DIR

    if args.mode == "fixtures":
        run_fixtures_mode(fixtures_dir, args.output, classify_bin)
    elif args.mode == "synthetic":
        run_synthetic_mode(args.output, classify_bin, args.api_key, args.samples_per_type, args.model)
    elif args.mode == "perturb":
        run_perturb_mode(args.output)
    elif args.mode == "test-set":
        run_test_set_mode(fixtures_dir, args.output)
    elif args.mode == "ambiguous-test-set":
        run_ambiguous_test_set_mode(args.output, args.api_key, args.model)
    elif args.mode == "golden-train":
        run_golden_train_mode(
            output_dir=args.output,
            samples_per_type=args.samples_per_type,
            api_key=args.api_key,
            model=args.model,
            dry_run=args.dry_run,
        )
    elif args.mode == "all":
        run_all_mode(
            fixtures_dir, args.output, classify_bin, args.api_key, args.samples_per_type, args.model
        )


if __name__ == "__main__":
    main()
