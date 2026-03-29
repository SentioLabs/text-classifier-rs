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
import json
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

    # Last resort: repair broken JSON by extracting quoted strings
    # This handles cases where generated text contains unescaped characters
    if start != -1 and end != -1:
        items = []
        in_string = False
        escape = False
        current = []
        for ch in text[start + 1 : end]:
            if escape:
                current.append(ch)
                escape = False
            elif ch == "\\":
                current.append(ch)
                escape = True
            elif ch == '"' and not in_string:
                in_string = True
                current = []
            elif ch == '"' and in_string:
                in_string = False
                items.append("".join(current))
            elif in_string:
                current.append(ch)
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
        choices=["all", "fixtures", "synthetic", "perturb", "test-set"],
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
        default="claude-sonnet-4-5-20241022",
        help="Claude model for synthetic generation (default: claude-sonnet-4-5-20241022)",
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

Vary the style, complexity, and content. Respond with ONLY a JSON array of strings."""


def run_synthetic_mode(
    output_dir: str,
    classify_bin: str = DEFAULT_CLASSIFY_BIN,
    api_key: str | None = None,
    samples_per_type: int = 50,
    model: str = "claude-sonnet-4-5-20241022",
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
    model: str = "claude-sonnet-4-5-20241022",
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
    elif args.mode == "all":
        run_all_mode(
            fixtures_dir, args.output, classify_bin, args.api_key, args.samples_per_type, args.model
        )


if __name__ == "__main__":
    main()
