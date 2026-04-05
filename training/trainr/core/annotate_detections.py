"""LLM-based multi-label sub-type annotation for training data.

Uses OpenRouter API to generate binary detection labels for each content
sub-type present in a text sample. Output Parquet gains det_* columns.

Usage via trainr CLI:
    trainr data annotate-detections --input train.parquet --output annotated.parquet
"""

import argparse
import concurrent.futures
import json
import re
import sys
import threading

import polars as pl
from tqdm import tqdm

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

from trainr.shared.api import get_openrouter_api_key

# ---------------------------------------------------------------------------
# Detection labels — matches all ContentSubType labels from the Rust side
# ---------------------------------------------------------------------------

DETECTION_LABELS: list[str] = [
    "plain", "markdown", "rst", "latex",
    "python", "javascript", "typescript", "rust", "go", "java", "sql", "shell", "css",
    "yaml", "toml", "ini", "dockerfile", "makefile",
    "html", "xml", "sgml",
    "csv", "tsv", "pipe_table", "fixed_width",
    "json", "jsonl", "key_value", "log_lines",
]

# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-5.4-nano"

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(text: str) -> str:
    """Build the LLM prompt for multi-label detection annotation.

    Asks the model to identify ALL content types present in the text and
    return a JSON object with label: 0 or 1 for each detection label.
    """
    labels_str = ", ".join(f'"{label}"' for label in DETECTION_LABELS)
    return f"""\
Analyze the following text and identify ALL content types present.
For each label, output 1 if that content type is present in the text, or 0 if it is not.

Labels: [{labels_str}]

Return ONLY a JSON object with each label as a key and 0 or 1 as the value.
Do not include any other text, explanation, or markdown formatting.

Example output format:
{{"plain": 0, "markdown": 1, "python": 1, "javascript": 0, ...}}

Text to analyze:
---
{text}
---"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_response(response: str) -> dict[str, int]:
    """Parse LLM response into a dict of label -> 0/1.

    Handles:
    - Valid JSON with all labels
    - JSON wrapped in markdown code blocks
    - Missing labels (default to 0)
    - Non-binary values (clamped to 0 or 1)
    - Malformed JSON (returns all zeros)
    """
    defaults = {label: 0 for label in DETECTION_LABELS}

    # Strip markdown code blocks if present
    cleaned = response.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    # Try to parse JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        obj_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                return defaults
        else:
            return defaults

    if not isinstance(data, dict):
        return defaults

    # Merge with defaults, clamping values to 0 or 1
    result = dict(defaults)
    for label in DETECTION_LABELS:
        if label in data:
            try:
                val = int(data[label])
                result[label] = 1 if val > 0 else 0
            except (ValueError, TypeError):
                result[label] = 0

    return result


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(text: str, model: str, api_key: str) -> dict[str, int]:
    """Call the LLM to annotate a single text sample.

    Returns a dict of label -> 0/1.
    """
    if openai is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = build_prompt(text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    content = response.choices[0].message.content or ""
    return parse_response(content)


# ---------------------------------------------------------------------------
# DataFrame annotation
# ---------------------------------------------------------------------------


def annotate_dataframe(
    df: pl.DataFrame,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    concurrency: int = 20,
) -> pl.DataFrame:
    """Annotate a DataFrame with det_* columns using concurrent LLM calls.

    Args:
        df: Input DataFrame with a 'text' column.
        model: OpenRouter model ID.
        api_key: OpenRouter API key.
        concurrency: Number of concurrent workers.

    Returns:
        DataFrame with additional det_* binary columns.
    """
    texts = df["text"].to_list()
    n = len(texts)

    # Pre-allocate results by index to preserve order
    annotations: list[dict[str, int] | None] = [None] * n
    lock = threading.Lock()
    pbar = tqdm(total=n, desc="Annotating", file=sys.stderr)

    def _annotate(idx: int, text: str) -> None:
        result = call_llm(text, model=model, api_key=api_key)
        with lock:
            annotations[idx] = result
            pbar.update(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_annotate, i, t) for i, t in enumerate(texts)]
        concurrent.futures.wait(futures)
        # Raise any exceptions from workers
        for f in futures:
            f.result()

    pbar.close()

    # Build columns from annotations
    det_columns: dict[str, list[int]] = {
        f"det_{label}": [] for label in DETECTION_LABELS
    }
    for ann in annotations:
        assert ann is not None
        for label in DETECTION_LABELS:
            det_columns[f"det_{label}"].append(ann.get(label, 0))

    # Create a DataFrame from detection columns and concatenate
    det_df = pl.DataFrame(det_columns)
    return pl.concat([df, det_df], how="horizontal")


def stratified_sample(
    df: pl.DataFrame,
    n: int = 10000,
    min_per_group: int = 50,
    group_col: str = "sub_type",
    seed: int = 42,
) -> pl.DataFrame:
    """Stratified sample with a floor per sub-type group.

    Ensures rare sub-types get at least min_per_group samples.
    Remaining budget is distributed proportionally.
    """
    counts = df.group_by(group_col).len().sort("len")
    total = len(df)
    groups = counts[group_col].to_list()
    group_counts = dict(zip(counts[group_col].to_list(), counts["len"].to_list()))

    # Calculate per-group allocation
    allocations: dict[str, int] = {}
    n_groups = len(groups)

    # Clamp floor so guaranteed minimums can't exceed total budget
    effective_floor = min(min_per_group, n // max(n_groups, 1))

    budget = n
    # First pass: guarantee floor
    for g in groups:
        alloc = min(effective_floor, group_counts[g])
        allocations[g] = alloc
        budget -= alloc

    # Second pass: distribute remaining proportionally
    remaining_groups = [g for g in groups if group_counts[g] > allocations[g]]
    remaining_total = sum(group_counts[g] - allocations[g] for g in remaining_groups)
    if budget > 0 and remaining_total > 0:
        for g in remaining_groups:
            extra = int(budget * (group_counts[g] - allocations[g]) / remaining_total)
            allocations[g] = min(allocations[g] + extra, group_counts[g])

    # Sample from each group
    sampled = []
    for g in groups:
        group_df = df.filter(pl.col(group_col) == g)
        sample_n = min(allocations.get(g, min_per_group), len(group_df))
        sampled.append(group_df.sample(n=sample_n, seed=seed))

    return pl.concat(sampled)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate training data with multi-label content detections via LLM.",
    )
    parser.add_argument(
        "--input", required=True, help="Path to input Parquet file.",
    )
    parser.add_argument(
        "--output", required=True, help="Path to output Parquet file.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"OpenRouter model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Number of concurrent workers (default: 20).",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Stratified sample N rows before annotating (0 = annotate all).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the annotation pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    api_key = get_openrouter_api_key()

    print(f"Reading {args.input}...", file=sys.stderr)
    df = pl.read_parquet(args.input)
    print(f"  {len(df)} samples loaded.", file=sys.stderr)

    if args.sample > 0 and args.sample < len(df):
        df = stratified_sample(df, n=args.sample)
        print(f"  Sampled {len(df)} rows (stratified by sub_type).", file=sys.stderr)

    result_df = annotate_dataframe(
        df, model=args.model, api_key=api_key, concurrency=args.concurrency,
    )

    print(f"Writing {args.output}...", file=sys.stderr)
    result_df.write_parquet(args.output)
    print(f"  Done. {len(result_df)} samples with {len(DETECTION_LABELS)} detection columns.", file=sys.stderr)
