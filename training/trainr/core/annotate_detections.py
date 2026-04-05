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


MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 15]


def call_llm(text: str, model: str, api_key: str) -> dict[str, int]:
    """Call the LLM to annotate a single text sample.

    Retries on transient errors (connection, rate-limit, server errors).
    Returns a dict of label -> 0/1.
    """
    import time

    if openai is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=30.0,
    )

    prompt = build_prompt(text)
    last_err: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            return parse_response(content)
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as e:
            last_err = e
            backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(
                f"  Retry {attempt + 1}/{MAX_RETRIES} after {type(e).__name__}: "
                f"waiting {backoff}s",
                file=sys.stderr,
            )
            time.sleep(backoff)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_err}") from last_err


# ---------------------------------------------------------------------------
# DataFrame annotation
# ---------------------------------------------------------------------------


CHECKPOINT_INTERVAL = 500
"""Flush checkpoint every N completed annotations."""


def _load_checkpoint(path: str) -> dict[int, dict[str, int]]:
    """Load existing checkpoint JSONL. Returns {row_index: annotation}."""
    result: dict[int, dict[str, int]] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                result[record["idx"]] = record["labels"]
    except FileNotFoundError:
        pass
    return result


def _flush_checkpoint(
    path: str,
    annotations: list[dict[str, int] | None],
    flushed_up_to: int,
) -> int:
    """Append newly completed annotations to the checkpoint file.

    Returns the new flushed_up_to watermark.
    """
    with open(path, "a") as f:
        for i in range(flushed_up_to, len(annotations)):
            if annotations[i] is not None:
                f.write(json.dumps({"idx": i, "labels": annotations[i]}) + "\n")
            else:
                break
            flushed_up_to = i + 1
    return flushed_up_to


def annotate_dataframe(
    df: pl.DataFrame,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    concurrency: int = 20,
    checkpoint_path: str | None = None,
) -> pl.DataFrame:
    """Annotate a DataFrame with det_* columns using concurrent LLM calls.

    Supports checkpointing: if checkpoint_path is set, completed annotations
    are flushed periodically and the run can be resumed after interruption.

    Args:
        df: Input DataFrame with a 'text' column.
        model: OpenRouter model ID.
        api_key: OpenRouter API key.
        concurrency: Number of concurrent workers.
        checkpoint_path: Path to checkpoint JSONL file (None to disable).

    Returns:
        DataFrame with additional det_* binary columns.
    """
    texts = df["text"].to_list()
    n = len(texts)

    # Load checkpoint if available
    prior: dict[int, dict[str, int]] = {}
    if checkpoint_path:
        prior = _load_checkpoint(checkpoint_path)
        if prior:
            print(
                f"  Resumed from checkpoint: {len(prior)}/{n} already annotated.",
                file=sys.stderr,
            )

    # Pre-allocate results, filling in checkpoint data
    annotations: list[dict[str, int] | None] = [None] * n
    for idx, labels in prior.items():
        if idx < n:
            annotations[idx] = labels

    # Identify remaining work
    todo = [(i, texts[i]) for i in range(n) if annotations[i] is None]

    if not todo:
        print("  All rows already annotated from checkpoint.", file=sys.stderr)
    else:
        lock = threading.Lock()
        completed_count = n - len(todo)
        flushed_up_to = 0
        pbar = tqdm(
            total=n, initial=completed_count, desc="Annotating", file=sys.stderr,
        )

        def _annotate(idx: int, text: str) -> None:
            nonlocal flushed_up_to, completed_count
            result = call_llm(text, model=model, api_key=api_key)
            with lock:
                annotations[idx] = result
                completed_count += 1
                pbar.update(1)
                # Periodic checkpoint flush
                if (
                    checkpoint_path
                    and completed_count % CHECKPOINT_INTERVAL == 0
                ):
                    flushed_up_to = _flush_checkpoint(
                        checkpoint_path, annotations, flushed_up_to,
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_annotate, i, t) for i, t in todo]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

        pbar.close()

        # Final checkpoint flush
        if checkpoint_path:
            _flush_checkpoint(checkpoint_path, annotations, flushed_up_to)

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
    from pathlib import Path

    parser = build_parser()
    args = parser.parse_args(argv)

    api_key = get_openrouter_api_key()

    print(f"Reading {args.input}...", file=sys.stderr)
    df = pl.read_parquet(args.input)
    print(f"  {len(df)} samples loaded.", file=sys.stderr)

    if args.sample > 0 and args.sample < len(df):
        df = stratified_sample(df, n=args.sample)
        print(f"  Sampled {len(df)} rows (stratified by sub_type).", file=sys.stderr)

    # Checkpoint file sits next to the output: output.parquet -> output.checkpoint.jsonl
    output_path = Path(args.output)
    checkpoint_path = str(output_path.with_suffix(".checkpoint.jsonl"))

    result_df = annotate_dataframe(
        df,
        model=args.model,
        api_key=api_key,
        concurrency=args.concurrency,
        checkpoint_path=checkpoint_path,
    )

    print(f"Writing {args.output}...", file=sys.stderr)
    result_df.write_parquet(args.output)

    # Clean up checkpoint on success
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        ckpt.unlink()
        print("  Checkpoint removed (run complete).", file=sys.stderr)

    print(f"  Done. {len(result_df)} samples with {len(DETECTION_LABELS)} detection columns.", file=sys.stderr)
