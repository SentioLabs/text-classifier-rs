"""Tier 1 voting pilot — measure agreement/escalation rates on a 5K sample.

Runs Tier 1 routing models on a stratified sample from golden_train.parquet
to measure per-sub_type agreement and projected escalation rates BEFORE
committing to full-scale voting.

Usage via CLI:
    cd training && uv run python -m trainr.core.voting_pilot --dry-run
    cd training && uv run python -m trainr.core.voting_pilot
"""

import argparse
import concurrent.futures
import json
import sys
import threading
from pathlib import Path

import polars as pl
from tqdm import tqdm

from trainr.core.annotate_detections import (
    DETECTION_LABELS,
    call_llm,
    parse_response,
)
from trainr.shared.api import get_anthropic_api_key, get_openrouter_api_key

# ---------------------------------------------------------------------------
# Tier 1 routing table: sub_type -> (model_id, backend)
# ---------------------------------------------------------------------------

TIER1_ROUTING: dict[str, tuple[str, str]] = {
    # cheap nano models for easy types
    "plain": ("openai/gpt-5.4-nano", "openrouter"),
    "css": ("openai/gpt-5.4-nano", "openrouter"),
    "dockerfile": ("openai/gpt-5.4-nano", "openrouter"),
    "java": ("openai/gpt-5.4-nano", "openrouter"),
    "log_lines": ("openai/gpt-5.4-nano", "openrouter"),
    "rust": ("openai/gpt-5.4-nano", "openrouter"),
    "sql": ("openai/gpt-5.4-nano", "openrouter"),
    # flash-lite for structured
    "csv": ("google/gemini-3.1-flash-lite-preview", "openrouter"),
    "fixed_width": ("google/gemini-3.1-flash-lite-preview", "openrouter"),
    "ini": ("google/gemini-3.1-flash-lite-preview", "openrouter"),
    "typescript": ("google/gemini-3.1-flash-lite-preview", "openrouter"),
    # gem3-flash
    "latex": ("google/gemini-3-flash-preview", "openrouter"),
    "yaml": ("google/gemini-3-flash-preview", "openrouter"),
    # Premium models (Tier 1 for these types)
    "markdown": ("claude-sonnet-4-6", "anthropic"),
    "go": ("claude-sonnet-4-6", "anthropic"),
    "html": ("claude-sonnet-4-6", "anthropic"),
    "jsonl": ("claude-sonnet-4-6", "anthropic"),
    "shell": ("claude-sonnet-4-6", "anthropic"),
    "javascript": ("openai/gpt-5.4", "openrouter"),
    "key_value": ("openai/gpt-5.4", "openrouter"),
    "pipe_table": ("openai/gpt-5.4", "openrouter"),
    "rst": ("openai/gpt-5.4", "openrouter"),
    "sgml": ("openai/gpt-5.4", "openrouter"),
    "toml": ("openai/gpt-5.4", "openrouter"),
    "tsv": ("openai/gpt-5.4", "openrouter"),
    "json": ("anthropic/claude-haiku-4.5", "openrouter"),
    "makefile": ("anthropic/claude-haiku-4.5", "openrouter"),
    "python": ("anthropic/claude-haiku-4.5", "openrouter"),
    "xml": ("openai/gpt-5.4-mini", "openrouter"),
    "unknown": ("openai/gpt-5.4-nano", "openrouter"),
}

# ---------------------------------------------------------------------------
# Checkpoint I/O (reuses pattern from annotate_detections)
# ---------------------------------------------------------------------------

CHECKPOINT_INTERVAL = 200


def _load_checkpoint(path: str) -> dict[int, dict]:
    """Load existing checkpoint JSONL. Returns {row_index: record}."""
    result: dict[int, dict] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                result[record["idx"]] = record
    except FileNotFoundError:
        pass
    return result


def _flush_checkpoint(
    path: str,
    results: list[dict | None],
    flushed_up_to: int,
) -> int:
    """Append newly completed results to the checkpoint file."""
    with open(path, "a") as f:
        for i in range(flushed_up_to, len(results)):
            if results[i] is not None:
                f.write(json.dumps({"idx": i, **results[i]}) + "\n")
            else:
                break
            flushed_up_to = i + 1
    return flushed_up_to


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    df: pl.DataFrame, n: int = 5000, seed: int = 42,
) -> pl.DataFrame:
    """Take a stratified sample ensuring representation of all sub_types."""
    groups = df.group_by("sub_type")
    n_groups = df["sub_type"].n_unique()
    per_group = max(1, n // n_groups)
    samples = []
    for (sub_type,), group_df in groups:
        k = min(per_group, group_df.height)
        samples.append(group_df.sample(n=k, seed=seed))
    result = pl.concat(samples)
    # If we haven't reached n, add more from largest groups
    if result.height < n:
        remaining = n - result.height
        sampled_texts = result["text"].implode()
        leftover = df.filter(~pl.col("text").is_in(sampled_texts))
        extra = leftover.sample(n=min(remaining, leftover.height), seed=seed)
        result = pl.concat([result, extra])
    return result


# ---------------------------------------------------------------------------
# Agreement check
# ---------------------------------------------------------------------------


def check_agreement(sub_type: str, detections: dict[str, int]) -> bool:
    """Check if the model detects the labeled sub_type.

    Args:
        sub_type: The ground-truth sub_type label for the row.
        detections: Dict of det_* keys to 0/1 values from parse_response.

    Returns:
        True if det_{sub_type} == 1, False otherwise.
    """
    key = f"det_{sub_type}"
    return detections.get(key, 0) == 1


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def compute_summary(results_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-sub_type agreement and escalation rates.

    Args:
        results_df: DataFrame with 'sub_type' and 'tier1_agrees' columns.

    Returns:
        Summary DataFrame with sub_type, count, agreement_rate, escalation_rate.
    """
    return (
        results_df
        .group_by("sub_type")
        .agg(
            pl.len().alias("count"),
            pl.col("tier1_agrees").mean().alias("agreement_rate"),
            (1.0 - pl.col("tier1_agrees").mean()).alias("escalation_rate"),
        )
        .sort("escalation_rate", descending=True)
    )


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def run_dry_run(df: pl.DataFrame) -> None:
    """Print the plan and sample distribution without making API calls."""
    print("=== Voting Pilot Dry Run ===\n")
    print(f"Total rows in sample: {df.height}\n")

    dist = (
        df.group_by("sub_type")
        .len()
        .sort("len", descending=True)
    )
    print("Sample distribution by sub_type:")
    print("-" * 50)
    for row in dist.iter_rows(named=True):
        st = row["sub_type"]
        count = row["len"]
        model_id, backend = TIER1_ROUTING.get(st, ("unknown", "unknown"))
        print(f"  {st:20s}  n={count:5d}  model={model_id}  backend={backend}")

    print()
    # Count by backend
    backend_counts: dict[str, int] = {}
    for row in dist.iter_rows(named=True):
        st = row["sub_type"]
        _, backend = TIER1_ROUTING.get(st, ("unknown", "unknown"))
        backend_counts[backend] = backend_counts.get(backend, 0) + row["len"]
    print("Calls by backend:")
    for backend, count in sorted(backend_counts.items()):
        print(f"  {backend}: {count}")
    print()


# ---------------------------------------------------------------------------
# Pilot runner
# ---------------------------------------------------------------------------


def run_pilot(
    df: pl.DataFrame,
    concurrency_openrouter: int = 20,
    concurrency_anthropic: int = 5,
    checkpoint_path: str | None = None,
) -> pl.DataFrame:
    """Run the Tier 1 voting pilot on the given DataFrame.

    For each row, calls the assigned Tier 1 model and checks if it detects
    the labeled sub_type.

    Args:
        df: Input DataFrame with 'text' and 'sub_type' columns.
        concurrency_openrouter: Max concurrent workers for OpenRouter calls.
        concurrency_anthropic: Max concurrent workers for Anthropic calls.
        checkpoint_path: Path to checkpoint JSONL (None to disable).

    Returns:
        DataFrame with original columns plus tier1_model, tier1_agrees,
        and all det_* columns.
    """
    openrouter_key = get_openrouter_api_key()
    anthropic_key = get_anthropic_api_key()

    texts = df["text"].to_list()
    sub_types = df["sub_type"].to_list()
    n = len(texts)

    # Load checkpoint
    prior: dict[int, dict] = {}
    if checkpoint_path:
        prior = _load_checkpoint(checkpoint_path)
        if prior:
            print(
                f"  Resumed from checkpoint: {len(prior)}/{n} already done.",
                file=sys.stderr,
            )

    # Pre-allocate results
    results: list[dict | None] = [None] * n
    for idx, record in prior.items():
        if idx < n:
            results[idx] = record

    # Split work by backend
    todo_openrouter = []
    todo_anthropic = []
    for i in range(n):
        if results[i] is not None:
            continue
        st = sub_types[i]
        model_id, backend = TIER1_ROUTING.get(st, ("openai/gpt-5.4-nano", "openrouter"))
        if backend == "anthropic":
            todo_anthropic.append((i, texts[i], st, model_id))
        else:
            todo_openrouter.append((i, texts[i], st, model_id))

    lock = threading.Lock()
    completed = n - len(todo_openrouter) - len(todo_anthropic)
    flushed_up_to = 0
    pbar = tqdm(total=n, initial=completed, desc="Voting pilot", file=sys.stderr)

    def _process(idx: int, text: str, sub_type: str, model_id: str, backend: str, api_key: str) -> None:
        nonlocal flushed_up_to, completed
        detections = call_llm(text, model=model_id, api_key=api_key, backend=backend)
        det_cols = {f"det_{label}": detections.get(label, 0) for label in DETECTION_LABELS}
        agrees = check_agreement(sub_type, det_cols)
        record = {
            "tier1_model": model_id,
            "tier1_agrees": agrees,
            **det_cols,
        }
        with lock:
            results[idx] = record
            completed += 1
            pbar.update(1)
            if checkpoint_path and completed % CHECKPOINT_INTERVAL == 0:
                flushed_up_to = _flush_checkpoint(checkpoint_path, results, flushed_up_to)

    # Run OpenRouter calls
    if todo_openrouter:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_openrouter) as executor:
            futures = [
                executor.submit(_process, i, text, st, model, "openrouter", openrouter_key)
                for i, text, st, model in todo_openrouter
            ]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

    # Run Anthropic calls
    if todo_anthropic:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_anthropic) as executor:
            futures = [
                executor.submit(_process, i, text, st, model, "anthropic", anthropic_key)
                for i, text, st, model in todo_anthropic
            ]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

    pbar.close()

    # Final checkpoint flush
    if checkpoint_path:
        _flush_checkpoint(checkpoint_path, results, flushed_up_to)

    # Build output columns
    tier1_models = []
    tier1_agrees = []
    det_columns: dict[str, list[int]] = {f"det_{label}": [] for label in DETECTION_LABELS}

    for rec in results:
        assert rec is not None, "All rows should have results"
        tier1_models.append(rec["tier1_model"])
        tier1_agrees.append(rec["tier1_agrees"])
        for label in DETECTION_LABELS:
            det_columns[f"det_{label}"].append(rec.get(f"det_{label}", 0))

    result_df = pl.concat([
        df,
        pl.DataFrame({
            "tier1_model": tier1_models,
            "tier1_agrees": tier1_agrees,
            **det_columns,
        }),
    ], how="horizontal")

    return result_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tier 1 voting pilot — measure agreement/escalation rates on a stratified sample.",
    )
    parser.add_argument(
        "--input", default="golden_train.parquet",
        help="Path to input Parquet file (default: golden_train.parquet).",
    )
    parser.add_argument(
        "--output", default="voting_pilot_results.parquet",
        help="Path to output Parquet file (default: voting_pilot_results.parquet).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=5000,
        help="Number of rows to sample (default: 5000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Max concurrent workers for OpenRouter (default: 20). Anthropic uses 5.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just show the plan and sample distribution — no API calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the voting pilot."""
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"Reading {args.input}...", file=sys.stderr)
    df = pl.read_parquet(args.input)
    print(f"  {df.height} rows loaded.", file=sys.stderr)

    # Sample
    if args.sample_size < df.height:
        sample_df = stratified_sample(df, n=args.sample_size, seed=args.seed)
        print(f"  Sampled {sample_df.height} rows (stratified by sub_type).", file=sys.stderr)
    else:
        sample_df = df

    if args.dry_run:
        run_dry_run(sample_df)
        return

    # Checkpoint file sits next to the output
    output_path = Path(args.output)
    checkpoint_path = str(output_path.with_suffix(".checkpoint.jsonl"))

    result_df = run_pilot(
        sample_df,
        concurrency_openrouter=args.concurrency,
        concurrency_anthropic=5,
        checkpoint_path=checkpoint_path,
    )

    # Print summary
    summary = compute_summary(result_df)
    print("\n=== Voting Pilot Results ===\n", file=sys.stderr)
    print(summary.to_pandas().to_string(index=False), file=sys.stderr)

    overall_agree = result_df["tier1_agrees"].mean()
    print(f"\nOverall agreement: {overall_agree:.1%}", file=sys.stderr)
    print(f"Projected escalation rate: {1.0 - overall_agree:.1%}", file=sys.stderr)

    # Save
    print(f"\nWriting {args.output}...", file=sys.stderr)
    result_df.write_parquet(args.output)

    # Clean up checkpoint on success
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        ckpt.unlink()
        print("  Checkpoint removed (run complete).", file=sys.stderr)

    print(f"  Done. {result_df.height} rows with results.", file=sys.stderr)


if __name__ == "__main__":
    main()
