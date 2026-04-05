"""Tiered voting pipeline for label cleanup.

Runs a two-pass voting pipeline over golden_train.parquet:
  Pass 1 (Tier 1): cheap models confirm/reject the existing sub_type label.
  Pass 2 (Tier 2): escalate disagreements to premium models.

Rows where both tiers disagree are either re-labeled or dropped.

Usage via CLI:
    cd training && uv run python -m trainr.core.vote_labels --dry-run
    cd training && uv run python -m trainr.core.vote_labels
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
from trainr.core.voting_pilot import TIER1_ROUTING, check_agreement
from trainr.shared.api import get_anthropic_api_key, get_openrouter_api_key

# ---------------------------------------------------------------------------
# Valid sub_types (the full set a relabel can produce)
# ---------------------------------------------------------------------------

VALID_SUB_TYPES: set[str] = {
    "plain", "markdown", "rst", "latex",
    "python", "javascript", "typescript", "rust", "go", "java", "sql", "shell", "css",
    "yaml", "toml", "ini", "dockerfile", "makefile",
    "html", "xml", "sgml",
    "csv", "tsv", "pipe_table", "fixed_width",
    "json", "jsonl", "key_value", "log_lines",
    "unknown",
}

# ---------------------------------------------------------------------------
# Tier 2 routing table: sub_type -> (model_id, backend)
# Only types where Tier 1 is a cheap model get a separate Tier 2.
# ---------------------------------------------------------------------------

TIER2_ROUTING: dict[str, tuple[str, str]] = {
    "go": ("claude-sonnet-4-6", "anthropic"),
    "html": ("claude-sonnet-4-6", "anthropic"),
    "jsonl": ("claude-sonnet-4-6", "anthropic"),
    "markdown": ("claude-sonnet-4-6", "anthropic"),
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
}

# ---------------------------------------------------------------------------
# Relabel prompt
# ---------------------------------------------------------------------------

RELABEL_PROMPT = """\
What content sub_type best describes this text? Choose exactly one:
plain, markdown, rst, latex, python, javascript, typescript, rust, go, \
java, sql, shell, css, yaml, toml, ini, dockerfile, makefile, html, xml, \
sgml, csv, tsv, pipe_table, fixed_width, json, jsonl, key_value, log_lines, unknown"""

# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def decide_action(
    sub_type: str,
    tier1_agrees: bool,
    tier2_agrees: bool | None,
    tier2_relabel: str | None,
) -> tuple[str, str | None]:
    """Decide what to do with a row based on voting results.

    Returns:
        (action, new_sub_type) where action is "keep", "relabel", or "drop".
    """
    if tier1_agrees:
        return ("keep", None)
    if tier2_agrees is not None:
        if tier2_agrees:
            return ("keep", None)  # Tier 1 was wrong
        elif tier2_relabel and tier2_relabel in VALID_SUB_TYPES:
            return ("relabel", tier2_relabel)
        else:
            return ("drop", None)
    else:
        # No Tier 2 available — Tier 1 disagree is final
        return ("drop", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def has_separate_tier2(sub_type: str) -> bool:
    """Check whether a sub_type has a separate Tier 2 model."""
    return sub_type in TIER2_ROUTING


def parse_relabel_response(response: str) -> str | None:
    """Extract a valid sub_type from a relabel LLM response.

    Handles bare sub_type strings, quoted strings, and sentences
    containing a valid sub_type.

    Returns:
        The extracted sub_type, or None if no valid type found.
    """
    cleaned = response.strip().strip('"').strip("'").strip()
    if cleaned in VALID_SUB_TYPES:
        return cleaned

    # Try to find any valid sub_type token in the response.
    # Check multi-word types first (e.g. pipe_table, log_lines, etc.)
    for st in sorted(VALID_SUB_TYPES, key=len, reverse=True):
        if st in response:
            return st

    return None


def build_voting_log(records: list[dict]) -> pl.DataFrame:
    """Build a voting log DataFrame from a list of record dicts.

    Text is truncated to 200 characters.

    Returns:
        Polars DataFrame with columns: text, original_sub_type,
        tier1_model, tier1_agrees, tier2_model, tier2_agrees,
        new_sub_type, action.
    """
    if not records:
        return pl.DataFrame(
            schema={
                "text": pl.Utf8,
                "original_sub_type": pl.Utf8,
                "tier1_model": pl.Utf8,
                "tier1_agrees": pl.Boolean,
                "tier2_model": pl.Utf8,
                "tier2_agrees": pl.Boolean,
                "new_sub_type": pl.Utf8,
                "action": pl.Utf8,
            },
        )

    truncated = []
    for r in records:
        row = dict(r)
        text = row.get("text", "")
        row["text"] = text[:200]
        truncated.append(row)

    return pl.DataFrame(truncated)


# ---------------------------------------------------------------------------
# Checkpoint I/O
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
# Pass 1 — Tier 1 voting
# ---------------------------------------------------------------------------


def run_tier1(
    df: pl.DataFrame,
    concurrency: int = 20,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """Run Tier 1 voting on the DataFrame.

    For each row, calls the assigned Tier 1 model and checks if it detects
    the labeled sub_type.

    Returns:
        List of result dicts with keys: tier1_model, tier1_agrees, detections.
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
                f"  Resumed Tier 1 from checkpoint: {len(prior)}/{n} done.",
                file=sys.stderr,
            )

    results: list[dict | None] = [None] * n
    for idx, record in prior.items():
        if idx < n:
            results[idx] = record

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
    pbar = tqdm(total=n, initial=completed, desc="Tier 1 voting", file=sys.stderr)

    def _process(
        idx: int, text: str, sub_type: str, model_id: str,
        backend: str, api_key: str,
    ) -> None:
        nonlocal flushed_up_to, completed
        detections = call_llm(text, model=model_id, api_key=api_key, backend=backend)
        det_cols = {f"det_{label}": detections.get(label, 0) for label in DETECTION_LABELS}
        agrees = check_agreement(sub_type, det_cols)
        record = {
            "tier1_model": model_id,
            "tier1_agrees": agrees,
            "detections": det_cols,
        }
        with lock:
            results[idx] = record
            completed += 1
            pbar.update(1)
            if checkpoint_path and completed % CHECKPOINT_INTERVAL == 0:
                flushed_up_to = _flush_checkpoint(checkpoint_path, results, flushed_up_to)

    # Run OpenRouter
    if todo_openrouter:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_process, i, text, st, model, "openrouter", openrouter_key)
                for i, text, st, model in todo_openrouter
            ]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

    # Run Anthropic (lower concurrency)
    if todo_anthropic:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, 5)) as executor:
            futures = [
                executor.submit(_process, i, text, st, model, "anthropic", anthropic_key)
                for i, text, st, model in todo_anthropic
            ]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

    pbar.close()

    if checkpoint_path:
        _flush_checkpoint(checkpoint_path, results, flushed_up_to)

    # Verify all rows completed
    for i, r in enumerate(results):
        assert r is not None, f"Row {i} missing Tier 1 result"

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Pass 2 — Tier 2 escalation
# ---------------------------------------------------------------------------


def run_tier2(
    df: pl.DataFrame,
    tier1_results: list[dict],
    concurrency: int = 20,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """Run Tier 2 voting on rows where Tier 1 disagreed.

    Only escalates rows that have a separate Tier 2 model.
    For rows where Tier 1 IS the premium model, the disagree is final.

    Returns:
        Updated list of result dicts with tier2_model, tier2_agrees,
        tier2_relabel added where applicable.
    """
    openrouter_key = get_openrouter_api_key()
    anthropic_key = get_anthropic_api_key()

    texts = df["text"].to_list()
    sub_types = df["sub_type"].to_list()
    n = len(texts)

    # Load checkpoint for Tier 2
    prior: dict[int, dict] = {}
    if checkpoint_path:
        prior = _load_checkpoint(checkpoint_path)
        if prior:
            print(
                f"  Resumed Tier 2 from checkpoint: {len(prior)}/{n} done.",
                file=sys.stderr,
            )

    # Identify escalation candidates
    escalation_indices = []
    for i in range(n):
        if tier1_results[i]["tier1_agrees"]:
            continue
        st = sub_types[i]
        if has_separate_tier2(st):
            escalation_indices.append(i)

    print(
        f"  Tier 2 escalation: {len(escalation_indices)} rows "
        f"(of {sum(1 for r in tier1_results if not r['tier1_agrees'])} disagreements).",
        file=sys.stderr,
    )

    # Pre-fill results from checkpoint
    tier2_results: dict[int, dict] = {}
    for idx, record in prior.items():
        if idx in set(escalation_indices):
            tier2_results[idx] = record

    todo_openrouter = []
    todo_anthropic = []
    for i in escalation_indices:
        if i in tier2_results:
            continue
        st = sub_types[i]
        model_id, backend = TIER2_ROUTING[st]
        if backend == "anthropic":
            todo_anthropic.append((i, texts[i], st, model_id))
        else:
            todo_openrouter.append((i, texts[i], st, model_id))

    lock = threading.Lock()
    completed = len(tier2_results)
    total = len(escalation_indices)
    flushed_up_to = 0
    pbar = tqdm(total=total, initial=completed, desc="Tier 2 escalation", file=sys.stderr)

    # Collect checkpoint results as a flat list for flushing
    ckpt_results: list[dict | None] = [None] * n
    for idx, rec in tier2_results.items():
        ckpt_results[idx] = rec

    def _process(
        idx: int, text: str, sub_type: str, model_id: str,
        backend: str, api_key: str,
    ) -> None:
        nonlocal flushed_up_to, completed
        detections = call_llm(text, model=model_id, api_key=api_key, backend=backend)
        det_cols = {f"det_{label}": detections.get(label, 0) for label in DETECTION_LABELS}
        agrees = check_agreement(sub_type, det_cols)

        # If both tiers disagree, ask for a relabel
        relabel = None
        if not agrees:
            relabel = _ask_relabel(text, model_id, api_key, backend)

        record = {
            "tier2_model": model_id,
            "tier2_agrees": agrees,
            "tier2_relabel": relabel,
        }
        with lock:
            tier2_results[idx] = record
            ckpt_results[idx] = record
            completed += 1
            pbar.update(1)
            if checkpoint_path and completed % CHECKPOINT_INTERVAL == 0:
                flushed_up_to = _flush_checkpoint(checkpoint_path, ckpt_results, flushed_up_to)

    # Run OpenRouter
    if todo_openrouter:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_process, i, text, st, model, "openrouter", openrouter_key)
                for i, text, st, model in todo_openrouter
            ]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

    # Run Anthropic
    if todo_anthropic:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, 5)) as executor:
            futures = [
                executor.submit(_process, i, text, st, model, "anthropic", anthropic_key)
                for i, text, st, model in todo_anthropic
            ]
            concurrent.futures.wait(futures)
            for f in futures:
                f.result()

    pbar.close()

    if checkpoint_path:
        _flush_checkpoint(checkpoint_path, ckpt_results, flushed_up_to)

    # Merge tier2 results into tier1 results
    merged = []
    for i in range(n):
        record = dict(tier1_results[i])
        if i in tier2_results:
            record.update(tier2_results[i])
        else:
            record["tier2_model"] = None
            record["tier2_agrees"] = None
            record["tier2_relabel"] = None
        merged.append(record)

    return merged


def _ask_relabel(text: str, model_id: str, api_key: str, backend: str) -> str | None:
    """Ask the Tier 2 model to suggest a new sub_type label.

    Uses a dedicated prompt asking the model to pick one sub_type.
    """
    from trainr.core.annotate_detections import _call_anthropic, _call_openrouter

    call_fn = _call_anthropic if backend == "anthropic" else _call_openrouter

    import time
    from trainr.core.annotate_detections import (
        MAX_RETRIES,
        RETRY_BACKOFF,
        _get_retryable_errors,
    )

    retryable = _get_retryable_errors(backend)

    # Build a simple user message with the relabel prompt
    # We call the raw backend function but with our custom prompt
    user_msg = f"{RELABEL_PROMPT}\n\n---\n{text}\n---"

    for attempt in range(MAX_RETRIES):
        try:
            # Use the raw call functions with a simpler prompt
            response = call_fn(text=user_msg, model=model_id, api_key=api_key)
            return parse_relabel_response(response)
        except retryable as e:
            backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(
                f"  Relabel retry {attempt + 1}/{MAX_RETRIES} after "
                f"{type(e).__name__}: waiting {backoff}s",
                file=sys.stderr,
            )
            time.sleep(backoff)

    return None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    df: pl.DataFrame,
    tier1_only: bool = False,
    concurrency: int = 20,
    checkpoint_dir: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the full tiered voting pipeline.

    Args:
        df: Input DataFrame with 'text' and 'sub_type' columns.
        tier1_only: If True, skip Tier 2 escalation.
        concurrency: Max concurrent API workers.
        checkpoint_dir: Directory for checkpoint files (None to disable).

    Returns:
        (cleaned_df, voting_log_df) where cleaned_df has updated labels
        and dropped rows removed, and voting_log_df is the full audit log.
    """
    t1_ckpt = None
    t2_ckpt = None
    if checkpoint_dir:
        t1_ckpt = str(Path(checkpoint_dir) / "tier1.checkpoint.jsonl")
        t2_ckpt = str(Path(checkpoint_dir) / "tier2.checkpoint.jsonl")

    # Pass 1
    tier1_results = run_tier1(df, concurrency=concurrency, checkpoint_path=t1_ckpt)

    # Pass 2
    if tier1_only:
        # Fill in empty tier2 fields
        for r in tier1_results:
            r["tier2_model"] = None
            r["tier2_agrees"] = None
            r["tier2_relabel"] = None
        merged = tier1_results
    else:
        merged = run_tier2(
            df, tier1_results,
            concurrency=concurrency, checkpoint_path=t2_ckpt,
        )

    # Apply decisions
    texts = df["text"].to_list()
    sub_types = df["sub_type"].to_list()
    log_records = []
    keep_indices = []
    new_sub_types = list(sub_types)

    for i in range(len(texts)):
        r = merged[i]
        action, new_st = decide_action(
            sub_types[i],
            r["tier1_agrees"],
            r.get("tier2_agrees"),
            r.get("tier2_relabel"),
        )
        log_records.append({
            "text": texts[i],
            "original_sub_type": sub_types[i],
            "tier1_model": r["tier1_model"],
            "tier1_agrees": r["tier1_agrees"],
            "tier2_model": r.get("tier2_model"),
            "tier2_agrees": r.get("tier2_agrees"),
            "new_sub_type": new_st,
            "action": action,
        })
        if action == "keep":
            keep_indices.append(i)
        elif action == "relabel":
            new_sub_types[i] = new_st
            keep_indices.append(i)
        # "drop" -> skip

    # Build cleaned DataFrame
    cleaned_df = df[keep_indices].with_columns(
        pl.Series("sub_type", [new_sub_types[i] for i in keep_indices]),
    )

    voting_log = build_voting_log(log_records)

    return cleaned_df, voting_log


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def run_dry_run(df: pl.DataFrame) -> None:
    """Print the voting plan without making API calls."""
    print("=== Vote Labels Dry Run ===\n")
    print(f"Total rows: {df.height}\n")

    dist = df.group_by("sub_type").len().sort("len", descending=True)
    print("Distribution by sub_type:")
    print("-" * 70)
    for row in dist.iter_rows(named=True):
        st = row["sub_type"]
        count = row["len"]
        t1_model, t1_backend = TIER1_ROUTING.get(st, ("unknown", "unknown"))
        has_t2 = has_separate_tier2(st)
        t2_info = ""
        if has_t2:
            t2_model, t2_backend = TIER2_ROUTING[st]
            t2_info = f"  T2={t2_model} ({t2_backend})"
        print(
            f"  {st:20s}  n={count:5d}  T1={t1_model} ({t1_backend})"
            f"{t2_info}"
        )

    # Counts by tier
    n_with_t2 = sum(
        row["len"]
        for row in dist.iter_rows(named=True)
        if has_separate_tier2(row["sub_type"])
    )
    print(f"\nRows with separate Tier 2: {n_with_t2}")
    print(f"Rows with Tier 1 only:     {df.height - n_with_t2}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tiered voting pipeline for label cleanup.",
    )
    parser.add_argument(
        "--input", default="data/curated/train/golden_train.parquet",
        help="Path to input Parquet file.",
    )
    parser.add_argument(
        "--output", default="data/curated/train/golden_train.parquet",
        help="Path to output Parquet file (overwrites input).",
    )
    parser.add_argument(
        "--voting-log", default="data/curated/train/voting_log.parquet",
        help="Path to voting log Parquet file.",
    )
    parser.add_argument(
        "--tier1-only", action="store_true",
        help="Skip Tier 2 escalation (for debugging).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Max concurrent API workers (default: 20).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the voting plan without making API calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the tiered voting pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"Reading {args.input}...", file=sys.stderr)
    df = pl.read_parquet(args.input)
    print(f"  {df.height} rows loaded.", file=sys.stderr)

    if args.dry_run:
        run_dry_run(df)
        return

    # Checkpoint directory sits next to the output
    output_path = Path(args.output)
    checkpoint_dir = str(output_path.parent / ".vote_labels_checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    cleaned_df, voting_log = run_pipeline(
        df,
        tier1_only=args.tier1_only,
        concurrency=args.concurrency,
        checkpoint_dir=checkpoint_dir,
    )

    # Summary
    n_kept = voting_log.filter(pl.col("action") == "keep").height
    n_relabeled = voting_log.filter(pl.col("action") == "relabel").height
    n_dropped = voting_log.filter(pl.col("action") == "drop").height
    print(f"\n=== Voting Summary ===", file=sys.stderr)
    print(f"  Kept:      {n_kept}", file=sys.stderr)
    print(f"  Relabeled: {n_relabeled}", file=sys.stderr)
    print(f"  Dropped:   {n_dropped}", file=sys.stderr)
    print(f"  Total out: {cleaned_df.height} (was {df.height})", file=sys.stderr)

    # Save
    print(f"\nWriting {args.voting_log}...", file=sys.stderr)
    voting_log.write_parquet(args.voting_log)

    print(f"Writing {args.output}...", file=sys.stderr)
    cleaned_df.write_parquet(args.output)

    # Clean up checkpoints on success
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.exists():
        for f in ckpt_dir.iterdir():
            f.unlink()
        ckpt_dir.rmdir()
        print("  Checkpoints removed (run complete).", file=sys.stderr)

    print("  Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
