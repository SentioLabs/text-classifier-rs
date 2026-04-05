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
# Per-sub_type model routing
# ---------------------------------------------------------------------------

ROUTING_TABLE: dict[str, tuple[str, str]] = {
    # (model_id, backend)
    "plain": ("openai/gpt-5.4-nano", "openrouter"),
    "css": ("openai/gpt-5.4-nano", "openrouter"),
    "dockerfile": ("openai/gpt-5.4-nano", "openrouter"),
    "java": ("openai/gpt-5.4-nano", "openrouter"),
    "log_lines": ("openai/gpt-5.4-nano", "openrouter"),
    "rust": ("openai/gpt-5.4-nano", "openrouter"),
    "sql": ("openai/gpt-5.4-nano", "openrouter"),
    "csv": ("google/gemini-3.1-flash-lite-preview", "openrouter"),
    "fixed_width": ("google/gemini-3.1-flash-lite-preview", "openrouter"),
    "ini": ("google/gemini-3.1-flash-lite-preview", "openrouter"),
    "typescript": ("google/gemini-3.1-flash-lite-preview", "openrouter"),
    "latex": ("google/gemini-3-flash-preview", "openrouter"),
    "yaml": ("google/gemini-3-flash-preview", "openrouter"),
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
    "json": ("anthropic/claude-haiku-4.5", "anthropic"),
    "makefile": ("anthropic/claude-haiku-4.5", "anthropic"),
    "python": ("anthropic/claude-haiku-4.5", "anthropic"),
    "xml": ("openai/gpt-5.4-mini", "openrouter"),
}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are a multi-label text content classifier. Analyze text and identify ALL \
content types present. A text can contain multiple types (e.g., a markdown \
document with embedded Python code blocks should have both "markdown": 1 and \
"python": 1).

Labels: plain, markdown, rst, latex, python, javascript, typescript, rust, go, \
java, sql, shell, css, yaml, toml, ini, dockerfile, makefile, html, xml, sgml, \
csv, tsv, pipe_table, fixed_width, json, jsonl, key_value, log_lines

For each label, output 1 if that content type is present in the text, or 0 if not.

## Important distinctions
- "jsonl" = JSON Lines: one JSON object per line, no wrapping array. If you see \
multiple JSON objects one-per-line, that is jsonl, not json.
- "rst" = reStructuredText: look for .. directives, :role:`text`, or heading \
underlines with === or ---
- "sgml" = SGML/DTD: look for <!DOCTYPE, <!ENTITY, or SGML declarations
- "fixed_width" = columns aligned by character position with spaces, no delimiter \
characters between fields
- "pipe_table" = fields separated by | pipe characters
- "csv" vs "tsv": csv uses commas, tsv uses tabs

## Rules
- When in doubt, label 1. It is better to include a borderline detection than \
to miss it.
- "plain" = natural language prose without markup. Only set to 1 if there are \
full sentences of readable narrative text.

Return ONLY a JSON object with each label as a key and 0 or 1 as the value. \
No explanation, no markdown formatting.

{"plain": 0, "markdown": 0, "rst": 0, "latex": 0, "python": 0, "javascript": 0, \
"typescript": 0, "rust": 0, "go": 0, "java": 0, "sql": 0, "shell": 0, "css": 0, \
"yaml": 0, "toml": 0, "ini": 0, "dockerfile": 0, "makefile": 0, "html": 0, \
"xml": 0, "sgml": 0, "csv": 0, "tsv": 0, "pipe_table": 0, "fixed_width": 0, \
"json": 0, "jsonl": 0, "key_value": 0, "log_lines": 0}"""


def build_prompt(text: str) -> str:
    """Build the user message for multi-label detection annotation.

    The static label definitions are in SYSTEM_PROMPT (sent as a cached
    system message). This function returns just the user message with the
    text to analyze.
    """
    return f"""\
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

# Shared client instances — avoids creating a new connection pool per call
_openrouter_clients: dict[str, object] = {}  # keyed by api_key
_anthropic_clients: dict[str, object] = {}
_client_lock = threading.Lock()


def _get_openrouter_client(api_key: str):
    """Get or create a shared OpenRouter client for this API key."""
    if api_key not in _openrouter_clients:
        with _client_lock:
            if api_key not in _openrouter_clients:
                if openai is None:
                    raise RuntimeError("openai package not installed. Run: pip install openai")
                _openrouter_clients[api_key] = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                    timeout=30.0,
                )
    return _openrouter_clients[api_key]


def _get_anthropic_client(api_key: str):
    """Get or create a shared Anthropic client for this API key."""
    if api_key not in _anthropic_clients:
        with _client_lock:
            if api_key not in _anthropic_clients:
                try:
                    import anthropic
                except ImportError:
                    raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
                _anthropic_clients[api_key] = anthropic.Anthropic(
                    api_key=api_key, timeout=30.0,
                )
    return _anthropic_clients[api_key]


def _call_openrouter(text: str, model: str, api_key: str) -> str:
    """Call LLM via OpenRouter (OpenAI-compatible API)."""
    client = _get_openrouter_client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(text)},
        ],
        temperature=0.0,
        extra_body={
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
    )
    if not getattr(response, "choices", None):
        return ""
    return response.choices[0].message.content or ""


def _call_anthropic(text: str, model: str, api_key: str) -> str:
    """Call LLM via Anthropic API with native prompt caching."""
    client = _get_anthropic_client(api_key)
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {"role": "user", "content": build_prompt(text)},
        ],
        temperature=0.0,
    )
    return response.content[0].text if response.content else ""


# Transient errors to retry on, by backend
_OPENROUTER_RETRYABLE = None  # lazy-loaded
_ANTHROPIC_RETRYABLE = None


def _get_retryable_errors(backend: str) -> tuple:
    """Get the retryable exception types for the given backend."""
    global _OPENROUTER_RETRYABLE, _ANTHROPIC_RETRYABLE
    if backend == "anthropic":
        if _ANTHROPIC_RETRYABLE is None:
            import anthropic as anth
            _ANTHROPIC_RETRYABLE = (
                anth.APIConnectionError,
                anth.APITimeoutError,
                anth.RateLimitError,
                anth.InternalServerError,
            )
        return _ANTHROPIC_RETRYABLE
    else:
        if _OPENROUTER_RETRYABLE is None:
            _OPENROUTER_RETRYABLE = (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            )
        return _OPENROUTER_RETRYABLE


def call_llm(
    text: str, model: str, api_key: str, backend: str = "openrouter",
) -> dict[str, int]:
    """Call the LLM to annotate a single text sample.

    Args:
        text: The text to classify.
        model: Model ID (OpenRouter format or Anthropic format).
        api_key: API key for the chosen backend.
        backend: "openrouter" or "anthropic".

    Retries on transient errors. Returns a dict of label -> 0/1.
    """
    import time

    call_fn = _call_anthropic if backend == "anthropic" else _call_openrouter
    retryable = _get_retryable_errors(backend)
    last_err: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            content = call_fn(text, model=model, api_key=api_key)
            return parse_response(content)
        except retryable as e:
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
    backend: str = "openrouter",
    routing: bool = False,
    anthropic_api_key: str = "",
) -> pl.DataFrame:
    """Annotate a DataFrame with det_* columns using concurrent LLM calls.

    Supports checkpointing: if checkpoint_path is set, completed annotations
    are flushed periodically and the run can be resumed after interruption.

    Args:
        df: Input DataFrame with a 'text' column.
        model: Model ID (used when routing is False, or as fallback).
        api_key: API key for the chosen backend (or openrouter key when routing).
        concurrency: Number of concurrent workers (ignored per-backend when routing).
        checkpoint_path: Path to checkpoint JSONL file (None to disable).
        backend: "openrouter" or "anthropic" (used when routing is False).
        routing: When True, route each row to a model based on its sub_type.
        anthropic_api_key: API key for Anthropic backend (used when routing).

    Returns:
        DataFrame with additional det_* binary columns.
    """
    texts = df["text"].to_list()
    n = len(texts)

    # Build per-row routing info when routing is enabled
    row_routes: list[tuple[str, str, str]] = []  # (model, backend, api_key)
    if routing:
        sub_types = df["sub_type"].to_list()
        for st in sub_types:
            if st in ROUTING_TABLE:
                r_model, r_backend = ROUTING_TABLE[st]
            else:
                r_model, r_backend = model, "openrouter"
            r_key = anthropic_api_key if r_backend == "anthropic" else api_key
            row_routes.append((r_model, r_backend, r_key))

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
            if routing:
                r_model, r_backend, r_key = row_routes[idx]
                result = call_llm(text, model=r_model, api_key=r_key, backend=r_backend)
            else:
                result = call_llm(text, model=model, api_key=api_key, backend=backend)
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
        group_df = df.filter(pl.col(group_col) == g if g is not None else pl.col(group_col).is_null())
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
        "--model", default=None,
        help="Model ID. Defaults: openrouter=gpt-5.4-nano, anthropic=claude-haiku-4-5-20251001.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20, help="Number of concurrent workers (default: 20).",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Stratified sample N rows before annotating (0 = annotate all).",
    )
    parser.add_argument(
        "--backend", default="openrouter", choices=["openrouter", "anthropic"],
        help="API backend (default: openrouter).",
    )
    parser.add_argument(
        "--routing", action="store_true",
        help="Enable per-sub_type model routing (uses ROUTING_TABLE).",
    )
    return parser


BACKEND_DEFAULT_MODELS = {
    "openrouter": DEFAULT_MODEL,
    "anthropic": "claude-haiku-4-5-20251001",
}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the annotation pipeline."""
    from pathlib import Path

    parser = build_parser()
    args = parser.parse_args(argv)

    backend = args.backend
    model = args.model or BACKEND_DEFAULT_MODELS.get(backend, DEFAULT_MODEL)

    # Resolve API keys based on backend and routing mode
    anthropic_key = ""
    if backend == "anthropic" and not args.routing:
        from trainr.shared.api import get_anthropic_api_key
        api_key = get_anthropic_api_key()
    elif args.routing:
        from trainr.shared.api import get_anthropic_api_key
        api_key = get_openrouter_api_key()
        anthropic_key = get_anthropic_api_key()
    else:
        api_key = get_openrouter_api_key()

    if args.routing:
        print(f"Routing: enabled (per-sub_type model selection)", file=sys.stderr)
    else:
        print(f"Backend: {backend}, Model: {model}", file=sys.stderr)
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
        model=model,
        api_key=api_key,
        concurrency=args.concurrency,
        checkpoint_path=checkpoint_path,
        backend=backend,
        routing=args.routing,
        anthropic_api_key=anthropic_key,
    )

    print(f"Writing {args.output}...", file=sys.stderr)
    result_df.write_parquet(args.output)

    # Clean up checkpoint on success
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        ckpt.unlink()
        print("  Checkpoint removed (run complete).", file=sys.stderr)

    print(f"  Done. {len(result_df)} samples with {len(DETECTION_LABELS)} detection columns.", file=sys.stderr)
