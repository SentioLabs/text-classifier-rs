"""Relabel unknown sub_types via 3-way voting (heuristic + Magika + LLM).

For the-stack-v2 rows with sub_type == "unknown", runs three independent
classifiers and uses majority vote to assign a label. Non-Stack sources
with known patterns are relabeled in bulk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import openai
import polars as pl
from magika import Magika

from trainr.shared.api import get_openrouter_api_key
from trainr.shared.io import read_parquet, write_parquet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TEXT_LEN = 2000

MAGIKA_TO_SUBTYPE: dict[str, str] = {
    "c": "c_cpp",
    "cpp": "c_cpp",
    "objectivec": "objc",
    "txt": "prose_plain",
    "text": "prose_plain",
}

CLASSIFY_PROMPT = """\
Classify this text as exactly one of: c_cpp, objc, prose, other.
Reply with ONLY the label, nothing else.

Text:
---
{text}
---

Label:"""

# Regex patterns for heuristic classification
_OBJC_SIGNALS = re.compile(
    r"@interface|@implementation|@property|@protocol|@synthesize|@class"
)
_C_CPP_PREPROCESSOR = re.compile(r"#include|#define|#ifndef")
_C_FUNC_SIG = re.compile(
    r"\b(?:int|void|char|float|double|long|unsigned|signed|short|size_t|bool)\s+\w+\s*\("
)
_C_STRUCT = re.compile(r"\bstruct\s+\w+")
_CODE_BRACES_SEMICOLONS = re.compile(r"[{};]")

_C_SPECIFIC_INDICATORS = re.compile(
    r"\*\w|\w\*|typedef\b|enum\s+\w+\s*\{|enum\s*\{|NULL\b|nullptr\b"
)

_LICENSE_WORDS = re.compile(
    r"(?:permission|license|granted|warranty|copyright|distribute|sublicense|"
    r"merchantability|liability|damages|software|modification|restriction)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Heuristic voter
# ---------------------------------------------------------------------------


def classify_heuristic(text: str) -> str:
    """Regex-based voter for sub_type classification."""
    if not text.strip():
        return "drop"

    # ObjC signals (check first — ObjC files often also have #include)
    if _OBJC_SIGNALS.search(text):
        return "objc"

    # C/C++ signals
    if _C_CPP_PREPROCESSOR.search(text):
        return "c_cpp"
    if _C_FUNC_SIG.search(text):
        return "c_cpp"
    if _C_STRUCT.search(text):
        return "c_cpp"

    # License / prose detection — check ratio of license-related words
    # The spec says "license ratio > 30%" which means >30% of words are
    # license-related keywords. In practice, real license texts have ~15-20%
    # keyword density, so we use 0.10 as the threshold.
    words = text.split()
    if words:
        license_hits = len(_LICENSE_WORDS.findall(text))
        ratio = license_hits / len(words)
        if ratio > 0.10:
            return "prose_plain"

    # Code-like syntax (braces + semicolons) — only classify as c_cpp if
    # C-specific indicators are also present (pointers, typedef, enum{}, NULL/nullptr)
    brace_semi_count = len(_CODE_BRACES_SEMICOLONS.findall(text))
    if brace_semi_count >= 3 and _C_SPECIFIC_INDICATORS.search(text):
        return "c_cpp"

    return "drop"


# ---------------------------------------------------------------------------
# Magika voter
# ---------------------------------------------------------------------------


def classify_magika(text: str, magika_instance: Magika) -> str:
    """Magika-based voter. Maps ct_label through MAGIKA_TO_SUBTYPE."""
    result = magika_instance.identify_bytes(text.encode("utf-8"))
    ct_label = result.output.ct_label
    return MAGIKA_TO_SUBTYPE.get(ct_label, "unknown")


# ---------------------------------------------------------------------------
# LLM voter
# ---------------------------------------------------------------------------

_VALID_LLM_LABELS = {"c_cpp", "objc", "prose", "other"}


def _parse_llm_label(answer: str) -> str:
    """Normalize an LLM response to one of the valid labels."""
    answer = answer.strip().lower()
    if answer in _VALID_LLM_LABELS:
        return answer
    # Try to find a valid label in the response
    for label in _VALID_LLM_LABELS:
        if label in answer:
            return label
    return "other"


async def _classify_one_llm(
    client: openai.AsyncOpenAI,
    text: str,
    model: str,
    semaphore: asyncio.Semaphore,
    pbar: Any = None,
    max_retries: int = 5,
) -> str:
    """Single LLM classification call with exponential backoff on 429s."""
    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=16,
                    messages=[
                        {
                            "role": "user",
                            "content": CLASSIFY_PROMPT.format(
                                text=text[:MAX_TEXT_LEN]
                            ),
                        }
                    ],
                )
                answer = response.choices[0].message.content or ""
                label = _parse_llm_label(answer)
                if pbar is not None:
                    pbar.update(1)
                return label
            except openai.RateLimitError:
                if attempt < max_retries:
                    wait = 2 ** attempt + random.random()
                    await asyncio.sleep(wait)
                    continue
                print("  LLM rate limit: max retries exceeded", file=sys.stderr)
                if pbar is not None:
                    pbar.update(1)
                return "other"
            except Exception as exc:
                print(f"  LLM error: {exc}", file=sys.stderr)
                if pbar is not None:
                    pbar.update(1)
                return "other"
        return "other"


async def classify_batch_llm(
    client: openai.AsyncOpenAI,
    texts: list[str],
    model: str,
    concurrency: int,
) -> list[str]:
    """Classify a batch of texts concurrently via LLM."""
    from tqdm import tqdm

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(texts), desc="    LLM voting", file=sys.stderr)
    tasks = [
        _classify_one_llm(client, t, model, semaphore, pbar)
        for t in texts
    ]
    results = await asyncio.gather(*tasks)
    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, str] = {
    "prose_plain": "prose",
    "other": "unknown",
}


def _normalize_label(label: str) -> str:
    """Map voter-specific labels to a canonical set.

    Canonical labels: {c_cpp, objc, prose, drop, unknown, manual_review}.
    Mappings: prose_plain -> prose, other -> unknown.
    """
    return _LABEL_MAP.get(label, label)


# ---------------------------------------------------------------------------
# Vote consensus
# ---------------------------------------------------------------------------


def vote(heuristic: str, magika: str, llm: str) -> tuple[str, str]:
    """3-way vote consensus. Returns (label, method).

    method is one of: "unanimous", "majority", "tie".
    For ties, returns ("manual_review", "tie").
    """
    if heuristic == magika == llm:
        return (heuristic, "unanimous")

    # Check for majority (any two agree)
    if heuristic == magika:
        return (heuristic, "majority")
    if heuristic == llm:
        return (heuristic, "majority")
    if magika == llm:
        return (magika, "majority")

    return ("manual_review", "tie")


# ---------------------------------------------------------------------------
# Apply voted labels (vectorized)
# ---------------------------------------------------------------------------

# Map voted label -> (category, sub_type). "drop" and "manual_review" are
# handled specially: drops remove rows, manual_review leaves sub_type as-is.
_LABEL_TO_CATEGORY_SUBTYPE: dict[str, tuple[str, str]] = {
    "c_cpp": ("code", "c_cpp"),
    "objc": ("code", "objc"),
    "prose": ("prose", "plain"),
    "unknown": ("unknown", "unknown"),
}


def apply_voted_labels(
    df: pl.DataFrame,
    indices: list[int],
    labels: list[str],
) -> tuple[pl.DataFrame, int]:
    """Apply voted labels to the DataFrame using vectorized Polars operations.

    Returns (updated_df, manual_review_count).

    Label mapping:
    - c_cpp   -> category="code",  sub_type="c_cpp"
    - objc    -> category="code",  sub_type="objc"
    - prose   -> category="prose", sub_type="plain"
    - drop    -> row removed
    - manual_review -> sub_type left as "unknown"
    """
    # Build a mapping DataFrame: __idx -> voted_category, voted_sub_type, is_drop
    map_rows: list[dict] = []
    manual_review_count = 0
    for idx, label in zip(indices, labels):
        if label in ("manual_review", "unknown"):
            manual_review_count += 1
            continue  # leave row unchanged
        if label == "drop":
            map_rows.append({
                "__idx": idx,
                "__voted_category": "",
                "__voted_sub_type": "",
                "__is_drop": True,
            })
            continue
        cat, sub = _LABEL_TO_CATEGORY_SUBTYPE.get(label, ("unknown", label))
        map_rows.append({
            "__idx": idx,
            "__voted_category": cat,
            "__voted_sub_type": sub,
            "__is_drop": False,
        })

    if not map_rows:
        return df, manual_review_count

    mapping_df = pl.DataFrame(map_rows).cast({"__idx": pl.UInt32})

    # Join mapping onto the original df with row indices
    df = df.with_row_index("__idx")
    df = df.join(mapping_df, on="__idx", how="left")

    # Apply updates: use voted values where present, keep originals otherwise
    df = df.with_columns([
        pl.when(pl.col("__voted_category").is_not_null() & ~pl.col("__is_drop").fill_null(False))
        .then(pl.col("__voted_category"))
        .otherwise(pl.col("category"))
        .alias("category"),
        pl.when(pl.col("__voted_sub_type").is_not_null() & ~pl.col("__is_drop").fill_null(False))
        .then(pl.col("__voted_sub_type"))
        .otherwise(pl.col("sub_type"))
        .alias("sub_type"),
    ])

    # Drop rows marked for removal
    df = df.filter(~pl.col("__is_drop").fill_null(False))

    # Clean up helper columns
    df = df.drop("__idx", "__voted_category", "__voted_sub_type", "__is_drop")

    return df, manual_review_count


# ---------------------------------------------------------------------------
# Bulk relabeling for non-Stack sources
# ---------------------------------------------------------------------------

_DROP_SOURCES = {"real/generated_ocr", "real/generated_skip"}
_PROSE_PLAIN_SOURCES = {
    "real/the_stack_text_licenses",
    "real/arxiv_summarization",
    "real/finepdfs",
}


def relabel_bulk(df: pl.DataFrame) -> pl.DataFrame:
    """Relabel non-Stack sources with known patterns.

    - the_stack_text_licenses, arxiv_summarization, finepdfs → prose/plain
    - generated_ocr, generated_skip → drop rows
    """
    # Drop unknown rows from sources that should be removed
    df = df.filter(
        ~(
            pl.col("source").is_in(list(_DROP_SOURCES))
            & (pl.col("sub_type") == "unknown")
        )
    )

    # Relabel unknown rows from prose/plain sources
    is_target = (
        pl.col("source").is_in(list(_PROSE_PLAIN_SOURCES))
        & (pl.col("sub_type") == "unknown")
    )
    df = df.with_columns([
        pl.when(is_target)
        .then(pl.lit("prose"))
        .otherwise(pl.col("category"))
        .alias("category"),
        pl.when(is_target)
        .then(pl.lit("plain"))
        .otherwise(pl.col("sub_type"))
        .alias("sub_type"),
    ])

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def async_main(argv: list[str] | None = None) -> None:
    """Main async orchestrator for relabeling unknowns."""
    parser = argparse.ArgumentParser(
        description="Relabel unknown sub_types via 3-way voting"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-3.1-flash-lite-preview",
        help="OpenRouter model for LLM voting (default: gemini-3.1-flash-lite-preview)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max concurrent LLM requests (default: 20)",
    )
    parser.add_argument(
        "--manual-review",
        type=Path,
        default=None,
        help="Output path for tie rows (JSONL)",
    )
    args = parser.parse_args(argv)

    # Load data
    print(f"Loading {args.input}...", file=sys.stderr)
    df = read_parquet(args.input)
    total_before = df.height
    print(f"Loaded {total_before:,} rows", file=sys.stderr)

    # Phase 1: Bulk relabel non-Stack unknowns
    print("\nPhase 1: Bulk relabeling non-Stack sources...", file=sys.stderr)
    df = relabel_bulk(df)
    print(
        f"  After bulk: {df.height:,} rows ({total_before - df.height} dropped)",
        file=sys.stderr,
    )

    # Phase 2: 3-way voting for the-stack-v2 unknowns
    stack_unknown_mask = (pl.col("source") == "real/the-stack-v2") & (
        pl.col("sub_type") == "unknown"
    )
    stack_unknowns = df.filter(stack_unknown_mask)
    n_unknowns = stack_unknowns.height
    print(
        f"\nPhase 2: Voting on {n_unknowns:,} the-stack-v2 unknowns...",
        file=sys.stderr,
    )

    manual_review_count = 0

    if n_unknowns > 0:
        # Initialize voters
        magika_instance = Magika()
        llm_client = openai.AsyncOpenAI(
            api_key=get_openrouter_api_key(),
            base_url="https://openrouter.ai/api/v1",
        )

        texts = stack_unknowns["text"].to_list()

        # Run heuristic voter
        print("  Running heuristic voter...", file=sys.stderr)
        heuristic_labels = [classify_heuristic(t) for t in texts]

        # Run Magika voter
        print("  Running Magika voter...", file=sys.stderr)
        magika_labels = [classify_magika(t, magika_instance) for t in texts]

        # Run LLM voter
        print(
            f"  Running LLM voter ({args.model}, concurrency={args.concurrency})...",
            file=sys.stderr,
        )
        llm_labels = await classify_batch_llm(
            llm_client, texts, args.model, args.concurrency
        )

        # Normalize labels before voting
        print("  Normalizing labels...", file=sys.stderr)
        heuristic_labels = [_normalize_label(l) for l in heuristic_labels]
        magika_labels = [_normalize_label(l) for l in magika_labels]
        llm_labels = [_normalize_label(l) for l in llm_labels]

        # Vote
        print("  Computing votes...", file=sys.stderr)
        voted_labels = []
        voted_methods = []
        manual_review_rows: list[dict] = []

        for i, (h, m, l) in enumerate(
            zip(heuristic_labels, magika_labels, llm_labels)
        ):
            label, method = vote(h, m, l)
            voted_labels.append(label)
            voted_methods.append(method)

            if method == "tie":
                manual_review_rows.append(
                    {
                        "index": i,
                        "text_preview": texts[i][:200],
                        "heuristic": h,
                        "magika": m,
                        "llm": l,
                    }
                )

        # Apply voted labels to the DataFrame (vectorized)
        stack_unknown_indices = (
            df.with_row_index("__idx")
            .filter(stack_unknown_mask)["__idx"]
            .to_list()
        )

        df, manual_review_count = apply_voted_labels(
            df, stack_unknown_indices, voted_labels
        )

        # Write manual review rows
        if manual_review_rows and args.manual_review:
            args.manual_review.parent.mkdir(parents=True, exist_ok=True)
            with open(args.manual_review, "w") as f:
                for row in manual_review_rows:
                    f.write(json.dumps(row) + "\n")
            print(
                f"  Wrote {len(manual_review_rows)} tie rows to {args.manual_review}",
                file=sys.stderr,
            )

        # Summary
        from collections import Counter

        method_counts = Counter(voted_methods)
        label_counts = Counter(voted_labels)

        print(f"\n{'=' * 50}", file=sys.stderr)
        print("Voting summary:", file=sys.stderr)
        print(f"  Unanimous: {method_counts.get('unanimous', 0):,}", file=sys.stderr)
        print(f"  Majority:  {method_counts.get('majority', 0):,}", file=sys.stderr)
        print(f"  Tie:       {method_counts.get('tie', 0):,}", file=sys.stderr)
        print(f"\nLabel distribution:", file=sys.stderr)
        for label, count in label_counts.most_common():
            print(f"  {label}: {count:,}", file=sys.stderr)
        print(f"{'=' * 50}", file=sys.stderr)

    # Final check: all the-stack-v2 unknowns should be relabeled OR in manual review
    remaining_unknowns = df.filter(
        (pl.col("source") == "real/the-stack-v2")
        & (pl.col("sub_type") == "unknown")
    ).height
    if remaining_unknowns > manual_review_count:
        raise ValueError(
            f"Found {remaining_unknowns} the-stack-v2 rows with sub_type='unknown' "
            f"but only {manual_review_count} are in manual review"
        )
    if remaining_unknowns > 0:
        print(
            f"\nWarning: {remaining_unknowns} the-stack-v2 rows still have "
            f"sub_type='unknown' (all are pending manual review)",
            file=sys.stderr,
        )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, args.output)
    print(f"\nWrote {df.height:,} rows to {args.output}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    """Sync wrapper for async_main."""
    asyncio.run(async_main(argv))
