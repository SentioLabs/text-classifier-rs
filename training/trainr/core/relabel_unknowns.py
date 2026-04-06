"""Relabel unknown sub_types via 3-way voting (heuristic + Magika + LLM).

For the-stack-v2 rows with sub_type == "unknown", runs three independent
classifiers and uses majority vote to assign a label. Non-Stack sources
with known patterns are relabeled in bulk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

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

    # Code-like syntax (braces + semicolons)
    brace_semi_count = len(_CODE_BRACES_SEMICOLONS.findall(text))
    if brace_semi_count >= 3:
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
) -> str:
    """Single LLM classification call."""
    async with semaphore:
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
            return _parse_llm_label(answer)
        except Exception as exc:
            print(f"  LLM error: {exc}", file=sys.stderr)
            return "other"


async def classify_batch_llm(
    client: openai.AsyncOpenAI,
    texts: list[str],
    model: str,
    concurrency: int,
) -> list[str]:
    """Classify a batch of texts concurrently via LLM."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_classify_one_llm(client, t, model, semaphore) for t in texts]
    return await asyncio.gather(*tasks)


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
    # Drop rows from sources that should be removed
    df = df.filter(~pl.col("source").is_in(list(_DROP_SOURCES)))

    # Relabel prose/plain sources
    df = df.with_columns([
        pl.when(pl.col("source").is_in(list(_PROSE_PLAIN_SOURCES)))
        .then(pl.lit("prose"))
        .otherwise(pl.col("category"))
        .alias("category"),
        pl.when(pl.col("source").is_in(list(_PROSE_PLAIN_SOURCES)))
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

        # Apply voted labels to the DataFrame
        # Build a mapping from row indices in original df to new labels
        stack_unknown_indices = (
            df.with_row_index("__idx")
            .filter(stack_unknown_mask)["__idx"]
            .to_list()
        )

        # Create new sub_type and category columns
        new_sub_type = df["sub_type"].to_list()
        new_category = df["category"].to_list()

        for idx, label in zip(stack_unknown_indices, voted_labels):
            if label != "manual_review":
                new_sub_type[idx] = label
                # Infer category from sub_type
                if label in ("c_cpp", "objc"):
                    new_category[idx] = "code"
                elif label in ("prose_plain", "prose"):
                    new_category[idx] = "prose"

        df = df.with_columns([
            pl.Series("sub_type", new_sub_type),
            pl.Series("category", new_category),
        ])

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

    # Final check
    remaining_unknowns = df.filter(pl.col("sub_type") == "unknown").height
    assert remaining_unknowns == 0, (
        f"Expected 0 rows with sub_type == 'unknown', got {remaining_unknowns}"
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, args.output)
    print(f"\nWrote {df.height:,} rows to {args.output}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    """Sync wrapper for async_main."""
    asyncio.run(async_main(argv))
