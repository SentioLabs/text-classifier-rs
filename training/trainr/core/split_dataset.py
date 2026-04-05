#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["polars"]
# ///
"""Split a raw JSONL dataset into eval and training sets.

Produces three output files:
  - eval clear JSONL   (stratified by expected_category)
  - eval boundary JSONL (stratified by boundary_pair)
  - training Parquet    (all remaining samples)

Stratified sampling preserves model diversity within each stratum.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Taxonomy overrides — sub-types whose parent category has been reclassified
# ---------------------------------------------------------------------------

SUBTYPE_CATEGORY_OVERRIDES = {
    "yaml": "structured",
    "toml": "structured",
    "ini": "structured",
    "pdf_dump": None,      # reclassified by heuristic
    "ocr_garbage": None,   # reclassified by heuristic
    "boilerplate": None,   # reclassified by heuristic
    "skip": None,          # reclassified by heuristic
}

VALID_CATEGORIES = frozenset({"prose", "code", "structured"})

# ---------------------------------------------------------------------------
# Content-based reclassification
# ---------------------------------------------------------------------------


def reclassify_by_content(text: str) -> str:
    """Reclassify a sample using structural text features.

    Used for samples whose original category was ``"artifact"`` or ``"skip"``.
    Returns one of ``"prose"``, ``"code"``, or ``"structured"``.
    """
    from trainr.core.featurize import (
        sentence_coherence_score,
        dictionary_word_ratio,
        alpha_ratio,
        key_value_ratio,
        comment_ratio,
        delimiter_consistency,
    )

    scs = sentence_coherence_score(text)
    dwr = dictionary_word_ratio(text)
    ar = alpha_ratio(text)
    kvr = key_value_ratio(text)
    cr = comment_ratio(text)
    dc = delimiter_consistency(text)

    if scs > 0.3 and dwr > 0.5:
        return "prose"
    elif kvr > 0.3 or dc > 0.5:
        return "structured"
    elif cr > 0.2:
        return "code"
    elif dwr > 0.5 and ar > 0.8:
        return "prose"
    else:
        return "structured"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    samples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def stratified_sample(
    samples: list[dict],
    key: str,
    n_per_group: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split *samples* into (selected, remainder), stratified by *key*.

    Within each stratum, if samples have a ``"model"`` field the selection
    is sub-stratified by model to preserve diversity.  Uses
    ``random.Random(seed)`` for determinism.
    """
    rng = random.Random(seed)

    # Group by key
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        groups[s[key]].append(s)

    selected: list[dict] = []
    remainder: list[dict] = []

    for _group_key in sorted(groups):
        group = groups[_group_key]
        n = min(n_per_group, len(group))

        # Check if samples have a model field for sub-stratification
        has_model = any("model" in s for s in group)

        if has_model:
            picked = _sub_stratify_by_model(group, n, rng)
        else:
            rng.shuffle(group)
            picked = group[:n]

        picked_set = {id(s) for s in picked}
        selected.extend(picked)
        remainder.extend(s for s in group if id(s) not in picked_set)

    return selected, remainder


def _sub_stratify_by_model(
    group: list[dict], n: int, rng: random.Random
) -> list[dict]:
    """Select *n* items from *group*, drawing evenly across model values."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for s in group:
        by_model[s.get("model", "__none__")].append(s)

    # Shuffle within each model bucket
    for bucket in by_model.values():
        rng.shuffle(bucket)

    # Round-robin across models
    picked: list[dict] = []
    model_keys = sorted(by_model)
    idx = {k: 0 for k in model_keys}

    while len(picked) < n:
        made_progress = False
        for mk in model_keys:
            if len(picked) >= n:
                break
            if idx[mk] < len(by_model[mk]):
                picked.append(by_model[mk][idx[mk]])
                idx[mk] += 1
                made_progress = True
        if not made_progress:
            break  # all buckets exhausted

    return picked


def split_dataset(
    input_path: str,
    eval_clear_path: str | None,
    eval_boundary_path: str | None,
    train_path: str,
    eval_per_category: int = 1000,
    eval_per_pair: int = 1000,
    seed: int = 42,
    max_per_category: int = 0,
) -> None:
    """Split *input_path* JSONL into eval and training sets."""
    all_samples = load_jsonl(input_path)

    # Apply taxonomy overrides before stratification
    for s in all_samples:
        sub_type = s.get("expected_sub_type", s.get("sub_type", ""))
        override = SUBTYPE_CATEGORY_OVERRIDES.get(sub_type)
        if override is not None:
            s["expected_category"] = override

    # Reclassify artifact/skip samples using content-based heuristics
    for s in all_samples:
        cat = s.get("expected_category", "")
        if cat in ("artifact", "skip"):
            s["expected_category"] = reclassify_by_content(s.get("text", ""))

    # Drop samples whose category is not in the valid 3-category set
    all_samples = [
        s for s in all_samples
        if s.get("expected_category", "") in VALID_CATEGORIES
    ]

    # Separate clear vs boundary
    clear: list[dict] = []
    boundary: list[dict] = []
    for s in all_samples:
        if s.get("boundary_pair") is None:
            clear.append(s)
        else:
            boundary.append(s)

    # Stratified sample for eval
    eval_clear, train_clear = stratified_sample(
        clear, key="expected_category", n_per_group=eval_per_category, seed=seed
    )
    eval_boundary, train_boundary = stratified_sample(
        boundary, key="boundary_pair", n_per_group=eval_per_pair, seed=seed
    )

    training = train_clear + train_boundary

    # Downsample overrepresented categories if requested
    if max_per_category > 0:
        training_by_cat: dict[str, list[dict]] = defaultdict(list)
        for s in training:
            training_by_cat[s.get("expected_category", "")].append(s)
        rng = random.Random(seed)
        downsampled: list[dict] = []
        for cat in sorted(training_by_cat):
            samples = training_by_cat[cat]
            if len(samples) > max_per_category:
                rng.shuffle(samples)
                downsampled.extend(samples[:max_per_category])
            else:
                downsampled.extend(samples)
        training = downsampled

    # Write eval clear JSONL (skip if path is None — frozen eval mode)
    if eval_clear_path is not None:
        _write_jsonl(eval_clear_path, eval_clear)

    # Write eval boundary JSONL (skip if path is None — frozen eval mode)
    if eval_boundary_path is not None:
        _write_jsonl(eval_boundary_path, eval_boundary)

    # Write training Parquet — preserve original source and model provenance
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([
        {
            "text": s.get("text", ""),
            "category": s.get("expected_category", ""),
            "sub_type": s.get("sub_type", ""),
            "source": s.get("source", "unknown"),
            "model": s.get("model", "unknown"),
        }
        for s in training
    ]).write_parquet(train_path)

    # Print summary
    _print_summary(all_samples, eval_clear, eval_boundary, training)


def _write_jsonl(path: str, samples: list[dict]) -> None:
    """Write samples as JSONL (one JSON object per line)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def _print_summary(
    all_samples: list[dict],
    eval_clear: list[dict],
    eval_boundary: list[dict],
    training: list[dict],
) -> None:
    """Print a human-readable split summary."""
    print(f"Total input samples:   {len(all_samples)}")
    print(f"Eval clear samples:    {len(eval_clear)}")
    print(f"Eval boundary samples: {len(eval_boundary)}")
    print(f"Training samples:      {len(training)}")

    # Per-category breakdown for training
    from collections import Counter

    cats = Counter(s.get("expected_category", "unknown") for s in training)
    print("\nTraining per-category breakdown:")
    for cat in sorted(cats):
        print(f"  {cat}: {cats[cat]}")


def verify_diversity(samples: list[dict]) -> list[str]:
    """Check if any model exceeds 15% in any sub_type slice.

    Returns a list of warning strings (empty means all good).
    """
    warnings: list[str] = []

    # Group by sub_type
    by_sub_type: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        st = s.get("sub_type", "unknown")
        by_sub_type[st].append(s)

    for st, group in sorted(by_sub_type.items()):
        total = len(group)
        if total == 0:
            continue

        from collections import Counter

        model_counts = Counter(s.get("model", "unknown") for s in group)
        for model, count in model_counts.items():
            pct = count / total
            if pct > 0.15:
                warnings.append(
                    f"sub_type={st!r}: model {model!r} has {count}/{total} "
                    f"samples ({pct:.1%}), exceeds 15% threshold"
                )

    return warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split raw JSONL dataset into eval and training sets"
    )
    parser.add_argument(
        "--input",
        default="data/source/merged/raw_samples.jsonl",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--eval-output",
        default="data/eval/clear.jsonl",
        help="Path for eval clear output JSONL",
    )
    parser.add_argument(
        "--eval-boundary-output",
        default="data/eval/boundary.jsonl",
        help="Path for eval boundary output JSONL",
    )
    parser.add_argument(
        "--train-output",
        default="data/curated/train/golden_raw.parquet",
        help="Path for training output Parquet",
    )
    parser.add_argument(
        "--eval-per-category",
        type=int,
        default=1000,
        help="Number of eval samples per category",
    )
    parser.add_argument(
        "--eval-per-pair",
        type=int,
        default=1000,
        help="Number of eval samples per boundary pair",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=0,
        help="Max training samples per category (0 = no limit)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval set generation (for frozen eval mode)",
    )
    args = parser.parse_args(argv)

    eval_clear_path = None if args.skip_eval else args.eval_output
    eval_boundary_path = None if args.skip_eval else args.eval_boundary_output

    split_dataset(
        input_path=args.input,
        eval_clear_path=eval_clear_path,
        eval_boundary_path=eval_boundary_path,
        train_path=args.train_output,
        eval_per_category=args.eval_per_category,
        eval_per_pair=args.eval_per_pair,
        seed=args.seed,
        max_per_category=args.max_per_category,
    )

    # Verify diversity on eval sets (skip if frozen eval mode)
    if not args.skip_eval:
        eval_clear = load_jsonl(args.eval_output)
        eval_boundary = load_jsonl(args.eval_boundary_output)

        for label, data in [("eval clear", eval_clear), ("eval boundary", eval_boundary)]:
            warnings = verify_diversity(data)
            if warnings:
                print(f"\nDiversity warnings for {label}:")
                for w in warnings:
                    print(f"  WARNING: {w}")
            else:
                print(f"\nDiversity check passed for {label}")


