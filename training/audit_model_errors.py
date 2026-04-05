#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic", "openai"]
# ///
"""Audit remaining model errors by asking an LLM to break the tie.

For each sample where the model's prediction disagrees with the label,
asks an LLM whether the model or the label is correct. This identifies
samples where the model is actually right and the label needs fixing.

Reads eval_predictions JSONL (output of eval_onnx.py) and produces a
votes JSONL that can be fed to apply_corrections.py.

Usage (single-LLM):
    source ~/.a.sh
    uv run training/audit_model_errors.py \
        --predictions training/output/eval_predictions.clear_v2.jsonl \
        --output training/output/model_error_votes.jsonl \
        --backend openrouter --model openai/gpt-5.4-mini --concurrency 20

Usage (dual-LLM 4-way vote):
    source ~/.a.sh
    uv run training/audit_model_errors.py \
        --predictions training/output/eval_predictions.clear_v2.jsonl \
        --output training/output/model_error_votes.jsonl \
        --dual-llm --concurrency 20 \
        --filter-subtypes unknown,json,jsonl,plain,ini,csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import anthropic
import openai

CLASSIFY_PROMPT = """\
Classify this text into exactly one category. Reply with ONLY the category name, nothing else.

Categories:
- prose: Human-readable text meant for reading — articles, documentation, READMEs, papers, \
meeting notes, emails, markdown documents, reStructuredText, LaTeX papers. \
The key question: would a user searching for "discussion about X" want to find this?
- code: Programming source code, scripts, markup that is meant to be executed or rendered — \
Python, JavaScript, HTML, SQL, shell scripts, Dockerfiles, Makefiles, etc.
- structured: Data in a structured format — JSON, YAML, CSV, TSV, TOML, INI config files, \
log lines with timestamps, key-value pairs, tabular data, form fields.

Text to classify:
---
{text}
---

Category:"""

MAX_TEXT_LEN = 2000


def _parse_llm_answer(answer: str) -> str:
    answer = answer.strip().lower()
    if "prose" in answer:
        return "prose"
    elif "code" in answer:
        return "code"
    elif "structured" in answer:
        return "structured"
    return "unknown"


def _compute_verdict(current_label, model_pred, haiku_cat, gpt_cat):
    """4-way vote: 3/4 majority auto-corrects, 2-2 tie -> TIE."""
    voters = {"current_label": current_label, "model": model_pred, "haiku": haiku_cat, "gpt": gpt_cat}
    counts = Counter(voters.values())
    winner, winner_count = counts.most_common(1)[0]
    if winner_count >= 3:
        return ("KEEP_LABEL" if winner == current_label else "FIX_LABEL"), winner, dict(counts)
    if len(counts) == 2:  # 2-2 split
        return "TIE", winner, dict(counts)
    # 2-1-1: plurality wins
    return ("KEEP_LABEL" if winner == current_label else "FIX_LABEL"), winner, dict(counts)


async def _classify_one_anthropic(client, text, model, sem):
    async with sem:
        try:
            r = await client.messages.create(
                model=model, max_tokens=10,
                messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:MAX_TEXT_LEN])}],
            )
            return _parse_llm_answer(r.content[0].text)
        except Exception as e:
            print(f"  Anthropic error: {e}", file=sys.stderr)
            return "unknown"


async def _classify_one_openrouter(client, text, model, sem):
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model, max_tokens=16,
                messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:MAX_TEXT_LEN])}],
            )
            return _parse_llm_answer(r.choices[0].message.content or "")
        except Exception as e:
            print(f"  OpenRouter error: {e}", file=sys.stderr)
            return "unknown"


async def _classify_dual_llm(client, text, haiku_model, gpt_model, sem):
    """Call both LLMs concurrently via OpenRouter, return (haiku_cat, gpt_cat).

    Each LLM call acquires its own semaphore slot so --concurrency reflects
    the actual number of in-flight API requests (not pairs).
    """
    prompt = CLASSIFY_PROMPT.format(text=text[:MAX_TEXT_LEN])

    async def _call(model_name):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name, max_tokens=16,
                    messages=[{"role": "user", "content": prompt}],
                )
                return _parse_llm_answer(r.choices[0].message.content or "")
            except Exception as e:
                print(f"  Dual-LLM error ({model_name}): {e}", file=sys.stderr)
                return "unknown"

    haiku_cat, gpt_cat = await asyncio.gather(_call(haiku_model), _call(gpt_model))
    return (haiku_cat, gpt_cat)


async def classify_batch(backend, client, texts, model, concurrency):
    sem = asyncio.Semaphore(concurrency)
    if backend == "anthropic":
        tasks = [_classify_one_anthropic(client, t, model, sem) for t in texts]
    else:
        tasks = [_classify_one_openrouter(client, t, model, sem) for t in texts]
    return await asyncio.gather(*tasks)


async def classify_batch_dual(client, texts, haiku_model, gpt_model, concurrency):
    """Classify all texts with both LLMs concurrently."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [_classify_dual_llm(client, t, haiku_model, gpt_model, sem) for t in texts]
    return await asyncio.gather(*tasks)


async def async_main(argv=None):
    parser = argparse.ArgumentParser(description="Audit model errors with LLM tie-breaking")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="eval_predictions JSONL from eval_onnx.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output votes JSONL")

    # Single-LLM args
    parser.add_argument("--backend", choices=["anthropic", "openrouter"], default="openrouter")
    parser.add_argument("--model", default=None)

    # Dual-LLM args
    parser.add_argument("--dual-llm", action="store_true",
                        help="Use dual-LLM 4-way voting (mutually exclusive with --backend)")
    parser.add_argument("--ties-output", type=Path, default=None,
                        help="Output file for TIE verdicts (default: manual_review.jsonl next to --output)")
    parser.add_argument("--haiku-model", default="anthropic/claude-haiku-4-5",
                        help="Haiku model for dual-LLM mode")
    parser.add_argument("--gpt-model", default="openai/gpt-5.4-mini",
                        help="GPT model for dual-LLM mode")

    # Filtering
    parser.add_argument("--filter-subtypes", default=None,
                        help="Comma-separated list of sub-types to audit (default: all)")

    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args(argv)

    # Validate mutual exclusivity
    if args.dual_llm and (args.model or args.backend != "openrouter"):
        parser.error("--dual-llm is mutually exclusive with --backend and --model")

    # Set ties-output default
    if args.ties_output is None:
        args.ties_output = args.output.parent / "manual_review.jsonl"

    # Parse filter-subtypes
    filter_subtypes = None
    if args.filter_subtypes:
        filter_subtypes = {s.strip() for s in args.filter_subtypes.split(",")}

    # Set up client
    model = ""  # overwritten below; silences type-checker for single-LLM path
    if args.dual_llm:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            sys.exit("OPENROUTER_API_KEY not set")
        client = openai.AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        print(f"Dual-LLM mode: {args.haiku_model} + {args.gpt_model}", file=sys.stderr)
    elif args.backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY not set")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        model = args.model or "claude-haiku-4-5-20251001"
        print(f"Backend: {args.backend} ({model})", file=sys.stderr)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            sys.exit("OPENROUTER_API_KEY not set")
        client = openai.AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        model = args.model or "openai/gpt-5.4-mini"
        print(f"Backend: {args.backend} ({model})", file=sys.stderr)

    # Load only misclassified samples
    errors = []
    for i, line in enumerate(args.predictions.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["expected_category"] != r["predicted_category"]:
            sub_type = r.get("sub_type", "unknown")
            if filter_subtypes and sub_type not in filter_subtypes:
                continue
            errors.append((i, r))

    print(f"Found {len(errors)} model errors to audit", file=sys.stderr)

    if not errors:
        print("No errors to audit.", file=sys.stderr)
        return

    texts = [r["text"] for _, r in errors]
    start = time.time()

    if args.dual_llm:
        # Dual-LLM path
        dual_results = await classify_batch_dual(
            client, texts, args.haiku_model, args.gpt_model, args.concurrency
        )
        elapsed = time.time() - start
        print(f"Dual-LLM classification: {len(errors)} samples in {elapsed:.1f}s "
              f"({len(errors)/elapsed:.1f}/s)", file=sys.stderr)

        votes = []
        tie_votes = []
        verdicts = Counter()

        for (idx, sample), (haiku_cat, gpt_cat) in zip(errors, dual_results):
            label = sample["expected_category"]
            predicted = sample["predicted_category"]
            sub_type = sample.get("sub_type", "unknown")

            verdict, winner, vote_counts = _compute_verdict(label, predicted, haiku_cat, gpt_cat)
            verdicts[verdict] += 1

            record = {
                "index": idx,
                "current_category": label,
                "predicted_category": predicted,
                "haiku_category": haiku_cat,
                "gpt_category": gpt_cat,
                "vote_counts": vote_counts,
                "winner": winner,
                "sub_type": sub_type,
                "verdict": verdict,
                "magika_category": winner,  # for apply_corrections compat
                "text_preview": sample.get("text", "")[:150].replace("\n", "\\n"),
            }

            if verdict == "TIE":
                tie_votes.append(record)
            else:
                votes.append(record)

        # Write main output (remap FIX_LABEL -> CORRECT for apply_corrections.py)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for v in votes:
                out = dict(v)
                if out["verdict"] == "FIX_LABEL":
                    out["verdict"] = "CORRECT"
                f.write(json.dumps(out) + "\n")

        # Write ties output
        args.ties_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.ties_output, "w") as f:
            for v in tie_votes:
                f.write(json.dumps(v) + "\n")

        # Summary
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Total model errors audited:  {len(errors)}", file=sys.stderr)
        print(f"FIX_LABEL (label needs fix): {verdicts['FIX_LABEL']}", file=sys.stderr)
        print(f"KEEP_LABEL (model wrong):    {verdicts['KEEP_LABEL']}", file=sys.stderr)
        print(f"TIE (2-2 split):             {verdicts['TIE']}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Votes written to: {args.output}", file=sys.stderr)
        print(f"Ties written to:  {args.ties_output}", file=sys.stderr)

    else:
        # Single-LLM path (original behavior)
        llm_results = await classify_batch(args.backend, client, texts, model, args.concurrency)
        elapsed = time.time() - start
        print(f"LLM classification: {len(errors)} samples in {elapsed:.1f}s "
              f"({len(errors)/elapsed:.1f}/s)", file=sys.stderr)

        votes = []
        verdicts = Counter()

        for (idx, sample), llm_cat in zip(errors, llm_results):
            label = sample["expected_category"]
            predicted = sample["predicted_category"]
            sub_type = sample.get("sub_type", "unknown")

            # Three-way: label vs model vs LLM
            if llm_cat == predicted and llm_cat != label:
                verdict = "FIX_LABEL"  # Model + LLM agree -> label is wrong
            elif llm_cat == label and llm_cat != predicted:
                verdict = "MODEL_WRONG"  # Label + LLM agree -> model is wrong
            elif llm_cat != label and llm_cat != predicted:
                verdict = "AMBIGUOUS"  # All three disagree
            else:
                verdict = "MODEL_WRONG"  # Default

            verdicts[verdict] += 1
            votes.append({
                "index": idx,
                "current_category": label,
                "predicted_category": predicted,
                "llm_category": llm_cat,
                "sub_type": sub_type,
                "verdict": verdict,
                # Fields needed by apply_corrections.py (verdict=FIX_LABEL -> CORRECT)
                "magika_category": predicted,  # use model's prediction as the correction target
                "text_preview": sample.get("text", "")[:150].replace("\n", "\\n"),
            })

        # Write votes (remap FIX_LABEL -> CORRECT for apply_corrections.py compatibility)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for v in votes:
                out = dict(v)
                if out["verdict"] == "FIX_LABEL":
                    out["verdict"] = "CORRECT"
                f.write(json.dumps(out) + "\n")

        # Summary
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Total model errors audited:  {len(errors)}", file=sys.stderr)
        print(f"FIX_LABEL (model is right):  {verdicts['FIX_LABEL']}", file=sys.stderr)
        print(f"MODEL_WRONG (label is right): {verdicts['MODEL_WRONG']}", file=sys.stderr)
        print(f"AMBIGUOUS (all disagree):     {verdicts['AMBIGUOUS']}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        fix_labels = [v for v in votes if v["verdict"] == "FIX_LABEL"]
        if fix_labels:
            by_dir = Counter((v["current_category"], v["predicted_category"]) for v in fix_labels)
            print("\nLabel fixes (model was right):", file=sys.stderr)
            for (old, new), count in by_dir.most_common():
                print(f"  {old} -> {new}: {count}", file=sys.stderr)

            by_sub = Counter(v["sub_type"] for v in fix_labels)
            print("\nBy sub_type:", file=sys.stderr)
            for st, count in by_sub.most_common(10):
                print(f"  {st}: {count}", file=sys.stderr)


def main(argv=None):
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
