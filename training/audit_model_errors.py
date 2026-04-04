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

Usage:
    source ~/.a.sh
    uv run training/audit_model_errors.py \
        --predictions training/output/eval_predictions.clear_v2.jsonl \
        --output training/output/model_error_votes.jsonl \
        --backend openrouter --model openai/gpt-5.4-mini --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
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


async def classify_batch(backend, client, texts, model, concurrency):
    sem = asyncio.Semaphore(concurrency)
    if backend == "anthropic":
        tasks = [_classify_one_anthropic(client, t, model, sem) for t in texts]
    else:
        tasks = [_classify_one_openrouter(client, t, model, sem) for t in texts]
    return await asyncio.gather(*tasks)


async def async_main(argv=None):
    parser = argparse.ArgumentParser(description="Audit model errors with LLM tie-breaking")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="eval_predictions JSONL from eval_onnx.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output votes JSONL")
    parser.add_argument("--backend", choices=["anthropic", "openrouter"], default="openrouter")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args(argv)

    if args.backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY not set")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        model = args.model or "claude-haiku-4-5-20251001"
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
            errors.append((i, r))

    print(f"Found {len(errors)} model errors to audit", file=sys.stderr)

    if not errors:
        print("No errors to audit.", file=sys.stderr)
        return

    # Classify with LLM
    texts = [r["text"] for _, r in errors]
    start = time.time()
    llm_results = await classify_batch(args.backend, client, texts, model, args.concurrency)
    elapsed = time.time() - start
    print(f"LLM classification: {len(errors)} samples in {elapsed:.1f}s "
          f"({len(errors)/elapsed:.1f}/s)", file=sys.stderr)

    # Build votes
    from collections import Counter
    votes = []
    verdicts = Counter()

    for (idx, sample), llm_cat in zip(errors, llm_results):
        label = sample["expected_category"]
        predicted = sample["predicted_category"]
        sub_type = sample.get("sub_type", "unknown")

        # Three-way: label vs model vs LLM
        if llm_cat == predicted and llm_cat != label:
            verdict = "FIX_LABEL"  # Model + LLM agree → label is wrong
        elif llm_cat == label and llm_cat != predicted:
            verdict = "MODEL_WRONG"  # Label + LLM agree → model is wrong
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
            # Fields needed by apply_corrections.py (verdict=FIX_LABEL → CORRECT)
            "magika_category": predicted,  # use model's prediction as the correction target
            "text_preview": sample.get("text", "")[:150].replace("\n", "\\n"),
        })

    # Write votes (remap FIX_LABEL → CORRECT for apply_corrections.py compatibility)
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
            print(f"  {old} → {new}: {count}", file=sys.stderr)

        by_sub = Counter(v["sub_type"] for v in fix_labels)
        print("\nBy sub_type:", file=sys.stderr)
        for st, count in by_sub.most_common(10):
            print(f"  {st}: {count}", file=sys.stderr)


def main(argv=None):
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
