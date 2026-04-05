"""Three-way label audit: current label vs Magika vs LLM (Haiku or OpenRouter).

For each sample where Magika disagrees with the current label, asks an LLM
for a second opinion. Only flags for correction when Magika + LLM agree
against the current label.

Supports concurrent LLM calls via --concurrency for faster processing.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import openai
from magika import Magika

# ---------------------------------------------------------------------------
# Magika → our taxonomy (from src/types.rs)
# ---------------------------------------------------------------------------

MAGIKA_TO_CATEGORY: dict[str, str] = {
    # Prose
    "txt": "prose", "text": "prose", "rtf": "prose", "pdf": "prose",
    "doc": "prose", "docx": "prose", "epub": "prose", "email": "prose",
    "eml": "prose", "mbox": "prose", "latex": "prose", "tex": "prose",
    "rst": "prose", "markdown": "prose", "asciidoc": "prose",
    # Code
    "python": "code", "javascript": "code", "typescript": "code",
    "java": "code", "c": "code", "cpp": "code", "csharp": "code",
    "go": "code", "rust": "code", "ruby": "code", "php": "code",
    "perl": "code", "scala": "code", "kotlin": "code", "swift": "code",
    "r": "code", "lua": "code", "shell": "code", "bash": "code",
    "powershell": "code", "sql": "code", "css": "code", "scss": "code",
    "less": "code", "html": "code", "xml": "code", "svg": "code",
    "dockerfile": "code", "makefile": "code", "cmake": "code",
    "hcl": "code", "terraform": "code", "groovy": "code", "dart": "code",
    "elixir": "code", "erlang": "code", "haskell": "code",
    "clojure": "code", "lisp": "code", "matlab": "code",
    "fortran": "code", "cobol": "code", "assembly": "code",
    "protobuf": "code", "thrift": "code", "graphql": "code",
    "smali": "code", "webassembly": "code", "batch": "code",
    "prolog": "code", "bazel": "code", "vhdl": "code", "verilog": "code",
    "actionscript": "code", "visual_basic": "code", "asp": "code",
    "jsp": "code", "objectivec": "code", "ocaml": "code",
    "pascal": "code", "tcl": "code", "solidity": "code",
    # Structured
    "json": "structured", "jsonl": "structured", "ndjson": "structured",
    "yaml": "structured", "toml": "structured", "ini": "structured",
    "csv": "structured", "tsv": "structured", "plist": "structured",
    "properties": "structured",
    # Unknown
    "unknown": "unknown", "empty": "unknown",
}

CLASSIFY_PROMPT = """\
Classify this text into exactly one category. Reply with ONLY the category name, nothing else.

Categories:
- prose: Human-readable text meant for reading — articles, documentation, READMEs, papers, \
meeting notes, emails, markdown documents, reStructuredText, LaTeX papers. \
The key question: would a user searching for "discussion about X" want to find this?
- code: Programming source code, scripts, markup that is meant to be executed or rendered — \
Python, JavaScript, HTML, SQL, shell scripts, Dockerfiles, Makefiles, etc.
- structured: Data in a structured format — JSON, YAML, CSV, TSV, TOML, INI config files, \
log lines with timestamps, key-value pairs, tabular data.

Text to classify:
---
{text}
---

Category:"""

MAX_TEXT_LEN = 2000


def _parse_llm_answer(answer: str) -> str:
    """Normalize an LLM classification response to a category."""
    answer = answer.strip().lower()
    if "prose" in answer:
        return "prose"
    elif "code" in answer:
        return "code"
    elif "structured" in answer:
        return "structured"
    return "unknown"


# ---------------------------------------------------------------------------
# Async LLM classification
# ---------------------------------------------------------------------------


async def _classify_one_anthropic(
    client: anthropic.AsyncAnthropic,
    text: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:MAX_TEXT_LEN])}],
            )
            return _parse_llm_answer(response.content[0].text)
        except Exception as exc:
            print(f"  Anthropic error: {exc}", file=sys.stderr)
            return "unknown"


async def _classify_one_openrouter(
    client: openai.AsyncOpenAI,
    text: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:MAX_TEXT_LEN])}],
            )
            answer = response.choices[0].message.content or ""
            return _parse_llm_answer(answer)
        except Exception as exc:
            print(f"  OpenRouter error: {exc}", file=sys.stderr)
            return "unknown"


async def classify_batch_async(
    backend: str,
    client: anthropic.AsyncAnthropic | openai.AsyncOpenAI,
    texts: list[str],
    model: str,
    concurrency: int,
) -> list[str]:
    """Classify a list of texts concurrently."""
    semaphore = asyncio.Semaphore(concurrency)

    if backend == "anthropic":
        tasks = [_classify_one_anthropic(client, t, model, semaphore) for t in texts]
    else:
        tasks = [_classify_one_openrouter(client, t, model, semaphore) for t in texts]

    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_samples(input_path: Path) -> list[dict]:
    """Load samples from JSONL or CSV."""
    samples = []
    if input_path.suffix == ".jsonl":
        for line in input_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    elif input_path.suffix == ".csv":
        csv.field_size_limit(sys.maxsize)
        with open(input_path, newline="") as f:
            for row in csv.DictReader(f):
                samples.append(row)
    return samples


def get_category(sample: dict) -> str:
    """Get the category label from a sample (handles both eval and training formats)."""
    return sample.get("expected_category", sample.get("category", sample.get("label", "")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Three-way label audit with Magika + LLM")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--magika-min-confidence", type=float, default=0.50,
        help="Minimum Magika confidence to trigger LLM vote (default: 0.50)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max samples to process (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--backend", choices=["anthropic", "openrouter"], default="anthropic",
        help="LLM backend to use (default: anthropic)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override LLM model (default: haiku for anthropic, gpt-5.4-mini for openrouter)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent LLM requests (default: 10)",
    )
    parser.add_argument(
        "--progress-interval", type=int, default=100,
        help="Print progress every N LLM calls (default: 100)",
    )
    args = parser.parse_args(argv)

    # Set up async LLM client
    if args.backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set. Run: source ~/.a.sh", file=sys.stderr)
            sys.exit(1)
        llm_client = anthropic.AsyncAnthropic(api_key=api_key)
        llm_model = args.model or "claude-haiku-4-5-20251001"
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY not set. Run: source ~/.a.sh", file=sys.stderr)
            sys.exit(1)
        llm_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        llm_model = args.model or "openai/gpt-5.4-mini"

    print(f"Backend: {args.backend} ({llm_model})", file=sys.stderr)
    print(f"Concurrency: {args.concurrency}", file=sys.stderr)
    magika = Magika()

    print(f"Loading {args.input}...", file=sys.stderr)
    samples = load_samples(args.input)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples):,} samples", file=sys.stderr)

    # Phase 1: Run Magika on all samples, identify disagreements
    print("\nPhase 1: Magika classification...", file=sys.stderr)
    disagreements: list[tuple[int, dict, str, float]] = []

    for i, sample in enumerate(samples):
        text = sample.get("text", "")
        current_cat = get_category(sample)
        result = magika.identify_bytes(text.encode("utf-8"))
        magika_label = result.prediction.output.label
        magika_score = result.prediction.score
        magika_cat = MAGIKA_TO_CATEGORY.get(magika_label, "unknown")

        if magika_cat != "unknown" and magika_cat != current_cat and magika_score >= args.magika_min_confidence:
            disagreements.append((i, sample, magika_cat, magika_score))

        if (i + 1) % 10000 == 0:
            print(f"  Magika: {i + 1:,}/{len(samples):,}, {len(disagreements)} disagreements", file=sys.stderr)

    print(f"  Magika found {len(disagreements):,} disagreements (score >= {args.magika_min_confidence})", file=sys.stderr)

    if not disagreements:
        print("No disagreements found. Nothing to vote on.", file=sys.stderr)
        return

    # Phase 2: Concurrent LLM voting
    texts = [s.get("text", "") for _, s, _, _ in disagreements]
    print(f"\nPhase 2: LLM voting on {len(disagreements):,} samples "
          f"({args.concurrency} concurrent)...", file=sys.stderr)

    start_time = time.time()

    # Process in chunks for progress reporting
    chunk_size = args.progress_interval
    all_llm_results: list[str] = []

    for chunk_start in range(0, len(texts), chunk_size):
        chunk = texts[chunk_start:chunk_start + chunk_size]
        chunk_results = await classify_batch_async(
            args.backend, llm_client, chunk, llm_model, args.concurrency,
        )
        all_llm_results.extend(chunk_results)

        done = len(all_llm_results)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(texts) - done) / rate if rate > 0 else 0
        print(
            f"  {done:,}/{len(texts):,} ({done / len(texts) * 100:.0f}%) "
            f"@ {rate:.1f}/s  ETA {eta / 60:.1f}m",
            file=sys.stderr,
        )

    total_elapsed = time.time() - start_time

    # Build votes
    votes: list[dict] = []
    for (idx, sample, magika_cat, magika_score), llm_cat in zip(disagreements, all_llm_results):
        current_cat = get_category(sample)
        sub_type = sample.get("sub_type", "unknown")

        if magika_cat == llm_cat and magika_cat != current_cat:
            verdict = "CORRECT"
        elif magika_cat != llm_cat and llm_cat == current_cat:
            verdict = "KEEP"
        elif magika_cat != llm_cat and llm_cat != current_cat:
            verdict = "AMBIGUOUS"
        else:
            verdict = "KEEP"

        votes.append({
            "index": idx,
            "current_category": current_cat,
            "magika_category": magika_cat,
            "magika_score": round(magika_score, 4),
            "llm_category": llm_cat,
            "sub_type": sub_type,
            "verdict": verdict,
            "text_preview": sample.get("text", "")[:150].replace("\n", "\\n"),
        })

    # Write results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for vote in votes:
            f.write(json.dumps(vote) + "\n")

    # Summary
    from collections import Counter
    verdicts = Counter(v["verdict"] for v in votes)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Total Magika disagreements:     {len(disagreements):,}", file=sys.stderr)
    print(f"CORRECT (Magika+LLM agree):     {verdicts.get('CORRECT', 0):,}", file=sys.stderr)
    print(f"KEEP (LLM agrees w/ label):     {verdicts.get('KEEP', 0):,}", file=sys.stderr)
    print(f"AMBIGUOUS (all disagree):        {verdicts.get('AMBIGUOUS', 0):,}", file=sys.stderr)
    print(f"Total time:                      {total_elapsed / 60:.1f} min", file=sys.stderr)
    print(f"Effective rate:                  {len(disagreements) / total_elapsed:.1f} samples/sec", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    corrections = [v for v in votes if v["verdict"] == "CORRECT"]
    if corrections:
        by_dir = Counter((v["current_category"], v["magika_category"]) for v in corrections)
        print("\nConfirmed corrections:", file=sys.stderr)
        for (old, new), count in by_dir.most_common():
            print(f"  {old} → {new}: {count}", file=sys.stderr)

        by_sub = Counter(v["sub_type"] for v in corrections)
        print("\nBy sub_type:", file=sys.stderr)
        for st, count in by_sub.most_common(15):
            print(f"  {st}: {count}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))
