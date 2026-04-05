#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["magika"]
# ///
"""Correct mislabeled samples using Magika as an independent oracle.

For each sample, runs Magika to identify the content type, maps it to our
3-category taxonomy (prose/code/structured), and corrects the label when
Magika confidently disagrees with the current label.

Supports both JSONL (eval) and CSV (training) formats.

Usage:
    # Correct eval JSONL
    uv run training/correct_labels.py --input eval/v1/clear.jsonl --output eval/v2/clear.jsonl --format jsonl

    # Correct training CSV
    uv run training/correct_labels.py --input training/data/curated/train/golden_raw.csv \
        --output training/data/curated/train/golden_raw_corrected.csv --format csv

    # Dry run (report corrections without writing)
    uv run training/correct_labels.py --input eval/v1/clear.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from magika import Magika

# ---------------------------------------------------------------------------
# Magika label → our taxonomy mapping
#
# Our taxonomy (from src/types.rs):
#   Prose:      plain, markdown, rst, latex
#   Code:       python, javascript, typescript, rust, go, java, sql, shell,
#               css, dockerfile, makefile, html, xml, sgml
#   Structured: csv, tsv, pipe_table, fixed_width, json, jsonl, key_value,
#               log_lines, yaml, toml, ini
# ---------------------------------------------------------------------------

MAGIKA_TO_CATEGORY: dict[str, str] = {
    # Prose (human-readable text for reading/discussion)
    "txt": "prose",
    "text": "prose",
    "rtf": "prose",
    "pdf": "prose",
    "doc": "prose",
    "docx": "prose",
    "epub": "prose",
    "email": "prose",
    "eml": "prose",
    "mbox": "prose",
    "latex": "prose",
    "tex": "prose",
    "rst": "prose",
    "markdown": "prose",
    "asciidoc": "prose",

    # Code (formal syntax, programming languages, markup)
    "python": "code",
    "javascript": "code",
    "typescript": "code",
    "java": "code",
    "c": "code",
    "cpp": "code",
    "csharp": "code",
    "go": "code",
    "rust": "code",
    "ruby": "code",
    "php": "code",
    "perl": "code",
    "scala": "code",
    "kotlin": "code",
    "swift": "code",
    "r": "code",
    "lua": "code",
    "shell": "code",
    "bash": "code",
    "powershell": "code",
    "sql": "code",
    "css": "code",
    "scss": "code",
    "less": "code",
    "html": "code",
    "xml": "code",
    "svg": "code",
    "dockerfile": "code",
    "makefile": "code",
    "cmake": "code",
    "hcl": "code",
    "terraform": "code",
    "groovy": "code",
    "dart": "code",
    "elixir": "code",
    "erlang": "code",
    "haskell": "code",
    "clojure": "code",
    "lisp": "code",
    "matlab": "code",
    "fortran": "code",
    "cobol": "code",
    "assembly": "code",
    "protobuf": "code",
    "thrift": "code",
    "graphql": "code",
    "smali": "code",
    "webassembly": "code",
    "actionscript": "code",
    "visual_basic": "code",
    "asp": "code",
    "jsp": "code",
    "objectivec": "code",
    "ocaml": "code",
    "pascal": "code",
    "tcl": "code",
    "solidity": "code",
    "vhdl": "code",
    "verilog": "code",
    "batch": "code",
    "prolog": "code",
    "bazel": "code",

    # Structured data
    "json": "structured",
    "jsonl": "structured",
    "ndjson": "structured",
    "yaml": "structured",
    "toml": "structured",
    "ini": "structured",
    "csv": "structured",
    "tsv": "structured",
    "plist": "structured",
    "properties": "structured",

    # Unknown / empty → don't correct
    "unknown": "unknown",
    "empty": "unknown",
}

# Magika label → our sub_type (where there's a clean 1:1 mapping)
MAGIKA_TO_SUBTYPE: dict[str, str] = {
    "txt": "plain",
    "text": "plain",
    "markdown": "markdown",
    "rst": "rst",
    "latex": "latex",
    "tex": "latex",
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "rust": "rust",
    "go": "go",
    "sql": "sql",
    "shell": "shell",
    "bash": "shell",
    "css": "css",
    "html": "html",
    "xml": "xml",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "json": "json",
    "jsonl": "jsonl",
    "ndjson": "jsonl",
    "yaml": "yaml",
    "toml": "toml",
    "ini": "ini",
    "csv": "csv",
    "tsv": "tsv",
}

# Minimum Magika confidence to accept a correction
MIN_CONFIDENCE = 0.50


def map_magika(magika_label: str) -> tuple[str, str | None]:
    """Map a Magika label to (category, sub_type_or_None)."""
    category = MAGIKA_TO_CATEGORY.get(magika_label, "unknown")
    sub_type = MAGIKA_TO_SUBTYPE.get(magika_label)
    return category, sub_type


def should_correct(
    current_category: str,
    magika_category: str,
    magika_score: float,
    min_confidence: float,
) -> bool:
    """Decide whether to correct a label based on Magika's opinion."""
    if magika_category == "unknown":
        return False
    if magika_category == current_category:
        return False
    if magika_score < min_confidence:
        return False
    return True


def correct_jsonl(
    input_path: Path,
    output_path: Path | None,
    magika: Magika,
    min_confidence: float,
    dry_run: bool,
) -> dict:
    """Correct labels in a JSONL file (eval format)."""
    samples = []
    for line_num, line in enumerate(input_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"  Warning: skipping malformed line {line_num}: {exc}", file=sys.stderr)
            continue
    stats = {"total": len(samples), "corrected": 0, "skipped_low_confidence": 0, "corrections": []}

    for sample in samples:
        text = sample.get("text", "")
        current_category = sample.get("expected_category", sample.get("label", ""))

        result = magika.identify_bytes(text.encode("utf-8"))
        magika_label = result.prediction.output.label
        magika_score = result.prediction.score
        magika_category, magika_sub_type = map_magika(magika_label)

        if magika_category != "unknown" and magika_category != current_category:
            if magika_score >= min_confidence:
                correction = {
                    "old_category": current_category,
                    "new_category": magika_category,
                    "old_sub_type": sample.get("sub_type"),
                    "magika_label": magika_label,
                    "magika_score": round(magika_score, 4),
                    "text_preview": text[:100].replace("\n", "\\n"),
                }

                if "expected_category" in sample:
                    sample["expected_category"] = magika_category
                elif "label" in sample:
                    sample["label"] = magika_category

                # Update sub_type if Magika gives us a clean mapping
                if magika_sub_type:
                    correction["new_sub_type"] = magika_sub_type
                    sample["sub_type"] = magika_sub_type

                stats["corrected"] += 1
                stats["corrections"].append(correction)
            else:
                stats["skipped_low_confidence"] += 1

    if not dry_run and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return stats


def correct_csv(
    input_path: Path,
    output_path: Path | None,
    magika: Magika,
    min_confidence: float,
    dry_run: bool,
) -> dict:
    """Correct labels in a CSV file (training format)."""
    csv.field_size_limit(sys.maxsize)

    stats = {"total": 0, "corrected": 0, "skipped_low_confidence": 0, "corrections": []}
    rows: list[dict] = []

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            stats["total"] += 1
            text = row.get("text", "")
            current_category = row.get("category", "")

            result = magika.identify_bytes(text.encode("utf-8"))
            magika_label = result.prediction.output.label
            magika_score = result.prediction.score
            magika_category, magika_sub_type = map_magika(magika_label)

            if magika_category != "unknown" and magika_category != current_category:
                if magika_score >= min_confidence:
                    correction = {
                        "old_category": current_category,
                        "new_category": magika_category,
                        "old_sub_type": row.get("sub_type"),
                        "magika_label": magika_label,
                        "magika_score": round(magika_score, 4),
                        "text_preview": text[:100].replace("\n", "\\n"),
                    }

                    row["category"] = magika_category

                    if magika_sub_type:
                        correction["new_sub_type"] = magika_sub_type
                        row["sub_type"] = magika_sub_type

                    stats["corrected"] += 1
                    stats["corrections"].append(correction)
                else:
                    stats["skipped_low_confidence"] += 1

            rows.append(row)

            if stats["total"] % 10000 == 0:
                print(
                    f"  Processed {stats['total']:,} samples, "
                    f"{stats['corrected']:,} corrected so far...",
                    file=sys.stderr,
                )

    if not dry_run and output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Correct mislabeled samples using Magika")
    parser.add_argument("--input", type=Path, required=True, help="Input file (JSONL or CSV)")
    parser.add_argument("--output", type=Path, default=None, help="Output file (default: auto)")
    parser.add_argument(
        "--format",
        choices=["jsonl", "csv", "auto"],
        default="auto",
        help="File format (default: auto-detect from extension)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help=f"Minimum Magika confidence for correction (default: {MIN_CONFIDENCE})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report corrections without writing")
    args = parser.parse_args(argv)

    # Auto-detect format
    fmt = args.format
    if fmt == "auto":
        if args.input.suffix == ".jsonl":
            fmt = "jsonl"
        elif args.input.suffix == ".csv":
            fmt = "csv"
        else:
            parser.error(f"Cannot auto-detect format for {args.input.suffix}; use --format")

    # Auto-generate output path
    output = args.output
    if not output and not args.dry_run:
        stem = args.input.stem
        suffix = args.input.suffix
        output = args.input.parent / f"{stem}_corrected{suffix}"

    print(f"Input:  {args.input}", file=sys.stderr)
    print(f"Output: {output or '(dry run)'}", file=sys.stderr)
    print(f"Format: {fmt}", file=sys.stderr)
    print(f"Min confidence: {args.min_confidence}", file=sys.stderr)
    print(file=sys.stderr)

    magika = Magika()

    if fmt == "jsonl":
        stats = correct_jsonl(args.input, output, magika, args.min_confidence, args.dry_run)
    else:
        stats = correct_csv(args.input, output, magika, args.min_confidence, args.dry_run)

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Total samples:          {stats['total']:,}", file=sys.stderr)
    print(f"Labels corrected:       {stats['corrected']:,}", file=sys.stderr)
    print(f"Skipped (low conf):     {stats['skipped_low_confidence']:,}", file=sys.stderr)
    print(f"Correction rate:        {stats['corrected']/max(stats['total'],1)*100:.2f}%", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Print correction breakdown
    from collections import Counter
    if stats["corrections"]:
        by_direction = Counter(
            (c["old_category"], c["new_category"]) for c in stats["corrections"]
        )
        print("Correction breakdown:", file=sys.stderr)
        for (old, new), count in by_direction.most_common():
            print(f"  {old} → {new}: {count}", file=sys.stderr)

    # Write detailed corrections log
    corrections_log = (output.parent / f"{output.stem}_corrections.json") if output else None
    if corrections_log and not args.dry_run:
        corrections_log.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
        print(f"\nCorrections log: {corrections_log}", file=sys.stderr)


if __name__ == "__main__":
    main()
