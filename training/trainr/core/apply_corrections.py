"""Apply voted label corrections AND remap ghost sub-types.

Two operations in one pass:
1. Apply category corrections from audit_labels_vote.py (vote JSONL)
2. Remap ghost sub-types (pdf_dump, ocr_garbage, boilerplate, skip) to real
   ContentSubType values using Magika as an oracle.

Supports both JSONL (eval) and CSV (training) formats.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from magika import Magika

# ---------------------------------------------------------------------------
# Ghost sub-types to remap (not in Rust ContentSubType enum)
# ---------------------------------------------------------------------------

GHOST_SUBTYPES = {"pdf_dump", "ocr_garbage", "boilerplate", "skip"}

# Real sub-types from src/types.rs ContentSubType enum
REAL_SUBTYPES = {
    "plain", "markdown", "rst", "latex",
    "python", "javascript", "typescript", "rust", "go", "java", "sql",
    "shell", "css", "dockerfile", "makefile", "html", "xml", "sgml",
    "csv", "tsv", "pipe_table", "fixed_width", "json", "jsonl",
    "key_value", "log_lines", "yaml", "toml", "ini",
    "unknown",
}

# Magika label → real sub_type
MAGIKA_TO_SUBTYPE: dict[str, str] = {
    "txt": "plain", "text": "plain",
    "markdown": "markdown", "rst": "rst",
    "latex": "latex", "tex": "latex",
    "python": "python", "javascript": "javascript",
    "typescript": "typescript", "java": "java", "rust": "rust",
    "go": "go", "sql": "sql", "shell": "shell", "bash": "shell",
    "css": "css", "html": "html", "xml": "xml", "svg": "xml",
    "dockerfile": "dockerfile", "makefile": "makefile",
    "json": "json", "jsonl": "jsonl", "ndjson": "jsonl",
    "yaml": "yaml", "toml": "toml", "ini": "ini",
    "csv": "csv", "tsv": "tsv",
}

# Magika label → our category (same mapping as audit scripts)
MAGIKA_TO_CATEGORY: dict[str, str] = {
    "txt": "prose", "text": "prose", "rtf": "prose", "pdf": "prose",
    "doc": "prose", "docx": "prose", "epub": "prose", "email": "prose",
    "eml": "prose", "mbox": "prose", "latex": "prose", "tex": "prose",
    "rst": "prose", "markdown": "prose", "asciidoc": "prose",
    "python": "code", "javascript": "code", "typescript": "code",
    "java": "code", "c": "code", "cpp": "code", "csharp": "code",
    "go": "code", "rust": "code", "ruby": "code", "php": "code",
    "perl": "code", "scala": "code", "kotlin": "code", "swift": "code",
    "r": "code", "lua": "code", "shell": "code", "bash": "code",
    "powershell": "code", "sql": "code", "css": "code", "scss": "code",
    "less": "code", "html": "code", "xml": "code", "svg": "code",
    "dockerfile": "code", "makefile": "code", "cmake": "code",
    "hcl": "code", "terraform": "code", "batch": "code",
    "prolog": "code", "bazel": "code",
    "json": "structured", "jsonl": "structured", "ndjson": "structured",
    "yaml": "structured", "toml": "structured", "ini": "structured",
    "csv": "structured", "tsv": "structured", "plist": "structured",
    "properties": "structured",
    "unknown": "unknown", "empty": "unknown",
}


def load_corrections(votes_path: Path | None) -> dict[int, dict]:
    """Load CORRECT verdicts from votes JSONL, keyed by sample index."""
    if not votes_path or not votes_path.exists():
        return {}
    corrections = {}
    for line in votes_path.read_text().splitlines():
        if not line.strip():
            continue
        vote = json.loads(line)
        if vote.get("verdict") == "CORRECT":
            corrections[vote["index"]] = vote
    return corrections


def remap_ghost_subtype(text: str, current_subtype: str, magika: Magika) -> str:
    """Remap a ghost sub-type to a real one using Magika."""
    if current_subtype not in GHOST_SUBTYPES:
        return current_subtype

    result = magika.identify_bytes(text.encode("utf-8"))
    magika_label = result.prediction.output.label
    new_subtype = MAGIKA_TO_SUBTYPE.get(magika_label)

    if new_subtype and new_subtype in REAL_SUBTYPES:
        return new_subtype

    # Fallback: plain for prose-like, unknown for everything else
    return "plain" if result.prediction.output.group == "text" else "unknown"


def get_category_field(sample: dict) -> str:
    """Detect which field holds the category label."""
    if "expected_category" in sample:
        return "expected_category"
    elif "category" in sample:
        return "category"
    elif "label" in sample:
        return "label"
    return "category"


def process_jsonl(
    input_path: Path,
    output_path: Path,
    corrections: dict[int, dict],
    magika: Magika,
    remap_subtypes: bool,
) -> dict:
    """Process a JSONL file with corrections and sub-type remapping."""
    stats = {"total": 0, "cat_corrected": 0, "subtype_remapped": 0, "skipped_lines": 0}
    output_lines = []

    idx = 0
    for line in input_path.read_text().splitlines():
        line = line.strip()
        if not line:
            output_lines.append("")
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            output_lines.append(line)
            stats["skipped_lines"] += 1
            idx += 1
            continue

        stats["total"] += 1
        cat_field = get_category_field(sample)

        # 1. Apply category correction if voted
        if idx in corrections:
            corr = corrections[idx]
            sample[cat_field] = corr["magika_category"]
            stats["cat_corrected"] += 1

        # 2. Remap ghost sub-type
        if remap_subtypes and sample.get("sub_type", "") in GHOST_SUBTYPES:
            old_st = sample["sub_type"]
            new_st = remap_ghost_subtype(sample.get("text", ""), old_st, magika)
            sample["sub_type"] = new_st
            stats["subtype_remapped"] += 1

        output_lines.append(json.dumps(sample, ensure_ascii=False))
        idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n")
    return stats


def process_csv(
    input_path: Path,
    output_path: Path,
    corrections: dict[int, dict],
    magika: Magika,
    remap_subtypes: bool,
) -> dict:
    """Process a CSV file with corrections and sub-type remapping."""
    csv.field_size_limit(sys.maxsize)
    stats = {"total": 0, "cat_corrected": 0, "subtype_remapped": 0}
    rows = []

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for i, row in enumerate(reader):
            stats["total"] += 1

            # 1. Apply category correction if voted
            if i in corrections:
                corr = corrections[i]
                row["category"] = corr["magika_category"]
                stats["cat_corrected"] += 1

            # 2. Remap ghost sub-type
            if remap_subtypes and row.get("sub_type", "") in GHOST_SUBTYPES:
                old_st = row["sub_type"]
                new_st = remap_ghost_subtype(row.get("text", ""), old_st, magika)
                row["sub_type"] = new_st
                stats["subtype_remapped"] += 1

            rows.append(row)

            if (i + 1) % 10000 == 0:
                print(
                    f"  {i + 1:,} processed, "
                    f"{stats['cat_corrected']:,} cat fixes, "
                    f"{stats['subtype_remapped']:,} subtype remaps",
                    file=sys.stderr,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return stats


def process_parquet(
    input_path: Path,
    output_path: Path,
    corrections: dict[int, dict],
    magika: Magika,
    remap_subtypes: bool,
) -> dict:
    """Process a Parquet file with corrections and sub-type remapping."""
    import polars as pl

    df = pl.read_parquet(input_path)
    stats = {"total": len(df), "cat_corrected": 0, "subtype_remapped": 0}

    categories = df["category"].to_list()
    sub_types = df["sub_type"].to_list()
    texts = df["text"].to_list()

    for i in range(len(df)):
        if i in corrections:
            corr = corrections[i]
            categories[i] = corr["magika_category"]
            stats["cat_corrected"] += 1

        if remap_subtypes and sub_types[i] in GHOST_SUBTYPES:
            new_st = remap_ghost_subtype(texts[i] or "", sub_types[i], magika)
            sub_types[i] = new_st
            stats["subtype_remapped"] += 1

        if (i + 1) % 10000 == 0:
            print(
                f"  {i + 1:,} processed, "
                f"{stats['cat_corrected']:,} cat fixes, "
                f"{stats['subtype_remapped']:,} subtype remaps",
                file=sys.stderr,
            )

    df = df.with_columns([
        pl.Series("category", categories),
        pl.Series("sub_type", sub_types),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply category corrections + remap ghost sub-types"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input file (JSONL, CSV, or Parquet)")
    parser.add_argument("--output", type=Path, required=True, help="Output file")
    parser.add_argument("--votes", type=Path, default=None, help="Votes JSONL from audit_labels_vote.py")
    parser.add_argument(
        "--remap-subtypes-only", action="store_true",
        help="Only remap ghost sub-types, skip category corrections",
    )
    parser.add_argument(
        "--no-remap-subtypes", action="store_true",
        help="Only apply category corrections, skip sub-type remapping",
    )
    args = parser.parse_args(argv)

    corrections = {}
    if not args.remap_subtypes_only:
        corrections = load_corrections(args.votes)
        print(f"Loaded {len(corrections):,} category corrections", file=sys.stderr)

    remap_subtypes = not args.no_remap_subtypes
    if remap_subtypes:
        print("Ghost sub-type remapping: ENABLED", file=sys.stderr)
    else:
        print("Ghost sub-type remapping: DISABLED", file=sys.stderr)

    magika = Magika()

    print(f"Processing {args.input}...", file=sys.stderr)

    if args.input.suffix == ".jsonl":
        stats = process_jsonl(args.input, args.output, corrections, magika, remap_subtypes)
    elif args.input.suffix == ".csv":
        stats = process_csv(args.input, args.output, corrections, magika, remap_subtypes)
    elif args.input.suffix == ".parquet":
        stats = process_parquet(args.input, args.output, corrections, magika, remap_subtypes)
    else:
        print(f"Unsupported format: {args.input.suffix}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 50}", file=sys.stderr)
    print(f"Total samples:       {stats['total']:,}", file=sys.stderr)
    print(f"Categories corrected: {stats['cat_corrected']:,}", file=sys.stderr)
    print(f"Sub-types remapped:   {stats['subtype_remapped']:,}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
