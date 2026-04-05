"""Cross-validate eval set labels against Google's Magika file-type detector.

Writes each sample to a temp file (no extension, to avoid biasing Magika),
classifies with Magika, then maps Magika's label to our 3-category taxonomy
(prose/code/structured) and compares against the dataset label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from magika import Magika

# ---------------------------------------------------------------------------
# Magika label → our taxonomy mapping
# ---------------------------------------------------------------------------

# Magika outputs fine-grained labels like "python", "yaml", "json", "html", etc.
# We map them to our 3-category system.

MAGIKA_TO_CATEGORY: dict[str, str] = {
    # Code
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
    "latex": "prose",  # Our taxonomy: LaTeX → prose
    "tex": "prose",    # Our taxonomy: TeX → prose
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
    "vhdl": "code",
    "verilog": "code",
    "rst": "prose",       # Our taxonomy: reStructuredText → prose
    "markdown": "prose",  # Our taxonomy: Markdown → prose
    "asciidoc": "prose",  # Closest to our RST/Markdown → prose
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

    # Structured data
    "json": "structured",
    "jsonl": "structured",
    "ndjson": "structured",
    "yaml": "structured",
    "toml": "structured",
    "ini": "structured",
    "csv": "structured",
    "tsv": "structured",
    "xls": "structured",
    "xlsx": "structured",
    "ods": "structured",
    "parquet": "structured",
    "avro": "structured",
    "arrow": "structured",
    "plist": "structured",
    "properties": "structured",

    # Prose / text
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

    # Unknown / empty
    "unknown": "unknown",
    "empty": "unknown",
}


def map_magika_to_category(magika_label: str) -> str:
    """Map a Magika label to our 3-category taxonomy."""
    # Direct lookup
    if magika_label in MAGIKA_TO_CATEGORY:
        return MAGIKA_TO_CATEGORY[magika_label]

    # Check Magika group as fallback
    return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-validate labels with Magika")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training/output/eval_predictions.clear.jsonl"),
        help="JSONL file with eval predictions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: stdout)",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only output samples where our label disagrees with Magika",
    )
    args = parser.parse_args(argv)

    magika = Magika()
    samples = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]

    out = open(args.output, "w") if args.output else sys.stdout

    # Stats
    total = 0
    magika_agrees_with_label = 0
    magika_agrees_with_prediction = 0
    magika_disagrees_both = 0
    # Where our model was wrong but Magika agrees with our model (→ likely mislabel)
    mislabel_candidates = 0
    # Where our model was wrong and Magika agrees with the label (→ real model error)
    real_model_errors = 0

    for i, sample in enumerate(samples):
        text = sample["text"]
        expected = sample["expected_category"]
        predicted = sample["predicted_category"]
        sub_type = sample.get("sub_type", "unknown")

        # Classify with Magika
        result = magika.identify_bytes(text.encode("utf-8"))
        magika_label = result.prediction.output.label
        magika_group = result.prediction.output.group
        magika_score = result.prediction.score
        magika_category = map_magika_to_category(magika_label)

        total += 1
        was_error = expected != predicted

        if magika_category == expected:
            magika_agrees_with_label += 1
        if magika_category == predicted:
            magika_agrees_with_prediction += 1
        if magika_category != expected and magika_category != predicted:
            magika_disagrees_both += 1

        if was_error:
            if magika_category == predicted:
                mislabel_candidates += 1
            elif magika_category == expected:
                real_model_errors += 1

        record = {
            "index": i,
            "expected": expected,
            "predicted": predicted,
            "sub_type": sub_type,
            "magika_label": magika_label,
            "magika_group": magika_group,
            "magika_category": magika_category,
            "magika_score": round(magika_score, 4),
            "was_error": was_error,
            "text_preview": text[:150].replace("\n", "\\n"),
        }

        if was_error:
            if magika_category == predicted:
                record["audit_verdict"] = "LIKELY_MISLABEL"
            elif magika_category == expected:
                record["audit_verdict"] = "REAL_MODEL_ERROR"
            else:
                record["audit_verdict"] = "AMBIGUOUS"

        if not args.errors_only or was_error:
            out.write(json.dumps(record) + "\n")

    # Summary
    n_errors = sum(1 for s in samples if s["expected_category"] != s["predicted_category"])
    summary = {
        "_summary": True,
        "total_samples": total,
        "total_errors": n_errors,
        "magika_agrees_with_label": magika_agrees_with_label,
        "magika_agrees_with_prediction": magika_agrees_with_prediction,
        "magika_disagrees_both": magika_disagrees_both,
        "mislabel_candidates": mislabel_candidates,
        "real_model_errors": real_model_errors,
        "ambiguous_errors": n_errors - mislabel_candidates - real_model_errors,
        "estimated_true_accuracy": round(
            (total - n_errors + mislabel_candidates) / total, 4
        ),
    }

    print(json.dumps(summary, indent=2), file=sys.stderr)

    if args.output:
        out.close()
