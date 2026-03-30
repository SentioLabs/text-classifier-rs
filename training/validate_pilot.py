#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Pilot validation script for auditing generated sample batches.

Checks model diversity, template coverage, and classifier performance
for a JSONL file of pilot samples.

Usage:
    python validate_pilot.py --input data/pilot_samples.jsonl
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFY_BIN = str(PROJECT_ROOT / "target" / "release" / "classify")

REQUIRED_LENGTH_BUCKETS = {"short", "medium", "long"}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_samples(path: str) -> list[dict]:
    """Read a JSONL file and return a list of sample dicts."""
    samples: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def diversity_report(samples: list[dict]) -> dict:
    """Compute per sub-type diversity statistics.

    Returns a dict keyed by sub_type, each containing:
      - sample_count: int
      - model_distribution: dict[str, int]
      - temperature_values: set[float]
      - prompt_templates: set[str]
      - length_buckets: set[str]
      - content_domains: set[str]
    """
    by_sub_type: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_sub_type[s.get("sub_type", "unknown")].append(s)

    report: dict = {}
    for sub_type, sub_samples in by_sub_type.items():
        model_dist: dict[str, int] = defaultdict(int)
        temps: set[float] = set()
        templates: set[str] = set()
        buckets: set[str] = set()
        domains: set[str] = set()

        for s in sub_samples:
            model = s.get("model", "unknown")
            model_dist[model] += 1
            if "temperature" in s:
                temps.add(s["temperature"])
            if "prompt_template" in s:
                templates.add(s["prompt_template"])
            if "length_bucket" in s:
                buckets.add(s["length_bucket"])
            if "content_domain" in s:
                domains.add(s["content_domain"])

        report[sub_type] = {
            "sample_count": len(sub_samples),
            "model_distribution": dict(model_dist),
            "temperature_values": temps,
            "prompt_templates": templates,
            "length_buckets": buckets,
            "content_domains": domains,
        }

    return report


def check_diversity_checklist(
    sub_type_stats: dict,
) -> list[tuple[str, bool, str]]:
    """Run the 7-item diversity checklist against stats for a single sub-type.

    Returns a list of (check_name, passed, detail) tuples:
      1. at_least_5_models
      2. no_model_exceeds_15_pct
      3. at_least_3_temperatures
      4. at_least_4_templates
      5. all_length_buckets_present
      6. at_least_3_domains
      7. reasoning_mode_included
    """
    results: list[tuple[str, bool, str]] = []

    model_dist = sub_type_stats.get("model_distribution", {})
    total = sub_type_stats.get("sample_count", 0)
    temps = sub_type_stats.get("temperature_values", set())
    templates = sub_type_stats.get("prompt_templates", set())
    buckets = sub_type_stats.get("length_buckets", set())
    domains = sub_type_stats.get("content_domains", set())
    has_reasoning = sub_type_stats.get("has_reasoning_mode", False)

    # 1. At least 5 distinct models
    n_models = len(model_dist)
    results.append((
        "at_least_5_models",
        n_models >= 5,
        f"{n_models} models found",
    ))

    # 2. No single model exceeds 15%
    max_pct = 0.0
    max_model = ""
    if total > 0:
        for model, count in model_dist.items():
            pct = count / total * 100
            if pct > max_pct:
                max_pct = pct
                max_model = model
    results.append((
        "no_model_exceeds_15_pct",
        max_pct <= 15.0,
        f"max is {max_model} at {max_pct:.1f}%",
    ))

    # 3. At least 3 temperature values
    n_temps = len(temps)
    results.append((
        "at_least_3_temperatures",
        n_temps >= 3,
        f"{n_temps} temperature values",
    ))

    # 4. At least 4 prompt templates
    n_templates = len(templates)
    results.append((
        "at_least_4_templates",
        n_templates >= 4,
        f"{n_templates} templates",
    ))

    # 5. Short, medium, and long length buckets present
    missing = REQUIRED_LENGTH_BUCKETS - buckets
    results.append((
        "all_length_buckets_present",
        len(missing) == 0,
        f"missing: {missing}" if missing else "all present",
    ))

    # 6. Content domain varied (3+ domains)
    n_domains = len(domains)
    results.append((
        "at_least_3_domains",
        n_domains >= 3,
        f"{n_domains} domains",
    ))

    # 7. Reasoning-mode model included (for Prose sub-types)
    results.append((
        "reasoning_mode_included",
        bool(has_reasoning),
        "present" if has_reasoning else "not found",
    ))

    return results


def run_classifier_audit(
    samples: list[dict],
    classify_bin: str = DEFAULT_CLASSIFY_BIN,
) -> dict | None:
    """Run the classify binary on each sample and compare to expected_category.

    Returns a dict with confusion_matrix and per_sub_type_accuracy, or None
    if the binary is not found.
    """
    if not os.path.isfile(classify_bin):
        print(
            f"Warning: classify binary not found at {classify_bin}, skipping audit.",
            file=sys.stderr,
        )
        return None

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sub_type_correct: dict[str, int] = defaultdict(int)
    sub_type_total: dict[str, int] = defaultdict(int)

    for sample in samples:
        text = sample.get("text", "")
        expected = sample.get("expected_category", "")
        sub_type = sample.get("sub_type", "unknown")

        # Write text to a temp file and classify it
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [classify_bin, "file", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                predicted = "error"
            else:
                # Parse output — expect JSON with "category" field
                try:
                    output = json.loads(result.stdout.strip().splitlines()[-1])
                    predicted = output.get("category", "unknown")
                except (json.JSONDecodeError, IndexError):
                    predicted = "unknown"
        except (subprocess.TimeoutExpired, OSError):
            predicted = "error"
        finally:
            os.unlink(tmp_path)

        confusion[expected][predicted] += 1
        sub_type_total[sub_type] += 1
        if predicted == expected:
            sub_type_correct[sub_type] += 1

    # Compute per-sub-type accuracy and flag issues
    per_sub_type: dict[str, dict] = {}
    flags: list[str] = []
    for st in sorted(sub_type_total):
        total = sub_type_total[st]
        correct = sub_type_correct[st]
        accuracy = correct / total if total > 0 else 0.0
        per_sub_type[st] = {"accuracy": accuracy, "total": total, "correct": correct}
        if accuracy > 0.99:
            flags.append(f"{st}: accuracy {accuracy:.1%} — too easy")
        elif accuracy < 0.50:
            flags.append(f"{st}: accuracy {accuracy:.1%} — too hard")

    return {
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "per_sub_type_accuracy": per_sub_type,
        "flags": flags,
    }


def print_audit_samples(samples: list[dict], n: int = 50) -> None:
    """Print n random stratified samples (1-2 per sub-type) for manual review."""
    by_sub_type: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_sub_type[s.get("sub_type", "unknown")].append(s)

    selected: list[dict] = []
    sub_types = sorted(by_sub_type.keys())

    # First pass: pick 1-2 per sub-type, up to n
    per_type = max(1, min(2, n // max(len(sub_types), 1)))
    for st in sub_types:
        pool = by_sub_type[st]
        pick = min(per_type, len(pool))
        selected.extend(random.sample(pool, pick))
        if len(selected) >= n:
            break

    selected = selected[:n]

    print(f"\n{'=' * 72}")
    print(f"  AUDIT SAMPLES ({len(selected)} of {len(samples)} total)")
    print(f"{'=' * 72}")

    for i, s in enumerate(selected, 1):
        text_preview = s.get("text", "")[:200]
        print(f"\n--- Sample {i} ---")
        print(f"  sub_type:          {s.get('sub_type', 'N/A')}")
        print(f"  expected_category: {s.get('expected_category', 'N/A')}")
        print(f"  model:             {s.get('model', 'N/A')}")
        print(f"  temperature:       {s.get('temperature', 'N/A')}")
        print(f"  prompt_template:   {s.get('prompt_template', 'N/A')}")
        print(f"  length_bucket:     {s.get('length_bucket', 'N/A')}")
        print(f"  content_domain:    {s.get('content_domain', 'N/A')}")
        print(f"  text (preview):    {text_preview}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a pilot sample batch for diversity and classifier performance.",
    )
    parser.add_argument(
        "--input",
        default="data/pilot_samples.jsonl",
        help="Path to the JSONL file of pilot samples (default: data/pilot_samples.jsonl)",
    )
    return parser


def _pass_fail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading samples from {args.input} ...")
    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples.\n")

    report = diversity_report(samples)

    # Determine which sub-types are prose (for reasoning mode check)
    prose_sub_types = {"plain", "markdown", "rst", "latex"}

    print(f"{'=' * 72}")
    print("  DIVERSITY CHECKLIST (per sub-type)")
    print(f"{'=' * 72}")

    all_passed = True
    for sub_type in sorted(report):
        stats = report[sub_type]

        # Add has_reasoning_mode for prose sub-types
        if sub_type in prose_sub_types:
            has_reasoning = any(
                s.get("reasoning_mode") is True
                for s in samples
                if s.get("sub_type") == sub_type
            )
            stats["has_reasoning_mode"] = has_reasoning

        checks = check_diversity_checklist(stats)

        print(f"\n  [{sub_type}] ({stats['sample_count']} samples)")
        for name, passed, detail in checks:
            indicator = _pass_fail(passed)
            print(f"    [{indicator}] {name}: {detail}")
            if not passed:
                all_passed = False

    print(f"\n{'=' * 72}")
    if all_passed:
        print("  OVERALL: ALL CHECKS PASSED")
    else:
        print("  OVERALL: SOME CHECKS FAILED")
    print(f"{'=' * 72}")

    # Classifier audit
    print("\nRunning classifier audit ...")
    audit = run_classifier_audit(samples)
    if audit is not None:
        print(f"\n{'=' * 72}")
        print("  CLASSIFIER AUDIT")
        print(f"{'=' * 72}")
        for st, acc_info in sorted(audit["per_sub_type_accuracy"].items()):
            print(
                f"  {st}: {acc_info['accuracy']:.1%} "
                f"({acc_info['correct']}/{acc_info['total']})"
            )
        if audit["flags"]:
            print("\n  Flags:")
            for flag in audit["flags"]:
                print(f"    {flag}")
    else:
        print("  Classifier audit skipped (binary not found).")

    # Manual review samples
    print_audit_samples(samples)


if __name__ == "__main__":
    main()
