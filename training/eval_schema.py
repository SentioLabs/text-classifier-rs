"""Eval JSONL schema validation for hierarchical taxonomy training data.

Defines the canonical schema contract for eval/training JSONL files and
provides validation utilities used by all downstream generation scripts.
"""

import json
import sys

VALID_CATEGORIES: set[str] = {"prose", "code", "structured", "artifact"}

VALID_BOUNDARY_PAIRS: set[str] = {
    "prose_code",
    "prose_structured",
    "prose_artifact",
    "code_structured",
    "code_artifact",
    "structured_artifact",
}


def validate_sample(sample: dict) -> bool:
    """Check whether a single sample dict conforms to the eval JSONL schema.

    Requirements:
      - ``text`` field exists and is a non-empty string
      - ``expected_category`` is in ``VALID_CATEGORIES``
      - ``boundary_pair`` is either absent/None or in ``VALID_BOUNDARY_PAIRS``
      - If ``boundary_pair`` is set, ``expected_category`` must be one of the
        two categories named in the pair
    """
    # text: must be a non-empty string
    text = sample.get("text")
    if not isinstance(text, str) or not text:
        return False

    # expected_category: must be a recognised category
    category = sample.get("expected_category")
    if category not in VALID_CATEGORIES:
        return False

    # boundary_pair: optional, but must be valid when present
    boundary_pair = sample.get("boundary_pair")
    if boundary_pair is not None:
        if boundary_pair not in VALID_BOUNDARY_PAIRS:
            return False
        pair_categories = boundary_pair.split("_")
        if category not in pair_categories:
            return False

    return True


def validate_file(path: str) -> tuple[int, list[str]]:
    """Read a JSONL file and validate every line against the eval schema.

    Returns a tuple of ``(valid_count, error_messages)`` where each error
    message identifies the problematic line number and reason.
    """
    valid_count = 0
    errors: list[str] = []

    with open(path) as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                sample = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_num}: invalid JSON — {exc}")
                continue

            if validate_sample(sample):
                valid_count += 1
            else:
                errors.append(f"Line {line_num}: sample failed schema validation")

    return valid_count, errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-jsonl>", file=sys.stderr)
        sys.exit(2)

    file_path = sys.argv[1]
    valid, errs = validate_file(file_path)

    print(f"Valid samples: {valid}")
    if errs:
        print(f"Errors ({len(errs)}):")
        for err in errs:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("No errors found.")
