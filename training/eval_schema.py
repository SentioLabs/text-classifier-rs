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

PROVENANCE_FIELDS: set[str] = {
    "model",
    "temperature",
    "prompt_template",
    "content_domain",
    "length_bucket",
    "reasoning_mode",
    "sub_type",
}

VALID_LENGTH_BUCKETS: set[str] = {"short", "medium", "long"}


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


def validate_provenance(sample: dict) -> bool:
    """Check whether a sample dict contains valid provenance metadata.

    All provenance fields must be present with correct types:
      - ``model``: non-empty string
      - ``temperature``: float or int between 0.0 and 2.0
      - ``prompt_template``: non-empty string
      - ``content_domain``: non-empty string
      - ``length_bucket``: one of "short", "medium", "long"
      - ``reasoning_mode``: bool
      - ``sub_type``: non-empty string
    """
    # Check all fields present
    for field in PROVENANCE_FIELDS:
        if field not in sample:
            return False

    # model: non-empty string
    model = sample["model"]
    if not isinstance(model, str) or not model:
        return False

    # temperature: numeric, 0.0 <= t <= 2.0
    temperature = sample["temperature"]
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        return False
    if temperature < 0.0 or temperature > 2.0:
        return False

    # prompt_template: non-empty string
    prompt_template = sample["prompt_template"]
    if not isinstance(prompt_template, str) or not prompt_template:
        return False

    # content_domain: non-empty string
    content_domain = sample["content_domain"]
    if not isinstance(content_domain, str) or not content_domain:
        return False

    # length_bucket: one of the valid buckets
    length_bucket = sample["length_bucket"]
    if length_bucket not in VALID_LENGTH_BUCKETS:
        return False

    # reasoning_mode: bool
    reasoning_mode = sample["reasoning_mode"]
    if not isinstance(reasoning_mode, bool):
        return False

    # sub_type: non-empty string
    sub_type = sample["sub_type"]
    if not isinstance(sub_type, str) or not sub_type:
        return False

    return True


def diversity_report(path: str) -> dict:
    """Read a JSONL file and produce a diversity report over provenance fields.

    Returns a dict with:
      - ``total_samples``: total number of samples read
      - ``per_sub_type``: dict mapping each sub_type to a summary dict with:
          - ``model_distribution``: dict of model -> count
          - ``temperature_values``: set of temperature floats
          - ``template_count``: number of unique prompt_template values
          - ``domain_count``: number of unique content_domain values
          - ``length_buckets``: set of length_bucket values seen
    """
    total = 0
    per_sub_type: dict[str, dict] = {}

    with open(path) as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            sample = json.loads(raw_line)
            total += 1

            sub_type = sample.get("sub_type")
            if sub_type is None:
                continue

            if sub_type not in per_sub_type:
                per_sub_type[sub_type] = {
                    "model_distribution": {},
                    "temperature_values": set(),
                    "templates": set(),
                    "domains": set(),
                    "length_buckets": set(),
                }

            entry = per_sub_type[sub_type]

            model = sample.get("model")
            if model is not None:
                entry["model_distribution"][model] = entry["model_distribution"].get(model, 0) + 1

            temp = sample.get("temperature")
            if temp is not None:
                entry["temperature_values"].add(temp)

            template = sample.get("prompt_template")
            if template is not None:
                entry["templates"].add(template)

            domain = sample.get("content_domain")
            if domain is not None:
                entry["domains"].add(domain)

            bucket = sample.get("length_bucket")
            if bucket is not None:
                entry["length_buckets"].add(bucket)

    # Convert internal sets to counts for templates/domains
    for entry in per_sub_type.values():
        entry["template_count"] = len(entry.pop("templates"))
        entry["domain_count"] = len(entry.pop("domains"))

    return {"total_samples": total, "per_sub_type": per_sub_type}


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
