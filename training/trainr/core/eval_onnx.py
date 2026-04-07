"""Evaluate a trained ONNX model against eval JSONL or Parquet files.

Loads the ONNX model and model_config.json, runs inference on each eval
sample, and reports accuracy, per-category precision/recall/F1, and a
confusion matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from trainr.core.schema import CATEGORY_ORDER, build_prediction_record

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore[override]
        return iterable


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict[str, Any]:
    """Load model_config.json from *path*."""
    with open(path) as f:
        return json.load(f)


def load_eval_samples(paths: list[str]) -> list[dict[str, Any]]:
    """Load evaluation samples from one or more JSONL or Parquet files."""
    samples: list[dict[str, Any]] = []
    for p in paths:
        if p.endswith(".parquet"):
            import polars as pl

            df = pl.read_parquet(p)
            # Training Parquet uses "category"; eval pipeline expects "expected_category"
            if "category" in df.columns and "expected_category" not in df.columns:
                df = df.rename({"category": "expected_category"})
            samples.extend(df.to_dicts())
        else:
            # Existing JSONL logic unchanged
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return samples


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------


def normalize_features(
    raw: dict[str, float], config: dict[str, Any]
) -> np.ndarray:
    """Z-score normalize raw features using config's mean/std.

    Returns a float32 array of shape ``[1, n_features]``.
    """
    names = config["feature_names"]
    mean = config["feature_mean"]
    std = config["feature_std"]

    values = []
    for i, name in enumerate(names):
        val = raw.get(name, 0.0)
        s = std[i]
        if s == 0.0:
            values.append(0.0)
        else:
            values.append((val - mean[i]) / s)

    return np.array([values], dtype=np.float32)


# ---------------------------------------------------------------------------
# Category map helpers
# ---------------------------------------------------------------------------


def invert_category_map(category_map: dict[str, int]) -> dict[int, str]:
    """Invert category_map from {name: index} to {index: name}."""
    return {v: k for k, v in category_map.items()}


def ordered_categories(records: list[dict[str, Any]]) -> list[str]:
    """Return categories in stable order for the given prediction records."""
    seen = {
        record["expected_category"]
        for record in records
        if record.get("expected_category") is not None
    }
    seen.update(
        record["predicted_category"]
        for record in records
        if record.get("predicted_category") is not None
    )

    ordered = [category for category in CATEGORY_ORDER if category in seen]
    extras = sorted(seen.difference(ordered))
    return ordered + extras


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


# Maps sub_type labels to their canonical category, matching
# ContentSubType::category() in Rust types.rs.
SUB_TYPE_TO_CATEGORY: dict[str, str] = {
    # prose
    "plain": "prose",
    "markdown": "prose",
    "rst": "prose",
    "latex": "prose",
    # code
    "python": "code",
    "javascript": "code",
    "typescript": "code",
    "rust": "code",
    "go": "code",
    "java": "code",
    "sql": "code",
    "shell": "code",
    "css": "code",
    "dockerfile": "code",
    "makefile": "code",
    "html": "code",
    "xml": "code",
    "sgml": "code",
    "c_cpp": "code",
    "objc": "code",
    # structured
    "csv": "structured",
    "tsv": "structured",
    "pipe_table": "structured",
    "fixed_width": "structured",
    "json": "structured",
    "jsonl": "structured",
    "key_value": "structured",
    "log_lines": "structured",
    "yaml": "structured",
    "toml": "structured",
    "ini": "structured",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-x))


def predict_samples(
    session: Any,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    feature_extractor: Callable[[str], dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Run ONNX inference for each sample and return prediction records."""
    inv_sub_map = invert_category_map(config["sub_type_map"])
    detection_map: dict[str, int] | None = config.get("detection_map")
    inv_detection_map: dict[int, str] | None = None
    if detection_map is not None:
        inv_detection_map = {v: k for k, v in detection_map.items()}

    det_threshold: float = config.get("detection_threshold", 0.3)

    predictions: list[dict[str, Any]] = []
    if feature_extractor is None:
        from trainr.core.featurize import extract_all as feature_extractor

    for sample in tqdm(samples, desc="Evaluating"):
        text = sample.get("text", "")

        raw_features = feature_extractor(text)
        normalized = normalize_features(raw_features, config)

        outputs = session.run(None, {"features": normalized})
        # Marginalize category from sub-type probabilities
        sub_type_logits = outputs[1]
        shifted = sub_type_logits - sub_type_logits.max(axis=1, keepdims=True)
        sub_probs = np.exp(shifted) / np.exp(shifted).sum(
            axis=1, keepdims=True
        )
        cat_accum: dict[str, float] = {}
        for idx in range(sub_probs.shape[1]):
            label = inv_sub_map.get(idx, "unknown")
            if label not in SUB_TYPE_TO_CATEGORY and label != "unknown":
                import warnings
                warnings.warn(
                    f"Sub-type '{label}' not in SUB_TYPE_TO_CATEGORY — "
                    f"probability will accumulate as 'prose' (fallback). "
                    f"Add it to the mapping.",
                    stacklevel=2,
                )
            category = SUB_TYPE_TO_CATEGORY.get(label, "prose")
            cat_accum[category] = cat_accum.get(category, 0.0) + float(
                sub_probs[0, idx]
            )
        predicted = max(cat_accum, key=cat_accum.get)  # type: ignore[arg-type]
        record = build_prediction_record(sample, predicted)

        # Extract detection head outputs when available
        if len(outputs) >= 3 and inv_detection_map is not None:
            det_logits = outputs[2]
            det_probs = _sigmoid(det_logits)[0]
            detected: list[str] = []
            scores: dict[str, float] = {}
            for idx, label in sorted(inv_detection_map.items()):
                score = float(det_probs[idx])
                scores[label] = score
                if score >= det_threshold:
                    detected.append(label)
            record["detected_subtypes"] = detected
            record["detection_scores"] = scores

        predictions.append(record)

    return predictions


def run_evaluation(
    session: Any,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    feature_extractor: Callable[[str], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run ONNX inference on all samples and compute metrics."""
    predictions = predict_samples(
        session,
        config,
        samples,
        feature_extractor=feature_extractor,
    )
    categories = ordered_categories(predictions)

    y_true = [record["expected_category"] for record in predictions]
    y_pred = [record["predicted_category"] for record in predictions]

    return compute_metrics(y_true, y_pred, categories)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    categories: list[str],
) -> dict[str, Any]:
    """Compute accuracy, per-category precision/recall/F1, and confusion matrix.

    Returns a dict with keys: ``overall_accuracy``, ``per_category``,
    ``confusion_matrix``, ``categories``, ``total_samples``.
    """
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    overall_accuracy = correct / max(n, 1)

    # Build confusion matrix: rows = true, cols = predicted
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)
    cm = [[0] * n_cats for _ in range(n_cats)]
    for t, p in zip(y_true, y_pred):
        ti = cat_to_idx.get(t)
        pi = cat_to_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti][pi] += 1

    # Per-category metrics
    per_category: dict[str, dict[str, float]] = {}
    for i, cat in enumerate(categories):
        tp = cm[i][i]
        fn = sum(cm[i][j] for j in range(n_cats)) - tp
        fp = sum(cm[j][i] for j in range(n_cats)) - tp
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_category[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": support,
        }

    return {
        "overall_accuracy": overall_accuracy,
        "per_category": per_category,
        "confusion_matrix": cm,
        "categories": categories,
        "total_samples": n,
    }


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


def compute_detection_metrics(
    predictions: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    detection_map: dict[str, int],
) -> dict[str, Any]:
    """Compute per-label P/R/F1, micro/macro F1, and hamming loss.

    Args:
        predictions: Prediction records with ``detected_subtypes`` lists.
        samples: Ground-truth eval samples with ``det_*`` binary columns.
        detection_map: Mapping of label name to index from model config.

    Returns:
        Dict with ``per_label``, ``micro_f1``, ``macro_f1``, ``hamming_loss``.
    """
    labels = sorted(detection_map.keys(), key=lambda k: detection_map[k])
    n_samples = len(predictions)
    n_labels = len(labels)

    # Accumulators for micro averaging
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    total_hamming_errors = 0

    # Per-label accumulators
    label_tp: dict[str, int] = {l: 0 for l in labels}
    label_fp: dict[str, int] = {l: 0 for l in labels}
    label_fn: dict[str, int] = {l: 0 for l in labels}
    label_support: dict[str, int] = {l: 0 for l in labels}

    for pred, sample in zip(predictions, samples):
        detected_set = set(pred.get("detected_subtypes", []))
        for label in labels:
            gt = sample.get(f"det_{label}", 0)
            pred_val = 1 if label in detected_set else 0

            if gt == 1:
                label_support[label] += 1

            if pred_val == 1 and gt == 1:
                label_tp[label] += 1
                micro_tp += 1
            elif pred_val == 1 and gt == 0:
                label_fp[label] += 1
                micro_fp += 1
                total_hamming_errors += 1
            elif pred_val == 0 and gt == 1:
                label_fn[label] += 1
                micro_fn += 1
                total_hamming_errors += 1

    # Per-label metrics
    per_label: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for label in labels:
        tp = label_tp[label]
        fp = label_fp[label]
        fn = label_fn[label]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": label_support[label],
        }
        f1_scores.append(f1)

    # Micro F1
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # Macro F1
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Hamming loss
    total_slots = n_samples * n_labels
    hamming_loss = total_hamming_errors / total_slots if total_slots > 0 else 0.0

    return {
        "per_label": per_label,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "hamming_loss": hamming_loss,
    }


def format_detection_report(metrics: dict[str, Any]) -> str:
    """Format detection metrics as a human-readable report string."""
    lines: list[str] = []

    lines.append("")
    lines.append("── Detection Metrics " + "─" * 35)
    lines.append(
        f"{'Label':<18}{'Precision':>10}{'Recall':>10}{'F1':>8}{'N':>8}"
    )

    for label, m in metrics["per_label"].items():
        lines.append(
            f"{label:<18}{m['precision']:>10.2f}{m['recall']:>10.2f}"
            f"{m['f1']:>8.2f}{m['n']:>8}"
        )

    lines.append(
        f"Micro F1: {metrics['micro_f1']:.2f}    "
        f"Macro F1: {metrics['macro_f1']:.2f}    "
        f"Hamming: {metrics['hamming_loss']:.2f}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_report(metrics: dict[str, Any], eval_file: str) -> str:
    """Format metrics as a human-readable report string."""
    lines: list[str] = []

    lines.append(f"\n── Eval: {eval_file} {'─' * max(1, 50 - len(eval_file))}")
    lines.append(f"  Total samples:     {metrics.get('total_samples', 'N/A')}")
    lines.append(f"  Overall accuracy:  {metrics['overall_accuracy']:.3f}")

    lines.append("")
    lines.append("── Per-Category " + "─" * 40)
    lines.append(f"{'Category':<13}{'Precision':>10}{'Recall':>10}{'F1':>8}{'N':>8}")

    for cat, m in metrics["per_category"].items():
        lines.append(
            f"{cat:<13}{m['precision']:>10.2f}{m['recall']:>10.2f}"
            f"{m['f1']:>8.2f}{m['n']:>8}"
        )

    lines.append("")
    lines.append("── Confusion Matrix " + "─" * 36)

    categories = metrics["categories"]
    cm = metrics["confusion_matrix"]
    col_width = max(len(c) for c in categories) + 2
    header = " " * col_width + "".join(c.rjust(col_width) for c in categories)
    lines.append(header)

    for i, cat in enumerate(categories):
        row = cat.ljust(col_width) + "".join(
            str(cm[i][j]).rjust(col_width) for j in range(len(categories))
        )
        lines.append(row)

    return "\n".join(lines)


def format_json_report(metrics: dict[str, Any], eval_file: str) -> str:
    """Format metrics as a JSON string."""
    output = {
        "eval_file": eval_file,
        "overall_accuracy": metrics["overall_accuracy"],
        "total_samples": metrics.get("total_samples"),
        "per_category": metrics["per_category"],
        "confusion_matrix": metrics["confusion_matrix"],
        "categories": metrics["categories"],
    }
    return json.dumps(output, indent=2)


def write_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write prediction records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX model against eval JSONL or Parquet files."
    )
    parser.add_argument(
        "--model",
        default="output/model.onnx",
        help="Path to ONNX model (default: output/model.onnx)",
    )
    parser.add_argument(
        "--config",
        default="output/model_config.json",
        help="Path to model config JSON (default: output/model_config.json)",
    )
    parser.add_argument(
        "--eval",
        action="append",
        required=True,
        help="Path to eval JSONL or Parquet file (can be specified multiple times)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON instead of human-readable table",
    )
    parser.add_argument(
        "--predictions-output-dir",
        default=None,
        help=(
            "Optional directory for per-sample prediction JSONL files. "
            "Files are written as eval_predictions.<eval-stem>.jsonl."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import onnxruntime as ort

    args = parse_args(argv)

    config = load_config(args.config)
    session = ort.InferenceSession(args.model)

    for eval_path in args.eval:
        samples = load_eval_samples([eval_path])
        predictions = predict_samples(session, config, samples)
        metrics = compute_metrics(
            [record["expected_category"] for record in predictions],
            [record["predicted_category"] for record in predictions],
            ordered_categories(predictions),
        )

        if args.predictions_output_dir:
            output_path = (
                Path(args.predictions_output_dir)
                / f"eval_predictions.{Path(eval_path).stem}.jsonl"
            )
            write_prediction_records(output_path, predictions)

        # Auto-detect detection metrics
        det_metrics = None
        detection_map = config.get("detection_map")
        has_det_columns = any(
            key.startswith("det_") for sample in samples for key in sample
        )
        if detection_map and has_det_columns:
            det_metrics = compute_detection_metrics(
                predictions, samples, detection_map
            )

        if args.json:
            output = {
                "eval_file": Path(eval_path).name,
                "overall_accuracy": metrics["overall_accuracy"],
                "total_samples": metrics.get("total_samples"),
                "per_category": metrics["per_category"],
                "confusion_matrix": metrics["confusion_matrix"],
                "categories": metrics["categories"],
            }
            if det_metrics is not None:
                output["detection_metrics"] = det_metrics
            print(json.dumps(output, indent=2))
        else:
            print(format_report(metrics, Path(eval_path).name))
            if det_metrics is not None:
                print(format_detection_report(det_metrics))
