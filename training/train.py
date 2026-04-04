#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "pandas", "numpy", "onnx", "onnxscript", "onnxruntime", "scikit-learn"]
# ///
"""Train a dual-head feedforward neural network on structural text features.

Reads CSV files produced by generate.py and exports the trained model to ONNX
format along with configuration and metrics JSON files.

Usage:
    python training/train.py --data training/data/combined.csv \
        --output training/models/ [--epochs 200] [--batch-size 64] \
        [--lr 0.001] [--patience 15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    from .featurize import FEATURES
except ImportError:
    from featurize import FEATURES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = tuple(FEATURES.keys())

CATEGORY_MAP: dict[str, int] = {
    "prose": 0,
    "code": 1,
    "structured": 2,
}

NUM_CATEGORIES = len(CATEGORY_MAP)


def _parse_drop_features(raw_drop_features: list[str]) -> list[str]:
    """Parse repeated/comma-separated --drop-features argument values."""
    parsed: list[str] = []
    for value in raw_drop_features:
        for feature_name in value.split(","):
            stripped = feature_name.strip()
            if stripped:
                parsed.append(stripped)
    return parsed


def resolve_feature_columns(raw_drop_features: list[str]) -> tuple[str, ...]:
    """Resolve active feature columns after dropping requested features."""
    drop_features = _parse_drop_features(raw_drop_features)
    drop_set = set(drop_features)

    unknown = sorted(drop_set - set(FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown features in --drop-features: {', '.join(unknown)}")

    active_features = tuple(feature for feature in FEATURE_COLUMNS if feature not in drop_set)
    if not active_features:
        raise ValueError(
            "All features were dropped via --drop-features; at least one feature must remain."
        )

    return active_features


def _row_level_train_val_split(
    indices: np.ndarray,
    y_cat: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows with stratification; fall back to random split if needed."""
    try:
        return train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=y_cat,
        )
    except ValueError as exc:
        print(
            "Warning: stratified split failed "
            f"({exc}). Falling back to non-stratified row split.",
            file=sys.stderr,
        )
        return train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            stratify=None,
        )


def _grouped_train_val_split(
    indices: np.ndarray,
    y_cat: np.ndarray,
    source_values: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a grouped split that approximates row-level val_fraction."""
    unique_sources = np.unique(source_values)
    if len(unique_sources) < 2:
        return None

    n_rows = len(indices)
    n_sources = len(unique_sources)
    overall_dist = np.bincount(y_cat, minlength=NUM_CATEGORIES).astype(np.float64) / n_rows
    rng = np.random.default_rng(seed)
    n_trials = max(64, min(512, len(unique_sources) * 32))
    candidate_group_fractions = [k / n_sources for k in range(1, n_sources)]
    trials_per_fraction = max(4, min(32, n_trials // len(candidate_group_fractions)))
    best: dict[str, np.ndarray | float] | None = None

    for group_fraction in candidate_group_fractions:
        for _ in range(trials_per_fraction):
            trial_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=group_fraction,
                random_state=trial_seed,
            )
            try:
                train_idx, val_idx = next(splitter.split(indices, y=y_cat, groups=source_values))
            except ValueError:
                continue

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            val_ratio = len(val_idx) / n_rows
            row_error = abs(val_ratio - val_fraction)

            train_dist = np.bincount(y_cat[train_idx], minlength=NUM_CATEGORIES).astype(np.float64)
            train_dist /= len(train_idx)
            val_dist = np.bincount(y_cat[val_idx], minlength=NUM_CATEGORIES).astype(np.float64)
            val_dist /= len(val_idx)
            dist_error = float(
                np.abs(train_dist - overall_dist).mean() + np.abs(val_dist - overall_dist).mean()
            )

            if (
                best is None
                or row_error < best["row_error"]
                or (row_error == best["row_error"] and dist_error < best["dist_error"])
            ):
                best = {
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "row_error": row_error,
                    "dist_error": dist_error,
                    "val_ratio": val_ratio,
                }

    if best is None:
        return None

    max_row_drift = max(0.02, min(0.05, val_fraction * 0.25))
    if best["row_error"] > max_row_drift:
        return None

    return best["train_idx"], best["val_idx"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_prepare_data(
    csv_path: Path | str,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    val_fraction: float = 0.2,
    seed: int = 42,
    group_val_by_source: bool = False,
) -> dict:
    """Load CSV, encode labels, standardize features, and split."""
    df = pd.read_csv(csv_path)

    # Extract feature matrix
    X = df[list(feature_columns)].values.astype(np.float32)

    # Encode category labels
    y_cat = df["category"].map(CATEGORY_MAP).values.astype(np.int64)

    # Fill missing sub_type values with "unknown"
    df["sub_type"] = df["sub_type"].fillna("unknown")

    # Build sub_type label map from unique values in the data
    unique_sub_types = sorted(df["sub_type"].unique())
    sub_type_map: dict[str, int] = {st: i for i, st in enumerate(unique_sub_types)}
    y_sub = df["sub_type"].map(sub_type_map).values.astype(np.int64)

    indices = np.arange(len(df))
    source_values = (
        df["source"].fillna("unknown_source").astype(str).values if "source" in df.columns else None
    )

    # Split by grouped source when requested and viable; otherwise use row-level split.
    train_idx: np.ndarray
    val_idx: np.ndarray
    if group_val_by_source:
        if source_values is None:
            print(
                "Warning: --group-val-by-source requested but no source column found. "
                "Using row-level split.",
                file=sys.stderr,
            )
            train_idx, val_idx = _row_level_train_val_split(indices, y_cat, val_fraction, seed)
        else:
            grouped = _grouped_train_val_split(indices, y_cat, source_values, val_fraction, seed)
            if grouped is None:
                print(
                    "Warning: source-grouped validation split could not meet safety constraints. "
                    "Using row-level split.",
                    file=sys.stderr,
                )
                train_idx, val_idx = _row_level_train_val_split(indices, y_cat, val_fraction, seed)
            else:
                train_idx, val_idx = grouped
    else:
        train_idx, val_idx = _row_level_train_val_split(indices, y_cat, val_fraction, seed)

    X_train = X[train_idx]
    X_val = X[val_idx]
    y_cat_train = y_cat[train_idx]
    y_cat_val = y_cat[val_idx]
    y_sub_train = y_sub[train_idx]
    y_sub_val = y_sub[val_idx]

    # Z-score standardization (fit on training set only)
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    # Avoid division by zero for constant features
    feature_std[feature_std == 0] = 1.0

    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std

    # Inverse-frequency class weights so minority classes are penalized more
    class_counts = np.bincount(y_cat_train, minlength=NUM_CATEGORIES)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    result = {
        "X_train": X_train,
        "X_val": X_val,
        "y_cat_train": y_cat_train,
        "y_cat_val": y_cat_val,
        "y_sub_train": y_sub_train,
        "y_sub_val": y_sub_val,
        "feature_names": list(feature_columns),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "sub_type_map": sub_type_map,
        "class_weights": class_weights,
    }
    if source_values is not None:
        result["source_train"] = source_values[train_idx]
        result["source_val"] = source_values[val_idx]
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TextClassifier(nn.Module):
    """Dual-head feedforward network for category and sub-type classification."""

    def __init__(
        self,
        n_features: int = len(FEATURE_COLUMNS),
        n_categories: int = NUM_CATEGORIES,
        n_sub_types: int = 33,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.category_head = nn.Linear(32, n_categories)
        self.sub_type_head = nn.Linear(32, n_sub_types)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        return self.category_head(shared), self.sub_type_head(shared)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a torch.device.

    When *device_str* is ``"auto"``, CUDA is selected when available,
    otherwise CPU is used.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    data: dict,
    n_sub_types: int,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 15,
    device: torch.device | None = None,
) -> tuple[TextClassifier, dict]:
    """Train the model and return it along with metrics."""
    if device is None:
        device = resolve_device("auto")
    print(f"Training on: {device}")

    X_train = torch.from_numpy(data["X_train"]).to(device)
    X_val = torch.from_numpy(data["X_val"]).to(device)
    y_cat_train = torch.from_numpy(data["y_cat_train"]).to(device)
    y_cat_val = torch.from_numpy(data["y_cat_val"]).to(device)
    y_sub_train = torch.from_numpy(data["y_sub_train"]).to(device)
    y_sub_val = torch.from_numpy(data["y_sub_val"]).to(device)

    model = TextClassifier(
        n_features=len(data["feature_names"]),
        n_categories=NUM_CATEGORIES,
        n_sub_types=n_sub_types,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    weight_tensor = torch.tensor(data["class_weights"], dtype=torch.float32).to(device)
    cat_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    sub_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    total_epochs = 0

    dataset = torch.utils.data.TensorDataset(X_train, y_cat_train, y_sub_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_epochs = epoch

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_cat_correct = 0
        train_count = 0

        for batch_x, batch_y_cat, batch_y_sub in loader:
            optimizer.zero_grad()
            cat_logits, sub_logits = model(batch_x)
            loss = cat_criterion(cat_logits, batch_y_cat) + 0.3 * sub_criterion(
                sub_logits, batch_y_sub
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_x)
            train_cat_correct += (cat_logits.argmax(dim=1) == batch_y_cat).sum().item()
            train_count += len(batch_x)

        train_loss /= train_count
        train_acc = train_cat_correct / train_count

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_cat_logits, val_sub_logits = model(X_val)
            val_loss = (
                cat_criterion(val_cat_logits, y_cat_val)
                + 0.3 * sub_criterion(val_sub_logits, y_sub_val)
            ).item()
            val_cat_acc = (val_cat_logits.argmax(dim=1) == y_cat_val).float().mean().item()
            val_sub_acc = (val_sub_logits.argmax(dim=1) == y_sub_val).float().mean().item()

        print(
            f"epoch {epoch:>4d}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_cat_acc={val_cat_acc:.4f}  "
            f"val_sub_acc={val_sub_acc:.4f}",
            file=sys.stderr,
        )

        # --- Learning rate scheduling ---
        scheduler.step(val_loss)

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch} (patience={patience})",
                    file=sys.stderr,
                )
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics with best model
    model.eval()
    with torch.no_grad():
        val_cat_logits, val_sub_logits = model(X_val)
        final_cat_acc = (val_cat_logits.argmax(dim=1) == y_cat_val).float().mean().item()
        final_sub_acc = (val_sub_logits.argmax(dim=1) == y_sub_val).float().mean().item()

    metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "val_category_accuracy": final_cat_acc,
        "val_sub_type_accuracy": final_sub_acc,
    }

    return model, metrics


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_onnx(model: TextClassifier, output_path: Path, n_features: int) -> None:
    """Export the trained model to ONNX format.

    The model is moved to CPU before export so the ONNX graph is
    device-agnostic regardless of the training device.
    """
    model = model.cpu()
    model.eval()  # noqa: eval – this is nn.Module.eval(), not builtin eval
    dummy_input = torch.randn(1, n_features)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["category_logits", "sub_type_logits"],
        dynamic_axes={
            "features": {0: "batch"},
            "category_logits": {0: "batch"},
            "sub_type_logits": {0: "batch"},
        },
        opset_version=17,
    )


def save_config(
    output_path: Path,
    feature_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    sub_type_map: dict[str, int],
) -> None:
    """Save model configuration (feature stats, label maps) to JSON."""
    config = {
        "feature_names": feature_names,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "category_map": CATEGORY_MAP,
        "sub_type_map": sub_type_map,
    }
    output_path.write_text(json.dumps(config, indent=2) + "\n")


def save_metrics(output_path: Path, metrics: dict) -> None:
    """Save training metrics to JSON."""
    output_path.write_text(json.dumps(metrics, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dual-head text classifier and export to ONNX."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to combined CSV training data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write model.onnx, model_config.json, and metrics.json.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs (default: 200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15).",
    )
    parser.add_argument(
        "--drop-features",
        action="append",
        default=[],
        help=(
            "Feature names to remove from training (comma-separated and/or repeated). "
            "Example: --drop-features tab_density,page_number_density"
        ),
    )
    parser.add_argument(
        "--group-val-by-source",
        action="store_true",
        help=(
            "If a 'source' column exists, perform validation split by disjoint source groups "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for training: auto, cpu, cuda, cuda:0, etc. (default: auto)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    active_feature_columns = resolve_feature_columns(args.drop_features)

    print(f"Loading data from {args.data}", file=sys.stderr)
    data = load_and_prepare_data(
        args.data,
        feature_columns=active_feature_columns,
        group_val_by_source=args.group_val_by_source,
    )
    n_sub_types = len(data["sub_type_map"])

    print(
        f"Training: {len(data['X_train'])} train, {len(data['X_val'])} val, "
        f"{n_sub_types} sub-types",
        file=sys.stderr,
    )

    device = resolve_device(args.device)

    model, metrics = train_model(
        data,
        n_sub_types=n_sub_types,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )

    onnx_path = args.output / "model.onnx"
    export_onnx(model, onnx_path, n_features=len(data["feature_names"]))
    print(f"Exported ONNX model to {onnx_path}", file=sys.stderr)

    config_path = args.output / "model_config.json"
    save_config(
        config_path,
        data["feature_names"],
        data["feature_mean"],
        data["feature_std"],
        data["sub_type_map"],
    )
    print(f"Saved config to {config_path}", file=sys.stderr)

    metrics_path = args.output / "metrics.json"
    save_metrics(metrics_path, metrics)
    print(f"Saved metrics to {metrics_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
