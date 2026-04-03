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
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = (
    "line_length_cv",
    "char_entropy",
    "leading_whitespace_ratio",
    "tab_density",
    "sentence_punctuation_rate",
    "paragraph_break_rate",
    "alpha_ratio",
    "line_uniqueness",
    "short_line_ratio",
    "symbol_ratio",
    "delimiter_consistency",
    "json_brace_depth",
    "key_value_ratio",
    "xml_tag_ratio",
    "log_line_ratio",
    "comment_ratio",
    "numeric_field_ratio",
    "repetitive_structure_score",
    # Content-level features
    "dictionary_word_ratio",
    "encoding_error_ratio",
    "repeated_ngram_ratio",
    "sentence_coherence_score",
)

CATEGORY_MAP: dict[str, int] = {
    "prose": 0,
    "code": 1,
    "structured": 2,
    "artifact": 3,
    "skip": 4,
}

NUM_CATEGORIES = len(CATEGORY_MAP)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_prepare_data(
    csv_path: Path | str,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Load CSV, encode labels, standardize features, and split."""
    df = pd.read_csv(csv_path)

    # Extract feature matrix
    X = df[list(FEATURE_COLUMNS)].values.astype(np.float32)

    # Encode category labels
    y_cat = df["category"].map(CATEGORY_MAP).values.astype(np.int64)

    # Fill missing sub_type values with "unknown"
    df["sub_type"] = df["sub_type"].fillna("unknown")

    # Build sub_type label map from unique values in the data
    unique_sub_types = sorted(df["sub_type"].unique())
    sub_type_map: dict[str, int] = {st: i for i, st in enumerate(unique_sub_types)}
    y_sub = df["sub_type"].map(sub_type_map).values.astype(np.int64)

    # Stratified split on category
    X_train, X_val, y_cat_train, y_cat_val, y_sub_train, y_sub_val = train_test_split(
        X,
        y_cat,
        y_sub,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_cat,
    )

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

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_cat_train": y_cat_train,
        "y_cat_val": y_cat_val,
        "y_sub_train": y_sub_train,
        "y_sub_val": y_sub_val,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "sub_type_map": sub_type_map,
        "class_weights": class_weights,
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TextClassifier(nn.Module):
    """Dual-head feedforward network for category and sub-type classification."""

    def __init__(
        self,
        n_features: int = 22,
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
# Training
# ---------------------------------------------------------------------------


def train_model(
    data: dict,
    n_sub_types: int,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 15,
) -> tuple[TextClassifier, dict]:
    """Train the model and return it along with metrics."""
    device = torch.device("cpu")

    X_train = torch.from_numpy(data["X_train"]).to(device)
    X_val = torch.from_numpy(data["X_val"]).to(device)
    y_cat_train = torch.from_numpy(data["y_cat_train"]).to(device)
    y_cat_val = torch.from_numpy(data["y_cat_val"]).to(device)
    y_sub_train = torch.from_numpy(data["y_sub_train"]).to(device)
    y_sub_val = torch.from_numpy(data["y_sub_val"]).to(device)

    model = TextClassifier(
        n_features=len(FEATURE_COLUMNS),
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


def export_onnx(model: TextClassifier, output_path: Path) -> None:
    """Export the trained model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, len(FEATURE_COLUMNS))
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
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    sub_type_map: dict[str, int],
) -> None:
    """Save model configuration (feature stats, label maps) to JSON."""
    config = {
        "feature_names": list(FEATURE_COLUMNS),
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data}", file=sys.stderr)
    data = load_and_prepare_data(args.data)
    n_sub_types = len(data["sub_type_map"])

    print(
        f"Training: {len(data['X_train'])} train, {len(data['X_val'])} val, "
        f"{n_sub_types} sub-types",
        file=sys.stderr,
    )

    model, metrics = train_model(
        data,
        n_sub_types=n_sub_types,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    onnx_path = args.output / "model.onnx"
    export_onnx(model, onnx_path)
    print(f"Exported ONNX model to {onnx_path}", file=sys.stderr)

    config_path = args.output / "model_config.json"
    save_config(config_path, data["feature_mean"], data["feature_std"], data["sub_type_map"])
    print(f"Saved config to {config_path}", file=sys.stderr)

    metrics_path = args.output / "metrics.json"
    save_metrics(metrics_path, metrics)
    print(f"Saved metrics to {metrics_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
