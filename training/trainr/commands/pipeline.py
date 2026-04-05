import click


@click.group()
def pipeline():
    """Core training pipeline commands."""
    pass


@pipeline.command()
@click.option(
    "--input",
    "input_path",
    default="data/curated/train/golden_raw.parquet",
    help="Input parquet path (default: data/curated/train/golden_raw.parquet)",
)
@click.option(
    "--output",
    "output_path",
    default="data/curated/train/golden_featurized.parquet",
    help="Output parquet path (default: data/curated/train/golden_featurized.parquet)",
)
def featurize(input_path: str, output_path: str) -> None:
    """Compute structural text features and enrich a dataset."""
    from trainr.core.featurize import main as featurize_main

    argv = ["--input", input_path, "--output", output_path]
    featurize_main(argv)


@pipeline.command()
@click.option(
    "--input",
    "input_path",
    default="data/curated/train/golden_featurized.parquet",
    help="Path to input parquet (default: data/curated/train/golden_featurized.parquet)",
)
@click.option(
    "--output",
    "output_path",
    default="data/curated/train/golden_train.parquet",
    help="Path to output parquet (default: data/curated/train/golden_train.parquet)",
)
@click.option(
    "--feature-threshold",
    type=float,
    default=0.1,
    help="L2 distance threshold for feature dedup (default: 0.1)",
)
@click.option(
    "--semantic-threshold",
    type=float,
    default=0.9,
    help="Cosine similarity threshold for semantic dedup (default: 0.9)",
)
def dedup(
    input_path: str,
    output_path: str,
    feature_threshold: float,
    semantic_threshold: float,
) -> None:
    """FAISS two-layer deduplication pipeline."""
    from trainr.core.dedup import main as dedup_main

    import sys

    sys.argv = [
        "dedup",
        "--input", input_path,
        "--output", output_path,
        "--feature-threshold", str(feature_threshold),
        "--semantic-threshold", str(semantic_threshold),
    ]
    dedup_main()


@pipeline.command()
@click.option(
    "--data",
    required=True,
    help="Path to combined Parquet training data.",
)
@click.option(
    "--output",
    required=True,
    help="Directory to write model.onnx, model_config.json, and metrics.json.",
)
@click.option("--epochs", type=int, default=200, help="Maximum training epochs (default: 200).")
@click.option("--batch-size", type=int, default=64, help="Training batch size (default: 64).")
@click.option("--lr", type=float, default=0.001, help="Learning rate (default: 0.001).")
@click.option("--patience", type=int, default=15, help="Early stopping patience (default: 15).")
@click.option(
    "--drop-features",
    multiple=True,
    default=[],
    help="Feature names to drop (comma-separated and/or repeated).",
)
@click.option(
    "--group-val-by-source",
    is_flag=True,
    default=False,
    help="Perform validation split by disjoint source groups.",
)
@click.option("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0, etc. (default: auto)")
@click.option("--dropout", type=float, default=0.15, help="Dropout rate (default: 0.15).")
@click.option("--hidden-dim", type=int, default=256, help="First hidden layer width (default: 256).")
@click.option("--sub-type-weight", type=float, default=0.5, help="Sub-type loss weight (default: 0.5).")
@click.option("--no-batchnorm", is_flag=True, default=False, help="Disable BatchNorm layers.")
@click.option("--warmup-epochs", type=int, default=10, help="LR warmup epochs (default: 10).")
def train(
    data: str,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    drop_features: tuple[str, ...],
    group_val_by_source: bool,
    device: str,
    dropout: float,
    hidden_dim: int,
    sub_type_weight: float,
    no_batchnorm: bool,
    warmup_epochs: int,
) -> None:
    """Train a dual-head text classifier and export to ONNX."""
    from trainr.core.train import main as train_main

    argv = ["--data", data, "--output", output]
    argv += ["--epochs", str(epochs)]
    argv += ["--batch-size", str(batch_size)]
    argv += ["--lr", str(lr)]
    argv += ["--patience", str(patience)]
    argv += ["--device", device]
    argv += ["--dropout", str(dropout)]
    argv += ["--hidden-dim", str(hidden_dim)]
    argv += ["--sub-type-weight", str(sub_type_weight)]
    argv += ["--warmup-epochs", str(warmup_epochs)]
    if no_batchnorm:
        argv.append("--no-batchnorm")
    if group_val_by_source:
        argv.append("--group-val-by-source")
    for feat in drop_features:
        argv += ["--drop-features", feat]
    train_main(argv)
