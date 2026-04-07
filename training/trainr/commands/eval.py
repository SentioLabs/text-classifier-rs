import click

from trainr.core.eval_onnx import main as eval_onnx_main
from trainr.core.analyze_eval import main as analyze_eval_main


@click.group()
def eval_group():
    """Model evaluation commands."""
    pass


@eval_group.command(
    name="run",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def run(args: tuple[str, ...]) -> None:
    """Run ONNX model evaluation against eval JSONL or Parquet files."""
    eval_onnx_main(list(args))


@eval_group.command(
    name="analyze",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def analyze(args: tuple[str, ...]) -> None:
    """Analyze eval_onnx prediction records into slice reports."""
    analyze_eval_main(list(args))


@eval_group.command(name="verify")
@click.option(
    "--manifest",
    required=True,
    help="Path to training_manifest.json",
)
@click.option("--dataset", default=None, help="Path to training dataset (for SHA check)")
@click.option("--eval-clear", default=None, help="Path to clear eval JSONL")
@click.option("--eval-boundary", default=None, help="Path to boundary eval JSONL")
def verify(manifest: str, dataset: str | None, eval_clear: str | None, eval_boundary: str | None) -> None:
    """Verify eval artifacts are in sync with the training manifest."""
    from trainr.core.manifest import TrainingManifest, compute_file_sha256

    m = TrainingManifest.load(manifest)
    click.echo(f"Manifest: {m.run_id} ({m.timestamp})")
    click.echo(f"  Dataset: {m.dataset_rows} rows, SHA={m.dataset_sha256[:12]}...")
    click.echo(f"  Features: {m.feature_count} ({m.featurizer_version})")
    click.echo(f"  Model SHA: {m.model_sha256[:12]}...")

    current_dataset_sha = compute_file_sha256(dataset) if dataset else None
    current_clear_sha = compute_file_sha256(eval_clear) if eval_clear else None
    current_boundary_sha = compute_file_sha256(eval_boundary) if eval_boundary else None

    issues = m.verify(
        current_dataset_sha256=current_dataset_sha,
        current_eval_clear_sha256=current_clear_sha,
        current_eval_boundary_sha256=current_boundary_sha,
    )

    if issues:
        click.echo("\nDRIFT DETECTED:")
        for issue in issues:
            click.echo(f"  WARNING: {issue}")
        raise SystemExit(1)
    else:
        click.echo("\nAll checks passed — artifacts in sync.")
