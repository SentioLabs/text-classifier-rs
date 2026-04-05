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
