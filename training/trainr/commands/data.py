import click


@click.group()
def data():
    """Data sourcing and generation commands."""
    pass


@data.command("generate")
@click.option("--output", default=None, help="Output JSONL file path.")
@click.option("--total-samples", type=int, default=None, help="Total number of samples to generate.")
@click.option("--pilot", is_flag=True, default=False, help="Pilot mode: generate ~500 samples.")
@click.option("--dry-run", is_flag=True, default=False, help="Print plan without making API calls.")
@click.option("--resume", is_flag=True, default=False, help="Resume from existing output file.")
def generate_cmd(**kwargs):
    """Generate synthetic training data via OpenRouter multi-model API."""
    from trainr.core.generate_openrouter import main as _main

    argv = _build_argv(kwargs, flags=("pilot", "dry_run", "resume"))
    _main(argv)


@data.command("generate-eval")
@click.option("--mode", type=click.Choice(["clear", "boundary", "all"]), default=None, help="What to generate.")
@click.option("--output-dir", default=None, help="Directory to write eval JSONL files.")
@click.option("--samples-per-category", type=int, default=None, help="Clear samples per category.")
@click.option("--samples-per-pair", type=int, default=None, help="Boundary samples per pair.")
@click.option("--model", default=None, help="OpenAI model to use.")
@click.option("--dry-run", is_flag=True, default=False, help="Print plan without calling the API.")
def generate_eval_cmd(**kwargs):
    """Generate golden eval set using the OpenAI GPT-5.4 API."""
    from trainr.core.generate_eval import main as _main

    argv = _build_argv(kwargs, flags=("dry_run",))
    _main(argv)


@data.command("generate-fixtures")
@click.option("--mode", type=click.Choice(["all", "fixtures", "synthetic", "perturb", "test-set", "ambiguous-test-set", "golden-train"]), default=None, help="Generation mode.")
@click.option("--output", default=None, help="Output directory.")
@click.option("--samples-per-type", type=int, default=None, help="Samples per (category, sub_type) pair.")
@click.option("--api-key", default=None, help="Anthropic API key.")
@click.option("--model", default=None, help="Claude model for generation.")
@click.option("--dry-run", is_flag=True, default=False, help="Print plan without calling the API.")
def generate_fixtures_cmd(**kwargs):
    """Generate synthetic training data for the text classifier."""
    from trainr.core.generate import main as _main

    argv = _build_argv(kwargs, flags=("dry_run",))
    _main(argv)


@data.command("sample")
@click.option("--source", type=click.Choice(["stack", "real", "all"]), default="all", help="Which data source to sample.")
@click.option("--output", default=None, help="Output JSONL file path.")
@click.option("--dry-run", is_flag=True, default=False, help="Print plan without downloading.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
def sample_cmd(**kwargs):
    """Sample real data from The Stack and HuggingFace datasets."""
    from trainr.core.sample import main as _main

    argv = _build_argv(kwargs, flags=("dry_run",))
    _main(argv)


@data.command("split")
@click.option("--input", default=None, help="Path to input JSONL file.")
@click.option("--eval-output", default=None, help="Path for eval clear output JSONL.")
@click.option("--eval-boundary-output", default=None, help="Path for eval boundary output JSONL.")
@click.option("--train-output", default=None, help="Path for training output Parquet.")
@click.option("--max-per-category", type=int, default=None, help="Max training samples per category (0 = no limit).")
@click.option("--skip-eval", is_flag=True, default=False, help="Skip eval set generation.")
def split_cmd(**kwargs):
    """Split raw JSONL dataset into eval and training sets."""
    from trainr.core.split_dataset import main as _main

    argv = _build_argv(kwargs, flags=("skip_eval",))
    _main(argv)


def _build_argv(kwargs: dict, flags: tuple[str, ...] = ()) -> list[str]:
    """Convert click kwargs to an argv list for argparse-based entry points."""
    argv: list[str] = []
    for key, value in kwargs.items():
        if value is None:
            continue
        cli_key = "--" + key.replace("_", "-")
        if key in flags:
            if value:
                argv.append(cli_key)
        else:
            argv.append(cli_key)
            argv.append(str(value))
    return argv
