import click


@click.group()
def audit():
    """Label auditing and correction commands."""
    pass


@audit.command()
@click.option("--input", required=True, help="Input JSONL or CSV file")
@click.option("--output", required=True, help="Output votes JSONL")
@click.option("--magika-min-confidence", type=float, default=0.50,
              help="Minimum Magika confidence to trigger LLM vote (default: 0.50)")
@click.option("--limit", type=int, default=0,
              help="Max samples to process (0 = all)")
@click.option("--backend", type=click.Choice(["anthropic", "openrouter"]),
              default="anthropic", help="LLM backend (default: anthropic)")
@click.option("--model", default=None,
              help="Override LLM model")
@click.option("--concurrency", type=int, default=10,
              help="Max concurrent LLM requests (default: 10)")
@click.option("--progress-interval", type=int, default=100,
              help="Print progress every N LLM calls (default: 100)")
def labels(**kwargs):
    """Three-way label audit: current label vs Magika vs LLM."""
    from trainr.core.audit_labels_vote import main

    argv = []
    for key, value in kwargs.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(value))
    main(argv)


@audit.command()
@click.option("--predictions", required=True, help="eval_predictions JSONL from eval_onnx.py")
@click.option("--output", required=True, help="Output votes JSONL")
@click.option("--backend", type=click.Choice(["anthropic", "openrouter"]),
              default="openrouter", help="LLM backend (default: openrouter)")
@click.option("--model", default=None, help="Override LLM model")
@click.option("--dual-llm", is_flag=True, help="Use dual-LLM 4-way voting")
@click.option("--ties-output", default=None, help="Output file for TIE verdicts")
@click.option("--haiku-model", default="anthropic/claude-haiku-4-5",
              help="Haiku model for dual-LLM mode")
@click.option("--gpt-model", default="openai/gpt-5.4-mini",
              help="GPT model for dual-LLM mode")
@click.option("--filter-subtypes", default=None,
              help="Comma-separated list of sub-types to audit")
@click.option("--concurrency", type=int, default=20,
              help="Max concurrent LLM requests (default: 20)")
def errors(**kwargs):
    """Audit model errors with LLM tie-breaking."""
    from trainr.core.audit_model_errors import main

    argv = []
    for key, value in kwargs.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(value))
    main(argv)


@audit.command()
@click.option("--input", default="training/output/eval_predictions.clear.jsonl",
              help="JSONL file with eval predictions")
@click.option("--output", default=None, help="Output JSONL file (default: stdout)")
@click.option("--errors-only", is_flag=True,
              help="Only output samples where our label disagrees with Magika")
def magika(**kwargs):
    """Cross-validate labels with Magika."""
    from trainr.core.audit_labels_magika import main

    argv = []
    for key, value in kwargs.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(value))
    main(argv)


@audit.command("apply")
@click.option("--input", required=True, help="Input file (JSONL, CSV, or Parquet)")
@click.option("--output", required=True, help="Output file")
@click.option("--votes", default=None, help="Votes JSONL from audit_labels_vote.py")
@click.option("--remap-subtypes-only", is_flag=True,
              help="Only remap ghost sub-types, skip category corrections")
@click.option("--no-remap-subtypes", is_flag=True,
              help="Only apply category corrections, skip sub-type remapping")
def apply_cmd(**kwargs):
    """Apply voted label corrections and remap ghost sub-types."""
    from trainr.core.apply_corrections import main

    argv = []
    for key, value in kwargs.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.append(flag)
            argv.append(str(value))
    main(argv)
