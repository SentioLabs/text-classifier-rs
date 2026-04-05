"""trainr — unified CLI for text-classifier training pipeline."""
import click


@click.group()
def main():
    """Training pipeline for text-classifier-rs."""
    pass


# Import and register subgroups
from trainr.commands.data import data
from trainr.commands.pipeline import pipeline
from trainr.commands.eval import eval_group
from trainr.commands.audit import audit

main.add_command(data)
main.add_command(pipeline)
main.add_command(eval_group, name="eval")
main.add_command(audit)
