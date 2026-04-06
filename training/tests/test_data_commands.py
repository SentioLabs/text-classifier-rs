"""Tests for trainr.commands.data — CLI subcommand wiring."""

from click.testing import CliRunner


class TestRelabelUnknownsCommand:
    def test_subcommand_registered(self):
        """relabel-unknowns appears in the data group's commands."""
        from trainr.commands.data import data

        assert "relabel-unknowns" in data.commands

    def test_help_exits_zero(self):
        """--help exits with code 0."""
        from trainr.commands.data import data

        runner = CliRunner()
        result = runner.invoke(data, ["relabel-unknowns", "--help"])
        assert result.exit_code == 0

    def test_help_shows_description(self):
        """--help includes the docstring."""
        from trainr.commands.data import data

        runner = CliRunner()
        result = runner.invoke(data, ["relabel-unknowns", "--help"])
        assert "Relabel unknown sub_type rows" in result.output

    def test_help_shows_all_options(self):
        """All five options appear in help output."""
        from trainr.commands.data import data

        runner = CliRunner()
        result = runner.invoke(data, ["relabel-unknowns", "--help"])
        assert "--input" in result.output
        assert "--output" in result.output
        assert "--model" in result.output
        assert "--concurrency" in result.output
        assert "--manual-review" in result.output

    def test_input_and_output_are_required(self):
        """Invoking without --input and --output should fail."""
        from trainr.commands.data import data

        runner = CliRunner()
        result = runner.invoke(data, ["relabel-unknowns"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_relabel_unknowns_in_data_help(self):
        """relabel-unknowns appears in the data group help listing."""
        from trainr.commands.data import data

        runner = CliRunner()
        result = runner.invoke(data, ["--help"])
        assert "relabel-unknowns" in result.output
