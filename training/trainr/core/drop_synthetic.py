#!/usr/bin/env python3
"""Drop synthetic (source=='unknown') rows from the golden training set.

Filters out all rows where ``source`` is exactly ``"unknown"`` (case-sensitive),
archives them separately, and overwrites the input file with the filtered data.
"""

import argparse
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def filter_synthetic(input_path: Path | str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read a parquet file and split into real and synthetic rows.

    Args:
        input_path: Path to the input parquet file.

    Returns:
        A tuple of (filtered, dropped) DataFrames. ``filtered`` contains rows
        where ``source != "unknown"``; ``dropped`` contains the removed rows.
    """
    df = pl.read_parquet(input_path)
    mask = pl.col("source") == "unknown"
    filtered = df.filter(~mask)
    dropped = df.filter(mask)
    return filtered, dropped


def run_drop_synthetic(
    input_path: Path | str,
    archive_dir: Path | str,
) -> None:
    """Run the full drop-synthetic pipeline: filter, archive, overwrite.

    Args:
        input_path: Path to the golden training parquet.
        archive_dir: Directory to write the archived synthetic rows.
    """
    input_path = Path(input_path)
    archive_dir = Path(archive_dir)

    df_original = pl.read_parquet(input_path)
    total_before = df_original.height

    filtered, dropped = filter_synthetic(input_path)
    total_after = filtered.height
    rows_dropped = dropped.height

    # Create archive directory if needed
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "synthetic_dropped.parquet"

    # Write archive first, then overwrite input
    dropped.write_parquet(archive_path)
    filtered.write_parquet(input_path)

    # Print summary
    print(f"Total before: {total_before}")
    print(f"Total after:  {total_after}")
    print(f"Rows dropped: {rows_dropped}")

    if "sub_type" in df_original.columns:
        print("\nPer-sub_type distribution BEFORE:")
        before_dist = (
            df_original.group_by("sub_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        print(before_dist)

        print("\nPer-sub_type distribution AFTER:")
        after_dist = (
            filtered.group_by("sub_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        print(after_dist)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Drop synthetic (source=='unknown') rows from golden training set"
    )
    parser.add_argument(
        "--input",
        default="data/curated/train/golden_train.parquet",
        help="Path to input parquet (default: data/curated/train/golden_train.parquet)",
    )
    parser.add_argument(
        "--archive-dir",
        default="data/archive",
        help="Directory to write archived synthetic rows (default: data/archive)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_drop_synthetic(
        input_path=args.input,
        archive_dir=args.archive_dir,
    )


if __name__ == "__main__":
    main()
