#!/usr/bin/env python3
"""Drop synthetic (source=='unknown') rows from the golden training set.

Filters out all rows where ``source`` is exactly ``"unknown"`` (case-sensitive),
archives them separately, and overwrites the input file with the filtered data.
"""

import argparse
from pathlib import Path

import polars as pl


def filter_synthetic(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a DataFrame into real and synthetic rows.

    Args:
        df: Input DataFrame that must contain a ``source`` column.

    Returns:
        A tuple of (filtered, dropped) DataFrames. ``filtered`` contains rows
        where ``source != "unknown"``; ``dropped`` contains the removed rows.
    """
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

    df = pl.read_parquet(input_path)
    total_before = df.height

    filtered, dropped = filter_synthetic(df)
    total_after = filtered.height
    rows_dropped = dropped.height

    # Write archive if there are rows to archive
    if rows_dropped > 0:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / "synthetic_dropped.parquet"
        dropped.write_parquet(archive_path)
    else:
        print("No synthetic rows found — skipping archive.")

    # Atomic overwrite: write to temp file, then rename
    tmp_path = input_path.with_suffix(".tmp.parquet")
    filtered.write_parquet(tmp_path)
    tmp_path.rename(input_path)

    # Print summary
    print(f"Total before: {total_before}")
    print(f"Total after:  {total_after}")
    print(f"Rows dropped: {rows_dropped}")

    if "sub_type" in df.columns:
        print("\nPer-sub_type distribution BEFORE:")
        before_dist = (
            df.group_by("sub_type")
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

    run_drop_synthetic(
        input_path=args.input,
        archive_dir=args.archive_dir,
    )


if __name__ == "__main__":
    main()
