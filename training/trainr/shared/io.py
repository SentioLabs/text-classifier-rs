"""Common I/O utilities for reading/writing JSONL and Parquet files."""
import json
from pathlib import Path

import polars as pl


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_parquet(path: str | Path) -> pl.DataFrame:
    """Read a Parquet file into a Polars DataFrame."""
    return pl.read_parquet(str(path))


def write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    """Write a Polars DataFrame to a Parquet file."""
    df.write_parquet(str(path))
