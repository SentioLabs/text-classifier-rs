# Python Example

Demonstrates using `text-classifier` from Python via [uv](https://docs.astral.sh/uv/).

## Setup

```bash
cd examples/python
uv sync
```

This installs `text-classifier` from the GitHub release tag. A Rust toolchain is required since maturin compiles the native extension locally.

## Run

```bash
uv run classify_demo.py
```
