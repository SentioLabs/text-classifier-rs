# Contributing

Thanks for your interest in contributing to text-classifier-rs!

## Getting Started

```bash
git clone https://github.com/SentioLabs/text-classifier-rs.git
cd text-classifier-rs
cargo build
cargo test
```

## Development

### Running checks locally

Before submitting a PR, make sure these pass:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

### Adding test fixtures

Tests use text files in `tests/fixtures/` organized by category (`prose/`, `code/`, `tabular/`, `pdf_dump/`). Add new fixtures there when testing edge cases.

## Submitting Changes

1. Fork the repo and create a feature branch
2. Make your changes
3. Ensure all checks pass (see above)
4. Open a pull request against `main`

Please keep PRs focused — one logical change per PR.
