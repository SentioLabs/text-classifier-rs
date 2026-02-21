.PHONY: build build-release test test-single fmt fmt-check clippy lint check clean \
	python-setup python-build \
	install run review release-local help

# Build metadata
VERSION ?= $(shell grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
COMMIT  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
HOST_TARGET := $(shell rustc -vV | grep '^host:' | cut -d' ' -f2)

# Release targets (mirrors .github/workflows/release.yml matrix)
RELEASE_TARGETS ?= x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-apple-darwin aarch64-apple-darwin
DIST_DIR := dist

# Default target
.DEFAULT_GOAL := help

#
# Build
#

build: ## Build in debug mode (Tier 1 only)
	cargo build

build-release: ## Build in release mode
	cargo build --release

build-model: ## Build with fasttext Tier 2 support
	cargo build --features model

install: ## Install the classify binary to ~/.cargo/bin
	cargo install --path .

clean: ## Remove build artifacts
	cargo clean

#
# Test
#

test: ## Run all tests
	cargo test

test-single: ## Run a single test by name (usage: make test-single T=test_name)
ifndef T
	$(error T is required. Usage: make test-single T=test_name)
endif
	cargo test $(T)

test-file: ## Run a single test file (usage: make test-file F=test_tier1)
ifndef F
	$(error F is required. Usage: make test-file F=test_tier1)
endif
	cargo test --test $(F)

#
# Lint & Format
#

fmt: ## Auto-format all Rust source files
	cargo fmt

fmt-check: ## Check formatting without modifying files
	cargo fmt --check

clippy: ## Run clippy lints (warnings are errors)
	cargo clippy --all-targets -- -D warnings

lint: fmt-check clippy ## Run all lints (format check + clippy)

#
# Combined Checks
#

check: lint test ## Run all checks (lint + test)

review: ## Full review with status reporting
	@echo "=== Format ===" && \
	if cargo fmt --check 2>&1; then \
		printf "\n  fmt ok\n\n"; FMT=0; \
	else \
		printf "\n  fmt FAILED\n\n"; FMT=1; \
	fi; \
	echo "=== Clippy ===" && \
	if cargo clippy --all-targets -- -D warnings 2>&1; then \
		printf "\n  clippy ok\n\n"; CLIPPY=0; \
	else \
		printf "\n  clippy FAILED\n\n"; CLIPPY=1; \
	fi; \
	echo "=== Tests ===" && \
	if cargo test 2>&1; then \
		printf "\n  tests ok\n\n"; TESTS=0; \
	else \
		printf "\n  tests FAILED\n\n"; TESTS=1; \
	fi; \
	echo "=== Summary ===" && \
	if [ $$FMT -eq 0 ] && [ $$CLIPPY -eq 0 ] && [ $$TESTS -eq 0 ]; then \
		echo "  All checks passed"; \
	else \
		echo "  Checks failed:"; \
		[ $$FMT -ne 0 ] && echo "    - fmt"; \
		[ $$CLIPPY -ne 0 ] && echo "    - clippy"; \
		[ $$TESTS -ne 0 ] && echo "    - tests"; \
		exit 1; \
	fi

#
# Python Extension
#

python-setup: ## Create venv and install maturin
	uv venv && uv pip install maturin

python-build: ## Build the Python extension (release mode)
	. .venv/bin/activate && maturin develop --release

#
# Run
#

run: ## Classify text from stdin (usage: echo "text" | make run)
	cargo run --release

run-file: ## Classify a JSONL file (usage: make run-file IN=input.jsonl OUT=output.jsonl)
ifndef IN
	$(error IN is required. Usage: make run-file IN=input.jsonl OUT=output.jsonl)
endif
ifndef OUT
	$(error OUT is required. Usage: make run-file IN=input.jsonl OUT=output.jsonl)
endif
	cargo run --release -- file $(IN) -o $(OUT)

run-filter: ## Filter JSONL into translatable/skipped (usage: make run-filter IN=input.jsonl TRANS=trans.jsonl SKIP=skip.jsonl)
ifndef IN
	$(error IN is required)
endif
ifndef TRANS
	$(error TRANS is required)
endif
ifndef SKIP
	$(error SKIP is required)
endif
	cargo run --release -- filter $(IN) --translatable $(TRANS) --skipped $(SKIP)

#
# Release
#

release-build: ## Build release binary for current platform
	@echo "Building release binary v$(VERSION) ($(COMMIT))..."
	cargo build --release
	@echo "Binary: target/release/classify"

release-local: ## Build + package release archives (like CI) for all targets
	@echo "Building release v$(VERSION) ($(COMMIT))"
	@echo "Targets: $(RELEASE_TARGETS)"
	@echo ""
	@rm -rf $(DIST_DIR)
	@for target in $(RELEASE_TARGETS); do \
		echo "=== $$target ==="; \
		if [ "$$target" = "$(HOST_TARGET)" ]; then \
			echo "  Building natively..."; \
			cargo build --release --target "$$target" 2>&1 || exit 1; \
		elif command -v cross >/dev/null 2>&1; then \
			echo "  Building with cross..."; \
			cross build --release --target "$$target" 2>&1 || exit 1; \
		else \
			echo "  Skipping (not host target and cross not installed)"; \
			echo "  Install cross: cargo install cross --git https://github.com/cross-rs/cross"; \
			continue; \
		fi; \
		ARCHIVE_NAME="text-classifier_$(VERSION)_$$target"; \
		mkdir -p "$(DIST_DIR)/$$ARCHIVE_NAME"; \
		cp "target/$$target/release/classify" "$(DIST_DIR)/$$ARCHIVE_NAME/"; \
		cp README.md LICENSE "$(DIST_DIR)/$$ARCHIVE_NAME/"; \
		(cd $(DIST_DIR) && tar czf "$$ARCHIVE_NAME.tar.gz" "$$ARCHIVE_NAME"); \
		(cd $(DIST_DIR) && sha256sum "$$ARCHIVE_NAME.tar.gz" > "$$ARCHIVE_NAME.tar.gz.sha256"); \
		rm -rf "$(DIST_DIR)/$$ARCHIVE_NAME"; \
		echo "  -> $(DIST_DIR)/$$ARCHIVE_NAME.tar.gz"; \
		echo ""; \
	done
	@echo "=== Done ==="
	@ls -lh $(DIST_DIR)/*.tar.gz 2>/dev/null || echo "No archives built"

# Help target for self-documentation
help: ## Display this help
	@echo "text-classifier-rs v$(VERSION)"
	@echo ""
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' Makefile | sort | awk 'BEGIN { FS=":.*?## " } { names[NR]=$$1; descs[NR]=$$2; if (length($$1)>max) max=length($$1) } END { w=max+2; for (i=1; i<=NR; i++) printf "  \033[36m%-*s\033[0m %s\n", w, names[i], descs[i] }'
