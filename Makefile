# text-classifier-rs Makefile
# Run `make` or `make help` for available targets.

VERSION ?= $(shell grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
COMMIT  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
HOST_TARGET := $(shell rustc -vV | grep '^host:' | cut -d' ' -f2)
RELEASE_TARGETS ?= x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-apple-darwin aarch64-apple-darwin
DIST_DIR := dist

.DEFAULT_GOAL := help

# ─── Build ──────────────────────────────────────────────────────────────────

.PHONY: build build-release build-model install clean

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

# ─── Test ───────────────────────────────────────────────────────────────────

.PHONY: test test-single test-file

test: ## Run all tests
	cargo test

test-single: ## Run a single test (T=test_name)
ifndef T
	$(error T is required — e.g. make test-single T=test_name)
endif
	cargo test $(T)

test-file: ## Run a single test file (F=test_tier1)
ifndef F
	$(error F is required — e.g. make test-file F=test_tier1)
endif
	cargo test --test $(F)

# ─── Lint & Format ──────────────────────────────────────────────────────────

.PHONY: fmt fmt-check clippy lint

fmt: ## Auto-format all Rust source files
	cargo fmt

fmt-check: ## Check formatting without modifying files
	cargo fmt --check

clippy: ## Run clippy lints (warnings are errors)
	cargo clippy --all-targets -- -D warnings

lint: fmt-check clippy ## Run all lints (format check + clippy)

# ─── Combined Checks ───────────────────────────────────────────────────────

.PHONY: check review

check: lint test ## Lint + test — the pre-commit gate

review: ## Full review with pass/fail status reporting
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

# ─── Python Extension ──────────────────────────────────────────────────────

.PHONY: python-setup python-build

python-setup: ## Create venv and install maturin
	uv venv && uv pip install maturin

python-build: ## Build the Python extension (release mode)
	. .venv/bin/activate && maturin develop --release

# ─── Training Pipeline ─────────────────────────────────────────────────────

.PHONY: training-setup generate-data generate-fixtures generate-test-set generate-ambiguous train validate test-model test-model-ambiguous train-pipeline build-onnx update-model

training-setup: ## Set up the training Python environment
	cd training && uv sync --group dev

generate-data: ## Generate all training data (fixtures + synthetic + perturbations + test set)
	cd training && uv run python generate.py --mode all --output data/ --samples-per-type 200

generate-fixtures: ## Generate training data from test fixtures only (no API key needed)
	cd training && uv run python generate.py --mode fixtures --output data/

generate-test-set: ## Generate labeled test set from fixtures for validation
	cd training && uv run python generate.py --mode test-set --output data/

train: ## Train the model and export to ONNX
	cd training && uv run python train.py --data data/combined.csv --output models/

validate: ## Validate classifier accuracy (INPUT=test.jsonl)
ifndef INPUT
	$(error INPUT is required — e.g. make validate INPUT=test.jsonl)
endif
	cargo run --release -- validate --input $(INPUT)

generate-ambiguous: ## Generate 100 ambiguous boundary-case test samples (requires API key)
	cd training && uv run python generate.py --mode ambiguous-test-set --output data/

test-model: generate-test-set ## Validate classifier with ONNX model against fixture test set
	cargo run --release --features onnx-model -- validate --input training/data/test_set.jsonl

test-model-ambiguous: generate-ambiguous ## Validate classifier against ambiguous test samples
	cargo run --release --features onnx-model -- validate --input training/data/ambiguous_test_set.jsonl

train-pipeline: generate-data train update-model test-model ## Full pipeline: generate → train → embed → validate

build-onnx: ## Build with embedded ONNX model (Tier 1 + 2)
	cargo build --release --features onnx-model

update-model: ## Copy trained model from training/models/ to src/ for embedding
	cd training && uv run python -c "import onnx; m = onnx.load('models/model.onnx', load_external_data=True); onnx.save(m, '../src/model.onnx')"
	cp training/models/model_config.json src/model_config.json
	@echo "Model files updated in src/. Rebuild with: make build-onnx"

# ─── Run ────────────────────────────────────────────────────────────────────

.PHONY: run run-file run-filter

run: ## Classify text from stdin
	cargo run --release

run-file: ## Classify a JSONL file (IN=input.jsonl OUT=output.jsonl)
ifndef IN
	$(error IN and OUT are required — e.g. make run-file IN=input.jsonl OUT=output.jsonl)
endif
ifndef OUT
	$(error IN and OUT are required — e.g. make run-file IN=input.jsonl OUT=output.jsonl)
endif
	cargo run --release -- file $(IN) -o $(OUT)

run-filter: ## Filter JSONL into prose/skipped (IN= PROSE= SKIP=)
ifndef IN
	$(error IN, PROSE, SKIP are required)
endif
ifndef PROSE
	$(error IN, PROSE, SKIP are required)
endif
ifndef SKIP
	$(error IN, PROSE, SKIP are required)
endif
	cargo run --release -- filter $(IN) --prose $(PROSE) --skipped $(SKIP)

# ─── Release ────────────────────────────────────────────────────────────────

.PHONY: release-build release-local release-list release-show release-download release-delete release-pr

release-build: ## Build release binary for current platform
	@echo "Building release binary v$(VERSION) ($(COMMIT))..."
	cargo build --release
	@echo "Binary: target/release/classify"

release-local: ## Build + package release archives for all targets
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

release-list: ## List recent GitHub releases
	gh release list --limit 10

release-show: ## Show release details (TAG=v0.1.0)
ifndef TAG
	$(error TAG is required — e.g. make release-show TAG=v0.1.0)
endif
	gh release view $(TAG)

release-download: ## Download release assets (TAG=v0.1.0)
ifndef TAG
	$(error TAG is required — e.g. make release-download TAG=v0.1.0)
endif
	@mkdir -p $(DIST_DIR)
	gh release download $(TAG) --dir $(DIST_DIR)
	@echo "Assets downloaded to $(DIST_DIR)/"
	@ls -lh $(DIST_DIR)/

release-delete: ## Delete a GitHub release (TAG=v0.1.0)
ifndef TAG
	$(error TAG is required — e.g. make release-delete TAG=v0.1.0)
endif
	gh release delete $(TAG) --cleanup-tag --yes

release-pr: ## Show the current release-please PR
	@gh pr list --label "autorelease: pending" --json number,title,url --template '{{range .}}#{{.number}} {{.title}}{{"\n"}}  {{.url}}{{"\n"}}{{else}}No pending release PR{{"\n"}}{{end}}'

# ─── Help & Completion ──────────────────────────────────────────────────────

.PHONY: help completion targets

help: ## Display this help
	@printf "text-classifier-rs v%s\n\n" "$(VERSION)"
	@awk 'BEGIN { FS = ":.*##"; section = "" } \
		/^# ─── / { \
			gsub(/^# ─── /, ""); gsub(/ ─+$$/, ""); section = $$0; next \
		} \
		/^[a-zA-Z0-9_-]+:.*##/ { \
			if (section != prev) { printf "\n\033[1m%s\033[0m\n", section; prev = section } \
			printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 \
		}' $(MAKEFILE_LIST)
	@echo ""

targets: ## List all target names (for shell completion)
	@awk -F: '/^[a-zA-Z0-9_-]+:.*##/ { print $$1 }' $(MAKEFILE_LIST)

completion: ## Print shell completion snippet (eval "$(make completion)")
	@echo '# Add to ~/.zshrc or ~/.bashrc:'
	@echo '#   eval "$$(make -C /path/to/text-classifier-rs completion)"'
	@echo ''
	@echo '_make_text_classifier() {'
	@echo '  local targets'
	@echo '  targets=$$(make -C "$${COMP_PROJECT_DIR:-.}" targets 2>/dev/null)'
	@echo '  if [ -n "$$ZSH_VERSION" ]; then'
	@echo '    _arguments "1:target:($$targets)"'
	@echo '  else'
	@echo '    COMPREPLY=($$(compgen -W "$$targets" -- "$${COMP_WORDS[COMP_CWORD]}"))'
	@echo '  fi'
	@echo '}'
	@echo ''
	@echo 'if [ -n "$$ZSH_VERSION" ]; then'
	@echo '  compdef _make_text_classifier make'
	@echo 'else'
	@echo '  complete -F _make_text_classifier make'
	@echo 'fi'
