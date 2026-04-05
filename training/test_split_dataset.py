"""Tests for training/split_dataset.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest

# Ensure the training directory is importable
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    text="hello world",
    expected_category="prose",
    sub_type="plain",
    model="claude-3",
    boundary_pair=None,
):
    """Create a minimal sample dict."""
    d = {
        "text": text,
        "expected_category": expected_category,
        "sub_type": sub_type,
        "model": model,
    }
    if boundary_pair is not None:
        d["boundary_pair"] = boundary_pair
    else:
        d["boundary_pair"] = None
    return d


def _write_jsonl(path, samples):
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# Tests for load_jsonl
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    def test_loads_valid_jsonl(self, tmp_path):
        from split_dataset import load_jsonl

        p = tmp_path / "data.jsonl"
        samples = [{"a": 1}, {"b": 2}]
        _write_jsonl(str(p), samples)

        result = load_jsonl(str(p))
        assert result == samples

    def test_loads_empty_file(self, tmp_path):
        from split_dataset import load_jsonl

        p = tmp_path / "empty.jsonl"
        p.write_text("")

        result = load_jsonl(str(p))
        assert result == []


# ---------------------------------------------------------------------------
# Tests for stratified_sample
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def test_correct_counts_per_group(self):
        from split_dataset import stratified_sample

        # 3 categories, 5 samples each = 15 total
        samples = []
        for cat in ["prose", "code", "structured"]:
            for i in range(5):
                samples.append(_make_sample(expected_category=cat, model=f"model_{i}"))

        selected, remainder = stratified_sample(
            samples, key="expected_category", n_per_group=2, seed=42
        )

        # Should select 2 per group = 6 total
        assert len(selected) == 6
        assert len(remainder) == 9

        # Verify 2 per category
        from collections import Counter

        cats = Counter(s["expected_category"] for s in selected)
        assert cats["prose"] == 2
        assert cats["code"] == 2
        assert cats["structured"] == 2

    def test_deterministic_with_seed(self):
        from split_dataset import stratified_sample

        samples = []
        for cat in ["prose", "code"]:
            for i in range(10):
                samples.append(_make_sample(expected_category=cat, model=f"m{i}"))

        s1, _ = stratified_sample(samples, "expected_category", 3, seed=99)
        s2, _ = stratified_sample(samples, "expected_category", 3, seed=99)

        assert s1 == s2

    def test_different_seed_gives_different_result(self):
        from split_dataset import stratified_sample

        # Use 2 models with many samples each so shuffle order matters
        samples = []
        for cat in ["prose", "code"]:
            for i in range(20):
                samples.append(
                    _make_sample(
                        text=f"{cat}_{i}",
                        expected_category=cat,
                        model=f"m{i % 2}",
                    )
                )

        s1, _ = stratified_sample(samples, "expected_category", 5, seed=1)
        s2, _ = stratified_sample(samples, "expected_category", 5, seed=2)

        # With different seeds the shuffled order differs, so selections differ
        texts1 = {s["text"] for s in s1}
        texts2 = {s["text"] for s in s2}
        assert texts1 != texts2

    def test_sub_stratifies_by_model(self):
        from split_dataset import stratified_sample

        # 1 category, 2 models, 6 samples each = 12 total
        samples = []
        for model in ["modelA", "modelB"]:
            for i in range(6):
                samples.append(
                    _make_sample(expected_category="prose", model=model, sub_type="plain")
                )

        selected, remainder = stratified_sample(
            samples, key="expected_category", n_per_group=4, seed=42
        )

        # Should select 4 from the "prose" group, drawing from both models
        assert len(selected) == 4
        models_in_selected = {s["model"] for s in selected}
        assert "modelA" in models_in_selected
        assert "modelB" in models_in_selected

    def test_handles_group_with_fewer_samples_than_requested(self):
        from split_dataset import stratified_sample

        # Only 2 samples in "code" but we request 5 per group
        samples = [
            _make_sample(expected_category="prose"),
            _make_sample(expected_category="prose"),
            _make_sample(expected_category="prose"),
            _make_sample(expected_category="prose"),
            _make_sample(expected_category="prose"),
            _make_sample(expected_category="code"),
            _make_sample(expected_category="code"),
        ]

        selected, remainder = stratified_sample(
            samples, key="expected_category", n_per_group=5, seed=42
        )

        from collections import Counter

        cats = Counter(s["expected_category"] for s in selected)
        # prose has 5, so 5 selected; code has 2, so all 2 selected
        assert cats["prose"] == 5
        assert cats["code"] == 2
        assert len(remainder) == 0


# ---------------------------------------------------------------------------
# Tests for split_dataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def test_split_small_dataset(self, tmp_path):
        from split_dataset import split_dataset

        # Build input: 5 clear prose, 5 clear code, 5 boundary
        samples = []
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"prose text {i}",
                    expected_category="prose",
                    sub_type="plain",
                    model=f"m{i}",
                )
            )
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"code text {i}",
                    expected_category="code",
                    sub_type="python",
                    model=f"m{i}",
                )
            )
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"boundary text {i}",
                    expected_category="prose",
                    sub_type="markdown",
                    model=f"m{i}",
                    boundary_pair="prose_code",
                )
            )

        input_path = str(tmp_path / "input.jsonl")
        eval_clear = str(tmp_path / "eval_clear.jsonl")
        eval_boundary = str(tmp_path / "eval_boundary.jsonl")
        train_path = str(tmp_path / "train.parquet")

        _write_jsonl(input_path, samples)

        split_dataset(
            input_path=input_path,
            eval_clear_path=eval_clear,
            eval_boundary_path=eval_boundary,
            train_path=train_path,
            eval_per_category=2,
            eval_per_pair=2,
            seed=42,
        )

        # Check eval clear
        with open(eval_clear) as f:
            eval_clear_data = [json.loads(line) for line in f if line.strip()]
        # 2 per category (prose, code) = 4
        assert len(eval_clear_data) == 4

        # Check eval boundary
        with open(eval_boundary) as f:
            eval_boundary_data = [json.loads(line) for line in f if line.strip()]
        # 2 per pair (prose_code) = 2
        assert len(eval_boundary_data) == 2

        # Check training Parquet
        train_df = pl.read_parquet(train_path)
        # Remaining: (5-2) + (5-2) + (5-2) = 9
        assert len(train_df) == 9
        # Verify Parquet columns include provenance fields
        assert set(train_df.columns) == {"text", "category", "sub_type", "source", "model"}
        # Sources should be preserved from input (not hardcoded)
        assert all(v != "" for v in train_df.get_column("source").to_list())

    def test_split_writes_valid_jsonl(self, tmp_path):
        from split_dataset import split_dataset

        samples = [_make_sample(text=f"text {i}") for i in range(5)]
        input_path = str(tmp_path / "input.jsonl")
        eval_clear = str(tmp_path / "eval_clear.jsonl")
        eval_boundary = str(tmp_path / "eval_boundary.jsonl")
        train_path = str(tmp_path / "train.parquet")

        _write_jsonl(input_path, samples)

        split_dataset(
            input_path=input_path,
            eval_clear_path=eval_clear,
            eval_boundary_path=eval_boundary,
            train_path=train_path,
            eval_per_category=2,
            eval_per_pair=2,
            seed=42,
        )

        # Each line should be valid JSON
        with open(eval_clear) as f:
            for line in f:
                if line.strip():
                    json.loads(line)  # should not raise


# ---------------------------------------------------------------------------
# Tests for verify_diversity
# ---------------------------------------------------------------------------


class TestVerifyDiversity:
    def test_passes_when_diverse(self):
        from split_dataset import verify_diversity

        # 10 samples per model across 5 models = even distribution (20% each)
        # But 20% > 15%, so we need 7+ models to be under 15% each
        # Actually, let's use 10 models to be clearly under 15%
        samples = []
        for i in range(10):
            samples.append(_make_sample(model=f"model_{i}", sub_type="plain"))

        warnings = verify_diversity(samples)
        assert warnings == []

    def test_warns_when_model_dominates(self):
        from split_dataset import verify_diversity

        # 9 samples from one model, 1 from another in same sub_type
        samples = []
        for i in range(9):
            samples.append(_make_sample(model="dominant", sub_type="plain"))
        samples.append(_make_sample(model="other", sub_type="plain"))

        warnings = verify_diversity(samples)
        assert len(warnings) > 0
        assert any("dominant" in w for w in warnings)

    def test_checks_per_sub_type(self):
        from split_dataset import verify_diversity

        # sub_type "plain": evenly distributed across 10 models
        samples = []
        for i in range(10):
            samples.append(_make_sample(model=f"model_{i}", sub_type="plain"))
        # sub_type "python": dominated by one model
        for i in range(9):
            samples.append(_make_sample(model="dominant", sub_type="python"))
        samples.append(_make_sample(model="other", sub_type="python"))

        warnings = verify_diversity(samples)
        # Should warn about "python" sub_type, not "plain"
        assert len(warnings) > 0
        assert any("python" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tests for downsampling (max_per_category)
# ---------------------------------------------------------------------------


class TestDownsampling:
    def test_downsampling_caps_large_categories(self, tmp_path):
        from split_dataset import split_dataset

        samples = []
        for i in range(10):
            samples.append(
                _make_sample(
                    text=f"code sample {i}",
                    expected_category="code",
                    sub_type="python",
                    model=f"m{i % 3}",
                )
            )
        for i in range(3):
            samples.append(
                _make_sample(
                    text=f"prose sample {i}",
                    expected_category="prose",
                    sub_type="plain",
                    model=f"m{i}",
                )
            )

        input_path = str(tmp_path / "input.jsonl")
        eval_clear = str(tmp_path / "eval_clear.jsonl")
        eval_boundary = str(tmp_path / "eval_boundary.jsonl")
        train_path = str(tmp_path / "train.parquet")

        _write_jsonl(input_path, samples)

        split_dataset(
            input_path=input_path,
            eval_clear_path=eval_clear,
            eval_boundary_path=eval_boundary,
            train_path=train_path,
            eval_per_category=0,
            eval_per_pair=0,
            seed=42,
            max_per_category=5,
        )

        train_df = pl.read_parquet(train_path)
        train_rows = train_df.to_dicts()

        from collections import Counter

        cats = Counter(r["category"] for r in train_rows)
        assert cats["code"] == 5
        assert cats["prose"] == 3

    def test_downsampling_zero_means_no_limit(self, tmp_path):
        from split_dataset import split_dataset

        samples = []
        for i in range(10):
            samples.append(
                _make_sample(
                    text=f"code sample {i}",
                    expected_category="code",
                    sub_type="python",
                    model=f"m{i % 3}",
                )
            )

        input_path = str(tmp_path / "input.jsonl")
        eval_clear = str(tmp_path / "eval_clear.jsonl")
        eval_boundary = str(tmp_path / "eval_boundary.jsonl")
        train_path = str(tmp_path / "train.parquet")

        _write_jsonl(input_path, samples)

        split_dataset(
            input_path=input_path,
            eval_clear_path=eval_clear,
            eval_boundary_path=eval_boundary,
            train_path=train_path,
            eval_per_category=0,
            eval_per_pair=0,
            seed=42,
            max_per_category=0,
        )

        train_df = pl.read_parquet(train_path)

        assert len(train_df) == 10

    def test_downsampling_is_deterministic(self, tmp_path):
        from split_dataset import split_dataset

        samples = []
        for i in range(20):
            samples.append(
                _make_sample(
                    text=f"code sample {i}",
                    expected_category="code",
                    sub_type="python",
                    model=f"m{i % 5}",
                )
            )

        input_path = str(tmp_path / "input.jsonl")
        _write_jsonl(input_path, samples)

        results = []
        for run in range(2):
            eval_clear = str(tmp_path / f"eval_clear_{run}.jsonl")
            eval_boundary = str(tmp_path / f"eval_boundary_{run}.jsonl")
            train_path = str(tmp_path / f"train_{run}.parquet")

            split_dataset(
                input_path=input_path,
                eval_clear_path=eval_clear,
                eval_boundary_path=eval_boundary,
                train_path=train_path,
                eval_per_category=0,
                eval_per_pair=0,
                seed=42,
                max_per_category=5,
            )

            train_df = pl.read_parquet(train_path)
            results.append(train_df.get_column("text").to_list())

        assert results[0] == results[1]


# ---------------------------------------------------------------------------
# Tests for SUBTYPE_CATEGORY_OVERRIDES updates
# ---------------------------------------------------------------------------


class TestSubtypeCategoryOverrides:
    def test_overrides_include_reclassified_subtypes(self):
        from split_dataset import SUBTYPE_CATEGORY_OVERRIDES

        # Original overrides
        assert SUBTYPE_CATEGORY_OVERRIDES["yaml"] == "structured"
        assert SUBTYPE_CATEGORY_OVERRIDES["toml"] == "structured"
        assert SUBTYPE_CATEGORY_OVERRIDES["ini"] == "structured"

        # New overrides: None means reclassified by heuristic
        assert SUBTYPE_CATEGORY_OVERRIDES["pdf_dump"] is None
        assert SUBTYPE_CATEGORY_OVERRIDES["ocr_garbage"] is None
        assert SUBTYPE_CATEGORY_OVERRIDES["boilerplate"] is None
        assert SUBTYPE_CATEGORY_OVERRIDES["skip"] is None


# ---------------------------------------------------------------------------
# Tests for reclassify_by_content
# ---------------------------------------------------------------------------


class TestReclassifyByContent:
    def test_high_coherence_and_dictionary_returns_prose(self):
        from split_dataset import reclassify_by_content

        # Craft text with high sentence_coherence_score and dictionary_word_ratio
        # Sentences that start uppercase and end with period, using dictionary words
        text = "The quick brown fox jumps over the lazy dog.\n" * 10
        result = reclassify_by_content(text)
        assert result == "prose"

    def test_high_kvr_returns_structured(self):
        from split_dataset import reclassify_by_content

        # Text with lots of key: value lines
        text = "name: John\nage: 30\ncity: Paris\ncolor: blue\n" * 10
        result = reclassify_by_content(text)
        assert result == "structured"

    def test_high_comment_ratio_returns_code(self):
        from split_dataset import reclassify_by_content

        # Text with lots of comment lines
        text = "# this is a comment\n# another comment\n# more comments\nx = 1\n# yet another\n" * 5
        result = reclassify_by_content(text)
        assert result == "code"

    def test_high_delimiter_consistency_returns_structured(self):
        from split_dataset import reclassify_by_content

        # CSV-like data with consistent delimiters
        text = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n" * 5
        result = reclassify_by_content(text)
        assert result == "structured"

    def test_fallback_returns_structured(self):
        from split_dataset import reclassify_by_content

        # Gibberish: no coherence, no dictionary words, no kv, no comments
        text = "xzq wrp tlm\nbrg fnd klt\nmxp qrs znt\n"
        result = reclassify_by_content(text)
        assert result == "structured"

    def test_returns_only_valid_categories(self):
        from split_dataset import reclassify_by_content

        # Any text should return one of the 3 valid categories
        for text in ["", "hello", "x=1\ny=2\n", "# comment\n"]:
            result = reclassify_by_content(text)
            assert result in ("prose", "code", "structured")


# ---------------------------------------------------------------------------
# Tests for reclassification in split_dataset
# ---------------------------------------------------------------------------


class TestReclassificationInSplit:
    def test_artifact_samples_are_reclassified(self, tmp_path):
        from split_dataset import split_dataset

        # Create samples with artifact category
        samples = []
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"The quick brown fox jumped over the lazy dog sentence {i}.\n" * 5,
                    expected_category="artifact",
                    sub_type="pdf_dump",
                    model=f"m{i}",
                )
            )
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"prose text {i}",
                    expected_category="prose",
                    sub_type="plain",
                    model=f"m{i}",
                )
            )

        input_path = str(tmp_path / "input.jsonl")
        train_path = str(tmp_path / "train.parquet")

        _write_jsonl(input_path, samples)

        split_dataset(
            input_path=input_path,
            eval_clear_path=None,
            eval_boundary_path=None,
            train_path=train_path,
            eval_per_category=0,
            eval_per_pair=0,
            seed=42,
        )

        train_df = pl.read_parquet(train_path)

        # No training row should have category "artifact"
        categories = set(train_df.get_column("category").to_list())
        assert "artifact" not in categories
        assert categories.issubset({"prose", "code", "structured"})

    def test_skip_samples_are_reclassified(self, tmp_path):
        from split_dataset import split_dataset

        samples = []
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"name: value{i}\nkey: data{i}\nfield: entry{i}\n" * 3,
                    expected_category="skip",
                    sub_type="whitespace",
                    model=f"m{i}",
                )
            )
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"code text {i}",
                    expected_category="code",
                    sub_type="python",
                    model=f"m{i}",
                )
            )

        input_path = str(tmp_path / "input.jsonl")
        train_path = str(tmp_path / "train.parquet")

        _write_jsonl(input_path, samples)

        split_dataset(
            input_path=input_path,
            eval_clear_path=None,
            eval_boundary_path=None,
            train_path=train_path,
            eval_per_category=0,
            eval_per_pair=0,
            seed=42,
        )

        train_df = pl.read_parquet(train_path)

        categories = set(train_df.get_column("category").to_list())
        assert "skip" not in categories
        assert categories.issubset({"prose", "code", "structured"})

    def test_samples_with_invalid_category_after_reclassification_are_dropped(self, tmp_path):
        """Samples whose expected_category is not in {prose, code, structured}
        after reclassification should be dropped."""
        from split_dataset import split_dataset

        samples = []
        for i in range(5):
            samples.append(
                _make_sample(
                    text=f"prose text {i}",
                    expected_category="prose",
                    sub_type="plain",
                    model=f"m{i}",
                )
            )
        # Add a sample with an unknown category
        samples.append(
            _make_sample(
                text="unknown category sample",
                expected_category="unknown_garbage",
                sub_type="mystery",
                model="m0",
            )
        )

        input_path = str(tmp_path / "input.jsonl")
        train_path = str(tmp_path / "train.parquet")

        _write_jsonl(input_path, samples)

        split_dataset(
            input_path=input_path,
            eval_clear_path=None,
            eval_boundary_path=None,
            train_path=train_path,
            eval_per_category=0,
            eval_per_pair=0,
            seed=42,
        )

        train_df = pl.read_parquet(train_path)

        # The unknown category sample should be dropped
        categories = set(train_df.get_column("category").to_list())
        assert "unknown_garbage" not in categories
        assert len(train_df) == 5
