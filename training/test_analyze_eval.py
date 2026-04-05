"""Tests for analyze_eval.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


def _record(
    *,
    expected: str,
    predicted: str,
    sub_type: str | None = None,
    boundary_pair: str | None = None,
    content_domain: str | None = None,
    length_bucket: str | None = None,
    model: str | None = None,
) -> dict:
    return {
        "text": "example",
        "expected_category": expected,
        "predicted_category": predicted,
        "sub_type": sub_type,
        "boundary_pair": boundary_pair,
        "content_domain": content_domain,
        "length_bucket": length_bucket,
        "model": model,
    }


class TestComputeSliceAccuracy:
    def test_groups_by_sub_type(self):
        from analyze_eval import compute_slice_accuracy

        records = [
            _record(expected="artifact", predicted="artifact", sub_type="pdf_dump"),
            _record(expected="artifact", predicted="structured", sub_type="pdf_dump"),
            _record(expected="prose", predicted="prose", sub_type="plain"),
        ]

        result = compute_slice_accuracy(records, "sub_type")

        assert result["pdf_dump"]["total"] == 2
        assert result["pdf_dump"]["correct"] == 1
        assert result["pdf_dump"]["accuracy"] == pytest.approx(0.5)
        assert result["plain"]["accuracy"] == 1.0

    def test_groups_by_boundary_pair(self):
        from analyze_eval import compute_slice_accuracy

        records = [
            _record(
                expected="artifact",
                predicted="artifact",
                boundary_pair="structured_artifact",
            ),
            _record(
                expected="structured",
                predicted="artifact",
                boundary_pair="structured_artifact",
            ),
        ]

        result = compute_slice_accuracy(records, "boundary_pair")
        assert result["structured_artifact"]["total"] == 2
        assert result["structured_artifact"]["correct"] == 1


class TestBuildTopConfusions:
    def test_category_pair_confusions(self):
        from analyze_eval import build_top_confusions

        records = [
            _record(expected="artifact", predicted="structured"),
            _record(expected="artifact", predicted="structured"),
            _record(expected="artifact", predicted="prose"),
        ]

        result = build_top_confusions(records, limit=2)
        assert result[0]["expected_category"] == "artifact"
        assert result[0]["predicted_category"] == "structured"
        assert result[0]["count"] == 2

    def test_field_specific_confusions(self):
        from analyze_eval import build_top_confusions

        records = [
            _record(
                expected="artifact",
                predicted="structured",
                sub_type="pdf_dump",
            ),
            _record(
                expected="artifact",
                predicted="structured",
                sub_type="pdf_dump",
            ),
            _record(
                expected="artifact",
                predicted="prose",
                sub_type="ocr_garbage",
            ),
        ]

        result = build_top_confusions(records, "sub_type", limit=2)
        assert result[0]["sub_type"] == "pdf_dump"
        assert result[0]["count"] == 2


class TestBuildSliceReport:
    def test_contains_per_boundary_pair_and_confusions(self):
        from analyze_eval import build_slice_report

        records = [
            _record(
                expected="artifact",
                predicted="structured",
                sub_type="pdf_dump",
                boundary_pair="structured_artifact",
                content_domain="finance",
                length_bucket="long",
                model="openai/gpt-5",
            ),
            _record(
                expected="artifact",
                predicted="artifact",
                sub_type="pdf_dump",
                boundary_pair="structured_artifact",
                content_domain="finance",
                length_bucket="long",
                model="openai/gpt-5",
            ),
            _record(
                expected="structured",
                predicted="structured",
                sub_type="json",
                boundary_pair="structured_artifact",
                content_domain="api",
                length_bucket="short",
                model="anthropic/claude-sonnet-4.6",
            ),
        ]

        report = build_slice_report(records, eval_file="boundary.jsonl", limit=5)

        assert report["eval_file"] == "boundary.jsonl"
        assert report["per_boundary_pair"]["structured_artifact"]["accuracy"] == pytest.approx(
            2 / 3
        )
        assert report["top_confusions"]["by_category_pair"][0]["expected_category"] == "artifact"
        assert report["top_confusions"]["by_sub_type"][0]["sub_type"] == "pdf_dump"
        assert report["top_confusions"]["by_content_domain"][0]["content_domain"] == "finance"
        assert report["top_confusions"]["by_length_bucket"][0]["length_bucket"] == "long"
        assert report["top_confusions"]["by_model"][0]["model"] == "openai/gpt-5"


class TestMain:
    def test_writes_output_file(self, tmp_path):
        from analyze_eval import main

        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(_record(expected="prose", predicted="prose", sub_type="plain"))
            + "\n"
        )

        output_path = tmp_path / "slice_report.json"
        main(
            [
                "--predictions",
                str(predictions_path),
                "--output",
                str(output_path),
            ]
        )

        parsed = json.loads(output_path.read_text())
        assert parsed["overall_accuracy"] == 1.0
        assert parsed["per_sub_type"]["plain"]["accuracy"] == 1.0
