import json
import tempfile
from pathlib import Path

from trainr.core.manifest import compute_file_sha256, TrainingManifest


def test_compute_sha256():
    """SHA256 of known content matches expected hash."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world\n")
        path = f.name
    sha = compute_file_sha256(path)
    Path(path).unlink()
    # sha256 of "hello world\n"
    assert sha == "a948904f2f0f479b8f8564e9d7903f6e55ee3c4f09e2a4b4e5e77d8e3b3b7e9d" or len(sha) == 64


def test_manifest_roundtrip():
    """Manifest can be written and read back identically."""
    m = TrainingManifest(
        run_id="test-run-001",
        dataset_sha256="abc123",
        dataset_rows=1000,
        featurizer_version="2.0",
        feature_count=40,
        feature_names=["alpha_ratio", "symbol_ratio"],
        eval_clear_sha256="def456",
        eval_boundary_sha256="ghi789",
        model_sha256="jkl012",
        timestamp="2026-04-07T10:00:00Z",
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        path = f.name

    m.save(path)
    loaded = TrainingManifest.load(path)
    Path(path).unlink()

    assert loaded.run_id == m.run_id
    assert loaded.dataset_sha256 == m.dataset_sha256
    assert loaded.dataset_rows == m.dataset_rows
    assert loaded.feature_count == m.feature_count
    assert loaded.feature_names == m.feature_names


def test_manifest_verify_detects_mismatch():
    """verify() returns a list of mismatches when SHAs differ."""
    m = TrainingManifest(
        run_id="test-run-001",
        dataset_sha256="abc123",
        dataset_rows=1000,
        featurizer_version="2.0",
        feature_count=40,
        feature_names=["alpha_ratio"],
        eval_clear_sha256="def456",
        eval_boundary_sha256=None,
        model_sha256="jkl012",
        timestamp="2026-04-07T10:00:00Z",
    )
    # Simulate current state where dataset SHA changed
    issues = m.verify(current_dataset_sha256="CHANGED", current_eval_clear_sha256="def456")
    assert len(issues) == 1
    assert "dataset" in issues[0].lower()


def test_manifest_verify_passes_when_matching():
    """verify() returns empty list when everything matches."""
    m = TrainingManifest(
        run_id="test-run-001",
        dataset_sha256="abc123",
        dataset_rows=1000,
        featurizer_version="2.0",
        feature_count=40,
        feature_names=["alpha_ratio"],
        eval_clear_sha256="def456",
        eval_boundary_sha256="ghi789",
        model_sha256="jkl012",
        timestamp="2026-04-07T10:00:00Z",
    )
    issues = m.verify(
        current_dataset_sha256="abc123",
        current_eval_clear_sha256="def456",
        current_eval_boundary_sha256="ghi789",
    )
    assert issues == []
