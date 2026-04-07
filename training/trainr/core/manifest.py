"""Training manifest for drift detection.

Records exact inputs to a training run (dataset SHA, feature config,
eval set SHAs) so that stale artifacts can be detected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class TrainingManifest:
    run_id: str
    dataset_sha256: str
    dataset_rows: int
    featurizer_version: str
    feature_count: int
    feature_names: list[str]
    eval_clear_sha256: str
    eval_boundary_sha256: str | None
    model_sha256: str
    timestamp: str

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TrainingManifest:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def verify(
        self,
        current_dataset_sha256: str | None = None,
        current_eval_clear_sha256: str | None = None,
        current_eval_boundary_sha256: str | None = None,
    ) -> list[str]:
        issues: list[str] = []
        if current_dataset_sha256 and current_dataset_sha256 != self.dataset_sha256:
            issues.append(
                f"Dataset SHA mismatch: manifest={self.dataset_sha256[:12]}... "
                f"current={current_dataset_sha256[:12]}..."
            )
        if current_eval_clear_sha256 and current_eval_clear_sha256 != self.eval_clear_sha256:
            issues.append(
                f"Eval clear SHA mismatch: manifest={self.eval_clear_sha256[:12]}... "
                f"current={current_eval_clear_sha256[:12]}..."
            )
        if (
            current_eval_boundary_sha256
            and self.eval_boundary_sha256
            and current_eval_boundary_sha256 != self.eval_boundary_sha256
        ):
            issues.append(
                f"Eval boundary SHA mismatch: manifest={self.eval_boundary_sha256[:12]}... "
                f"current={current_eval_boundary_sha256[:12]}..."
            )
        return issues
