"""A/B regression audit comparison — iter15 prompt vs iter16 prompt.

Given three sets of annotator parquets (iter15 before, iter16a after, and
iter16b noise-floor companion), computes per-label agreement delta,
prevalence ratio, and same-prompt noise floor, then emits a gate verdict.

See docs/superpowers/specs/2026-04-10-iter17-ab-regression-audit-design.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import polars as pl

from trainr.core.audit_semantic_labels import (
    compute_agreement_across_models,
    compute_prevalence_per_label,
    filter_for_agreement,
)

# Gate thresholds, from the spec.
AGREEMENT_DELTA_THRESHOLD = 0.005
PREVALENCE_RATIO_LOW = 0.5
PREVALENCE_RATIO_HIGH = 2.0


class LabelCategory(str, Enum):
    SHARED = "shared"
    ITER15_ONLY = "iter15-only"
    ITER16_ONLY = "iter16-only"


class LabelVerdict(str, Enum):
    PASS = "PASS"
    FAIL_AGREEMENT = "FAIL-agreement"
    WARN_PREVALENCE = "WARN-prevalence"
    FAIL_AND_WARN = "FAIL-agreement+WARN-prevalence"
    NO_VERDICT = "N/A"  # iter15-only / iter16-only rows


@dataclass
class LabelRow:
    label: str
    category: LabelCategory
    iter15_agreement: float | None
    iter16_agreement: float | None
    delta_agreement: float | None
    iter15_prevalence: float | None
    iter16_prevalence: float | None
    prevalence_ratio: float | None
    noise_floor: float | None
    verdict: LabelVerdict


@dataclass
class DeltaReport:
    labels: dict[str, LabelRow] = field(default_factory=dict)
    shared_labels: list[str] = field(default_factory=list)
    iter15_only_labels: list[str] = field(default_factory=list)
    iter16_only_labels: list[str] = field(default_factory=list)

    @property
    def fail_agreement_rows(self) -> list[LabelRow]:
        return [
            r for r in self.labels.values()
            if r.verdict in (LabelVerdict.FAIL_AGREEMENT, LabelVerdict.FAIL_AND_WARN)
        ]

    @property
    def warn_prevalence_rows(self) -> list[LabelRow]:
        return [
            r for r in self.labels.values()
            if r.verdict in (LabelVerdict.WARN_PREVALENCE, LabelVerdict.FAIL_AND_WARN)
        ]


def compare_prompt_versions(
    before_frames: dict[str, pl.DataFrame],
    after_frames: dict[str, pl.DataFrame],
    noise_floor_frames: dict[str, pl.DataFrame],
    input_frame: pl.DataFrame,
) -> DeltaReport:
    """Compute the A/B regression report.

    All three frame dicts must be keyed by the canonical model slugs
    ({"gemini3flash", "sonnet", "gpt54mini"}) and share row ordering with
    `input_frame`. The caller is responsible for loading and filtering —
    this function works in-memory only.

    Future tasks (2.2-2.6) extend this function; for now it handles the
    simple all-shared, all-pass case.
    """
    report = DeltaReport()

    # Stratified frames for agreement.
    before_strat = {k: filter_for_agreement(v) for k, v in before_frames.items()}
    after_strat = {k: filter_for_agreement(v) for k, v in after_frames.items()}
    noise_strat = {k: filter_for_agreement(v) for k, v in noise_floor_frames.items()}

    iter15_agr = compute_agreement_across_models(before_strat)
    iter16_agr = compute_agreement_across_models(after_strat)
    noise_agr = compute_agreement_across_models(noise_strat)

    # compute_prevalence_per_label filters internally, so pass raw frames.
    iter15_prev = compute_prevalence_per_label(before_frames)
    iter16_prev = compute_prevalence_per_label(after_frames)

    shared_labels = sorted(set(iter15_agr.keys()) & set(iter16_agr.keys()))
    report.shared_labels = shared_labels

    for label in shared_labels:
        row = LabelRow(
            label=label,
            category=LabelCategory.SHARED,
            iter15_agreement=iter15_agr[label],
            iter16_agreement=iter16_agr[label],
            delta_agreement=iter16_agr[label] - iter15_agr[label],
            iter15_prevalence=iter15_prev.get(label, 0.0),
            iter16_prevalence=iter16_prev.get(label, 0.0),
            prevalence_ratio=_compute_prev_ratio(
                iter15_prev.get(label, 0.0),
                iter16_prev.get(label, 0.0),
            ),
            noise_floor=abs(noise_agr[label] - iter16_agr[label]) if label in noise_agr else None,
            verdict=LabelVerdict.PASS,  # placeholder; Task 2.3 adds real verdict logic
        )
        report.labels[label] = row

    return report


def _compute_prev_ratio(iter15_prev: float, iter16_prev: float) -> float:
    """Zero-handling rules from the spec: 0/0 → 1.0, 0/>0 → inf, >0/0 → 0.0."""
    if iter15_prev == 0 and iter16_prev == 0:
        return 1.0
    if iter15_prev == 0:
        return math.inf
    return iter16_prev / iter15_prev
