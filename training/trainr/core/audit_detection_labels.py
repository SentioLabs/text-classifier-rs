#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["polars"]
# ///
"""Quantitative audit of detection labels using archived iter9 data.

Reads the four archived iter9 detection Parquet files (one per LLM model)
and produces a markdown report evaluating each detection label on:

1. Fire rate (how often the label fires as 1)
2. Self-match vs cross-fire split (label == row sub_type vs label != sub_type)
3. Inter-annotator agreement across the 4 model variants
4. Co-occurrence patterns with other labels
5. Weak-label candidates (low fire rate and/or low cross-fire)

Writes the report to docs/plans/detection-label-audit-report.md.

Usage:
    uv run --with polars python training/trainr/core/audit_detection_labels.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARCHIVE_DIR = Path("data/audit")
OUTPUT_PATH = Path("../docs/plans/detection-label-audit-report-v2.md")

MODELS: dict[str, str] = {
    "gemini3flash": "audit_5k_v2_gemini3flash.parquet",
    "sonnet": "audit_5k_v2_sonnet.parquet",
    "gpt54mini": "audit_5k_v2_gpt54mini.parquet",
}

# Thresholds for flagging labels
WEAK_FIRE_RATE_THRESHOLD = 0.01  # <1% fire rate
WEAK_CROSS_FIRE_THRESHOLD = 30  # <30 cross-fires across 5k rows
LOW_AGREEMENT_THRESHOLD = 0.70  # Fleiss-kappa-ish agreement floor

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def load_model_data() -> dict[str, pl.DataFrame]:
    """Load all four model annotation files."""
    dfs: dict[str, pl.DataFrame] = {}
    for name, filename in MODELS.items():
        dfs[name] = pl.read_parquet(ARCHIVE_DIR / filename)
    return dfs


def detection_labels(df: pl.DataFrame) -> list[str]:
    """Extract detection label names from det_* columns, sorted."""
    return sorted([c[4:] for c in df.columns if c.startswith("det_")])


def compute_fire_stats(df: pl.DataFrame, labels: list[str]) -> dict[str, dict]:
    """Compute per-label fire rate, self-match, cross-fire stats.

    Returns {label: {"total_fires": int, "self_match": int, "cross_fire": int,
                     "fire_rate": float, "cross_fire_ratio": float}}
    """
    n = len(df)
    sub_types = df["sub_type"].to_list()
    result: dict[str, dict] = {}

    for label in labels:
        col = f"det_{label}"
        values = df[col].to_list()
        total = sum(1 for v in values if v == 1)
        self_match = sum(
            1 for v, st in zip(values, sub_types) if v == 1 and st == label
        )
        cross = total - self_match
        result[label] = {
            "total_fires": total,
            "self_match": self_match,
            "cross_fire": cross,
            "fire_rate": total / n,
            "cross_fire_ratio": cross / total if total > 0 else 0.0,
        }
    return result


def compute_agreement(
    dfs: dict[str, pl.DataFrame], labels: list[str]
) -> dict[str, dict]:
    """Compute inter-annotator agreement across the 4 models.

    Uses a simple metric: for each row, the fraction of models that agree
    (majority vote) on each label. Averaged across rows gives a per-label
    agreement score. Also reports pairwise agreement.

    Returns {label: {"mean_agreement": float, "unanimous_rate": float,
                     "split_rate": float}}
    """
    model_names = list(dfs.keys())
    n_models = len(model_names)
    n_rows = len(dfs[model_names[0]])

    result: dict[str, dict] = {}
    for label in labels:
        col = f"det_{label}"
        # Collect per-row votes from each model
        votes = [dfs[m][col].to_list() for m in model_names]

        unanimous = 0
        split = 0
        agreement_sum = 0.0
        for row_idx in range(n_rows):
            row_votes = [votes[m][row_idx] for m in range(n_models)]
            ones = sum(1 for v in row_votes if v == 1)
            zeros = n_models - ones
            if ones == n_models or zeros == n_models:
                unanimous += 1
                agreement_sum += 1.0
            else:
                split += 1
                # Agreement on majority = max(ones, zeros) / n_models
                agreement_sum += max(ones, zeros) / n_models

        result[label] = {
            "mean_agreement": agreement_sum / n_rows,
            "unanimous_rate": unanimous / n_rows,
            "split_rate": split / n_rows,
        }
    return result


def compute_cooccurrence(
    df: pl.DataFrame, labels: list[str]
) -> dict[tuple[str, str], int]:
    """Count how often each pair of labels fires together on the same row.

    Returns {(label_a, label_b): count} for label_a < label_b.
    """
    cooc: dict[tuple[str, str], int] = defaultdict(int)
    det_cols = [f"det_{label}" for label in labels]
    values = df.select(det_cols).to_numpy()

    for row in values:
        fired = [labels[i] for i, v in enumerate(row) if v == 1]
        for i, a in enumerate(fired):
            for b in fired[i + 1 :]:
                key = (a, b) if a < b else (b, a)
                cooc[key] += 1
    return dict(cooc)


def compute_per_subtype_dist(
    df: pl.DataFrame, labels: list[str]
) -> dict[str, Counter]:
    """For each sub_type, count how often each detection label fires on rows
    of that sub_type. Returns {sub_type: Counter(label -> count)}."""
    result: dict[str, Counter] = defaultdict(Counter)
    sub_types = df["sub_type"].to_list()
    det_cols = [f"det_{label}" for label in labels]
    values = df.select(det_cols).to_numpy()

    for st, row in zip(sub_types, values):
        if st is None:
            continue
        for i, v in enumerate(row):
            if v == 1:
                result[st][labels[i]] += 1
    return dict(result)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def format_fire_table(
    stats: dict[str, dict],
    agreement: dict[str, dict],
    n_rows: int,
    model_name: str,
) -> str:
    """Format the per-label fire rate table as markdown."""
    rows = sorted(
        stats.items(), key=lambda kv: kv[1]["total_fires"], reverse=True
    )
    lines = [
        f"### Fire Stats ({model_name}, n={n_rows})",
        "",
        "| Label | Fires | Rate | Self-match | Cross-fire | Cross% | Agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in rows:
        agr = agreement[label]["mean_agreement"]
        lines.append(
            f"| `{label}` | {s['total_fires']} | "
            f"{s['fire_rate']:.3f} | "
            f"{s['self_match']} | "
            f"{s['cross_fire']} | "
            f"{s['cross_fire_ratio']:.2f} | "
            f"{agr:.3f} |"
        )
    return "\n".join(lines)


def format_agreement_table(
    agreement: dict[str, dict], stats: dict[str, dict]
) -> str:
    """Format the inter-annotator agreement table, sorted worst-first."""
    rows = sorted(
        agreement.items(), key=lambda kv: kv[1]["mean_agreement"]
    )
    lines = [
        "### Inter-Annotator Agreement (4 models, sorted worst-first)",
        "",
        "| Label | Mean Agreement | Unanimous % | Split % | Total Fires (gemini3flash) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, a in rows:
        fires = stats[label]["total_fires"]
        lines.append(
            f"| `{label}` | {a['mean_agreement']:.3f} | "
            f"{a['unanimous_rate']:.3f} | "
            f"{a['split_rate']:.3f} | "
            f"{fires} |"
        )
    return "\n".join(lines)


def format_cooccurrence_table(cooc: dict[tuple[str, str], int]) -> str:
    """Format top co-occurring label pairs."""
    top = sorted(cooc.items(), key=lambda kv: kv[1], reverse=True)[:25]
    lines = [
        "### Top 25 Co-occurring Label Pairs",
        "",
        "| Label A | Label B | Co-fire Count |",
        "|---|---|---:|",
    ]
    for (a, b), count in top:
        lines.append(f"| `{a}` | `{b}` | {count} |")
    return "\n".join(lines)


def format_per_subtype(
    per_st: dict[str, Counter], sub_type_counts: dict[str, int]
) -> str:
    """For each sub_type, show top 5 detection labels that fire on it."""
    lines = [
        "### Per-Sub_type Detection Distribution",
        "",
        "For each sub_type, shows the top 5 detection labels that fire on its rows.",
        "A sub_type whose top detection is itself is working as expected.",
        "A sub_type whose top detections are OTHER labels suggests semantic cross-cutting.",
        "",
        "| Sub_type | n | Top detections |",
        "|---|---:|---|",
    ]
    for st in sorted(per_st.keys()):
        n = sub_type_counts.get(st, 0)
        top5 = per_st[st].most_common(5)
        top_str = ", ".join(f"`{label}`({count})" for label, count in top5)
        lines.append(f"| `{st}` | {n} | {top_str} |")
    return "\n".join(lines)


def classify_labels(
    stats: dict[str, dict], agreement: dict[str, dict]
) -> dict[str, list[str]]:
    """Classify labels into strong/weak/suspicious buckets."""
    strong: list[str] = []
    weak: list[str] = []
    low_agreement: list[str] = []
    suspicious: list[str] = []

    for label, s in stats.items():
        total = s["total_fires"]
        cross = s["cross_fire"]
        rate = s["fire_rate"]
        agr = agreement[label]["mean_agreement"]

        if rate < WEAK_FIRE_RATE_THRESHOLD or cross < WEAK_CROSS_FIRE_THRESHOLD:
            weak.append(label)
        elif cross >= 100:
            strong.append(label)

        if agr < LOW_AGREEMENT_THRESHOLD:
            low_agreement.append(label)

        # Suspicious: low fire rate AND high self-match ratio
        # (i.e., it's only firing on its own sub_type and rarely even then)
        if (
            rate < 0.02
            and total > 0
            and s["cross_fire_ratio"] < 0.3
        ):
            suspicious.append(label)

    return {
        "strong": sorted(strong, key=lambda x: stats[x]["cross_fire"], reverse=True),
        "weak": sorted(weak, key=lambda x: stats[x]["total_fires"]),
        "low_agreement": sorted(
            low_agreement, key=lambda x: agreement[x]["mean_agreement"]
        ),
        "suspicious": sorted(suspicious),
    }


def build_report(
    stats_per_model: dict[str, dict[str, dict]],
    agreement: dict[str, dict],
    cooc: dict[tuple[str, str], int],
    per_st: dict[str, Counter],
    sub_type_counts: dict[str, int],
    labels: list[str],
    n_rows: int,
) -> str:
    """Assemble the full markdown report."""
    gemini_stats = stats_per_model["gemini3flash"]
    buckets = classify_labels(gemini_stats, agreement)

    lines = [
        "# Detection Label Audit Report",
        "",
        "**Generated:** 2026-04-10",
        "**Data source:** `training/data/audit/` (fresh annotations on current golden_train)",
        f"**Sample size:** {n_rows} stratified rows per model",
        f"**Labels analyzed:** {len(labels)}",
        f"**Models:** {', '.join(sorted(MODELS.keys()))}",
        "",
        "## Purpose",
        "",
        "Quantitative audit of the current detection label set to identify which labels earn their keep",
        "as cross-cutting semantic signals vs. which are dead weight that duplicates `sub_type_scores`.",
        "Data comes from fresh annotations on the current `golden_train.parquet` corpus, so findings",
        "reflect the state after post-iter9 data quality work. The purpose is to inform label-set",
        "decisions for the upcoming consolidated annotation run (which will introduce `log_content`",
        "and potentially additional labels).",
        "",
        "## TL;DR",
        "",
        "### Strong labels (keep)",
        "",
        "Labels with >=100 cross-fires — genuinely detecting embedded/mixed content beyond sub_type:",
        "",
    ]
    for label in buckets["strong"]:
        s = gemini_stats[label]
        lines.append(
            f"- `{label}` — {s['cross_fire']} cross-fires "
            f"({s['cross_fire_ratio']:.0%} of fires), agreement {agreement[label]['mean_agreement']:.3f}"
        )

    lines += [
        "",
        "### Weak labels (candidates for prompt refinement or retirement)",
        "",
        f"Labels with fire rate <{WEAK_FIRE_RATE_THRESHOLD:.0%} OR cross-fires <{WEAK_CROSS_FIRE_THRESHOLD}:",
        "",
    ]
    for label in buckets["weak"]:
        s = gemini_stats[label]
        lines.append(
            f"- `{label}` — {s['total_fires']} fires ({s['fire_rate']:.3f}), "
            f"{s['cross_fire']} cross-fires, agreement {agreement[label]['mean_agreement']:.3f}"
        )

    lines += [
        "",
        "### Suspicious labels (low fire + mostly self-match)",
        "",
        "Labels that barely fire AND when they do, only on their own sub_type:",
        "",
    ]
    if buckets["suspicious"]:
        for label in buckets["suspicious"]:
            s = gemini_stats[label]
            lines.append(
                f"- `{label}` — {s['total_fires']} fires, "
                f"{s['cross_fire_ratio']:.0%} cross-fire ratio"
            )
    else:
        lines.append("- None.")

    lines += [
        "",
        "### Low inter-annotator agreement",
        "",
        f"Labels where the 4 models disagree notably (mean agreement <{LOW_AGREEMENT_THRESHOLD:.0%}):",
        "",
    ]
    if buckets["low_agreement"]:
        for label in buckets["low_agreement"]:
            a = agreement[label]
            lines.append(
                f"- `{label}` — mean agreement {a['mean_agreement']:.3f}, "
                f"unanimous on {a['unanimous_rate']:.1%} of rows"
            )
    else:
        lines.append(
            f"- None. All labels achieve >={LOW_AGREEMENT_THRESHOLD:.0%} agreement."
        )

    model_names = sorted(stats_per_model.keys())
    n_models = len(model_names)
    lines += [
        "",
        "## Detailed Tables",
        "",
        format_fire_table(gemini_stats, agreement, n_rows, "gemini3flash"),
        "",
        format_agreement_table(agreement, gemini_stats),
        "",
        format_cooccurrence_table(cooc),
        "",
        format_per_subtype(per_st, sub_type_counts),
        "",
        "## Cross-Model Fire Rate Comparison",
        "",
        f"Sanity check: how consistent are fire rates across the {n_models} annotator models?",
        "Labels with large variance in fire rate are candidates for prompt clarification.",
        "",
        "| Label | " + " | ".join(model_names) + " | Max-Min |",
        "|---|" + "---:|" * n_models + "---:|",
    ]
    for label in sorted(labels):
        rates = {
            m: stats_per_model[m][label]["fire_rate"] for m in stats_per_model
        }
        span = max(rates.values()) - min(rates.values())
        row_cells = [f"{rates[m]:.3f}" for m in model_names]
        lines.append(
            f"| `{label}` | " + " | ".join(row_cells) + f" | {span:.3f} |"
        )

    lines += [
        "",
        "## Recommendations",
        "",
        "See buckets in TL;DR above. For the upcoming consolidated annotation run:",
        "",
        "1. **Strong labels** should stay in the label set without modification.",
        "2. **Suspicious labels** should be considered for retirement; their signal is",
        "   almost entirely duplicated by the sub_type head.",
        f"3. **Low-agreement labels** (mean agreement <{LOW_AGREEMENT_THRESHOLD:.0%}) need their",
        "   definitions tightened in `SYSTEM_PROMPT` before the next annotation run.",
        "   Disagreement across frontier-class models indicates a fuzzy definition,",
        "   not a model capability gap.",
        "4. **High-variance labels** (Max-Min > 0.05) also signal definition ambiguity.",
        "",
        "The new `log_content` label will be added alongside these changes. Additional",
        "new labels (`stack_trace`, `diff_patch`) should only be added if the audit",
        "confirms the detection head has capacity (i.e., existing labels aren't saturated).",
        "",
        "## Caveats",
        "",
        "- Fire rates are sensitive to sub_type distribution in the sampled rows. The",
        "  stratified sample aims to represent the full corpus, but rare sub_types",
        "  (e.g., `unknown` with 14 rows total) will have limited signal.",
        "- `plain` dominates the label distribution. Normalize comparisons against",
        "  sub_type_normalized rates rather than raw counts when possible.",
        "- Inter-annotator agreement uses a simple majority-vote metric rather than",
        "  Fleiss' kappa. Good enough for directional signal; not a formal statistical claim.",
        "- The annotator models (gemini3flash, sonnet, gpt-5.4-mini) are all frontier-class.",
        "  Disagreement between them reflects label ambiguity, not capability gaps.",
        "- Cross-fire counts below ~30 should be treated with suspicion — this may",
        "  reflect a dataset gap (missing examples of embedded-content scenarios) rather",
        "  than the label being genuinely low-value. Especially true for programming",
        "  language labels whose primary cross-cutting use case is 'prose documents with",
        "  embedded code blocks' — if that scenario is underrepresented in the corpus,",
        "  the audit cannot measure it.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Loading archived iter9 annotation data...")
    dfs = load_model_data()
    base_df = dfs["gemini3flash"]
    labels = detection_labels(base_df)
    n_rows = len(base_df)

    print(f"Loaded {len(dfs)} models, {n_rows} rows, {len(labels)} labels")

    print("Computing fire stats per model...")
    stats_per_model: dict[str, dict[str, dict]] = {}
    for name, df in dfs.items():
        stats_per_model[name] = compute_fire_stats(df, labels)

    print("Computing inter-annotator agreement...")
    agreement = compute_agreement(dfs, labels)

    print("Computing co-occurrence matrix...")
    cooc = compute_cooccurrence(base_df, labels)

    print("Computing per-sub_type distribution...")
    per_st = compute_per_subtype_dist(base_df, labels)
    sub_type_counts = {
        row[0]: row[1]
        for row in base_df.group_by("sub_type").len().iter_rows()
        if row[0] is not None
    }

    print("Building report...")
    report = build_report(
        stats_per_model=stats_per_model,
        agreement=agreement,
        cooc=cooc,
        per_st=per_st,
        sub_type_counts=sub_type_counts,
        labels=labels,
        n_rows=n_rows,
    )

    output_path = OUTPUT_PATH.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
