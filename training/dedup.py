#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["faiss-cpu", "sentence-transformers", "pandas", "numpy"]
# ///
"""FAISS two-layer deduplication pipeline.

Layer 1: Feature-space dedup using L2 distance on structural features.
Layer 2: Semantic dedup using cosine similarity on sentence-transformer embeddings.
"""

import argparse
import logging
import sys

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def feature_dedup(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Remove near-duplicate samples based on L2 distance in feature space.

    Builds a FAISS IndexFlatL2 from the specified feature columns. For each
    sample, queries the 2-nearest neighbors (self + nearest). Marks samples
    whose nearest-neighbor L2 distance is below *threshold* as duplicates,
    keeping the first occurrence.

    Args:
        df: Input dataframe.
        feature_cols: Column names to use as feature dimensions.
        threshold: L2 distance below which two samples are considered duplicates.

    Returns:
        Deduplicated dataframe with the original column order preserved.
    """
    n = len(df)
    if n <= 1 or threshold <= 0:
        logger.info("Feature dedup: removed 0 of %d samples (0.0%%) %s", n,
                     "(disabled)" if threshold <= 0 else "")
        return df.copy()

    features = df[feature_cols].values.astype(np.float32)
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)

    # Query 2-nearest neighbors: self (distance 0) + closest other
    distances, indices_arr = index.search(features, 2)

    # Find the nearest neighbor that is not self for each sample.
    nn_distances = np.empty(n, dtype=np.float32)
    nn_indices = np.empty(n, dtype=np.int64)
    for i in range(n):
        if int(indices_arr[i, 0]) != i:
            nn_distances[i] = distances[i, 0]
            nn_indices[i] = int(indices_arr[i, 0])
        else:
            nn_distances[i] = distances[i, 1]
            nn_indices[i] = int(indices_arr[i, 1])

    # Walk forward: if a sample's nearest neighbor is closer than threshold
    # and that neighbor has a *lower* index, mark the current sample as dup.
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        if nn_distances[i] < threshold:
            nn_idx = int(nn_indices[i])
            # Mark the later index as duplicate
            if nn_idx < i:
                keep[i] = False
            else:
                keep[nn_idx] = False

    removed = int(n - keep.sum())
    pct = removed / n * 100
    logger.info("Feature dedup: removed %d of %d samples (%.1f%%)", removed, n, pct)

    return df.loc[keep].reset_index(drop=True)


def semantic_dedup(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Remove near-duplicate samples based on cosine similarity of embeddings.

    Loads the ``all-MiniLM-L6-v2`` sentence-transformer, encodes all texts to
    384-dim embeddings, L2-normalizes them, and builds a FAISS IndexFlatIP
    (inner product = cosine similarity on normalized vectors). Samples whose
    nearest-neighbor cosine similarity exceeds *threshold* are marked as
    duplicates, keeping the first occurrence.

    Args:
        df: Input dataframe.
        text_col: Name of the column containing text.
        threshold: Cosine similarity above which two samples are duplicates.

    Returns:
        Deduplicated dataframe.
    """
    from sentence_transformers import SentenceTransformer

    n = len(df)
    if n <= 1:
        logger.info("Semantic dedup: removed 0 of %d samples (0.0%%)", n)
        return df.copy()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [str(t) if not isinstance(t, str) else t for t in df[text_col].tolist()]
    embeddings = model.encode(texts, convert_to_numpy=True).astype(np.float32)

    # L2 normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Query 2-nearest neighbors
    similarities, indices_arr = index.search(embeddings, 2)

    # Find the nearest neighbor that is not self for each sample.
    nn_sims = np.empty(n, dtype=np.float32)
    nn_indices = np.empty(n, dtype=np.int64)
    for i in range(n):
        if int(indices_arr[i, 0]) != i:
            nn_sims[i] = similarities[i, 0]
            nn_indices[i] = int(indices_arr[i, 0])
        else:
            nn_sims[i] = similarities[i, 1]
            nn_indices[i] = int(indices_arr[i, 1])

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        if nn_sims[i] > threshold:
            nn_idx = int(nn_indices[i])
            if nn_idx < i:
                keep[i] = False
            else:
                keep[nn_idx] = False

    removed = int(n - keep.sum())
    pct = removed / n * 100
    logger.info("Semantic dedup: removed %d of %d samples (%.1f%%)", removed, n, pct)

    return df.loc[keep].reset_index(drop=True)


def dedup_pipeline(
    input_csv: str,
    output_csv: str,
    feature_threshold: float = 0.1,
    semantic_threshold: float = 0.9,
) -> None:
    """Run the full two-layer dedup pipeline: feature dedup then semantic dedup.

    Reads *input_csv*, applies feature-space deduplication on all numeric
    columns (excluding the ``text`` column), then applies semantic
    deduplication on the ``text`` column. Writes the result to *output_csv*
    and prints a summary.

    Args:
        input_csv: Path to input CSV file.
        output_csv: Path to write deduplicated CSV.
        feature_threshold: L2 distance threshold for feature dedup.
        semantic_threshold: Cosine similarity threshold for semantic dedup.
    """
    df = pd.read_csv(input_csv)
    original_count = len(df)

    # Detect feature columns: all numeric columns except 'text'
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "text"
    ]

    if feature_cols:
        df_after_feat = feature_dedup(df, feature_cols, threshold=feature_threshold)
    else:
        logger.info("No numeric feature columns found — skipping feature dedup")
        df_after_feat = df
    after_feature_count = len(df_after_feat)

    df_after_sem = semantic_dedup(
        df_after_feat, text_col="text", threshold=semantic_threshold
    )
    final_count = len(df_after_sem)

    df_after_sem.to_csv(output_csv, index=False)

    print(f"Original count: {original_count}")
    print(f"After feature dedup: {after_feature_count}")
    print(f"After semantic dedup: {final_count}")
    print(f"Final count: {final_count}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FAISS two-layer deduplication pipeline"
    )
    parser.add_argument(
        "--input",
        default="training/data/golden_raw.csv",
        help="Path to input CSV (default: training/data/golden_raw.csv)",
    )
    parser.add_argument(
        "--output",
        default="training/data/golden_train.csv",
        help="Path to output CSV (default: training/data/golden_train.csv)",
    )
    parser.add_argument(
        "--feature-threshold",
        type=float,
        default=0.1,
        help="L2 distance threshold for feature dedup (default: 0.1)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for semantic dedup (default: 0.9)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dedup_pipeline(
        input_csv=args.input,
        output_csv=args.output,
        feature_threshold=args.feature_threshold,
        semantic_threshold=args.semantic_threshold,
    )


if __name__ == "__main__":
    main()
