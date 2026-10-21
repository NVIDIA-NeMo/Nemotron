# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic near-duplicate removal with an injectable embedding backend."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np


def greedy_keep_indices(similar_pairs: list[tuple[int, int]], count: int, seed: int) -> list[int]:
    adjacency: defaultdict[int, list[int]] = defaultdict(list)
    for left, right in similar_pairs:
        adjacency[left].append(right)
        adjacency[right].append(left)
    order = list(range(count))
    random.Random(seed).shuffle(order)
    removed: set[int] = set()
    kept: list[int] = []
    for index in order:
        if index in removed:
            continue
        kept.append(index)
        removed.update(candidate for candidate in adjacency[index] if candidate not in kept)
    return sorted(kept)


def component_keep_indices(similar_pairs: list[tuple[int, int]], count: int, seed: int) -> list[int]:
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left, right in similar_pairs:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[left_root] = right_root
    components: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(count):
        components[find(index)].append(index)
    rng = random.Random(seed)
    return sorted(rng.choice(members) for members in components.values())


def find_similar_pairs(embeddings: np.ndarray, threshold: float, chunk_size: int = 4096) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for start in range(0, len(embeddings), chunk_size):
        stop = min(start + chunk_size, len(embeddings))
        similarities = embeddings[start:stop] @ embeddings.T
        for local_index, row in enumerate(similarities):
            index = start + local_index
            matches = np.flatnonzero(row[index + 1 :] >= threshold)
            pairs.extend((index, index + 1 + int(offset)) for offset in matches)
    return pairs


def deduplicate_embeddings(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    threshold: float,
    method: str,
    seed: int,
    chunk_size: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if embeddings.shape[0] != len(records):
        raise ValueError("embedding row count does not match record count")
    pairs = find_similar_pairs(embeddings, threshold, chunk_size)
    if method == "greedy":
        keep = greedy_keep_indices(pairs, len(records), seed)
    elif method == "components":
        keep = component_keep_indices(pairs, len(records), seed)
    else:
        raise ValueError(f"Unknown semantic dedup method: {method!r}")
    return [records[index] for index in keep], {"loaded": len(records), "pairs": len(pairs), "accepted": len(keep)}


def embed_questions(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Semantic dedup requires the 'data-sdg' optional dependency") from exc
    model = SentenceTransformer(model_name, device=device)
    values = model.encode(
        [f"query: {record['question']}" for record in records],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(values)
