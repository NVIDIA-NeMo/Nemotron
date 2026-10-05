# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Curator-backed duplicate finder for BFCL Stage 11.

Importing this module pulls in NeMo Curator and Ray, so Stage 11 imports it only
when it is about to embed. The adapter reuses the shared semantic-dedup workflow
and adds what BFCL needs from it: BFCL's own id column, exact cluster membership
derived from the same embeddings, and a signature that pins the embeddings a run
actually used.
"""

from __future__ import annotations

import glob
import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.dedup_balancing import (
    DedupSettings,
    reconcile_curator_pairwise_artifacts,
)
from nemotron.steps.byob.runtime.deduplication import TextSemanticDeduplication


@dataclass(frozen=True)
class _CuratorConfig:
    """The attributes the shared Curator workflow reads off a config.

    Stage 11 clamps ``n_clusters`` to the rows on hand, so the workflow is handed
    the effective settings rather than the raw BFCL config.
    """

    output_dir: str
    expt_name: str
    semantic_deduplication_config: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, where: str) -> None:
    if missing := [column for column in columns if column not in frame.columns]:
        raise RuntimeError(
            f"Curator {where} output is missing {', '.join(missing)}; "
            f"observed columns: {', '.join(map(str, frame.columns))}"
        )


def _embedding_signature(vectors: dict[str, list[float]]) -> str:
    payload = canonical_json([[task_id, vectors[task_id]] for task_id in sorted(vectors)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TextSemanticDeduplicationBfcl(TextSemanticDeduplication):
    """Adapt the shared Curator workflow to BFCL's projected surfaces."""

    def prepare_input_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Project ``id`` and ``text``, the only columns the embedder reads."""
        return dataset[["id", "text"]].copy()

    def _mark_duplicates(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["is_duplicate"] = dataset["id"].isin(self._duplicate_ids())
        return dataset

    def analyze(self, dataset: pd.DataFrame, *, eps: float) -> dict[str, Any]:
        """Embed the projections and report duplicates, clusters, and lineage."""
        input_file = os.path.join(self.input_path, "projected_surfaces.parquet")
        self.prepare_input_data(dataset).to_parquet(input_file, index=False)
        self._compute_embeddings(input_file)
        self.workflow.run(kmeans_executor=self.kmeans_executor, pairwise_executor=self.executor)
        vectors = self._read_embeddings()
        duplicate_ids, cluster_by_id, pairwise_by_id = self._read_curator_decisions(
            task_ids=set(vectors),
            eps=eps,
        )
        return {
            "duplicate_ids": duplicate_ids,
            "cluster_by_id": cluster_by_id,
            "pairwise_by_id": pairwise_by_id,
            "embedding_signature": _embedding_signature(vectors),
        }

    def _read_embeddings(self) -> dict[str, list[float]]:
        paths = sorted(glob.glob(os.path.join(self.embeddings_path, "**", "*.parquet"), recursive=True))
        if not paths:
            raise RuntimeError(f"Curator wrote no embeddings under {self.embeddings_path}")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        _require_columns(frame, ("id", "embeddings"), where="embedding")
        ids = [str(task_id) for task_id in frame["id"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Curator embedding output contains repeated ids")
        vectors: dict[str, list[float]] = {}
        for task_id, embedding in zip(ids, frame["embeddings"], strict=True):
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Curator embedding output has a malformed vector for task {task_id!r}") from exc
            if not vector or any(not math.isfinite(value) for value in vector):
                raise RuntimeError(f"Curator embedding output requires a non-empty finite vector for task {task_id!r}")
            vectors[task_id] = vector
        if len({len(vector) for vector in vectors.values()}) != 1:
            raise RuntimeError("Curator embedding output contains inconsistent vector widths")
        return vectors

    def _duplicate_ids(self) -> list[str]:
        paths = sorted(glob.glob(os.path.join(self.output_path, "duplicates", "*.parquet")))
        if not paths:
            return []
        frame = pd.concat([pd.read_parquet(path) for path in paths])
        _require_columns(frame, ("id",), where="duplicates")
        return sorted(str(task_id) for task_id in frame["id"])

    def _read_curator_decisions(
        self,
        *,
        task_ids: set[str],
        eps: float,
    ) -> tuple[list[str], dict[str, str], dict[str, dict[str, Any]]]:
        """Build clusters from the exact Curator pairs that produced duplicates.

        Curator writes one best predecessor and similarity for every row in each
        K-means partition. Its duplicate stage removes precisely the rows whose
        score crosses ``1 - eps``. Joining those predecessor edges therefore
        recovers connected duplicate groups without a second similarity
        implementation or comparisons across Curator's candidate partitions.
        """
        paths = sorted(
            glob.glob(
                os.path.join(self.workflow.pairwise_output_path, "**", "*.parquet"),
                recursive=True,
            )
        )
        if not paths:
            raise RuntimeError(f"Curator wrote no pairwise decisions under {self.workflow.pairwise_output_path}")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        _require_columns(
            frame,
            ("id", "max_id", "cosine_sim_score"),
            where="pairwise",
        )
        duplicate_ids = self._duplicate_ids()
        pairs = [
            (str(task_id), str(predecessor), float(score))
            for task_id, predecessor, score in zip(
                frame["id"],
                frame["max_id"],
                frame["cosine_sim_score"],
                strict=True,
            )
        ]
        try:
            cluster_by_id = reconcile_curator_pairwise_artifacts(
                task_ids=task_ids,
                pairs=pairs,
                duplicate_ids=duplicate_ids,
                eps=eps,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        pairwise_by_id = {
            task_id: {
                "predecessor_id": predecessor,
                "similarity_score": score,
            }
            for task_id, predecessor, score in pairs
        }
        return duplicate_ids, cluster_by_id, pairwise_by_id


def curator_duplicate_finder(
    *,
    config: BfclConfig,
    settings: DedupSettings,
    n_clusters: int,
    rows: list[dict[str, str]],
    **_: Any,
) -> dict[str, Any]:
    """Run Curator over the projected surfaces of one Stage 11 input set."""
    adapter = TextSemanticDeduplicationBfcl(
        _CuratorConfig(
            output_dir=str(config.output_dir),
            expt_name=str(config.expt_name),
            semantic_deduplication_config={
                "model_identifier": settings.model_identifier,
                "n_clusters": n_clusters,
                "eps": settings.eps,
                "remove_duplicates": settings.remove_duplicates,
            },
        )
    )
    return adapter.analyze(pd.DataFrame(rows, columns=["id", "text"]), eps=settings.eps)
