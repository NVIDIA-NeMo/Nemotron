from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np

from data_designer.config.base import SingleColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell
from data_designer.plugins import Plugin, PluginType

logger = logging.getLogger(__name__)


class DDRetrievalDedupConfig(SingleColumnConfig):
    """Deduplicate QA pairs from a DD Retrieval set."""

    qa_pairs_column: str
    embedding_alias: str
    column_type: Literal["data-designer-retrieval-dedup"] = "data-designer-retrieval-dedup"
    dedupe_similarity_threshold: float = 0.9

    @property
    def required_columns(self) -> list[str]:
        return [self.qa_pairs_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class DDRetrievalDedup(ColumnGeneratorCellByCell[DDRetrievalDedupConfig]):

    @property
    def embedder(self):
        return self.resource_provider.model_registry.get_model(
            model_alias=self.config.embedding_alias)

    def _embed(self, text: str) -> list[float]:
        """Calculate an embedding of the text
        """
        response = self.embedder._router.embedding(
            input=text,
            model=self.embedder.model_name,
            encoding_format="float",
            extra_body=self.embedder._model_config.inference_parameters.
            extra_body)

        return response.data[0]["embedding"]

    def dedupe_qa_pairs(self, embeddings: list[list[float]]) -> list[int]:
        """Run a semantic dedupe of the qa pairs.

        Run embeddings and calculate pairwise cosine similarity with numpy.
        For every pair with similarity > self.config.dedupe_cosine_threshold, just take the 
        first one of the pair.

        Returns the index list of vectors to retain.
        """
        if not embeddings:
            return []

        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D array of shape (n, d).")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms

        cosine_sim = np.clip(normalized @ normalized.T, -1.0, 1.0)

        threshold = self.config.dedupe_similarity_threshold
        keep_indexes: list[int] = []
        dropped = np.zeros(len(embeddings), dtype=bool)

        for i in range(len(embeddings)):
            if dropped[i]:
                continue

            keep_indexes.append(i)
            if i == len(embeddings) - 1:
                continue

            close_matches = np.where(cosine_sim[i,
                                                i + 1:] > threshold)[0] + i + 1
            dropped[close_matches] = True

        return keep_indexes

    def generate(self, data: dict) -> dict:
        logger.debug("Deduplicating QA pairs from column: %s", self.config.qa_pairs_column)

        ## Acquiring QA pairs
        qa_pairs: list = data[self.config.qa_pairs_column]["pairs"]
        max_parallel = getattr(
            self.embedder._model_config.inference_parameters,
            "max_parallel_requests",
            None,
        )
        workers = max(1, max_parallel or 1)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            embeddings = list(
                executor.map(self._embed, [qa["question"] for qa in qa_pairs]))

        retained_qa_pair_indexes = self.dedupe_qa_pairs(embeddings)
        dropped = len(qa_pairs) - len(retained_qa_pair_indexes)
        if dropped > 0:
            logger.info(
                "Dedup: retained %d of %d QA pairs (%d duplicates removed)",
                len(retained_qa_pair_indexes),
                len(qa_pairs),
                dropped,
            )
        else:
            logger.debug(
                "Dedup: retained all %d QA pairs (no duplicates)",
                len(qa_pairs),
            )

        retained_qa_pairs = [qa_pairs[i] for i in retained_qa_pair_indexes]

        return data | {self.config.name: retained_qa_pairs}


dd_retrieval_dedup_plugin = Plugin(
    impl_qualified_name="retriever_sdg.deduplication.DDRetrievalDedup",
    config_qualified_name="retriever_sdg.deduplication.DDRetrievalDedupConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)