from __future__ import annotations

from typing import Literal

from data_designer.config.base import SingleColumnConfig
from data_designer.engine.column_generators.generators.base import ColumnGeneratorCellByCell
from data_designer.plugins import Plugin, PluginType


class DDRetrievalDedupConfig(SingleColumnConfig):
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
    def generate(self, data: dict) -> dict:
        qa_obj = data[self.config.qa_pairs_column]

        if isinstance(qa_obj, dict):
            pairs = qa_obj.get("pairs", qa_obj)
        else:
            pairs = getattr(qa_obj, "pairs", qa_obj)

        return data | {self.config.name: pairs}


dd_retrieval_dedup_plugin = Plugin(
    impl_qualified_name="retriever_sdg.deduplication.DDRetrievalDedup",
    config_qualified_name="retriever_sdg.deduplication.DDRetrievalDedupConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
