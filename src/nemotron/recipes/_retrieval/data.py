# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalize missing corpus modalities for the pinned AutoModel dataset.

AutoModel already decodes binary images. Its retrieval transform expects an
empty string for an absent image, while mixed plugin corpora use nulls.
"""

from dataclasses import dataclass
from typing import Any

from nemo_automodel.components.datasets.llm.retrieval_dataset import RetrievalDatasetConfig


@dataclass
class _NullableCorpus:
    source: Any

    @property
    def path(self) -> str:
        return self.source.path

    def get_all_ids(self) -> list[str]:
        return self.source.get_all_ids()

    def get_document_by_id(self, identifier: str) -> dict[str, Any]:
        document = self.source.get_document_by_id(identifier).copy()
        for field in ("text", "image"):
            if document.get(field) is None:
                document[field] = ""
        return document


class NullableImageDatasetConfig(RetrievalDatasetConfig):
    """Keep AutoModel configuration and loading, normalizing absent modalities."""

    def build(self) -> Any:
        dataset = super().build()
        # AutoModel installs this lazy transform before any documents are read.
        # Wrap only this dataset's corpus objects; leave its global registry alone.
        transform = dataset.format["format_kwargs"]["transform"]
        for info in transform.corpus_dict.values():
            info.corpus = _NullableCorpus(info.corpus)
        return dataset
