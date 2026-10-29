# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native AutoModel adapter for BEIR embedding evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemotron.recipes.embed.mining_encoder import CheckpointMiningEncoderConfig


class AutoModelBEIREncoder:
    """Encode BEIR queries and documents with saved AutoModel retrieval semantics."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        multimodal_config: CheckpointMiningEncoderConfig,
    ) -> None:
        import torch
        from nemo_automodel._transformers.auto_model import NeMoAutoModelBiEncoder

        model_options = {"use_liger_kernel": False, "use_sdpa_patching": True, "attn_implementation": "sdpa"}
        self.model = NeMoAutoModelBiEncoder.from_pretrained(str(model_path), **model_options)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device.type)
        self.model.eval()
        expected = ("avg", True, False)
        actual = (self.model.pooling, self.model.l2_normalize, self.model.is_causal)
        if actual != expected:
            raise ValueError(f"Multimodal checkpoint retrieval semantics must be {expected}; got {actual}")
        self.multimodal_encoder = multimodal_config.build(model=self.model, device=device)

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs: Any) -> Any:
        """Encode query strings using the configured query prefix."""
        return self.multimodal_encoder.encode_queries(queries, batch_size=batch_size)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, dict[str, str]],
        batch_size: int = 8,
        **kwargs: Any,
    ) -> Any:
        """Encode BEIR records through the configured text or multimodal processor."""
        documents = list(corpus.values()) if isinstance(corpus, dict) else corpus
        multimodal_documents = [
            {
                "title": document.get("title", ""),
                "text": document.get("text", ""),
                "image": document.get("image_path"),
            }
            for document in documents
        ]
        return self.multimodal_encoder.encode_documents(multimodal_documents, batch_size=batch_size)
