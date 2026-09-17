# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native AutoModel adapter for BEIR embedding evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_automodel._transformers.mining import CheckpointMiningEncoderConfig


class AutoModelBEIREncoder:
    """Encode BEIR queries and documents with saved AutoModel retrieval semantics."""

    def __init__(
        self,
        model_path: str | Path,
        max_length: int = 512,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        tokenizer_force_default: bool = False,
        multimodal_config: CheckpointMiningEncoderConfig | None = None,
    ) -> None:
        import torch
        from nemo_automodel._transformers.auto_model import NeMoAutoModelBiEncoder

        self.torch = torch
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        model_options = {"use_liger_kernel": False, "use_sdpa_patching": True}
        if multimodal_config is None:
            model_options.update(
                pooling="avg",
                l2_normalize=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
            )
        self.model = NeMoAutoModelBiEncoder.from_pretrained(str(model_path), **model_options)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device.type)
        self.model.eval()
        self.multimodal_encoder = None
        self.tokenizer = None
        if multimodal_config is not None:
            expected = ("avg", True, False)
            actual = (self.model.pooling, self.model.l2_normalize, self.model.is_causal)
            if actual != expected:
                raise ValueError(f"Multimodal checkpoint retrieval semantics must be {expected}; got {actual}")
            self.multimodal_encoder = multimodal_config.build(model=self.model, device=device)
        else:
            from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

            self.tokenizer = NeMoAutoTokenizer.from_pretrained(
                str(model_path),
                force_default=tokenizer_force_default,
                add_bos_token=True,
                add_eos_token=False,
            )

    def _encode(self, texts: list[str], prefix: str, batch_size: int) -> Any:
        """Encode prefixed text batches with native pooling and normalization."""
        embeddings = []
        device = next(self.model.parameters()).device
        for start in range(0, len(texts), batch_size):
            batch = [prefix + text for text in texts[start : start + batch_size]]
            tokenized = self.tokenizer(
                batch,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_token_type_ids=False,
            )
            features = [{key: values[index] for key, values in tokenized.items()} for index in range(len(batch))]
            inputs = self.tokenizer.pad(features, padding="longest", return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with self.torch.no_grad():
                embeddings.append(self.model.encode(inputs).detach().cpu())
        return self.torch.cat(embeddings)

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs: Any) -> Any:
        """Encode query strings using the configured query prefix."""
        if self.multimodal_encoder is not None:
            return self.multimodal_encoder.encode_queries(queries, batch_size=batch_size)
        return self._encode(queries, self.query_prefix, batch_size)

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, dict[str, str]],
        batch_size: int = 8,
        **kwargs: Any,
    ) -> Any:
        """Encode BEIR records through the configured text or multimodal processor."""
        documents = list(corpus.values()) if isinstance(corpus, dict) else corpus
        if self.multimodal_encoder is not None:
            multimodal_documents = [
                {
                    "title": document.get("title", ""),
                    "text": document.get("text", ""),
                    "image": document.get("image_path"),
                }
                for document in documents
            ]
            return self.multimodal_encoder.encode_documents(multimodal_documents, batch_size=batch_size)
        texts = [(document.get("title", "") + " " + document.get("text", "")).strip() for document in documents]
        return self._encode(texts, self.passage_prefix, batch_size)
