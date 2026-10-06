# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe preprocessing controls over AutoModel's native checkpoint encoder."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class CheckpointMiningEncoderConfig:
    """Keep mining and evaluation preprocessing aligned with recipe training."""

    q_max_length: int | None = None
    p_max_length: int | None = None
    query_prefix: str | None = None
    passage_prefix: str | None = None
    image_longest_edge: int | None = None
    use_images: bool = True
    use_text_in_document: bool = True

    def build(
        self,
        *,
        device: Any,
        model: Any = None,
        model_name_or_path: str | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> Any:
        from nemo_automodel import NeMoAutoModelBiEncoder
        from nemo_automodel.recipes.retrieval.mining_encoder import CheckpointMiningEncoder

        if model is None:
            options = {"use_liger_kernel": False, "use_sdpa_patching": True, "trust_remote_code": trust_remote_code}
            if attn_implementation is not None:
                options["attn_implementation"] = attn_implementation
            model = NeMoAutoModelBiEncoder.from_pretrained(model_name_or_path, **options).to(device)
            model.eval()
        target = getattr(model.model, "retrieval_processor_target", None)
        if target is None:
            raise ValueError("Checkpoint does not declare a supported retrieval processor")
        module_name, class_name = target.rsplit(".", 1)
        processor_class = getattr(import_module(module_name), class_name)
        options = {
            name: getattr(self, name)
            for name in ("q_max_length", "p_max_length", "query_prefix", "passage_prefix", "image_longest_edge")
            if getattr(self, name) is not None
        }
        if model.config._commit_hash is not None:
            options["revision"] = model.config._commit_hash
        processor = processor_class.from_pretrained(model.config.name_or_path or model.source_model_path, **options)
        use_images, use_text = self.use_images, self.use_text_in_document

        class RecipeCheckpointEncoder(CheckpointMiningEncoder):
            def encode_documents(self, documents, *, batch_size):
                prepared = []
                for document in documents:
                    document = dict(document)
                    if not use_images:
                        document.pop("image", None)
                    image = document.get("image")
                    has_image = image is not None and not (isinstance(image, str) and image == "")
                    if not use_text and has_image:
                        document["text"] = ""
                        document["title"] = ""
                    prepared.append(document)
                return super().encode_documents(prepared, batch_size=batch_size)

        return RecipeCheckpointEncoder(model=model, processor=processor, device=device)
