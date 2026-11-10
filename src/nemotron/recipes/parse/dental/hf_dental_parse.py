# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Expose native Parse's visual prefix as sequence positions for NeMo recipes.

Checkpoint tensor names and the native encoder/decoder modules are preserved.
"""

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from .hf_nemotron_parse_modeling import NemotronParseForConditionalGeneration


class DentalParseForConditionalGeneration(NemotronParseForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.configure_trainable_layers()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        # HF loading may replace Parameter objects after the constructor runs.
        model.configure_trainable_layers()
        return model

    def configure_trainable_layers(self):
        scope = getattr(self.config, "dental_train_scope", "full")
        if scope not in ("full", "decoder"):
            raise ValueError("Training scope must be full or decoder")
        self.requires_grad_(scope == "full")
        if scope == "decoder":
            self.decoder.requires_grad_(True)
            self.lm_head.requires_grad_(True)
        # The objective never executes auxiliary MTP heads. Excluding them
        # keeps optimizer checkpoint state aligned with parameters that train.
        self.decoder.extra_heads.requires_grad_(False)
        self.decoder.extra_proj.requires_grad_(False)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        content_sizes=None,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
        past_key_values=None,
        **kwargs,
    ):
        """Compute logits aligned with NeMo RL's expanded multimodal sequence.

        Args:
            input_ids: Long tensor [batch, sequence], beginning with one pad
                placeholder for every visual feature. Text follows that prefix.
            pixel_values: Packed float patches [patches, 768] or [1, patches, 768].
            image_sizes: Long tensor [batch, 2] of padded image height and width.
            content_sizes: Long tensor [batch, 2] of content height and width.
            attention_mask: Optional mask [batch, sequence]; visual positions
                must be valid. Right padding may be masked.
            use_cache: Use the native text-only cached-generation interface when true.
            past_key_values: Native generation cache; training does not supply one.
            return_dict: Return the standard model output when true.
            **kwargs: Position IDs are unused by the native position-free decoder.

        Returns:
            CausalLMOutputWithPast with logits [batch, sequence, vocab]. Visual
            prefix logits are zero placeholders, excluded by the recipe masks.
        """
        if use_cache or past_key_values is not None:
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                content_sizes=content_sizes,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=return_dict,
                **kwargs,
            )

        if input_ids is None or image_sizes is None or content_sizes is None:
            raise ValueError("Expanded input_ids and both image geometry tensors are required")
        counts_from_geometry = [
            (height // 16) * (width // 64) + 1 for height, width in image_sizes.tolist()
        ]
        if len(set(counts_from_geometry)) == 1:
            count = counts_from_geometry[0]
            if count >= input_ids.shape[1] or not torch.all(
                input_ids[:, :count] == self.config.pad_token_id
            ):
                raise ValueError("Input visual placeholders do not match the encoded image")
            if attention_mask is not None and not attention_mask[:, :count].bool().all():
                raise ValueError("Visual placeholders must remain valid in attention_mask")
            # Preserve the original forward/backward graph for homogeneous
            # geometry, including the single-image cookbook configuration.
            native = super().forward(
                input_ids=input_ids[:, count:],
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                content_sizes=content_sizes,
                attention_mask=None if attention_mask is None else attention_mask[:, count:],
                use_cache=False,
                return_dict=True,
            )
            output = CausalLMOutputWithPast(logits=F.pad(native.logits, (0, 0, count, 0)))
            return output if return_dict else (output.logits, None)
        vision, visual_mask = self.encoder(pixel_values, image_sizes, content_sizes)
        embeddings = self.decoder.embed_tokens(input_ids)
        mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        rows = []
        counts = visual_mask.sum(dim=1).tolist()
        for index, count in enumerate(counts):
            if count >= input_ids.shape[1] or not torch.all(
                input_ids[index, :count] == self.config.pad_token_id
            ):
                raise ValueError("Input visual placeholders do not match the encoded image")
            if not mask[index, :count].all():
                raise ValueError("Visual placeholders must remain valid in attention_mask")
            rows.append(
                torch.cat(
                    (vision[index, :count].to(embeddings.dtype), embeddings[index, count:]), dim=0
                )
            )
        hidden = self.decoder(torch.stack(rows), None, mask)
        # Native Parse projects text only. Visual positions have no targets.
        logits = torch.cat(
            [
                F.pad(self.lm_head(hidden[index : index + 1, count:]), (0, 0, count, 0))
                for index, count in enumerate(counts)
            ],
            dim=0,
        )
        output = CausalLMOutputWithPast(logits=logits, past_key_values=None)
        return output if return_dict else (output.logits, None)
