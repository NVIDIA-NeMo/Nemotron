#!/usr/bin/env python3

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

"""Shared vocab-padding helper for all init engines.

Megatron's VocabParallelEmbedding requires the embedding row count to be divisible
by the tensor-parallel size (TP=4 here, so we pad to a multiple of 4). The resized
vocab is arbitrary (base + extension, or survivors + fresh), so we pad the row count
up to a multiple here — uniformly for add and replace, every method — so
divisibility is decoupled from the (often odd) vocab. Padding rows sit above the
tokenizer's real ids, are filled with the base mean, and are never indexed by data.
"""

from __future__ import annotations


def pad_vocab_to_multiple(model, multiple: int) -> int:
    """Pad the model's input/output embeddings up to a multiple of `multiple`.

    Returns the final row count. No-op if `multiple` is falsy or already aligned.
    """
    import torch

    if not multiple:
        return model.get_input_embeddings().weight.shape[0]
    v = model.get_input_embeddings().weight.shape[0]
    if v % multiple == 0:
        print(f"  vocab {v:,} already a multiple of {multiple}; no padding")
        return v
    padded = ((v + multiple - 1) // multiple) * multiple
    with torch.no_grad():
        fill_in = model.get_input_embeddings().weight[:v].mean(dim=0).clone()
        fill_out = model.get_output_embeddings().weight[:v].mean(dim=0).clone()
    model.resize_token_embeddings(padded, mean_resizing=False)
    with torch.no_grad():
        model.get_input_embeddings().weight[v:] = fill_in
        model.get_output_embeddings().weight[v:] = fill_out
    print(f"  padded vocab {v:,} -> {padded:,} (multiple of {multiple}) for TP divisibility")
    return padded
