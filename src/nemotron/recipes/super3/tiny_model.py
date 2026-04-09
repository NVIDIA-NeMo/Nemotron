# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tiny Super3 model provider for single-GPU integration testing.

Defines :class:`Nemotron3SuperTinyProvider`, a scaled-down Super3 architecture
(~7M params) that exercises every Super3-specific code path:

- Hybrid Mamba + Attention layers  (pattern ``MEM*EME``)
- Mixture-of-Experts with latent routing  (``moe_latent_size``)
- Multi-Token Prediction  (``mtp_num_layers=2``)
- Shared expert  (``moe_shared_expert_intermediate_size``)

Used by ``test_train.py`` in both ``stage0_pretrain`` and ``stage1_sft``.
"""

from __future__ import annotations

from dataclasses import dataclass

from megatron.bridge.models.nemotronh import Nemotron3SuperProvider


@dataclass
class Nemotron3SuperTinyProvider(Nemotron3SuperProvider):
    """Architecturally-valid tiny Super3 for integration testing.

    ~7M params total.  Preserves every Super3-unique feature at minimal scale:

    ==================  ==========  ======  ==============================
    Parameter           Full Super  Tiny    Notes
    ==================  ==========  ======  ==============================
    num_layers          88          7       pattern ``MEM*EME``
    hidden_size         4096        256     divisible by num_attention_heads
    num_attention_heads 32          4       head_dim = 64
    kv_channels         128         64      matches head_dim
    mamba_num_heads     128         8
    mamba_head_dim      64          32
    num_moe_experts     512         16      fits on 1 GPU with EP=1
    moe_ffn_hidden_size 2688        384     scaled proportionally
    moe_latent_size     1024        128     exercises latent routing
    mtp_num_layers      2           2       unchanged — exercises MTP
    ==================  ==========  ======  ==============================
    """

    hybrid_override_pattern: str = "MEM*EME"
    num_layers: int = 7
    hidden_size: int = 256
    num_attention_heads: int = 4
    kv_channels: int = 64
    num_query_groups: int = 2
    mamba_num_heads: int = 8
    mamba_head_dim: int = 32
    mamba_state_dim: int = 128
    ffn_hidden_size: int = 384
    num_moe_experts: int = 16
    moe_ffn_hidden_size: int = 384
    moe_shared_expert_intermediate_size: int = 768  # 384 × 2
    moe_router_topk: int = 2
    moe_router_topk_scaling_factor: float = 2.5
    moe_latent_size: int = 128
    mtp_num_layers: int = 2
    mtp_hybrid_override_pattern: str = "*E"


def make_tiny_super3_model(seq_length: int = 8192) -> Nemotron3SuperTinyProvider:
    """Construct a tiny Super3 model configured for single-GPU execution.

    Returns a fully-configured :class:`Nemotron3SuperTinyProvider` with TP=1,
    EP=1, and all production model-init kwargs (attention backend, fusions,
    MTP settings, etc.) matching the production recipe.

    Args:
        seq_length: Sequence length for the model. Defaults to 8192.
    """
    # Only pass fields guaranteed to be on the dataclass hierarchy to __init__.
    # All other settings are applied post-construction as attributes, since
    # available fields vary across Megatron-Core / Megatron-Bridge versions.
    model = Nemotron3SuperTinyProvider(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        seq_length=seq_length,
    )

    # Match the production recipe's model-init settings.
    # Set as attributes to avoid constructor errors when fields don't exist
    # on TransformerConfig in the target container's Megatron-Core version.
    model.apply_rope_fusion = False
    model.async_tensor_model_parallel_allreduce = True
    model.attention_backend = "flash"
    model.gradient_accumulation_fusion = True
    model.init_method_std = 0.014
    model.use_fused_weighted_squared_relu = True
    model.keep_mamba_stack_attention_linear_in_bf16 = True
    model.keep_mtp_spec_in_bf16 = True
    model.calculate_per_token_loss = True
    model.mtp_loss_scaling_factor = 0.3
    model.moe_token_dispatcher_type = "alltoall"
    model.moe_shared_expert_overlap = False
    model.use_te_rng_tracker = True
    model.mtp_use_repeated_layer = True
    model.enable_cuda_graph = False

    return model
