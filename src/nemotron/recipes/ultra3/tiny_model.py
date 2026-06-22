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

"""Tiny Ultra3 model provider for single-GPU integration testing.

Defines :class:`Nemotron3UltraTinyProvider`, a scaled-down Ultra3/Nemotron-H
architecture (~7M params) that exercises every Ultra3-specific code path:

- Hybrid Mamba + Attention layers  (pattern ``MEM*EME``)
- Mixture-of-Experts with latent routing  (``moe_latent_size``)
- Multi-Token Prediction  (``mtp_num_layers=2``)
- Shared expert  (``moe_shared_expert_intermediate_size``)

Used by ``test_train.py`` in ``stage0_pretrain``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from megatron.core.activations import squared_relu
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider


@dataclass
class Nemotron3UltraTinyProvider(MambaModelProvider):
    """Architecturally-valid tiny Ultra3 for integration testing.

    ~7M params total.  Preserves every Ultra3/Nemotron-H feature at minimal
    scale:

    ==================  ==========  ======  ==============================
    Parameter           Full Ultra  Tiny    Notes
    ==================  ==========  ======  ==============================
    num_layers          108         7       pattern ``MEM*EME``
    hidden_size         6144        256     divisible by num_attention_heads
    num_attention_heads 48          4       head_dim = 64
    kv_channels         128         64      matches head_dim
    mamba_num_heads     192         8
    mamba_head_dim      64          32
    num_moe_experts     512         16      fits on 1 GPU with EP=1
    moe_ffn_hidden_size 3840        384     scaled down
    moe_latent_size     2048        128     exercises latent routing
    mtp_num_layers      2           2       unchanged — exercises MTP
    ==================  ==========  ======  ==============================
    """

    hybrid_layer_pattern: str = "MEM*EME"
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


def make_tiny_ultra3_model(seq_length: int = 8192) -> Nemotron3UltraTinyProvider:
    """Construct a tiny Ultra3 model configured for single-GPU execution.

    Returns a fully-configured :class:`Nemotron3UltraTinyProvider` with TP=1,
    EP=1, and all production model-init kwargs (attention backend, fusions,
    MTP settings, etc.) matching the production recipe.

    Args:
        seq_length: Sequence length for the model. Defaults to 8192.
    """
    # Only pass fields guaranteed to be on the dataclass hierarchy to __init__.
    # All other settings are applied post-construction as attributes, since
    # available fields vary across Megatron-Core / Megatron-Bridge versions.
    model = Nemotron3UltraTinyProvider(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        seq_length=seq_length,
    )

    # Match Nemotron-H / Ultra recipe model-init settings.
    # Set as attributes to avoid constructor errors when fields don't exist
    # on TransformerConfig in the target container's Megatron-Core version.
    model.pipeline_dtype = torch.bfloat16
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = 1
    model.expert_tensor_parallel_size = 1
    model.pipeline_model_parallel_layout = None
    model.position_embedding_type = "none"
    model.activation_func = squared_relu
    model.masked_softmax_fusion = True
    model.apply_query_key_layer_scaling = False
    model.persist_layer_norm = True
    model.attention_softmax_in_fp32 = False
    model.first_last_layers_bf16 = True
    model.is_hybrid_model = True
    model.moe_aux_loss_coeff = 0.0001
    model.moe_router_score_function = "sigmoid"
    model.moe_router_enable_expert_bias = True
    model.moe_router_load_balancing_type = "seq_aux_loss"
    model.moe_router_dtype = "fp32"
    model.moe_grouped_gemm = True
    model.moe_permute_fusion = True
    model.apply_rope_fusion = False
    model.attention_backend = "fused"
    model.gradient_accumulation_fusion = True
    model.init_method_std = 0.014
    model.use_fused_weighted_squared_relu = True
    model.keep_mamba_stack_attention_linear_in_bf16 = True
    model.keep_mtp_spec_in_bf16 = True
    model.calculate_per_token_loss = True
    model.mtp_num_layers = 2
    model.mtp_loss_scaling_factor = 0.3
    model.mtp_use_repeated_layer = True
    model.moe_token_dispatcher_type = "alltoall"
    model.moe_shared_expert_overlap = False
    model.moe_flex_dispatcher_backend = "hybridep"
    model.transformer_impl = "transformer_engine"
    model.cross_entropy_fusion_impl = "te"
    model.use_te_rng_tracker = True
    model.cuda_graph_impl = "none"
    model.cuda_graph_scope = []
    model.enable_cuda_graph = False

    return model
