#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Source this file inside the NeMo RL container. Set any variable before
# sourcing it to override the portable /shared default.

export NEMO_RL="${NEMO_RL:-/shared/code/RL}"
export NEMOTRON_REPO="${NEMOTRON_REPO:-/shared/code/Nemotron}"
export MODEL_DIR="${MODEL_DIR:-/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026}"
export RUN_DIR="${RUN_DIR:-/shared/runs/super35-star-count}"
export CACHE_DIR="${CACHE_DIR:-${RUN_DIR}/cache}"
export RECIPE="${RECIPE:-${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym/super_vl_3_5_star_count_megatron.yaml}"

mkdir -p \
  "${RUN_DIR}/logs" \
  "${CACHE_DIR}/hf_modules" \
  "${CACHE_DIR}/hf_config_locks" \
  "${CACHE_DIR}/megatron_ckpt" \
  "${CACHE_DIR}/vllm"

export HF_MODULES_CACHE="${CACHE_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${CACHE_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

# Make the runtime choices from the reference run explicit. NeMo RL defaults
# to vLLM V1 on this branch, and the recipe also sets FLASH_ATTN explicitly.
export NRL_VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
