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

"""HuggingFace runtime environment utilities for data prep pipelines.

This module centralizes HF environment variable handling and RuntimeEnv
construction for Xenna pipeline stages.
"""

from __future__ import annotations

import os

from cosmos_xenna.ray_utils.runtime_envs import RuntimeEnv


def detect_hf_env_vars() -> dict[str, str]:
    """Detect HuggingFace environment variables from the current environment.

    Returns:
        Dictionary containing HF_HOME and/or HF_TOKEN if present in the environment.
        Empty dict if neither is set.
    """
    env_vars: dict[str, str] = {}
    if os.environ.get("HF_HOME"):
        env_vars["HF_HOME"] = os.environ["HF_HOME"]
    if os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return env_vars


def make_hf_runtime_env(
    *,
    base_env_vars: dict[str, str] | None = None,
    extra_env_vars: dict[str, str] | None = None,
) -> RuntimeEnv:
    """Create a RuntimeEnv with HuggingFace environment variables for worker processes.

    Args:
        base_env_vars: Base environment variables (typically from detect_hf_env_vars()).
            If None, will auto-detect from current environment.
        extra_env_vars: Additional environment variables to include. These take
            precedence over base_env_vars values.

    Returns:
        RuntimeEnv configured with HuggingFace environment variables.
    """
    # Auto-detect if not provided
    if base_env_vars is None:
        base_env_vars = detect_hf_env_vars()

    # Merge base and extra (extra takes precedence)
    merged: dict[str, str] = {}
    if base_env_vars:
        merged.update(base_env_vars)
    if extra_env_vars:
        merged.update(extra_env_vars)

    return RuntimeEnv(extra_env_vars=merged) if merged else RuntimeEnv()
