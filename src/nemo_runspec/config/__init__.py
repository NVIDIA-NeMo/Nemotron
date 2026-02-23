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

"""Configuration loading, merging, and OmegaConf resolvers.

This package provides:
- Config loading pipeline (YAML, dotlist overrides, env profile merging)
- OmegaConf custom resolvers for artifact and git-mount resolution
"""

# Config loading and merging
from nemo_runspec.config.loader import (
    apply_dotlist_overrides,
    build_job_config,
    extract_train_config,
    find_config_file,
    generate_job_dir,
    load_config,
    parse_config,
    save_configs,
)

# OmegaConf resolvers
from nemo_runspec.config.resolvers import (
    ResolverMode,
    clear_artifact_cache,
    clear_git_mounts,
    get_git_mounts,
    register_auto_mount_resolver,
    register_resolvers,
    register_resolvers_from_config,
    resolve_artifact_pre_init,
)

__all__ = [
    # Loader
    "parse_config",
    "find_config_file",
    "load_config",
    "apply_dotlist_overrides",
    "build_job_config",
    "extract_train_config",
    "generate_job_dir",
    "save_configs",
    # Resolvers
    "ResolverMode",
    "register_resolvers",
    "register_resolvers_from_config",
    "register_auto_mount_resolver",
    "resolve_artifact_pre_init",
    "clear_artifact_cache",
    "get_git_mounts",
    "clear_git_mounts",
]
