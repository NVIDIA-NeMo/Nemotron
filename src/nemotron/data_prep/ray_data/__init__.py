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

"""Ray Data shard-task executor for data preparation.

This module provides Ray Data-based execution for shard processing, using
ActorPoolStrategy for actor lifecycle management and resource accounting.

Key components:
- ShardTask: Represents a single shard to process
- execute_shard_tasks: Main executor using Ray Data map_batches
- BinIdxShardTaskUDF: UDF for binidx (Megatron) format processing
"""

from nemotron.data_prep.ray_data.binidx_udf import BinIdxShardTaskUDF
from nemotron.data_prep.ray_data.executor import RayDataExecConfig, execute_shard_tasks
from nemotron.data_prep.ray_data.tasks import ShardTask

__all__ = [
    "ShardTask",
    "execute_shard_tasks",
    "RayDataExecConfig",
    "BinIdxShardTaskUDF",
]
