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

"""Input-side stages: dedup -> cluster -> sample the incoming queries.

The query source hands us a JSONL list of queries. These offline stages turn that
raw list into a diverse, deduplicated seed set for the conversation engine.
Trajectory shape is sampled per-row by the planner (no separate classification pass).
"""

from .dedup import dedup, normalize
from .cluster import cluster_queries
from .sample import sample

__all__ = ["dedup", "normalize", "cluster_queries", "sample"]
