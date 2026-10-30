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

"""Synthesize diverse seed queries from a chunked corpus.

Reads the on-disk chunk corpus directly (streaming), samples a bounded pool,
clusters it in embedding space, then generates end-user questions across topic
clusters x question kinds (multi-hop = related same-cluster chunks). Optionally
validates each query is answerable by the live retriever. Output feeds query_prep.

    corpus.jsonl -> reservoir_sample -> embed+cluster -> units(kind x chunks)
                 -> generate (LLM) -> validate -> queries.jsonl
"""

from .corpus import Chunk, count_chunks, reservoir_sample, stream_chunks
from .lancedb_source import read_lancedb, rows_to_pool
from .run import run_query_gen
from .sizing import plan_sizes

__all__ = ["Chunk", "stream_chunks", "reservoir_sample", "count_chunks",
           "read_lancedb", "rows_to_pool", "run_query_gen", "plan_sizes"]
