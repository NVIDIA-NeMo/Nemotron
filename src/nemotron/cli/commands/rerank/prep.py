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

"""Data preparation command for rerank recipe.

Delegates to the embed data prep implementation — the same training data
format (query + positive/negative documents) is used for both embedding
and reranking models.
"""

from nemotron.cli.commands.embed.prep import (
    META,
    _execute_prep,
    prep,
)

__all__ = ["META", "_execute_prep", "prep"]
