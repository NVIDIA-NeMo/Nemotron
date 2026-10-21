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

"""long_context_chat_sdg — the multi-turn retrieval-grounded conversation-generation engine.

Given a chunked corpus or JSONL query list, this package synthesizes and prepares
diverse seeds, then generates and evaluates multi-turn, multi-step tool-calling
conversations for SFT. Production retrieval is supplied by an external HTTP service;
explicit demo runs can use labeled LLM-simulated evidence.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
