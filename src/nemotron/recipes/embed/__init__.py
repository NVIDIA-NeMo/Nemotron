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

"""Embed recipe for fine-tuning embedding models.

Workflow stages:
- stage0_sdg: Generate synthetic Q&A pairs from documents
- stage1_data_prep: Prepare training data (convert, mine, unroll)
- stage2_finetune: Fine-tune the embedding model
- stage3_eval: Evaluate base vs fine-tuned models
"""
