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

"""Rerank artifacts - fine-tuned reranking model artifacts."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from nemotron.kit.artifacts.base import Artifact


class RerankModelArtifact(Artifact):
    """Fine-tuned cross-encoder reranking model checkpoint (output of stage0_finetune).

    The path points to the final model checkpoint directory.
    """

    base_model: Annotated[str, Field(description="Base model that was fine-tuned")]
    training_examples: Annotated[int, Field(ge=0, description="Number of training examples")]
    num_epochs: Annotated[int, Field(ge=0, description="Training epochs")]
    global_batch_size: Annotated[int, Field(ge=0, description="Global batch size")]
    learning_rate: Annotated[float, Field(description="Learning rate")]
    num_labels: Annotated[int, Field(ge=1, description="Number of classification labels")]
