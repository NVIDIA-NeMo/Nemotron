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

"""Embed artifacts - embedding data prep and fine-tuned model artifacts."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from nemotron.kit.artifacts.base import Artifact


class EmbedDataArtifact(Artifact):
    """Prepared embedding training data (output of stage1_data_prep).

    The path points to the output directory containing training data,
    eval BEIR data, and mined hard negatives.
    """

    training_examples: Annotated[int, Field(ge=0, description="Number of training examples")]
    eval_queries: Annotated[int, Field(ge=0, description="Number of evaluation queries")]
    base_model: Annotated[str, Field(description="Model used for hard negative mining")]
    quality_threshold: Annotated[float, Field(description="Minimum quality score")]
    hard_negatives_per_query: Annotated[int, Field(ge=0, description="Negatives mined per query")]


class EmbedModelArtifact(Artifact):
    """Fine-tuned embedding model checkpoint (output of stage2_finetune).

    The path points to the final model checkpoint directory.
    """

    base_model: Annotated[str, Field(description="Base model that was fine-tuned")]
    training_examples: Annotated[int, Field(ge=0, description="Number of training examples")]
    num_epochs: Annotated[int, Field(ge=0, description="Training epochs")]
    global_batch_size: Annotated[int, Field(ge=0, description="Global batch size")]
    learning_rate: Annotated[float, Field(description="Learning rate")]
    temperature: Annotated[float, Field(description="Contrastive loss temperature")]
