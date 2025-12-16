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

"""
Artifact classes for nemotron.kit.

This module provides typed artifact classes for tracking data and model outputs.
Each artifact type defines its own W&B file/reference handling.
"""

from nemotron.kit.artifacts.base import Artifact, TrackingInfo
from nemotron.kit.artifacts.data_blends import DataBlendsArtifact
from nemotron.kit.artifacts.model import ModelArtifact
from nemotron.kit.artifacts.pretrain_blends import PretrainBlendsArtifact
from nemotron.kit.artifacts.pretrain_data import PretrainDataArtifact
from nemotron.kit.artifacts.sft_data import SFTDataArtifact
from nemotron.kit.artifacts.split_jsonl import SplitJsonlDataArtifact

__all__ = [
    # Base
    "Artifact",
    "TrackingInfo",
    # Data artifacts
    "DataBlendsArtifact",
    "PretrainBlendsArtifact",
    "PretrainDataArtifact",
    "SFTDataArtifact",
    "SplitJsonlDataArtifact",
    # Model artifacts
    "ModelArtifact",
]
