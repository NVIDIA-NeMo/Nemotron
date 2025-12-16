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

"""PretrainBlendsArtifact - pretrain data blends with bin/idx files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from nemotron.kit.artifacts.base import Artifact
from nemotron.kit.trackers import InputDatasetInfo


class PretrainBlendsArtifact(Artifact):
    """Pretrain data blends artifact (output of pretrain data_prep).

    The path points to the output directory containing bin/idx files.
    The blend_path points to the blend.json file within that directory.

    Source URIs are tracked for W&B lineage:
    - source_datasets: Input datasets with metadata (or URIs for backwards compat)
    - tokenizer_uri: URI of the tokenizer model (hf://models/...)
    """

    total_tokens: Annotated[int, Field(ge=0, description="Total tokens processed")]
    total_sequences: Annotated[int, Field(ge=0, description="Total documents processed")]
    elapsed_sec: Annotated[
        float, Field(default=0.0, ge=0, description="Processing time in seconds")
    ]

    # Sharding configuration
    num_shards: Annotated[int, Field(ge=1, description="Number of output shards")]

    # Path to blend.json for Megatron-Bridge per_split_data_args_path
    blend_path: Annotated[str | None, Field(default=None, description="Path to blend.json file")]

    # Per-split token counts (optional, populated in per-split mode)
    train_tokens: Annotated[
        int | None, Field(default=None, ge=0, description="Tokens in train split")
    ]
    valid_tokens: Annotated[
        int | None, Field(default=None, ge=0, description="Tokens in valid split")
    ]
    test_tokens: Annotated[
        int | None, Field(default=None, ge=0, description="Tokens in test split")
    ]

    # Source datasets for lineage tracking
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]
    tokenizer_uri: Annotated[str | None, Field(default=None, description="URI of tokenizer model")]

    def get_wandb_files(self) -> list[tuple[str, str]]:
        """Return blend.json and metadata.json for upload."""
        files = []
        # Add blend.json if it exists
        if self.blend_path and Path(self.blend_path).exists():
            files.append((self.blend_path, "blend.json"))
        elif (self.path / "blend.json").exists():
            files.append((str(self.path / "blend.json"), "blend.json"))
        # Add metadata.json
        metadata_path = self.path / "metadata.json"
        if metadata_path.exists():
            files.append((str(metadata_path), "metadata.json"))
        return files

    def get_wandb_references(self) -> list[tuple[str, str]]:
        """Return reference to data directory on shared storage."""
        # Reference to directory containing bin/idx files
        return [(f"file://{self.path.resolve()}", "output")]

    def get_input_uris(self) -> list[str]:
        """Return URIs of input datasets and tokenizer for lineage."""
        uris = []
        for ds in self.source_datasets:
            if isinstance(ds, InputDatasetInfo):
                uris.append(ds.uri)
            else:
                uris.append(ds)
        if self.tokenizer_uri:
            uris.append(self.tokenizer_uri)
        return uris
