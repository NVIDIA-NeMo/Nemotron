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

"""SFTDataArtifact - packed SFT data for Megatron-Bridge."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from nemotron.kit.artifacts.base import Artifact
from nemotron.kit.trackers import InputDatasetInfo


class SFTDataArtifact(Artifact):
    """Packed SFT data artifact (output of SFT data_prep).

    Contains packed .npy files with tokenized and packed chat sequences.
    The path points to the output directory containing training_{pack_size}.npy, etc.

    Output files (Megatron-Bridge compatible):
    - training_{pack_size}.npy: Training data
    - validation_{pack_size}.npy: Validation data
    - test_{pack_size}.npy: Test data
    - {pack_size}_metadata.jsonl: Megatron-Bridge compatible packing metadata
    - metadata.json: Nemotron artifact metadata

    Source URIs are tracked for W&B lineage:
    - source_datasets: Input datasets with metadata (or URIs for backwards compat)
    - tokenizer_uri: URI of the tokenizer model (hf://models/...)
    """

    total_tokens: Annotated[int, Field(ge=0, description="Total tokens processed")]
    total_sequences: Annotated[int, Field(ge=0, description="Total sequences after packing")]
    elapsed_sec: Annotated[
        float, Field(default=0.0, ge=0, description="Processing time in seconds")
    ]

    # Packing configuration
    pack_size: Annotated[int, Field(ge=1, description="Maximum tokens per packed sequence")]

    # Explicit paths to split files (Megatron-Bridge compatible naming)
    training_path: Annotated[
        str | None, Field(default=None, description="Path to training_{pack_size}.npy")
    ]
    validation_path: Annotated[
        str | None, Field(default=None, description="Path to validation_{pack_size}.npy")
    ]
    test_path: Annotated[
        str | None, Field(default=None, description="Path to test_{pack_size}.npy")
    ]
    metadata_path: Annotated[
        str | None, Field(default=None, description="Path to {pack_size}_metadata.jsonl")
    ]

    # Source datasets for lineage tracking
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]
    tokenizer_uri: Annotated[str | None, Field(default=None, description="URI of tokenizer model")]

    def get_wandb_files(self) -> list[tuple[str, str]]:
        """Return metadata.json for upload (small file with artifact info)."""
        files = []
        metadata_path = self.path / "metadata.json"
        if metadata_path.exists():
            files.append((str(metadata_path), "metadata.json"))
        return files

    def get_wandb_references(self) -> list[tuple[str, str]]:
        """Return references to data files on shared storage.

        Data files (.npy) stay on shared storage and are not uploaded.
        Only metadata.json is uploaded for resolver field access.
        """
        refs = []
        # Add reference to the output directory containing .npy files
        refs.append((f"file://{self.path.resolve()}", "output"))
        return refs

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
