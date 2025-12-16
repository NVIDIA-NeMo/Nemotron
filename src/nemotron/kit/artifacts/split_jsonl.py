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

"""SplitJsonlDataArtifact - non-tokenized JSONL data for RL."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from nemotron.kit.artifacts.base import Artifact
from nemotron.kit.trackers import InputDatasetInfo


class SplitJsonlDataArtifact(Artifact):
    """Split JSONL data artifact (output of non-tokenized data_prep).

    Used for RL and other stages that output JSONL files without tokenization.
    The path points directly to the manifest.json file.

    Unlike DataBlendsArtifact, this does not track token counts since the
    data is not tokenized.

    Source URIs are tracked for W&B lineage:
    - source_datasets: Input datasets with metadata (or URIs for backwards compat)
    """

    total_sequences: Annotated[int, Field(ge=0, description="Total documents processed")]
    elapsed_sec: Annotated[
        float, Field(default=0.0, ge=0, description="Processing time in seconds")
    ]

    # Source datasets for lineage tracking
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]

    def _get_output_dir(self) -> Path:
        """SplitJsonlDataArtifact.path is manifest.json (a file), so use parent."""
        return self.path.parent

    def get_wandb_files(self) -> list[tuple[str, str]]:
        """Return manifest.json and metadata.json for upload."""
        files = []
        # SplitJsonlDataArtifact.path is the manifest.json file itself
        if self.path.exists():
            files.append((str(self.path), "manifest.json"))
        # metadata.json is in same directory
        metadata_path = self.path.parent / "metadata.json"
        if metadata_path.exists():
            files.append((str(metadata_path), "metadata.json"))
        return files

    def get_wandb_references(self) -> list[tuple[str, str]]:
        """Return reference to data directory on shared storage."""
        # Reference to parent directory containing JSONL files
        return [(f"file://{self.path.parent.resolve()}", "output")]

    def get_input_uris(self) -> list[str]:
        """Return URIs of input datasets for lineage."""
        uris = []
        for ds in self.source_datasets:
            if isinstance(ds, InputDatasetInfo):
                uris.append(ds.uri)
            else:
                uris.append(ds)
        return uris
