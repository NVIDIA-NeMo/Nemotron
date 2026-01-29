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

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from nemotron.kit.artifacts.base import Artifact
from nemotron.kit.trackers import InputDatasetInfo, tokenizer_to_uri

if TYPE_CHECKING:
    from nemotron.data_prep.blend import DataBlend
    from nemotron.data_prep.config import FormatResult


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
        str | None, Field(default=None, description="Path to training_{pack_size}.npy (legacy)")
    ]
    validation_path: Annotated[
        str | None, Field(default=None, description="Path to validation_{pack_size}.npy (legacy)")
    ]
    test_path: Annotated[
        str | None, Field(default=None, description="Path to test_{pack_size}.npy (legacy)")
    ]
    metadata_path: Annotated[
        str | None, Field(default=None, description="Path to {pack_size}_metadata.jsonl (legacy)")
    ]

    # Xenna-native output (packed Parquet shards + blend.json)
    blend_path: Annotated[
        str | None, Field(default=None, description="Path to blend.json (xenna-native output)")
    ]
    num_shards: Annotated[
        int | None, Field(default=None, description="Number of output shards (xenna-native)")
    ]
    data_format: Annotated[
        str, Field(default="packed_sft_parquet", description="Output format: 'packed_sft_parquet'")
    ]

    # Source datasets for lineage tracking
    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input datasets with metadata"),
    ]
    tokenizer_uri: Annotated[str | None, Field(default=None, description="URI of tokenizer model")]

    def get_wandb_files(self) -> list[tuple[str, str]]:
        """Return metadata files for upload (small files with artifact info)."""
        files = []
        metadata_path = self.path / "metadata.json"
        if metadata_path.exists():
            files.append((str(metadata_path), "metadata.json"))
        # Include blend.json for xenna-native output
        if self.blend_path:
            from pathlib import Path
            blend_file = Path(self.blend_path)
            if blend_file.exists():
                files.append((str(blend_file), "blend.json"))
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

    @classmethod
    def from_result(
        cls,
        format_result: "FormatResult",
        blend: "DataBlend",
        tokenizer_model: str,
        blend_json_path: str | Path,
        pack_size: int,
        *,
        messages_field_default: str = "messages",
        elapsed_sec: float = 0.0,
        name: str | None = None,
    ) -> "SFTDataArtifact":
        """Create artifact from xenna-native pipeline format result.

        This is a convenience constructor that builds the source_datasets
        and tokenizer_uri from the blend and tokenizer model.

        Args:
            format_result: Result from run_sft_pipeline
            blend: Input data blend
            tokenizer_model: HuggingFace model name (e.g., "nvidia/...")
            blend_json_path: Path to the blend.json file
            pack_size: Maximum tokens per packed sequence
            messages_field_default: Default messages field name for lineage
            elapsed_sec: Processing time in seconds
            name: Optional artifact name

        Returns:
            SFTDataArtifact ready to save
        """
        source_datasets = [
            InputDatasetInfo(
                uri=d.path,
                name=d.name,
                weight=d.weight,
                split=d.split,
                subset=d.subset,
                text_field=d.text_field or messages_field_default,
            )
            for d in blend.datasets
        ]

        # Point artifact path to splits directory for cleaner consumer config
        # (consumers can use ${art:data,path}/train/ directly)
        splits_dir = format_result.output_dir / "splits"
        artifact = cls(
            path=splits_dir.resolve(),
            total_tokens=format_result.total_tokens,
            total_sequences=format_result.total_sequences,
            elapsed_sec=elapsed_sec,
            pack_size=pack_size,
            source_datasets=source_datasets,
            tokenizer_uri=tokenizer_to_uri(tokenizer_model),
            # Xenna-native output fields
            blend_path=str(blend_json_path),
            num_shards=format_result.num_shards,
            data_format="packed_sft_parquet",
            # Legacy fields set to None (not used in xenna mode)
            training_path=None,
            validation_path=None,
            test_path=None,
            metadata_path=None,
        )
        if name:
            artifact.name = name
        return artifact
