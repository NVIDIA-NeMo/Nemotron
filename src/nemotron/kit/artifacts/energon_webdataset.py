# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""EnergonWebDatasetArtifact — sharded WebDataset for Energon-backed training."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from nemotron.kit.artifacts.base import Artifact
from nemotron.kit.trackers import InputDatasetInfo


class EnergonWebDatasetArtifact(Artifact):
    """Energon WebDataset artifact (output of multimodal SFT data prep).

    The path points to the Energon dataset root, which contains tar shards at
    the top level plus a ``.nv-meta/`` directory with the dataset.yaml,
    sqlite index, split.yaml, and per-shard sample counts.

    Output structure:
        path/
          {split}-shard-NNNNNN.tar      # one per shard
          .nv-meta/
            dataset.yaml                  # sample-type + field map
            split.yaml                    # train/val/test shard assignment
            .info.yaml                    # per-shard sample counts
            index.sqlite                  # byte-offset random-access index
            index.uuid

    Source URIs are tracked for W&B lineage:
        - source_datasets: input data references (tar URI, QA annotations URL)
    """

    train_samples: Annotated[
        int, Field(default=0, ge=0, description="Sample count in the train split"),
    ]
    val_samples: Annotated[
        int, Field(default=0, ge=0, description="Sample count in the val split"),
    ]
    test_samples: Annotated[
        int, Field(default=0, ge=0, description="Sample count in the test split"),
    ]
    num_shards: Annotated[
        int, Field(default=0, ge=0, description="Total tar shards across all splits"),
    ]
    elapsed_sec: Annotated[
        float, Field(default=0.0, ge=0, description="Build time in seconds"),
    ]

    data_format: Annotated[
        str,
        Field(
            default="energon-webdataset",
            description="Output format identifier (always 'energon-webdataset')",
        ),
    ]

    source_datasets: Annotated[
        list[InputDatasetInfo | str],
        Field(default_factory=list, description="Input source URIs for lineage"),
    ]

    @property
    def total_samples(self) -> int:
        """Total samples across all splits."""
        return self.train_samples + self.val_samples + self.test_samples

    def get_wandb_files(self) -> list[tuple[str, str]]:
        """Upload .nv-meta files (small, useful for resolver lookups)."""
        files: list[tuple[str, str]] = []
        meta_dir = self.path / ".nv-meta"
        for filename in ("dataset.yaml", "split.yaml", ".info.yaml", "index.uuid"):
            f = meta_dir / filename
            if f.exists():
                files.append((str(f), filename))
        artifact_metadata = self.path / "metadata.json"
        if artifact_metadata.exists():
            files.append((str(artifact_metadata), "metadata.json"))
        return files

    def get_wandb_references(self) -> list[tuple[str, str]]:
        """Reference the dataset root on shared storage (tar shards stay on disk)."""
        return [(f"file://{self.path.resolve()}", "output")]

    def get_input_uris(self) -> list[str]:
        """Return input URIs for lineage tracking."""
        uris: list[str] = []
        for ds in self.source_datasets:
            if isinstance(ds, InputDatasetInfo):
                uris.append(ds.uri)
            else:
                uris.append(ds)
        return uris

    @classmethod
    def from_run(
        cls,
        *,
        dataset_path: Path,
        split_sample_counts: dict[str, int],
        num_shards: int,
        source_datasets: list[InputDatasetInfo | str] | None = None,
        elapsed_sec: float = 0.0,
        name: str | None = None,
    ) -> "EnergonWebDatasetArtifact":
        """Build an artifact from the per-split sample counts emitted by finalize."""
        artifact = cls(
            path=dataset_path.resolve(),
            train_samples=int(split_sample_counts.get("train", 0)),
            val_samples=int(split_sample_counts.get("val", 0)),
            test_samples=int(split_sample_counts.get("test", 0)),
            num_shards=int(num_shards),
            elapsed_sec=float(elapsed_sec),
            source_datasets=source_datasets or [],
        )
        if name:
            artifact.name = name
        return artifact
