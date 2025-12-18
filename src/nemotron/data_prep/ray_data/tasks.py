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

"""ShardTask model for Ray Data shard-task execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ShardTask:
    """Represents a single shard to process.

    This is the unit of scheduling in Ray Data. Each task contains all
    information needed to process one shard independently.

    Attributes:
        dataset_name: Name of the dataset (for tracking/logging)
        plan_hash: Hash identifying the shard plan (for receipt)
        shard_index: Index of this shard within the plan
        assignment_json: ShardAssignment serialized as JSON string
            (JSON serialization ensures stable Arrow encoding)
        output_dir: Directory for .bin/.idx files
        receipts_dir: Directory for receipt JSON files
        fs_protocol: Filesystem protocol for OUTPUT (file, s3, gcs)
            Note: Input files use their own paths and may have different protocols
        kind: Output format (binidx for v1)
        text_field: Name of text column in input files
    """

    # Identity
    dataset_name: str
    plan_hash: str
    shard_index: int

    # Execution inputs
    # JSON-serialized for stable Arrow encoding across Ray versions
    assignment_json: str

    # Output locations (same as legacy layout)
    output_dir: str
    receipts_dir: str
    fs_protocol: str  # For OUTPUT only; inputs use per-file protocols

    # Format routing (v1 focuses on binidx)
    kind: Literal["binidx"] = "binidx"
    text_field: str = "text"

    @classmethod
    def from_assignment(
        cls,
        assignment: dict[str, Any],
        **kwargs: Any,
    ) -> ShardTask:
        """Create ShardTask from assignment dict, serializing to JSON.

        Args:
            assignment: ShardAssignment as dict (shard_index, files, total_bytes)
            **kwargs: Other ShardTask fields

        Returns:
            ShardTask with JSON-serialized assignment
        """
        return cls(
            assignment_json=json.dumps(assignment),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for Ray Data serialization.

        Returns:
            Dict with all fields suitable for ray.data.from_items()
        """
        return {
            "dataset_name": self.dataset_name,
            "plan_hash": self.plan_hash,
            "shard_index": self.shard_index,
            "assignment_json": self.assignment_json,  # Already JSON string
            "output_dir": self.output_dir,
            "receipts_dir": self.receipts_dir,
            "fs_protocol": self.fs_protocol,
            "kind": self.kind,
            "text_field": self.text_field,
        }

    def get_assignment(self) -> dict[str, Any]:
        """Deserialize assignment from JSON.

        Returns:
            ShardAssignment as dict
        """
        return json.loads(self.assignment_json)
