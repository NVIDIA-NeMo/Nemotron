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

"""Tier 1 static validation for the step index markdown."""

from pathlib import Path


def test_steps_index_exists(steps_root: Path) -> None:
    assert (steps_root / "STEPS.md").exists(), "src/nemotron/steps/STEPS.md does not exist"


def test_every_step_manifest_is_mentioned_in_steps_index(
    steps_root: Path,
    all_step_tomls: list[Path],
) -> None:
    steps_index_path = steps_root / "STEPS.md"
    assert steps_index_path.exists(), "src/nemotron/steps/STEPS.md does not exist"

    steps_index = steps_index_path.read_text(encoding="utf-8")

    for manifest_path in all_step_tomls:
        step_dir = manifest_path.parent.relative_to(steps_root).as_posix()
        assert step_dir in steps_index, (
            f"{manifest_path}: step directory {step_dir!r} is not mentioned in STEPS.md"
        )
