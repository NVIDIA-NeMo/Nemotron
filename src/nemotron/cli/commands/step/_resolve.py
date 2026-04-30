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

"""Shared step-id → StepInfo resolution helpers."""
from __future__ import annotations

import typer

from nemotron.steps.index import StepInfo, discover_steps


def resolve_step(step_id: str) -> StepInfo:
    """Find a step by id, with a helpful error if missing.

    Accepts the canonical id (peft/automodel) or the directory tail (automodel)
    when unambiguous.
    """
    steps = discover_steps()
    by_id = {s.id: s for s in steps}
    if step_id in by_id:
        return by_id[step_id]

    # Allow short form when unambiguous.
    tail_matches = [s for s in steps if s.path.name == step_id]
    if len(tail_matches) == 1:
        return tail_matches[0]

    available = ", ".join(sorted(by_id))
    typer.echo(f"Unknown step id: {step_id}", err=True)
    typer.echo(f"Available: {available}", err=True)
    raise typer.Exit(1)
