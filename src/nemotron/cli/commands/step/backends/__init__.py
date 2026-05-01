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

"""Backend protocol + registry for ``nemotron step run``.

A ``Backend`` knows how to take a parsed step (script + runspec + rendered
config + env profile) and submit it for execution. ``step run`` selects a
backend by name from the env profile's ``executor`` field and calls
:meth:`Backend.submit`. Adding a new backend is one new file under this
package — ``step run`` itself does not change.
"""
from __future__ import annotations

from nemotron.cli.commands.step.backends.base import Backend, JobContext
from nemotron.cli.commands.step.backends.cloud import CloudBackend
from nemotron.cli.commands.step.backends.local import LocalBackend
from nemotron.cli.commands.step.backends.registry import get_backend, register

# Built-in backends are registered here so a fresh import sees them all.
register("local", LocalBackend)
register("slurm", "nemotron.cli.commands.step.backends.slurm:SlurmBackend")
register("lepton", CloudBackend)
register("dgxcloud", CloudBackend)

__all__ = ["Backend", "CloudBackend", "JobContext", "LocalBackend", "get_backend", "register"]
