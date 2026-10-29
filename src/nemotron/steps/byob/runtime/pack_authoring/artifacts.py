# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic, byte-deterministic writes for every artifact either authoring phase emits.

Two properties matter to a reviewer. Rerunning against unchanged inputs has to produce
identical bytes, otherwise a diff cannot show what actually changed. And a run that dies
mid-write must not leave a half-written pack file that looks reviewable.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json


def sha256_text(text: str) -> str:
    """Digest exactly the bytes that were written."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def sha256_json(value: Any) -> str:
    """Digest a document through its canonical form, so key order cannot change it."""
    return sha256_text(canonical_json(value))


def write_text_atomic(text: str, path: Path) -> Path:
    """Replace ``path`` with ``text`` in one step, or leave it untouched."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_canonical_json(document: Any, path: Path) -> Path:
    """Write one JSON document in the canonical form its digest was taken over."""
    return write_text_atomic(canonical_json(document) + "\n", path)
