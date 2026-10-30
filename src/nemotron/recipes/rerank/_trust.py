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

"""Remote-code trust helpers for the rerank recipe."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

TRUSTED_REMOTE_CODE_PREFIXES = ("nvidia/",)


def _is_existing_local_path(model_ref: str) -> bool:
    """Return True when a model reference resolves to an existing local path."""
    return Path(model_ref).expanduser().exists()


def untrusted_remote_model_refs(model_refs: Iterable[str]) -> list[str]:
    """Return remote model refs that need explicit trust_remote_code opt-in."""
    untrusted = []
    for model_ref in model_refs:
        if _is_existing_local_path(model_ref):
            continue
        if model_ref.startswith(TRUSTED_REMOTE_CODE_PREFIXES):
            continue
        untrusted.append(model_ref)
    return untrusted


def validate_trust_remote_code(model_refs: Iterable[str], *, allow_untrusted_remote_code: bool) -> None:
    """Reject non-allowlisted remote refs when code loading is implicitly trusted."""
    untrusted = untrusted_remote_model_refs(model_refs)
    if allow_untrusted_remote_code or not untrusted:
        return
    refs = ", ".join(repr(ref) for ref in untrusted)
    raise ValueError(
        "trust_remote_code is enabled for rerank model loading; non-NVIDIA remote "
        f"model refs require allow_untrusted_remote_code=true. Ref(s): {refs}"
    )
