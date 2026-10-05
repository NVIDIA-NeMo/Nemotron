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

"""Work out a candidate's weight identity instead of asking an operator to type it.

Pinning weights is what makes a score reproducible, but the operator is rarely
the one who knows the pin. A registry already publishes the commit behind a tag;
a local checkout already contains the bytes; a hosted frontier route publishes
neither and never will. Asking a person to paste a value in all three cases
produces one correct answer, one tedious answer, and one invented answer, and the
invented one is the dangerous answer because nothing downstream can tell it from
a real pin.

So each case is resolved from the thing that actually knows it, and the case that
cannot be resolved is recorded as itself rather than filled in. Nothing here
derives a digest from a model name: a hash of ``gpt-5.6`` identifies the string
``gpt-5.6``, and reading it as weight identity would let a provider swap the
weights under a config that still claims to be reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import EvalConfigError
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    WEIGHT_MANIFEST_DIGEST_SCHEME,
    CandidateModelIdentity,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

# Registry lookups are pure metadata reads, so a caller can supply its own to
# keep a test offline.
RegistryResolver = Callable[[str, str, str | None], str]


class ModelIdentityResolutionError(ValueError):
    """Identity could not be resolved from the source that was supposed to know it."""


def _resolve_huggingface_revision(source: str, model: str, revision: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise ModelIdentityResolutionError(
            f"resolving {source} identities needs huggingface_hub; install it, or pass the commit directly"
        ) from exc
    try:
        info = HfApi().model_info(model, revision=revision)
    except Exception as exc:  # noqa: BLE001 - the client raises its own hierarchy
        raise ModelIdentityResolutionError(
            f"{source} could not resolve {model}@{revision or 'default'}: {type(exc).__name__}: {exc}"
        ) from exc
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha.strip():
        raise ModelIdentityResolutionError(
            f"{source} returned no commit for {model}@{revision or 'default'}, so nothing immutable was learned"
        )
    return sha.strip()


# Which registry each source is actually read from. A source is not a free label
# here: resolving ``modelscope`` against Hugging Face would answer with a commit
# from the wrong registry and then record it as ModelScope provenance, which is a
# false pin rather than a missing one. The client library is imported lazily
# inside each resolver, because it is an optional dependency of an evaluation
# that may never resolve anything.
_REGISTRY_RESOLVERS: Final[dict[str, RegistryResolver]] = {
    "huggingface": _resolve_huggingface_revision,
}


def resolve_registry_identity(
    *,
    source: str,
    model: str,
    revision: str | None = None,
    resolver: RegistryResolver | None = None,
) -> CandidateModelIdentity:
    """Turn a registry reference into the commit it currently points at.

    A moving reference is accepted as *input* on purpose: ``main`` is what an
    operator has, and resolving it once here is the whole point. What gets
    recorded is the commit it resolved to, so the config keeps meaning the same
    weights after the branch moves on.
    """
    lookup = resolver or _REGISTRY_RESOLVERS.get(source)
    if lookup is None:
        supported = ", ".join(sorted(_REGISTRY_RESOLVERS))
        raise ModelIdentityResolutionError(
            f"no registry client for source {source!r}; this build resolves {supported}. Record the commit "
            "that registry publishes directly, or resolve a provider-managed identity instead"
        )
    resolved = lookup(source, model, revision)
    return _identity(source=source, model=model, revision=resolved)


def digest_weight_directory(root: Path) -> str:
    """Digest a weight directory through a manifest of its files.

    Hashing a manifest rather than a concatenation keeps the digest stable
    against read order and makes a renamed file a different identity, which it
    is. Symlinks are refused rather than followed: a link can point outside the
    tree, and a digest that depends on where it pointed is not an identity.

    The result is labelled with its scheme instead of a bare ``sha256:``, because
    what it measures is wider than the weights: this manifest covers everything
    served alongside them, down to the file names. Two equal manifests are the
    same directory, but two unequal ones may be the same weights with a note
    added beside them, so the label is what lets the contamination gate decline
    that comparison instead of reading it as different weights and clearing a
    candidate it should not.
    """
    if not root.is_dir():
        raise ModelIdentityResolutionError(f"not a directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ModelIdentityResolutionError(
                f"refusing to digest a symlink, whose target is outside the identity: {path}"
            )
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not files:
        raise ModelIdentityResolutionError(f"no files to digest under {root}, so there are no weights to name")
    manifest = canonical_json({"scheme": WEIGHT_MANIFEST_DIGEST_SCHEME, "files": files})
    return f"{WEIGHT_MANIFEST_DIGEST_SCHEME}:{hashlib.sha256(manifest.encode('utf-8')).hexdigest()}"


def resolve_local_identity(*, source: str, model: str, weights_dir: Path) -> CandidateModelIdentity:
    """Name locally held weights by the bytes on disk."""
    return _identity(source=source, model=model, weights_digest=digest_weight_directory(weights_dir))


def provider_managed_identity(*, source: str, model: str) -> CandidateModelIdentity:
    """Record a hosted route whose provider pins nothing.

    This is a resolution result, not a failure to resolve. The run may proceed
    and the score is real; what it may not do is claim the weights will still be
    these weights, and the config carries that limit rather than an operator's
    memory of it.

    ``source`` and ``model`` must be the candidate's own ``provider`` and
    ``model``: with no pin, the route is the only thing that identifies this
    candidate, and the config contract refuses an unpinned identity that names
    anything else.
    """
    return _identity(source=source, model=model)


def _identity(
    *,
    source: str,
    model: str,
    revision: str | None = None,
    weights_digest: str | None = None,
) -> CandidateModelIdentity:
    # Nothing here is trusted to have produced a usable identity: a registry can
    # answer a branch name for a branch name, and the value has to survive the
    # same contract the config would apply to it hours later. EvalConfigError is
    # caught alongside ValueError because the contract deliberately raises
    # outside the ValueError hierarchy to keep its type through pydantic.
    try:
        return CandidateModelIdentity(
            source=source,
            model=model,
            revision=revision,
            weights_digest=weights_digest,
        )
    except (ValueError, EvalConfigError) as exc:
        raise ModelIdentityResolutionError(f"resolved an identity the eval contract refuses: {exc}") from exc


def identity_document(identity: CandidateModelIdentity) -> dict[str, Any]:
    """Render one resolved identity as the config block plus what it is worth.

    ``identity_publication_gate`` reports the one gate this identity decides.
    Publication also depends on scoring settings, the contamination policy, the
    artifacts a run writes, and whether the source run was gold-eligible, none of
    which are visible from here — so a satisfied gate is a precondition met, not
    permission to publish.
    """
    return {
        "model_identity": {
            "source": identity.source,
            "model": identity.model,
            "revision": identity.revision,
            "weights_digest": identity.weights_digest,
        },
        "assurance": identity.assurance,
        "canonical_id": identity.canonical_id,
        "identity_publication_gate": ("satisfied" if identity.assurance == "weights_pinned" else "blocked"),
    }
