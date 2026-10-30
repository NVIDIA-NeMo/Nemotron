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

"""Who a model is, and who touched the benchmark (schema 1.0).

A contamination gate asks one question: did the model under evaluation already
see these tasks while they were being built? Answering it requires comparing two
identities that were written by different parts of the system, for different
reasons, and that do not share a vocabulary.

*Generation* records the models it drove in ``run_manifest.json`` under
``models.<role>``: a serving route (``provider`` plus ``model``), an opaque
operator-chosen ``canonical_id``, and whatever weight identity the operator
supplied — often just the model name, because a generation run only needs to
name what it called, not prove which bytes answered.

*Evaluation* records candidates under the evaluation config contract, which is
stricter where it can be: a candidate pins an immutable revision or a weights
digest, because a score is claimed about specific weights. A hosted model whose
provider publishes neither is recorded as ``provider_managed`` instead, which is
weaker evidence here and costs that run its right to publish.

Neither identity is a superset of the other, so :func:`compare_model_identity`
weighs both axes and returns one of three verdicts. The asymmetry that matters
is which way an ambiguous comparison falls: here it falls to ``unknown``, never
to ``different``. Config validation compares candidates *to each other* and keeps identifiers
case-sensitive, because collapsing two case-variants would hide a real
difference between two candidates. This module compares a candidate to a model
that already read the benchmark, where the dangerous mistake is the opposite
one, so every string comparison here is case-insensitive and model names are
matched on a normalized form. An ``unknown`` verdict costs an operator a pinned
identity; a wrong ``different`` verdict costs the benchmark its validity.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    BROAD_SCOPE_DIGEST_SCHEMES,
    EvalCandidate,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

MODEL_IDENTITY_CONTRACT_VERSION: Final = "1.0"

# Every way a model can have read benchmark content while it was being built.
# ``translator`` is not a generation role: it belongs to the translation run that
# derived a localized benchmark from a published one, and it reads every row.
ExposureRole = Literal["profile", "paraphrase", "surface_judge", "translator"]

# Which published rows a role's exposure actually covers. Scope is derived from
# the rows themselves wherever the schema records it, and falls back to "all"
# only for roles that read the whole surface by construction.
ExposureScope = Literal[
    "all_published_rows",
    "profile_shaped_rows",
    "paraphrased_rows",
    "translated_rows",
]

# The verdict of comparing two model identities. ``unknown`` is a first-class
# outcome, not an error: it is what the pipeline can honestly say when one side
# pinned less than the other, and it is the verdict that blocks publication.
IdentityVerdict = Literal["match", "different", "unknown"]

_NON_ALPHANUMERIC: Final = re.compile(r"[^0-9a-z]+")


def _sha256_json(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()}"


def normalize_model_token(value: str | None) -> str | None:
    """Reduce a model name to a form two registries can be compared on.

    Registry prefix and punctuation are dropped, so ``meta/Llama-3.3-70B`` and
    ``meta-llama/llama_3.3_70b`` reduce to the same token. This deliberately
    over-matches: an organization's fine-tune of a base model normalizes to the
    base model's token, which turns a comparison that would have said
    "different" into "unknown". That direction is the safe one — it asks the
    operator for a pinned identity instead of clearing a candidate that may have
    written the rows it is being scored on.
    """
    if not value:
        return None
    tail = value.strip().rsplit("/", 1)[-1]
    token = _NON_ALPHANUMERIC.sub("", tail.casefold())
    return token or None


def _folded(value: str | None) -> str | None:
    return value.strip().casefold() if value and value.strip() else None


def _equal(left: str | None, right: str | None) -> bool:
    """True when both sides are present and equal, ignoring case."""
    folded_left, folded_right = _folded(left), _folded(right)
    return folded_left is not None and folded_left == folded_right


def _conflicts(left: str | None, right: str | None) -> bool:
    """True when both sides are present and differ; absence never conflicts."""
    folded_left, folded_right = _folded(left), _folded(right)
    return folded_left is not None and folded_right is not None and folded_left != folded_right


def _digest_parts(value: str) -> tuple[str | None, str]:
    """Split ``algorithm:hex`` apart, tolerating a digest recorded as bare hex.

    The candidate side is schema-checked, but only to one of the schemes the eval
    contract knows (``sha256`` over weight bytes, ``bfcl-weight-manifest-v1``
    over a manifest of weight files). A generation manifest carries whatever the
    pack config wrote, so the same bytes can arrive as ``sha256:AB…``, ``AB…``, or
    under another algorithm entirely.
    """
    text = value.strip().casefold()
    algorithm, _, body = text.rpartition(":")
    return (algorithm or None, body)


def _compare_digests(left: str, right: str) -> IdentityVerdict | None:
    """Compare two weight digests, or decline to when they are incomparable.

    Equal bodies are the same digest however each side spelled the algorithm.
    Unequal bodies only prove two different weights when the two digests measure
    the same thing: a different algorithm, or the same algorithm over a different
    byte scope, produces unequal digests for identical weights. A scheme that
    covers more than the weights is unequal for reasons that are not about the
    weights at all, so its inequality is declined even against itself. Declining
    (``None``) sends the comparison down to the name evidence, where it can still
    end in ``unknown`` — never in a clearance this could not support.
    """
    left_algorithm, left_body = _digest_parts(left)
    right_algorithm, right_body = _digest_parts(right)
    if left_body == right_body:
        return "match"
    if _conflicts(left_algorithm, right_algorithm) or len(left_body) != len(right_body):
        return None
    if {left_algorithm, right_algorithm} & BROAD_SCOPE_DIGEST_SCHEMES:
        return None
    return "different"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelIdentityClaim(_Frozen):
    """One model, named on every axis the artifact that recorded it carried.

    All fields are optional because the two writers pin different things, and a
    claim that is missing an axis must degrade a comparison to ``unknown``
    rather than fail to be constructed: an exposure this pipeline cannot fully
    identify is exactly the case the gate exists to catch.
    """

    provider: StrictStr | None = None
    served_model: StrictStr | None = None
    weight_source: StrictStr | None = None
    weight_model: StrictStr | None = None
    revision: StrictStr | None = None
    weights_digest: StrictStr | None = None
    # A canonical id, from whichever side recorded it. The two sides do not share
    # a namespace: generation carries the operator's own opaque string, while a
    # candidate's is derived as ``source:model@reference``. So the label is only
    # ever an equality signal, never parsed, and it only settles a comparison
    # when an operator deliberately made the two agree.
    label: StrictStr | None = None

    @property
    def names_a_model(self) -> bool:
        """Whether this claim identifies anything at all."""
        return any((self.weights_digest, self.weight_model, self.served_model, self.label))

    @property
    def model_tokens(self) -> frozenset[str]:
        """Normalized model names this claim could be known by."""
        tokens = {
            normalize_model_token(self.weight_model),
            normalize_model_token(self.served_model),
        }
        return frozenset(token for token in tokens if token)

    @property
    def display_name(self) -> str:
        """A stable, non-secret name for reports and error messages."""
        if self.label:
            return self.label
        if self.weight_model:
            reference = self.weights_digest or self.revision
            qualified = f"{self.weight_source}:{self.weight_model}" if self.weight_source else self.weight_model
            return f"{qualified}@{reference}" if reference else qualified
        if self.served_model:
            return f"{self.provider}/{self.served_model}" if self.provider else self.served_model
        return "unidentified"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "served_model": self.served_model,
            "weight_source": self.weight_source,
            "weight_model": self.weight_model,
            "revision": self.revision,
            "weights_digest": self.weights_digest,
            "label": self.label,
        }


def candidate_identity_claim(candidate: EvalCandidate) -> ModelIdentityClaim:
    """Read an evaluation candidate as an identity claim.

    Both axes are kept: the serving route is what a generation run would have
    called, and the weight identity is what the score is claimed about. A
    candidate that was served through the same route as the paraphraser is
    contaminated even if the two configs describe their weights differently.
    """
    identity = candidate.model_identity
    return ModelIdentityClaim(
        provider=candidate.provider,
        served_model=candidate.model,
        weight_source=identity.source,
        weight_model=identity.model,
        revision=identity.revision,
        weights_digest=identity.weights_digest,
        label=identity.canonical_id,
    )


def compare_model_identity(
    left: ModelIdentityClaim | None,
    right: ModelIdentityClaim | None,
) -> IdentityVerdict:
    """Decide whether two claims name the same weights.

    Evidence is weighed strongest first. A weights digest names bytes, so two
    digests that measure the same thing settle the question in either direction;
    two that do not are set aside rather than read as a difference. Below that,
    only agreement can be positive: an operator label or a full serving route
    that match are taken as the same model, while names that merely fail to
    match are not taken as different unless both sides named a model and no
    normalized name is shared.
    """
    if left is None or right is None or not left.names_a_model or not right.names_a_model:
        return "unknown"
    if left.weights_digest and right.weights_digest:
        if (verdict := _compare_digests(left.weights_digest, right.weights_digest)) is not None:
            return verdict
    if _equal(left.label, right.label):
        return "match"
    if _equal(left.served_model, right.served_model) and not _conflicts(left.provider, right.provider):
        return "match"
    if left.model_tokens & right.model_tokens:
        if _conflicts(left.weight_source, right.weight_source):
            # The same model name from two registries. One of them is a mirror,
            # a local copy, or the same registry spelled differently, and this
            # module cannot tell which — a generation manifest names the weight
            # source in whatever words the pack config used.
            return "unknown"
        if left.revision and right.revision:
            return "match" if _equal(left.revision, right.revision) else "different"
        # The same model name under two different pins, or under none: this is
        # the case the operator has to resolve, not one the pipeline may clear.
        return "unknown"
    if left.model_tokens and right.model_tokens:
        return "different"
    return "unknown"


class VerifiedModelExposure(_Frozen):
    """A model that read benchmark content, and the rows it read.

    An exposure is only recorded when it covers at least one published row: a
    role that was configured but shaped nothing cannot have leaked anything, and
    recording it would let ``fail_run`` abort an evaluation over a model that
    never touched the benchmark being scored.

    ``identity`` is ``None`` when the artifact that recorded the exposure did not
    say which model it was. That is not an error to be swallowed — it makes every
    comparison against it ``unknown``, which is what blocks publication.
    """

    schema_version: Literal["1.0"] = MODEL_IDENTITY_CONTRACT_VERSION
    role: ExposureRole
    scope: ExposureScope
    identity: ModelIdentityClaim | None
    task_ids: tuple[StrictStr, ...]
    evidence: StrictStr

    @model_validator(mode="after")
    def _covers_rows(self) -> VerifiedModelExposure:
        if not self.task_ids:
            raise ValueError("an exposure that covers no published row is not recorded")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("exposure task ids must be unique")
        return self

    @property
    def identified(self) -> bool:
        return self.identity is not None and self.identity.names_a_model

    @property
    def task_ids_hash(self) -> str:
        return _sha256_json(list(self.task_ids))

    @property
    def display_name(self) -> str:
        return self.identity.display_name if self.identity is not None else "undeclared"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "scope": self.scope,
            "identity": self.identity.semantic_payload() if self.identity is not None else None,
            "identified": self.identified,
            "task_count": len(self.task_ids),
            "task_ids_hash": self.task_ids_hash,
            "evidence": self.evidence,
        }

    def as_document(self) -> dict[str, Any]:
        return {
            **self.semantic_payload(),
            "model": self.display_name,
            "task_ids": list(self.task_ids),
        }
