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

"""Versioned contract for ``eval_config.yaml`` (schemas 1.1 and 1.2).

An eval config decides what a score *means*: which benchmark rows are answered,
which model answered them, how a tool call is compared against the gold call, and
whether the result may be published. Every one of those is therefore pinned here
rather than defaulted at runtime, and the whole resolved config collapses into a
single :attr:`BfclEvalConfig.eval_config_hash` that an eval manifest can carry.

Three properties drive the shape of these models.

*Strictness.* Pydantic's default coercion turns ``"false"`` into ``True`` and
``"0.7"`` into a float. In a config that gates publication, a quoted boolean that
silently flips ``allow_llm_repair`` on is the difference between a benchmark and a
demo, so every scalar is a strict type and every enum is a ``Literal``.

*Immutable candidate identity.* A serving route (``provider``/``model``) says
where the request went, not which weights answered it. Scores are only comparable
when the weights are named by something that cannot move, so a candidate pins an
immutable ``revision`` or a ``weights_digest``, and branch names such as ``main``
are refused. A hosted route that publishes neither may leave both null under
schema 1.2 and is recorded as ``provider_managed``: it names the route it was
served by, and it cannot publish.

*Path-free semantics.* The hash covers *what the run evaluates*, never *where the
files live*: referenced files enter as content hashes and ``outputs.output_dir``
does not enter at all. Moving a checkout must not fork the identity of an
evaluation, while editing the scoring contract must.

Credentials live in the environment. The config names the variable
(``api_key_env``); it never holds a value, and nothing here is serialized in a
way that could put one in a manifest or a log.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Final, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import (
    CandidateIdentityError,
    EvalConfigSchemaError,
    MutableCandidateRevisionError,
    PublicationPolicyError,
    UnsupportedEvalModeError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.held_out_eval import (
    HeldOutEvalConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    ContentHash,
    FrozenDict,
    NonNegativeInt,
    PositiveInt,
    freeze_json,
    thaw_json,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

# The version a config written today declares. 1.2 added two things a 1.1 reader
# would refuse: a candidate that pins nothing, and a weights digest that names its
# scheme. Both are widenings, so every 1.1 config still loads and still means
# exactly what it meant — which is why the declared version is carried through to
# the resolved config and its hash rather than being restamped as the current one.
EVAL_CONFIG_SCHEMA_VERSION: Final = "1.2"
EVAL_CONFIG_SCHEMA_VERSIONS: Final = ("1.1", "1.2")
EvalConfigSchemaVersion = Literal["1.1", "1.2"]

# Canonical mode order. ``trace`` scores the calls a model proposed against the
# gold trace; ``executable`` additionally replays them against the oracle. The
# order is fixed so two configs that request the same work hash the same.
EVAL_MODES: Final = ("trace", "executable", "held_out_eval")
EvalMode = Literal["trace", "executable", "held_out_eval"]

# Refs that name "whatever is current" rather than a fixed set of weights. A
# score taken against one of these cannot be reproduced next week, so a candidate
# may not pin one. Branch refs are rejected by prefix as well, since any branch is
# by definition a moving pointer.
MUTABLE_REVISIONS: Final = frozenset(
    {
        "main",
        "master",
        "head",
        "latest",
        "dev",
        "devel",
        "develop",
        "trunk",
        "stable",
        "nightly",
        "edge",
        "tip",
        "current",
        "default",
        "release",
        "prod",
        "production",
    }
)
_MUTABLE_REF_PREFIXES: Final = ("refs/heads/", "refs/remotes/")
_IMMUTABLE_REVISION_PATTERN: Final = re.compile(r"^[0-9a-fA-F]{40,64}$")

# How much a candidate's identity can promise. ``weights_pinned`` names bytes or
# the immutable commit they came from, so the same config scores the same weights
# next year. ``provider_managed`` names only a route a provider may re-point
# underneath it, which several hosted frontier models are the honest case for:
# they publish no commit and no digest, so demanding one buys a value nobody can
# verify. The level is derived from the evidence rather than declared, because an
# operator claim about assurance would be the one part of the identity that is
# not checkable.
IdentityAssurance = Literal["weights_pinned", "provider_managed"]

# What a canonical identity says where a pin would go. It reads as a word rather
# than a hash so that no consumer can mistake an unpinned candidate for a pinned
# one, and so two unpinned candidates on the same route collide the way two
# indistinguishable candidates should.
UNPINNED_REFERENCE: Final = "provider_managed"

# A digest over a manifest of weight files rather than over weight bytes. The two
# schemes disagree for identical weights, and the contamination gate reads two
# unequal digests of the same algorithm and length as *different weights*. So the
# scheme is part of the value: a manifest digest compared against a raw one has
# conflicting algorithms, which sends that comparison to "unknown" — the operator
# is asked to pin comparably instead of a candidate being cleared by an artifact
# of how someone hashed it. What the scheme measures is recorded separately, in
# ``BROAD_SCOPE_DIGEST_SCHEMES``.
WEIGHT_MANIFEST_DIGEST_SCHEME: Final = "bfcl-weight-manifest-v1"
WEIGHTS_DIGEST_SCHEMES: Final = ("sha256", WEIGHT_MANIFEST_DIGEST_SCHEME)

# Schemes whose byte scope is wider than the weights, so two unequal digests do
# not establish two different sets of weights. The manifest scheme covers every
# file in a served directory, including its names: a note dropped beside
# untouched weights moves the digest. Equality still proves sameness, because
# nothing outside the weights can make two directories agree by accident. It is
# inequality that carries no information about the weights, and the
# contamination gate must not read it as a difference — that reading is what
# would clear a candidate that wrote the rows it is being scored on.
BROAD_SCOPE_DIGEST_SCHEMES: Final = frozenset({WEIGHT_MANIFEST_DIGEST_SCHEME})
_WEIGHTS_DIGEST_PATTERN: Final = (
    r"^(?:" + "|".join(re.escape(scheme) for scheme in WEIGHTS_DIGEST_SCHEMES) + r"):[0-9a-f]{64}$"
)
WeightsDigest = Annotated[StrictStr, Field(pattern=_WEIGHTS_DIGEST_PATTERN)]

# The scoring settings a publishable run must use. These are the correctness
# gates: an operator who relaxes one gets a higher number for the same model, so
# relaxing any of them makes the run debug-only rather than publishable.
# ``intermediate_text_matching`` is pinned for the opposite reason: demanding the
# gold wording measures how a reply was phrased rather than what the candidate
# did, so a verbatim run is not comparable to a publication run either.
LOCKED_PUBLICATION_SCORING: Final = FrozenDict(
    {
        "argument_matching": "schema_then_canonical",
        "respect_call_order": True,
        "respect_call_group": True,
        "allow_llm_repair": False,
        "task_success": "all_applicable_gates",
        "intermediate_text_matching": "structural",
    }
)

# The contamination settings a publishable run must use: enforcement on, a
# violation aborts the run, and every candidate is scored on the same task set.
LOCKED_PUBLICATION_CONTAMINATION: Final = FrozenDict(
    {
        "enforce": True,
        "on_violation": "fail_run",
        "comparison_set": "common_intersection",
    }
)

_ALIAS_PATTERN: Final = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_ENV_NAME_PATTERN: Final = r"^[A-Z][A-Z0-9_]{0,63}$"
_IDENTITY_TOKEN_PATTERN: Final = r"^[a-z0-9][a-z0-9._-]{0,63}$"
# Provider-specific inference knobs live in a versioned namespace so that adding
# one is a declared contract change instead of an unvalidated passthrough.
_EXTENSION_NAMESPACE_PATTERN: Final = r"^[a-z0-9][a-z0-9_]*\.v[0-9]+$"

Alias = Annotated[StrictStr, Field(pattern=_ALIAS_PATTERN)]
EnvVarName = Annotated[StrictStr, Field(pattern=_ENV_NAME_PATTERN)]
IdentityToken = Annotated[StrictStr, Field(pattern=_IDENTITY_TOKEN_PATTERN)]


def _numeric(value: Any) -> Any:
    """Accept only real numbers, so ``"0.7"`` and ``True`` never become floats."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("must be finite")
    return number


FiniteFloat = Annotated[float, BeforeValidator(_numeric)]
PositiveSeconds = Annotated[float, BeforeValidator(_numeric), Field(gt=0)]


def _sha256_json(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()}"


class _Strict(BaseModel):
    """Frozen, closed, non-coercing base for every eval config model."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


class EvalFileRef(_Strict):
    """A file this config depends on, identified by its bytes rather than its path.

    Both halves are kept: ``path`` is what the runner opens, ``content_hash`` is
    what the semantic hash and the eval manifest record. That split is what lets
    the same logical config keep its identity across checkouts while still
    changing identity when the referenced bytes change.
    """

    path: Path
    content_hash: ContentHash

    @field_validator("path")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"must be an absolute path, got {value!s}")
        return value

    def semantic_payload(self) -> dict[str, Any]:
        return {"content_hash": self.content_hash}


class EvalOracleResource(_Strict):
    """A resolvable execution resource for ``executable`` evaluation.

    ``oracle.kind`` in a generation manifest is only a lineage label. It does
    not tell a later process where the Python backend or endpoint configuration
    lives. This model carries both the pack manifest and the concrete execution
    resource by content hash, while the resolved paths remain available to the
    runner.
    """

    kind: Literal["python", "endpoint"]
    pack_manifest: EvalFileRef
    execution_resource: EvalFileRef
    pack_id: StrictStr
    pack_version: StrictStr
    expected_pack_content_hash: ContentHash

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pack_manifest": self.pack_manifest.semantic_payload(),
            "execution_resource": self.execution_resource.semantic_payload(),
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "expected_pack_content_hash": self.expected_pack_content_hash,
        }


class EvalSource(_Strict):
    """The published generation run this evaluation reads.

    The config names a ``run_manifest.json``, never a parquet: the publication
    manifest is the commit marker, so a directory that holds a benchmark without
    one holds an unpublished artifact. Everything else — which table to read, whether
    the run is gold-eligible, whether an oracle exists — is taken from the
    manifest instead of restated by the operator, so the two cannot disagree.
    """

    run_manifest: EvalFileRef
    benchmark: EvalFileRef
    translation_manifest: EvalFileRef | None = None
    run_id: StrictStr
    benchmark_schema_version: StrictStr
    publication_dir: Path
    gold_eligible: StrictBool
    lineage_policy: StrictStr
    oracle: EvalOracleResource | None = None

    @field_validator("publication_dir")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"must be an absolute path, got {value!s}")
        return value

    def semantic_payload(self) -> dict[str, Any]:
        # The run id and the manifest hash both appear: the id says which run was
        # evaluated, the hash says the manifest still says what it said then.
        return {
            "run_id": self.run_id,
            "run_manifest": self.run_manifest.semantic_payload(),
            "benchmark": self.benchmark.semantic_payload(),
            "benchmark_schema_version": self.benchmark_schema_version,
            "translation_manifest": (
                self.translation_manifest.semantic_payload() if self.translation_manifest is not None else None
            ),
            "oracle": self.oracle.semantic_payload() if self.oracle is not None else None,
        }

    @property
    def oracle_kind(self) -> str | None:
        return self.oracle.kind if self.oracle is not None else None


class EvalSettings(_Strict):
    """Which evaluations run against the source benchmark.

    The tuple is typed as strings rather than as a ``Literal`` so that an unknown
    mode is reported as an unsupported mode, naming the modes that do exist,
    instead of as a generic type mismatch.
    """

    modes: tuple[StrictStr, ...]

    @field_validator("modes")
    @classmethod
    def _canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise UnsupportedEvalModeError(
                "eval.mode",
                "no evaluation mode was requested",
                value=list(value),
                expected=f"a non-empty subset of {list(EVAL_MODES)}",
                recovery="set eval.mode to [trace] or [trace, executable]",
            )
        unknown = sorted(set(value) - set(EVAL_MODES))
        if unknown:
            raise UnsupportedEvalModeError(
                "eval.mode",
                f"unknown evaluation mode(s): {', '.join(unknown)}",
                value=list(value),
                expected=f"only {list(EVAL_MODES)}",
                recovery="remove the unknown mode; scoring dimensions are declared by the benchmark, not by the mode",
            )
        if len(set(value)) != len(value):
            raise UnsupportedEvalModeError(
                "eval.mode",
                "a mode is repeated",
                value=list(value),
                expected="each mode at most once",
                recovery="list every mode once; running one twice would double-count its tasks",
            )
        return tuple(mode for mode in EVAL_MODES if mode in set(value))

    @property
    def executable(self) -> bool:
        return "executable" in self.modes or "held_out_eval" in self.modes

    @property
    def held_out_eval(self) -> bool:
        return "held_out_eval" in self.modes

    @property
    def scope(self) -> str:
        """How far a result from this config may claim to reach."""
        return "trace_and_executable" if self.executable else "trace_only"

    def semantic_payload(self) -> dict[str, Any]:
        return {"modes": list(self.modes)}


class EvalScoringConfig(_Strict):
    """How a proposed tool call is compared against the gold call.

    ``contract`` points at the prose that defines the comparison. It is
    content-hashed, so editing the document changes ``eval_config_hash``: two runs
    that agree on every flag but disagree on what "argument match" means are not
    the same evaluation.
    """

    contract: EvalFileRef
    argument_matching: Literal["schema_then_canonical", "canonical_only"]
    insert_declared_defaults: StrictBool
    respect_call_order: StrictBool
    respect_call_group: StrictBool
    allow_llm_repair: StrictBool
    task_success: Literal["all_applicable_gates", "assertions_only"]
    # What an intermediate text-only assistant turn must reproduce. ``structural``
    # asks what the turn did — words rather than a call, and words that say
    # something — which is the behavior a pack declares its milestones in.
    # ``verbatim`` additionally demands the recorded sentence, so it scores one
    # translation of one phrasing and is kept only for reproducing older runs.
    intermediate_text_matching: Literal["structural", "verbatim"] = "structural"

    def publication_deviations(self) -> tuple[str, ...]:
        """Name the locked gates this scoring config relaxes, if any."""
        return tuple(
            f"scoring.{field}"
            for field, required in LOCKED_PUBLICATION_SCORING.items()
            if getattr(self, field) != required
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract.semantic_payload(),
            "argument_matching": self.argument_matching,
            "insert_declared_defaults": self.insert_declared_defaults,
            "respect_call_order": self.respect_call_order,
            "respect_call_group": self.respect_call_group,
            "allow_llm_repair": self.allow_llm_repair,
            "task_success": self.task_success,
            "intermediate_text_matching": self.intermediate_text_matching,
        }

    @property
    def scoring_policy_hash(self) -> str:
        """Content identity of the scoring rules, independent of config paths."""
        return _sha256_json(self.semantic_payload())


class EvalLimits(_Strict):
    """Runtime bounds, all pinned; nothing is inherited from a provider default.

    Limits are part of the measurement: a model cut off at 1024 tokens or two
    turns did not answer the same question as one given ten, so these values enter
    the hash exactly like the scoring flags do.
    """

    max_turns: PositiveInt
    tool_timeout_s: PositiveSeconds
    candidate_timeout_s: PositiveSeconds
    episode_timeout_s: PositiveSeconds
    max_parallel_tasks: PositiveInt
    max_retries: NonNegativeInt

    @model_validator(mode="after")
    def _episode_covers_a_turn(self) -> EvalLimits:
        if self.episode_timeout_s < max(self.tool_timeout_s, self.candidate_timeout_s):
            raise EvalConfigSchemaError(
                "limits.episode_timeout_s",
                "an episode may not time out before a single tool call or candidate call can finish",
                value=self.episode_timeout_s,
                expected=(
                    "at least max(limits.tool_timeout_s, limits.candidate_timeout_s) = "
                    f"{max(self.tool_timeout_s, self.candidate_timeout_s)}"
                ),
                recovery="raise limits.episode_timeout_s, or lower the per-call timeouts",
            )
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "max_turns": self.max_turns,
            "tool_timeout_s": self.tool_timeout_s,
            "candidate_timeout_s": self.candidate_timeout_s,
            "episode_timeout_s": self.episode_timeout_s,
            "max_parallel_tasks": self.max_parallel_tasks,
            "max_retries": self.max_retries,
        }


class CandidateApi(_Strict):
    """Where a candidate is served, and which environment variable holds its key."""

    base_url: StrictStr
    api_key_env: EnvVarName

    @field_validator("base_url")
    @classmethod
    def _http_endpoint(cls, value: StrictStr) -> str:
        normalized = value.strip().rstrip("/")
        parts = urlsplit(normalized)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("must be an http(s) URL")
        if "@" in parts.netloc:
            raise ValueError("must not embed credentials in the URL; use api_key_env")
        if parts.query or parts.fragment:
            raise ValueError("must not carry a query string or fragment")
        return normalized

    def semantic_payload(self) -> dict[str, Any]:
        # The variable *name* is part of the config; its value never is.
        return {"base_url": self.base_url, "api_key_env": self.api_key_env}


class CandidateModelIdentity(_Strict):
    """The weights that answered, named by whatever can honestly name them.

    A registry model can be pinned to bytes or to the commit they came from, and
    pinning it is what makes a score reproducible. A hosted frontier model often
    cannot: the provider publishes neither a digest nor an immutable revision, so
    the strongest true statement is the route. Both are accepted here and told
    apart by :attr:`assurance`, because the alternative — demanding a pin that
    does not exist — produces invented values, and an invented digest is worse
    than an absent one: it claims a reproducibility nothing backs.

    What an unpinned identity may not do is publish. That gate lives in
    :class:`BfclEvalConfig`, so the cost of not pinning is paid once, visibly, in
    ``non_publication_reasons`` rather than by refusing the run outright.
    """

    source: IdentityToken
    model: StrictStr
    revision: StrictStr | None = None
    weights_digest: WeightsDigest | None = None

    @field_validator("model")
    @classmethod
    def _non_empty(cls, value: StrictStr) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("revision")
    @classmethod
    def _immutable_revision(cls, value: StrictStr | None) -> str | None:
        if value is None:
            return None
        revision = value.strip()
        if not revision:
            raise ValueError("must be a non-empty string when present")
        lowered = revision.lower()
        if lowered in MUTABLE_REVISIONS or lowered.startswith(_MUTABLE_REF_PREFIXES):
            raise MutableCandidateRevisionError(
                "candidates[].model_identity.revision",
                "the revision names a moving pointer, so the same config would score different weights later",
                value=revision,
                expected="an immutable revision such as a commit sha, or a weights_digest instead",
                recovery="pin the commit the weights were served from, set weights_digest to sha256:<64 hex>, "
                "or leave both null if the provider publishes neither and accept a non-publishable run",
            )
        return revision

    @model_validator(mode="after")
    def _a_stated_revision_is_immutable(self) -> CandidateModelIdentity:
        """Refuse a revision that names a moving target, but not a missing one.

        Stating nothing and stating ``release-2026`` are different failures. The
        first is an operator who cannot pin, and the run records that. The second
        is a pin that will silently mean other weights later, which is the thing
        this field exists to prevent.
        """
        if (
            self.weights_digest is None
            and self.revision is not None
            and _IMMUTABLE_REVISION_PATTERN.fullmatch(self.revision) is None
        ):
            raise MutableCandidateRevisionError(
                "candidates[].model_identity.revision",
                "the revision is not a verifiable immutable commit identifier",
                value=self.revision,
                expected="40-64 hexadecimal commit characters, or a weights_digest",
                recovery="pin the full immutable commit, set weights_digest to sha256:<64 hex>, or leave both "
                "null and accept a non-publishable run; branch and tag names can move",
            )
        return self

    @property
    def assurance(self) -> IdentityAssurance:
        """How much this identity survives the endpoint changing underneath it."""
        if self.weights_digest is not None or self.revision is not None:
            return "weights_pinned"
        return "provider_managed"

    @property
    def canonical_id(self) -> str:
        """Source-qualified identity used to tell two candidates apart.

        The digest wins when both are present: it names the bytes, while a
        revision only names where they came from. Model and revision stay
        case-sensitive because the config supports arbitrary registries, and
        not every registry case-folds identifiers. An identity with neither
        reads as ``provider_managed``, so the string itself says the weights
        were never pinned instead of looking like a pin that happens to be short.
        """
        reference = self.weights_digest or self.revision or UNPINNED_REFERENCE
        return f"{self.source}:{self.model}@{reference}"

    def semantic_payload(self) -> dict[str, Any]:
        # The payload keeps the shape it had before assurance existed. The level
        # is derived from these same fields, so recording it would add no
        # information while changing the hash of every config already published.
        return {
            "source": self.source,
            "model": self.model,
            "revision": self.revision,
            "weights_digest": self.weights_digest,
            "canonical_id": self.canonical_id,
        }


class CandidateInference(_Strict):
    """Decoding parameters, pinned rather than inherited from the provider.

    Provider defaults change without notice, and a temperature that drifted from
    0 to 0.7 between two runs explains a score difference no code review would
    catch. Anything a provider supports beyond these fields goes into
    ``provider_extensions`` under a versioned namespace, never at the top level.
    """

    temperature: Annotated[FiniteFloat, Field(ge=0)]
    top_p: Annotated[FiniteFloat, Field(gt=0, le=1)] | None
    max_tokens: PositiveInt
    seed: StrictInt | None = None
    tool_choice: Literal["auto", "required", "none"]
    provider_extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _nucleus_sampling_is_unset_only_where_it_cannot_matter(self) -> CandidateInference:
        """Let ``top_p`` be null at temperature 0, and only there.

        Some providers refuse a request that carries both fields; every Anthropic
        model does, on Bedrock and on Azure alike. Without a way to omit one, that
        whole family cannot be evaluated at all.

        Omitting it is safe exactly when the decode is greedy: at temperature 0 the
        distribution is a point mass, so no nucleus cutoff can change the token that
        is chosen, and a provider default this config never saw cannot influence the
        result either. Above 0 the default would decide real sampling behaviour, which
        is the drift this class exists to prevent, so the field stays mandatory there.

        ``top_p`` is nullable but has no default: a config must still say which case
        it is in, and ``null`` is recorded in the semantic payload, so the hash tells
        an omitted top_p apart from any pinned value.
        """
        if self.top_p is None and self.temperature != 0:
            raise ValueError(
                "top_p may only be null when temperature is 0, where a greedy decode makes nucleus "
                f"sampling inert; at temperature {self.temperature} an unset top_p would inherit a "
                "provider default that decides sampling behaviour. Pin top_p, or pin temperature to 0."
            )
        return self

    @field_validator("provider_extensions")
    @classmethod
    def _versioned_namespaces(cls, value: Mapping[str, Any]) -> FrozenDict:
        frozen: dict[str, Any] = {}
        for namespace, settings in value.items():
            name = str(namespace)
            if not re.match(_EXTENSION_NAMESPACE_PATTERN, name):
                raise ValueError(
                    f"namespace {name!r} must be versioned, such as 'nvidia.v1', "
                    "so an unsupported field is a declared contract change"
                )
            if not isinstance(settings, Mapping):
                raise ValueError(f"{name} must map to a mapping, got {type(settings).__name__}")
            validate_json_value(settings, label=f"inference.provider_extensions.{name}")
            frozen[name] = freeze_json(settings)
        return FrozenDict(frozen)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "tool_choice": self.tool_choice,
            "provider_extensions": thaw_json(self.provider_extensions),
        }

    @property
    def inference_parameters_hash(self) -> str:
        return _sha256_json(self.semantic_payload())


class EvalCandidate(_Strict):
    """One model under evaluation: where it is served, and which weights it serves.

    ``provider`` and ``model`` are operational identity — the route a request
    takes. ``model_identity`` is weight identity. Keeping them apart is what makes
    a score comparable after an endpoint is renamed or a route is re-pointed.
    """

    alias: Alias
    model: StrictStr
    provider: IdentityToken
    provider_api_version: StrictStr
    api: CandidateApi
    model_identity: CandidateModelIdentity
    inference: CandidateInference

    @field_validator("model", "provider_api_version")
    @classmethod
    def _non_empty(cls, value: StrictStr) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @model_validator(mode="after")
    def _an_unpinned_identity_names_the_route_it_was_served_by(self) -> EvalCandidate:
        """Hold an unpinned candidate to the only evidence it has: the route.

        A pinned candidate may describe its weights however the registry does,
        because the pin is what tells two candidates apart. An unpinned one has
        no pin, so its canonical identity is built from ``source`` and ``model``
        alone — and those are free text. Two candidates on one endpoint could
        then be given two spellings and pass the duplicate check that exists to
        stop exactly that, each reporting a score for "different weights" that
        the same deployment produced. Requiring the identity to restate the route
        makes the collision happen where it should.
        """
        identity = self.model_identity
        if identity.assurance == "weights_pinned":
            return self
        if identity.source != self.provider or identity.model != self.model:
            raise CandidateIdentityError(
                "candidates[].model_identity",
                "a candidate that pins no weights must name the route that answered, and this one names "
                f"{identity.source}:{identity.model} while the request goes to {self.provider}:{self.model}",
                expected=f"source {self.provider!r} and model {self.model!r}, matching provider and model",
                recovery="restate the serving route as the identity, or pin the revision or weights_digest that "
                "makes this candidate a different one",
            )
        return self

    @property
    def canonical_model_identity(self) -> str:
        return self.model_identity.canonical_id

    @property
    def inference_parameters_hash(self) -> str:
        return self.inference.inference_parameters_hash

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "model": self.model,
            "provider": self.provider,
            "provider_api_version": self.provider_api_version,
            "api": self.api.semantic_payload(),
            "model_identity": self.model_identity.semantic_payload(),
            "inference": self.inference.semantic_payload(),
            "inference_parameters_hash": self.inference_parameters_hash,
        }


class ContaminationPolicy(_Strict):
    """What happens when a candidate turns out to have seen a task.

    Config resolution only pins the policy. The contamination gate detects
    collisions and computes the comparable task set; this model prevents it from
    receiving a policy that would quietly drop the evidence.
    """

    enforce: StrictBool
    on_violation: Literal["fail_run", "exclude_row"]
    comparison_set: Literal["common_intersection", "per_candidate"]

    def publication_deviations(self) -> tuple[str, ...]:
        return tuple(
            f"contamination.{field}"
            for field, required in LOCKED_PUBLICATION_CONTAMINATION.items()
            if getattr(self, field) != required
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "enforce": self.enforce,
            "on_violation": self.on_violation,
            "comparison_set": self.comparison_set,
        }


class EvalPublicationPolicy(_Strict):
    """Whether the operator is asking for a publishable result."""

    requested: StrictBool
    require_same_task_ids: StrictBool

    @field_validator("require_same_task_ids")
    @classmethod
    def _always_required(cls, value: StrictBool) -> bool:
        if value is not True:
            raise PublicationPolicyError(
                "publication.require_same_task_ids",
                "candidates scored on different task sets produce numbers that cannot be compared",
                value=value,
                expected="true",
                recovery="leave publication.require_same_task_ids at true; use contamination.comparison_set to "
                "decide how the shared task set is derived",
            )
        return value

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "require_same_task_ids": self.require_same_task_ids,
        }


class EvalOutputConfig(_Strict):
    """Where eval artifacts go, and which ones are written.

    The directory is deliberately absent from the semantic hash: two operators
    running the same evaluation into different directories ran the same
    evaluation. Which artifacts get written is not, because a run that skipped its
    per-task results cannot be audited afterwards.
    """

    output_dir: Path
    write_task_results: StrictBool
    write_eval_manifest: StrictBool
    cache_candidate_responses: StrictBool
    cache_tool_results: StrictBool

    @field_validator("output_dir")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(f"must be an absolute path, got {value!s}")
        return value

    def publication_deviations(self) -> tuple[str, ...]:
        return tuple(
            f"outputs.{field}"
            for field in (
                "write_task_results",
                "write_eval_manifest",
                "cache_candidate_responses",
                "cache_tool_results",
            )
            if getattr(self, field) is not True
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "write_task_results": self.write_task_results,
            "write_eval_manifest": self.write_eval_manifest,
            "cache_candidate_responses": self.cache_candidate_responses,
            "cache_tool_results": self.cache_tool_results,
        }


class BfclEvalConfig(_Strict):
    """A resolved, frozen eval config with one hash that stands for all of it."""

    schema_version: EvalConfigSchemaVersion
    config_status: Literal["resolved"]
    source: EvalSource
    settings: EvalSettings
    scoring: EvalScoringConfig
    limits: EvalLimits
    candidates: tuple[EvalCandidate, ...]
    contamination: ContaminationPolicy
    publication: EvalPublicationPolicy
    outputs: EvalOutputConfig
    held_out_eval: HeldOutEvalConfig | None = None

    @field_validator("candidates")
    @classmethod
    def _distinguishable(cls, value: tuple[EvalCandidate, ...]) -> tuple[EvalCandidate, ...]:
        if not value:
            raise CandidateIdentityError(
                "candidates",
                "no candidate was declared, so there is nothing to evaluate",
                expected="at least one candidate",
                recovery="add a candidate with an alias, a serving route, and an immutable model_identity",
            )
        aliases = [candidate.alias for candidate in value]
        repeated_aliases = sorted({alias for alias in aliases if aliases.count(alias) > 1})
        if repeated_aliases:
            raise CandidateIdentityError(
                "candidates[].alias",
                f"alias(es) reused: {', '.join(repeated_aliases)}; per-candidate artifacts would overwrite each other",
                expected="a unique alias per candidate",
                recovery="rename the duplicate alias",
            )
        identities = [candidate.canonical_model_identity for candidate in value]
        repeated = sorted({identity for identity in identities if identities.count(identity) > 1})
        if repeated:
            colliding = sorted(
                candidate.alias for candidate in value if candidate.canonical_model_identity in set(repeated)
            )
            raise CandidateIdentityError(
                "candidates[].model_identity",
                f"candidates {', '.join(colliding)} resolve to the same weights, so a comparison between them "
                "would report a difference the weights cannot explain",
                expected="a distinct canonical model identity per candidate",
                recovery="drop the duplicate candidate, or pin the revision/weights_digest that actually differs",
            )
        return value

    @model_validator(mode="after")
    def _coherent(self) -> BfclEvalConfig:
        if self.settings.executable and self.source.oracle is None:
            raise EvalConfigSchemaError(
                "eval.mode",
                "executable evaluation needs a resolvable oracle pack and execution resource",
                value=list(self.settings.modes),
                expected="source_oracle with pack_manifest, kind, and resource, or eval.mode limited to [trace]",
                recovery="evaluate trace-only, or configure source_oracle for the exact pack and backend/endpoint "
                "used by the source run",
            )
        if self.settings.held_out_eval != (self.held_out_eval is not None):
            raise EvalConfigSchemaError(
                "held_out_eval",
                "must be present exactly when eval.mode includes held_out_eval",
                value=self.held_out_eval is not None,
                expected="a versioned policy pin for held_out_eval, otherwise no section",
                recovery="add held_out_eval with the exact policy hash, ids, seed, and pack version",
            )
        if self.settings.held_out_eval:
            if self.source.translation_manifest is not None:
                raise EvalConfigSchemaError(
                    "translation_manifest",
                    "held_out_eval cannot compare a translated seen slice with an untranslated private slice",
                    expected="the source benchmark directly, with no translation_manifest",
                    recovery="evaluate the frozen source publication, or add a separately pinned private "
                    "translation contract before comparing translated slices",
                )
            if self.contamination.comparison_set != "common_intersection":
                raise EvalConfigSchemaError(
                    "contamination.comparison_set",
                    "held_out_eval requires one common seen task set for every candidate",
                    value=self.contamination.comparison_set,
                    expected="common_intersection",
                    recovery="use common_intersection so every seen/private gap compares the same strata",
                )
            unsafe = [
                name
                for name in (
                    "write_task_results",
                    "cache_candidate_responses",
                    "cache_tool_results",
                )
                if getattr(self.outputs, name)
            ]
            if unsafe:
                raise EvalConfigSchemaError(
                    "outputs",
                    "held_out_eval may not persist private tasks, prompts, or replay caches: " + ", ".join(unsafe),
                    expected="write_task_results, cache_candidate_responses, and cache_tool_results all false",
                    recovery="publish only the aggregate held-out generalization report",
                )
        if self.publication.requested:
            deviations = self.non_publication_reasons
            if deviations:
                raise PublicationPolicyError(
                    "publication.requested",
                    "publication was requested with settings that weaken what the score means: "
                    + ", ".join(deviations),
                    value=True,
                    expected="every correctness, contamination, and artifact gate at its locked value, and "
                    "weights pinned on every candidate",
                    recovery="restore the locked values and pin each candidate's revision or weights_digest, or "
                    "set publication.requested to false and treat this run as debug-only",
                )
        return self

    @property
    def non_publication_reasons(self) -> tuple[str, ...]:
        """Every reason this config's results may not be published, in field order.

        Computed independently of ``publication.requested`` so a debug config
        reports *why* it is debug-only rather than merely that it is.
        """
        reasons: list[str] = []
        reasons.extend(self.scoring.publication_deviations())
        # A candidate the provider may re-point cannot carry a published claim:
        # the number would be about whatever answered that day. The run is still
        # allowed, because measuring a hosted model is a real thing to want; what
        # it may not do is present the result as reproducible.
        reasons.extend(
            f"candidates[{candidate.alias}].model_identity"
            for candidate in self.candidates
            if candidate.model_identity.assurance != "weights_pinned"
        )
        reasons.extend(self.contamination.publication_deviations())
        reasons.extend(self.outputs.publication_deviations())
        if self.settings.executable and not self.source.gold_eligible:
            # Executable claims replay against the oracle, and only gold rows were
            # validated that way at generation time.
            reasons.append("source.gold_eligible")
        if not self.publication.requested:
            reasons.append("publication.requested")
        return tuple(reasons)

    @property
    def publication_allowed(self) -> bool:
        return not self.non_publication_reasons

    @property
    def publication_scope(self) -> str:
        return self.settings.scope

    @property
    def candidate_aliases(self) -> tuple[str, ...]:
        return tuple(candidate.alias for candidate in self.candidates)

    def candidate(self, alias: str) -> EvalCandidate:
        for candidate in self.candidates:
            if candidate.alias == alias:
                return candidate
        raise KeyError(alias)

    def semantic_payload(self) -> dict[str, Any]:
        """What this config asks for, with nothing that depends on the machine.

        Candidates are sorted by alias: their order in the YAML is presentation,
        not meaning, so reordering two candidates must not fork the hash. Modes
        are canonically ordered by :class:`EvalSettings`. Absolute paths, output
        locations, timestamps, and secret values are all absent by construction.
        """
        return {
            "schema_version": self.schema_version,
            "source": self.source.semantic_payload(),
            "eval": self.settings.semantic_payload(),
            "held_out_eval": (self.held_out_eval.semantic_payload() if self.held_out_eval is not None else None),
            "scoring": self.scoring.semantic_payload(),
            "limits": self.limits.semantic_payload(),
            "candidates": [
                candidate.semantic_payload() for candidate in sorted(self.candidates, key=lambda item: item.alias)
            ],
            "contamination": self.contamination.semantic_payload(),
            "publication": {
                **self.publication.semantic_payload(),
                "allowed": self.publication_allowed,
                "scope": self.publication_scope,
            },
            "outputs": self.outputs.semantic_payload(),
        }

    @property
    def eval_config_hash(self) -> str:
        return _sha256_json(self.semantic_payload())
