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

"""Typed failures for evaluation source verification.

A config error says the operator wrote something invalid. These errors say
something else: the *source* an already-valid config points at is not the
publication the config resolved, or is not safe to evaluate in the requested
mode. Those are separate families on purpose, because the recovery differs —
one is an edit, the other is "regenerate, or evaluate the run you actually
published".

Every failure names the artifact, states the constraint, and reports the
observed evidence. Evidence is rendered by :func:`render_evidence`, which
prints content hashes and counts verbatim — a mismatch nobody can see the two
sides of is not a diagnosis — and reduces everything else to its shape. Nothing
here ever receives a credential: endpoint configs are resolved through the
loader that keeps only environment variable *names*, so no token, header value,
or API key exists at this layer to leak.

Identifiers minted by this pipeline (run ids, task ids, pack ids, file names)
do appear in the prose, because a report that will not name the run it rejected
cannot be acted on.
"""

from __future__ import annotations

import re
from typing import Any, Final

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import redact_value

_UNSET: Final = object()
_CONTENT_HASH: Final = re.compile(r"^sha256:[0-9a-f]{64}$")


def render_evidence(value: Any) -> str:
    """Render observed evidence for a verification failure.

    Content hashes survive intact: the whole point of a drift report is to let a
    reader compare the two digests. Everything else falls back to the config
    layer's redaction, so an arbitrary string can never be echoed.
    """
    if value is _UNSET:
        return "<missing>"
    if isinstance(value, str) and _CONTENT_HASH.fullmatch(value):
        return value
    return redact_value(value)


class SourceVerificationError(Exception):
    """The evaluation source is not the publication it claims to be."""

    code: str = "eval_source_invalid"

    def __init__(
        self,
        artifact: str,
        problem: str,
        *,
        expected: str,
        recovery: str,
        actual: Any = _UNSET,
    ) -> None:
        self.artifact = artifact
        self.problem = problem
        self.expected = expected
        self.recovery = recovery
        self.rendered_actual = render_evidence(actual)
        super().__init__(
            f"{artifact}: {problem} (observed {self.rendered_actual}); "
            f"expected {expected}. Fix: {recovery}"
        )

    def as_report(self) -> dict[str, str]:
        """Structured form for a diagnostic artifact or a step report."""
        return {
            "code": self.code,
            "artifact": self.artifact,
            "problem": self.problem,
            "actual": self.rendered_actual,
            "expected": self.expected,
            "recovery": self.recovery,
        }


class SourceManifestSchemaError(SourceVerificationError):
    """``run_manifest.json`` is absent, unreadable, or not a BFCL commit marker."""

    code = "eval_source_manifest_invalid"


class SourceManifestDriftError(SourceVerificationError):
    """The manifest bytes changed between config resolution and verification."""

    code = "eval_source_manifest_drift"


class BenchmarkHashMismatchError(SourceVerificationError):
    """A benchmark table's bytes do not match the hash the publication declares."""

    code = "eval_source_benchmark_hash_mismatch"


class BenchmarkSchemaMismatchError(SourceVerificationError):
    """The parquet on disk is not written with the benchmark schema this build reads."""

    code = "eval_source_benchmark_schema_mismatch"


class PublicationSemanticsError(SourceVerificationError):
    """The raw and published tables do not satisfy the publication contract."""

    code = "eval_source_publication_invalid"


class SourceTaskIndexError(SourceVerificationError):
    """The published rows cannot be indexed into a task set an eval run can address."""

    code = "eval_source_task_index_invalid"


class ModelExposureError(SourceVerificationError):
    """The publication cannot say which models read its rows.

    This is a source failure rather than a contamination one: the contamination
    contamination gate decides what an exposure *means*, but it can only do that if the
    publication declared every model that shaped the rows it ships. A manifest
    that omits a role, names a model for no role, or contradicts its own rows
    leaves a gap that would read as "no contamination found".
    """

    code = "eval_source_model_exposure_invalid"


class OraclePackDriftError(SourceVerificationError):
    """The oracle pack's content changed after the source run was generated."""

    code = "eval_source_oracle_pack_drift"


class OracleResourceMismatchError(SourceVerificationError):
    """The declared oracle resource is not the one the source run executed."""

    code = "eval_source_oracle_resource_mismatch"


class TranslationLineageError(SourceVerificationError):
    """A translated benchmark does not derive from this source run, or changed its truth."""

    code = "eval_source_translation_lineage_invalid"


class SourceChangedDuringEvalError(SourceVerificationError):
    """The source moved after it was verified, so the run would score two benchmarks."""

    code = "eval_source_changed_during_eval"


def describe_source_verification_error(exc: Exception) -> str:
    """One-line, secret-free summary for a CLI or a step report."""
    if isinstance(exc, SourceVerificationError):
        report = exc.as_report()
        return f"[{report['code']}] {report['artifact']}: {report['problem']}"
    return f"[eval_source_invalid] {type(exc).__name__}: {redact_value(str(exc))}"
