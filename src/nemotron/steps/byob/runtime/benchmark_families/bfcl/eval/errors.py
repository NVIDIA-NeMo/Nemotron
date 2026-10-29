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

"""Typed failures for the BFCL eval config contract.

Every error names the config field, what the value was (redacted), the
constraint that was expected, and the edit that fixes it. An eval config is the
one place in this pipeline that carries credentials by reference, so no error
message, log line, or diagnostic here may echo a value verbatim: values are
rendered through :func:`redact_value`, which reports type and size for anything
that is not a small scalar.

These are deliberately *not* :class:`ValueError` subclasses. Pydantic wraps
``ValueError`` raised inside a validator into a ``ValidationError``, which would
erase the error type a caller needs to distinguish "you typed a bad key" from
"you pinned a mutable model revision". Raising from ``Exception`` keeps the type
intact through model construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

_UNSET: Final = object()


def redact_value(value: Any, *, secret: bool = False) -> str:
    """Render a config value for an error message without leaking it.

    ``secret=True`` is used for fields that may hold a credential even when the
    contract says they should not, so the report states the shape only.
    """
    if secret:
        return "<redacted>"
    if value is _UNSET:
        return "<missing>"
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        # Config strings include URLs, path-embedded credentials, model routes,
        # and arbitrary provider extensions. A short string is not safer than a
        # long one ("p4ss" is short), so diagnostics report shape, never bytes.
        return f"<str len={len(value)}>"
    if isinstance(value, Mapping):
        return f"<mapping with {len(value)} keys>"
    if isinstance(value, (Sequence, set, frozenset)):
        return f"<{type(value).__name__} with {len(value)} items>"
    return f"<{type(value).__name__}>"


class EvalConfigError(Exception):
    """An eval config the pipeline refuses to run.

    Attributes are kept structured so a CLI, a step report, or a test can read
    the field path and the machine-readable ``code`` without parsing prose.
    """

    code: str = "eval_config_invalid"

    def __init__(
        self,
        field: str,
        problem: str,
        *,
        expected: str,
        recovery: str,
        value: Any = _UNSET,
        secret: bool = False,
    ) -> None:
        self.field = field
        self.problem = problem
        self.expected = expected
        self.recovery = recovery
        self.rendered_value = redact_value(value, secret=secret)
        self.other_violations: tuple[EvalConfigError, ...] = ()
        message = (
            f"{field}: {problem} (got {self.rendered_value}); "
            f"expected {expected}. Fix: {recovery}"
        )
        super().__init__(message)

    def also_report(self, others: Sequence[EvalConfigError]) -> EvalConfigError:
        """Carry the independent violations found alongside this one.

        Validation used to stop at the first refusal, so a config with three
        unrelated problems cost three runs to learn about — and a preflight
        makes no candidate request, so there was nothing to save by stopping.
        This one stays the raised error, keeping its code and exit status, and
        the rest ride along in the message and the report.
        """
        if not others:
            return self
        self.other_violations = (*self.other_violations, *others)
        listed = "\n".join(f"  also: {other.args[0]}" for other in others)
        self.args = (f"{self.args[0]}\n{listed}",)
        return self

    def as_report(self) -> dict[str, Any]:
        """Structured form for step reports, with the value already redacted."""
        report: dict[str, Any] = {
            "code": self.code,
            "field": self.field,
            "problem": self.problem,
            "value": self.rendered_value,
            "expected": self.expected,
            "recovery": self.recovery,
        }
        if self.other_violations:
            report["other_violations"] = [other.as_report() for other in self.other_violations]
        return report


class EvalConfigSchemaError(EvalConfigError):
    """The YAML shape, a key name, or a value type is not a schema this build reads."""

    code = "eval_config_schema_invalid"


class EvalConfigPathError(EvalConfigError):
    """A referenced file or directory is missing, wrong-typed, or out of bounds."""

    code = "eval_config_path_invalid"


class CandidateIdentityError(EvalConfigError):
    """A candidate cannot be told apart from another one, or names no weights."""

    code = "candidate_identity_invalid"


class MutableCandidateRevisionError(CandidateIdentityError):
    """A candidate pins a moving target such as ``main`` instead of a commit."""

    code = "candidate_revision_mutable"


class SecretInConfigError(EvalConfigError):
    """A credential value was written into the config instead of its env var name."""

    code = "secret_in_eval_config"


class PublicationPolicyError(EvalConfigError):
    """Publication was requested with a gate that would weaken the score."""

    code = "eval_publication_policy_violation"


class UnsupportedEvalModeError(EvalConfigError):
    """The requested evaluation modes are empty, repeated, or unknown."""

    code = "unsupported_eval_mode"
