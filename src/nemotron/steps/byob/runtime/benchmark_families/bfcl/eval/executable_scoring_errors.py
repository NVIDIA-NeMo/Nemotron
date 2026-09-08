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

"""Typed, secret-safe failures raised before executable scoring."""

from __future__ import annotations

from typing import Any, Final

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import redact_value

_UNSET: Final = object()


class ExecutableScoringError(Exception):
    """Evidence or policy that cannot produce a meaningful executable score."""

    code = "eval_executable_scoring_invalid"

    def __init__(
        self,
        subject: str,
        problem: str,
        *,
        expected: str,
        recovery: str,
        actual: Any = _UNSET,
        secret: bool = False,
    ) -> None:
        self.subject = subject
        self.problem = problem
        self.expected = expected
        self.recovery = recovery
        self.rendered_actual = (
            redact_value(actual, secret=secret) if actual is not _UNSET else "<missing>"
        )
        super().__init__(
            f"{subject}: {problem} (observed {self.rendered_actual}); "
            f"expected {expected}. Fix: {recovery}"
        )

    def as_report(self) -> dict[str, str]:
        return {
            "code": self.code,
            "subject": self.subject,
            "problem": self.problem,
            "actual": self.rendered_actual,
            "expected": self.expected,
            "recovery": self.recovery,
        }


class ExecutableEvidenceError(ExecutableScoringError):
    """The task, episode, source, and plan do not identify one evaluation."""

    code = "eval_executable_evidence_mismatch"


class ExecutableScoringPolicyError(ExecutableScoringError):
    """The pinned policy asks for unsupported executable scoring behavior."""

    code = "eval_executable_scoring_policy_unsupported"


class ExecutableAggregationError(ExecutableScoringError):
    """Per-task scores do not form one complete authorized candidate run."""

    code = "eval_executable_aggregation_invalid"


def describe_executable_scoring_error(exc: Exception) -> str:
    if isinstance(exc, ExecutableScoringError):
        report = exc.as_report()
        return f"[{report['code']}] {report['subject']}: {report['problem']}"
    return (
        f"[eval_executable_scoring_invalid] {type(exc).__name__}: "
        f"{redact_value(str(exc))}"
    )


__all__ = [
    "ExecutableAggregationError",
    "ExecutableEvidenceError",
    "ExecutableScoringError",
    "ExecutableScoringPolicyError",
    "describe_executable_scoring_error",
]
