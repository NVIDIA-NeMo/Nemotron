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

"""Secret-free typed failures for the evaluation conversation driver."""

from __future__ import annotations

from typing import Any, Final

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.errors import redact_value

_UNSET: Final = object()


class ConversationDriverError(Exception):
    """An episode could not be scripted, authorized, or advanced."""

    code = "eval_conversation_invalid"

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


class ConversationScriptError(ConversationDriverError):
    """A published row does not describe a conversation this driver can replay."""

    code = "eval_conversation_script_invalid"


class ConversationAuthorizationError(ConversationDriverError):
    """A task or candidate was not the one the contamination gate authorized."""

    code = "eval_conversation_unauthorized"


class ConversationLeakageError(ConversationDriverError):
    """Something that is not model-facing reached, or nearly reached, a prompt."""

    code = "eval_conversation_answer_key_leak"


class ConversationTransitionError(ConversationDriverError):
    """The driver was asked to make a move its own state does not allow."""

    code = "eval_conversation_transition_invalid"


def describe_conversation_error(exc: Exception) -> str:
    if isinstance(exc, ConversationDriverError):
        report = exc.as_report()
        return f"[{report['code']}] {report['subject']}: {report['problem']}"
    return f"[eval_conversation_invalid] {type(exc).__name__}: {redact_value(str(exc))}"


__all__ = [
    "ConversationAuthorizationError",
    "ConversationDriverError",
    "ConversationLeakageError",
    "ConversationScriptError",
    "ConversationTransitionError",
    "describe_conversation_error",
]
