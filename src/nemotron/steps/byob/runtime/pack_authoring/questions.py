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

"""Bounded, evidence-bound questions for unresolved assisted-authoring semantics."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.authoring_workflow.revision_store import (
    RevisionStore,
    RevisionStoreError,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.untrusted_text import quote_untrusted
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    EvidenceRevisionLink,
    SemanticAnswer,
    SemanticAnswerValue,
    SourceEvidenceDocument,
    UnsignedSourceEvidence,
    build_source_evidence,
    load_source_evidence,
    validate_semantic_target,
)

OPEN_QUESTIONS_VERSION: Literal["bfcl-open-questions-v1"] = "bfcl-open-questions-v1"
ANSWER_SET_VERSION: Literal["bfcl-question-answers-v1"] = "bfcl-question-answers-v1"
EVIDENCE_REVISION_RECORD_VERSION: Literal["bfcl-evidence-revision-record-v1"] = (
    "bfcl-evidence-revision-record-v1"
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_QUESTION_ID = re.compile(r"^q_[0-9a-f]{24}$")
_MAX_QUESTIONS = 100
_MAX_ARTIFACT_BYTES = 256 * 1024
_MAX_PROMPT_BYTES = 4096
_MAX_ENUM_VALUES = 64
_MAX_ENUM_STRING_BYTES = 512


class QuestionError(ValueError):
    """Raised when an open-question artifact cannot be trusted."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


AnswerScalar = StrictBool | StrictInt | StrictFloat | StrictStr


class AnswerDomain(_StrictModel):
    """A closed answer space; unconstrained prose is intentionally unsupported."""

    kind: Literal["boolean", "integer_range", "number_range", "enum"]
    enum_values: tuple[AnswerScalar, ...] = ()
    minimum: StrictFloat | StrictInt | None = None
    maximum: StrictFloat | StrictInt | None = None

    @field_validator("enum_values")
    @classmethod
    def _enum_values(
        cls,
        value: tuple[AnswerScalar, ...],
    ) -> tuple[AnswerScalar, ...]:
        if len(value) > _MAX_ENUM_VALUES:
            raise ValueError("answer enum exceeds 64 values")
        canonical = [sha256_json({"value": item}) for item in value]
        if len(canonical) != len(set(canonical)):
            raise ValueError("answer enum values must be unique")
        if canonical != sorted(canonical):
            raise ValueError("answer enum values must be in canonical digest order")
        for item in value:
            if isinstance(item, str) and len(item.encode("utf-8")) > _MAX_ENUM_STRING_BYTES:
                raise ValueError("answer enum string exceeds 512 bytes")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("answer enum numbers must be finite")
        return value

    @model_validator(mode="after")
    def _shape(self) -> AnswerDomain:
        if self.kind == "boolean":
            if self.enum_values or self.minimum is not None or self.maximum is not None:
                raise ValueError("boolean answer domain takes no constraints")
        elif self.kind == "enum":
            if (
                not self.enum_values
                or self.minimum is not None
                or self.maximum is not None
            ):
                raise ValueError("enum answer domain requires only enum_values")
        else:
            if self.enum_values:
                raise ValueError("numeric answer domains cannot contain enum_values")
            if self.minimum is None or self.maximum is None:
                raise ValueError("numeric answer domains require minimum and maximum")
            if not math.isfinite(float(self.minimum)) or not math.isfinite(
                float(self.maximum)
            ):
                raise ValueError("numeric answer bounds must be finite")
            if self.minimum > self.maximum:
                raise ValueError("answer minimum cannot exceed maximum")
            if self.kind == "integer_range" and (
                isinstance(self.minimum, float) and not self.minimum.is_integer()
                or isinstance(self.maximum, float) and not self.maximum.is_integer()
            ):
                raise ValueError("integer answer bounds must be integral")
        return self


class QuestionImpact(_StrictModel):
    consequence: Literal["blocks", "degrades_coverage", "requires_review"]
    stages: tuple[Literal["drafting", "validation", "freeze"], ...]
    artifacts: tuple[
        Literal[
            "manifest",
            "tools",
            "fixtures",
            "task_templates",
            "validation_cases",
            "assertions",
            "endpoint_config",
        ],
        ...,
    ] = ()

    @field_validator("stages", "artifacts")
    @classmethod
    def _ordered_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("question impact lists cannot be empty")
        if tuple(sorted(set(value))) != value:
            raise ValueError("question impact lists must be sorted and unique")
        return value


class OpenQuestion(_StrictModel):
    question_id: StrictStr
    question_digest: StrictStr
    target_path: StrictStr
    prompt: StrictStr
    evidence_refs: tuple[StrictStr, ...]
    answer_domain: AnswerDomain
    impact: QuestionImpact
    answer: None = None

    @field_validator("target_path")
    @classmethod
    def _semantic_target_only(cls, value: str) -> str:
        return validate_semantic_target(value)

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question prompt must be non-empty")
        if len(value.encode("utf-8")) > _MAX_PROMPT_BYTES:
            raise ValueError("question prompt exceeds 4096 bytes")
        if not value.startswith("<untrusted-data>\n") or not value.endswith(
            "\n</untrusted-data>"
        ):
            raise ValueError("question prompt must be fenced as untrusted data")
        return value

    @field_validator("evidence_refs")
    @classmethod
    def _evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) > 16:
            raise ValueError("question requires between 1 and 16 evidence references")
        if tuple(sorted(set(value))) != value:
            raise ValueError("question evidence references must be sorted and unique")
        if any(
            not re.fullmatch(r"#/unresolved_gaps/(0|[1-9][0-9]*)", item)
            or len(item) > 512
            or any(ord(character) < 32 for character in item)
            for item in value
        ):
            raise ValueError(
                "question evidence references must name unresolved-gap JSON pointers"
            )
        return value

    @model_validator(mode="after")
    def _identity(self) -> OpenQuestion:
        unsigned = self.model_dump(
            mode="json",
            exclude={"question_id", "question_digest", "answer"},
        )
        observed = sha256_json(unsigned)
        if self.question_digest != observed:
            raise ValueError("open question digest mismatch")
        if self.question_id != f"q_{observed.removeprefix('sha256:')[:24]}":
            raise ValueError("open question identity mismatch")
        return self


class OpenQuestionsArtifact(_StrictModel):
    schema_version: Literal["bfcl-open-questions-v1"]
    evidence_digest: StrictStr
    questions: tuple[OpenQuestion, ...]
    artifact_digest: StrictStr

    @field_validator("evidence_digest", "artifact_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("question artifact digests must be sha256 values")
        return value

    @field_validator("questions")
    @classmethod
    def _questions(cls, value: tuple[OpenQuestion, ...]) -> tuple[OpenQuestion, ...]:
        if len(value) > _MAX_QUESTIONS:
            raise ValueError("open question artifact exceeds 100 questions")
        identifiers = [item.question_id for item in value]
        if any(not _QUESTION_ID.fullmatch(identifier) for identifier in identifiers):
            raise ValueError("open question identifier has an invalid shape")
        if identifiers != sorted(set(identifiers)):
            raise ValueError("open questions must be sorted with unique identities")
        targets = [item.target_path for item in value]
        if len(targets) != len(set(targets)):
            raise ValueError("open questions cannot write one semantic target twice")
        return value

    @model_validator(mode="after")
    def _artifact_digest(self) -> OpenQuestionsArtifact:
        unsigned = self.model_dump(mode="json", exclude={"artifact_digest"})
        if self.artifact_digest != sha256_json(unsigned):
            raise ValueError("open question artifact digest mismatch")
        if len(
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ) > _MAX_ARTIFACT_BYTES:
            raise ValueError("open question artifact exceeds 256 KiB")
        return self


class QuestionCandidate(_StrictModel):
    target_path: StrictStr
    prompt: StrictStr
    evidence_refs: tuple[StrictStr, ...]
    answer_domain: AnswerDomain
    impact: QuestionImpact


def build_open_question(candidate: QuestionCandidate) -> OpenQuestion:
    prompt = quote_untrusted(candidate.prompt)
    unsigned = {
        "target_path": candidate.target_path,
        "prompt": prompt,
        "evidence_refs": list(candidate.evidence_refs),
        "answer_domain": candidate.answer_domain.model_dump(mode="json"),
        "impact": candidate.impact.model_dump(mode="json"),
    }
    digest = sha256_json(unsigned)
    return OpenQuestion.model_validate(
        {
            **unsigned,
            "question_id": f"q_{digest.removeprefix('sha256:')[:24]}",
            "question_digest": digest,
            "answer": None,
        }
    )


def build_open_questions(
    *,
    evidence_digest: str,
    candidates: Sequence[QuestionCandidate],
) -> OpenQuestionsArtifact:
    questions = tuple(
        sorted(
            (build_open_question(candidate) for candidate in candidates),
            key=lambda item: item.question_id,
        )
    )
    document: dict[str, Any] = {
        "schema_version": OPEN_QUESTIONS_VERSION,
        "evidence_digest": evidence_digest,
        "questions": [question.model_dump(mode="json") for question in questions],
    }
    document["artifact_digest"] = sha256_json(document)
    return OpenQuestionsArtifact.model_validate(document)


def load_open_questions(
    path: Path,
    *,
    evidence_digest: str,
) -> OpenQuestionsArtifact:
    source = path.resolve()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QuestionError(f"open question artifact repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        raw = source.read_bytes()
        if len(raw) > _MAX_ARTIFACT_BYTES:
            raise QuestionError("open question artifact exceeds 256 KiB")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        artifact = OpenQuestionsArtifact.model_validate(document)
    except QuestionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise QuestionError(f"cannot load open questions {source}: {exc}") from exc
    if artifact.evidence_digest != evidence_digest:
        raise QuestionError("open question artifact names stale evidence")
    return artifact


def write_open_questions(artifact: OpenQuestionsArtifact, path: Path) -> Path:
    return write_canonical_json(artifact.model_dump(mode="json"), path)


class AnswerSubmission(_StrictModel):
    question_id: StrictStr
    value: SemanticAnswerValue

    @field_validator("question_id")
    @classmethod
    def _question_id(cls, value: str) -> str:
        if not _QUESTION_ID.fullmatch(value):
            raise ValueError("answer has an invalid question identity")
        return value


class AnswerSetArtifact(_StrictModel):
    schema_version: Literal["bfcl-question-answers-v1"]
    evidence_digest: StrictStr
    question_artifact_digest: StrictStr
    answers: tuple[AnswerSubmission, ...]
    answer_set_digest: StrictStr

    @field_validator(
        "evidence_digest",
        "question_artifact_digest",
        "answer_set_digest",
    )
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("answer artifact digests must be sha256 values")
        return value

    @field_validator("answers")
    @classmethod
    def _answers(
        cls,
        value: tuple[AnswerSubmission, ...],
    ) -> tuple[AnswerSubmission, ...]:
        identities = [item.question_id for item in value]
        if identities != sorted(set(identities)):
            raise ValueError("answers must be sorted with unique question identities")
        if len(value) > _MAX_QUESTIONS:
            raise ValueError("answer artifact exceeds 100 answers")
        return value

    @model_validator(mode="after")
    def _answer_digest(self) -> AnswerSetArtifact:
        unsigned = self.model_dump(mode="json", exclude={"answer_set_digest"})
        if self.answer_set_digest != sha256_json(unsigned):
            raise ValueError("answer set digest mismatch")
        return self


class EvidenceRevisionRecord(_StrictModel):
    schema_version: Literal["bfcl-evidence-revision-record-v1"]
    parent_bundle_digest: StrictStr
    revised_bundle_digest: StrictStr
    question_artifact_digest: StrictStr
    answer_set_digest: StrictStr
    applied_question_ids: tuple[StrictStr, ...]
    record_digest: StrictStr

    @field_validator(
        "parent_bundle_digest",
        "revised_bundle_digest",
        "question_artifact_digest",
        "answer_set_digest",
        "record_digest",
    )
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST.fullmatch(value):
            raise ValueError("evidence revision record contains an invalid digest")
        return value

    @field_validator("applied_question_ids")
    @classmethod
    def _question_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("applied question identities must be sorted and unique")
        return value

    @model_validator(mode="after")
    def _record_digest(self) -> EvidenceRevisionRecord:
        unsigned = self.model_dump(mode="json", exclude={"record_digest"})
        if self.record_digest != sha256_json(unsigned):
            raise ValueError("evidence revision record digest mismatch")
        return self


@dataclass(frozen=True)
class AppliedEvidenceRevision:
    evidence: SourceEvidenceDocument
    record: EvidenceRevisionRecord


def build_answer_set(
    *,
    evidence_digest: str,
    question_artifact_digest: str,
    answers: Sequence[AnswerSubmission],
) -> AnswerSetArtifact:
    canonical = tuple(sorted(answers, key=lambda item: item.question_id))
    document: dict[str, Any] = {
        "schema_version": ANSWER_SET_VERSION,
        "evidence_digest": evidence_digest,
        "question_artifact_digest": question_artifact_digest,
        "answers": [answer.model_dump(mode="json") for answer in canonical],
    }
    document["answer_set_digest"] = sha256_json(document)
    return AnswerSetArtifact.model_validate(document)


def _validate_answer(question: OpenQuestion, value: SemanticAnswerValue) -> None:
    domain = question.answer_domain
    if domain.kind == "boolean":
        if not isinstance(value, bool):
            raise QuestionError(f"{question.question_id} requires a boolean answer")
        return
    if domain.kind == "integer_range":
        if isinstance(value, bool) or not isinstance(value, int):
            raise QuestionError(f"{question.question_id} requires an integer answer")
        assert domain.minimum is not None and domain.maximum is not None
        if not float(domain.minimum) <= value <= float(domain.maximum):
            raise QuestionError(f"{question.question_id} answer is outside its range")
        return
    if domain.kind == "number_range":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QuestionError(f"{question.question_id} requires a numeric answer")
        assert domain.minimum is not None and domain.maximum is not None
        if not float(domain.minimum) <= float(value) <= float(domain.maximum):
            raise QuestionError(f"{question.question_id} answer is outside its range")
        return
    allowed = {
        sha256_json({"value": item})
        for item in domain.enum_values
    }
    if sha256_json({"value": value}) not in allowed:
        raise QuestionError(f"{question.question_id} answer is outside its enum")


def _verify_evidence_references(
    evidence: SourceEvidenceDocument,
    questions: OpenQuestionsArtifact,
) -> None:
    for question in questions.questions:
        for reference in question.evidence_refs:
            index = int(reference.rsplit("/", 1)[1])
            if index >= len(evidence.unresolved_gaps):
                raise QuestionError(
                    f"{question.question_id} references a missing unresolved gap"
                )


def apply_answers(
    evidence: SourceEvidenceDocument,
    questions: OpenQuestionsArtifact,
    answers: AnswerSetArtifact,
    *,
    root_bundle_digest: str | None = None,
) -> AppliedEvidenceRevision:
    if questions.evidence_digest != evidence.bundle_digest:
        raise QuestionError("open questions are stale for this evidence bundle")
    if answers.evidence_digest != evidence.bundle_digest:
        raise QuestionError("answer set is stale for this evidence bundle")
    if answers.question_artifact_digest != questions.artifact_digest:
        raise QuestionError("answer set names a different question artifact")
    _verify_evidence_references(evidence, questions)
    expected = {question.question_id: question for question in questions.questions}
    if not expected:
        raise QuestionError("there are no open questions to answer")
    supplied = {answer.question_id: answer for answer in answers.answers}
    missing = sorted(set(expected) - set(supplied))
    extra = sorted(set(supplied) - set(expected))
    if missing or extra:
        raise QuestionError(
            f"answer set must be exact; missing={missing!r}, extra={extra!r}"
        )
    semantic_answers: list[SemanticAnswer] = []
    for question_id, question in sorted(expected.items()):
        submission = supplied[question_id]
        _validate_answer(question, submission.value)
        semantic_answers.append(
            SemanticAnswer(
                question_id=question.question_id,
                question_digest=question.question_digest,
                target_path=question.target_path,
                value=submission.value,
                evidence_refs=question.evidence_refs,
            )
        )
    inherited_root = (
        evidence.revision.root_bundle_digest if evidence.revision is not None else None
    )
    if (
        inherited_root is not None
        and root_bundle_digest is not None
        and root_bundle_digest != inherited_root
    ):
        raise QuestionError("revision root cannot change across evidence revisions")
    link = EvidenceRevisionLink(
        root_bundle_digest=inherited_root or root_bundle_digest or evidence.bundle_digest,
        parent_bundle_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answer_set_digest=answers.answer_set_digest,
    )
    unsigned_document = evidence.model_dump(mode="json", exclude={"bundle_digest"})
    combined_answers = tuple(
        sorted(
            (*evidence.semantic_answers, *semantic_answers),
            key=lambda item: item.question_id,
        )
    )
    unsigned_document["semantic_answers"] = [
        answer.model_dump(mode="json") for answer in combined_answers
    ]
    unsigned_document["revision"] = link.model_dump(mode="json")
    revised = build_source_evidence(
        UnsignedSourceEvidence.model_validate(unsigned_document)
    )
    record_document: dict[str, Any] = {
        "schema_version": EVIDENCE_REVISION_RECORD_VERSION,
        "parent_bundle_digest": evidence.bundle_digest,
        "revised_bundle_digest": revised.bundle_digest,
        "question_artifact_digest": questions.artifact_digest,
        "answer_set_digest": answers.answer_set_digest,
        "applied_question_ids": sorted(expected),
    }
    record_document["record_digest"] = sha256_json(record_document)
    return AppliedEvidenceRevision(
        evidence=revised,
        record=EvidenceRevisionRecord.model_validate(record_document),
    )


def load_answer_set(
    path: Path,
    *,
    evidence_digest: str,
    question_artifact_digest: str,
) -> AnswerSetArtifact:
    source = path.resolve()

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise QuestionError(f"answer artifact repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        raw = source.read_bytes()
        if len(raw) > _MAX_ARTIFACT_BYTES:
            raise QuestionError("answer artifact exceeds 256 KiB")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        artifact = AnswerSetArtifact.model_validate(document)
    except QuestionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise QuestionError(f"cannot load answer set {source}: {exc}") from exc
    if artifact.evidence_digest != evidence_digest:
        raise QuestionError("answer artifact names stale evidence")
    if artifact.question_artifact_digest != question_artifact_digest:
        raise QuestionError("answer artifact names stale questions")
    return artifact


def write_answer_set(artifact: AnswerSetArtifact, path: Path) -> Path:
    return write_canonical_json(artifact.model_dump(mode="json"), path)


def write_evidence_revision(
    revision: AppliedEvidenceRevision,
    revisions_root: Path,
) -> Path:
    """Write once under the content address; existing revisions are never replaced."""

    try:
        return RevisionStore(revisions_root).put_json(
            revision.evidence.bundle_digest,
            {
                "evidence.json": revision.evidence.model_dump(mode="json"),
                "revision_record.json": revision.record.model_dump(mode="json"),
            },
        )
    except RevisionStoreError as exc:
        raise QuestionError(exc.detail) from exc


def verify_answered_revision(
    revised: SourceEvidenceDocument,
    *,
    parent_evidence_path: Path | None,
    open_questions_path: Path | None,
    answer_set_path: Path | None,
    expected_root_digest: str | None = None,
    expected_normalized_origin_digest: str | None = None,
) -> None:
    """Fail closed unless an answered revision can be deterministically replayed."""

    supplied = (parent_evidence_path, open_questions_path, answer_set_path)
    if revised.revision is None:
        if any(path is not None for path in supplied):
            raise QuestionError(
                "question resume artifacts require an answered evidence revision"
            )
        return
    if any(path is None for path in supplied):
        raise QuestionError(
            "answered evidence revision requires parent, questions, and answer set"
        )
    assert parent_evidence_path is not None
    assert open_questions_path is not None
    assert answer_set_path is not None
    parent = load_source_evidence(parent_evidence_path)
    if (
        expected_normalized_origin_digest is not None
        and parent.revision is None
        and parent.bundle_digest != expected_normalized_origin_digest
    ):
        raise QuestionError("revision parent is not the migration-normalized origin")
    if parent.bundle_digest != revised.revision.parent_bundle_digest:
        raise QuestionError("answered revision names a different parent evidence bundle")
    questions = load_open_questions(
        open_questions_path,
        evidence_digest=parent.bundle_digest,
    )
    if questions.artifact_digest != revised.revision.question_artifact_digest:
        raise QuestionError("answered revision names a different question artifact")
    answers = load_answer_set(
        answer_set_path,
        evidence_digest=parent.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
    )
    if answers.answer_set_digest != revised.revision.answer_set_digest:
        raise QuestionError("answered revision names a different answer set")
    replayed = apply_answers(
        parent,
        questions,
        answers,
        root_bundle_digest=expected_root_digest,
    )
    if replayed.evidence != revised:
        raise QuestionError("answered evidence revision does not match deterministic replay")
