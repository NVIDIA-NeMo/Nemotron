from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.pack_authoring.artifacts import (
    sha256_json,
    sha256_text,
    write_canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.questions import (
    AnswerDomain,
    AnswerSubmission,
    OpenQuestion,
    OpenQuestionsArtifact,
    QuestionCandidate,
    QuestionError,
    QuestionImpact,
    apply_answers,
    build_answer_set,
    build_open_question,
    build_open_questions,
    load_answer_set,
    load_open_questions,
    verify_answered_revision,
    write_answer_set,
    write_evidence_revision,
    write_open_questions,
)
from nemotron.steps.byob.runtime.source_adapters.contract import (
    AdapterCapability,
    AdapterDescriptor,
    CleanupKind,
    CleanupSemantics,
    FixtureAccessKind,
    FixtureAccessPolicy,
    ProbeSafetyKind,
    ProbeSafetyPolicy,
)
from nemotron.steps.byob.runtime.source_adapters.domain_brief import DomainBriefEvidence
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    CapabilityEvidence,
    CertificationReference,
    ConfirmationVocabulary,
    FixtureEvidence,
    PackIdentity,
    SemanticAnswer,
    SourceEvidenceDocument,
    SourceIdentity,
    ToolEvidence,
    UnresolvedGap,
    UnsignedSourceEvidence,
    UntrustedText,
    build_source_evidence,
)
from nemotron.steps.byob.runtime.source_adapters.held_out import (
    build_not_applicable_decision,
)
from nemotron.steps.byob.runtime.source_adapters.migration import (
    MIGRATION_APPROVAL_VERSION,
    EvidenceMigrationError,
    NormalizedSourceEvidence,
    load_normalized_approval,
)

EVIDENCE_A = "sha256:" + "a" * 64
EVIDENCE_B = "sha256:" + "b" * 64


def _impact() -> QuestionImpact:
    return QuestionImpact(
        consequence="blocks",
        stages=("drafting", "validation"),
        artifacts=("task_templates", "validation_cases"),
    )


def _candidate(
    *,
    target: str = "/semantic/business_rules/checkout_limit",
    prompt: str = "How many books may one patron check out?",
    reference: str = "#/unresolved_gaps/0",
    domain: AnswerDomain | None = None,
) -> QuestionCandidate:
    return QuestionCandidate(
        target_path=target,
        prompt=prompt,
        evidence_refs=(reference,),
        answer_domain=domain
        or AnswerDomain(
            kind="integer_range",
            minimum=1,
            maximum=20,
        ),
        impact=_impact(),
    )


def _evidence() -> SourceEvidenceDocument:
    descriptor = AdapterDescriptor(
        contract_version="bfcl-source-adapter-v1",
        kind="fixture",
        implementation_name="bfcl.fixture",
        implementation_version="1.0.0",
        capabilities=(
            AdapterCapability.DESCRIBE_TOOLS,
            AdapterCapability.PIN_IDENTITY,
        ),
        fixture_access=FixtureAccessPolicy(
            kind=FixtureAccessKind.READ_ONLY,
            supports_redaction=True,
        ),
        probe_safety=ProbeSafetyPolicy(
            kind=ProbeSafetyKind.IDENTITY_ONLY,
            max_calls=1,
            timeout_s=1.0,
        ),
        cleanup=CleanupSemantics(kind=CleanupKind.NONE, timeout_s=1.0),
    )
    descriptor_digest = sha256_json(descriptor.model_dump(mode="json"))
    return build_source_evidence(
        UnsignedSourceEvidence(
            schema_version="bfcl-source-evidence-v2",
            source_adapter=descriptor,
            certification=CertificationReference(
                reference_version="bfcl-adapter-certification-reference-v1",
                report_schema_version="bfcl-adapter-certification-report-v1",
                report_digest=EVIDENCE_A,
                descriptor_digest=descriptor_digest,
                issuer="bfcl-source-adapter-verifier-v1",
                profile_id="fixture-a0",
                attained_tier="A0",
            ),
            pack=PackIdentity(pack_id="question-fixture", version="1.0.0"),
            domain_brief=DomainBriefEvidence(
                schema_version="bfcl-domain-brief-v1",
                language="en",
                untrusted_text="Evaluate reviewed checkout semantics.",
                source_digest=EVIDENCE_A,
                content_digest=sha256_text(
                    "Evaluate reviewed checkout semantics."
                ),
                redaction_report_digest=EVIDENCE_B,
            ),
            identity=SourceIdentity(
                subject="question fixture",
                effective_content_digest=EVIDENCE_A,
                source_config_digest=EVIDENCE_B,
            ),
            capabilities=(
                CapabilityEvidence(
                    capability=AdapterCapability.DESCRIBE_TOOLS,
                    status="observed",
                    evidence_digests=(EVIDENCE_A,),
                ),
                CapabilityEvidence(
                    capability=AdapterCapability.PIN_IDENTITY,
                    status="observed",
                    evidence_digests=(EVIDENCE_B,),
                ),
            ),
            vocabulary=ConfirmationVocabulary(),
            fixtures=FixtureEvidence(
                direction="read_only",
                content_digest=None,
                held_out=build_not_applicable_decision(
                    "Question fixture has no held-out evaluation.",
                    reviewed_by="question-tests",
                ),
            ),
            tools=(
                ToolEvidence(
                    published_name="library.lookup",
                    source_name="library.lookup",
                    description=UntrustedText(untrusted_text="Look up a book."),
                    parameter_schema={"type": "object", "properties": {}},
                    output_schema=None,
                    annotations=None,
                    mutates=False,
                    requires_confirmation=False,
                    raw_digest=EVIDENCE_A,
                ),
            ),
            unresolved_gaps=(
                UnresolvedGap(
                    code="checkout_limit_unknown",
                    field="checkout_limit",
                    reason="The source does not declare a checkout limit.",
                    evidence_refs=(),
                ),
            ),
        )
    )


def test_question_identity_is_stable_fenced_and_order_independent() -> None:
    first = _candidate()
    second = _candidate(
        target="/semantic/safety_intent/delete_confirmation",
        prompt="Must deletion require confirmation?",
        reference="#/unresolved_gaps/1",
        domain=AnswerDomain(kind="boolean"),
    )

    question = build_open_question(first)
    forward = build_open_questions(
        evidence_digest=EVIDENCE_A,
        candidates=(first, second),
    )
    reverse = build_open_questions(
        evidence_digest=EVIDENCE_A,
        candidates=(second, first),
    )

    assert question.prompt.startswith("<untrusted-data>\n")
    assert question.prompt.endswith("\n</untrusted-data>")
    assert question.question_id.startswith("q_")
    assert forward == reverse
    assert forward.artifact_digest == reverse.artifact_digest


def test_question_targets_cannot_override_executable_or_authority_fields() -> None:
    for target in (
        "/identity/effective_content_digest",
        "/semantic/certification/attained_tier",
        "/semantic/source_adapter/capabilities",
        "/semantic/tool/identity/name",
    ):
        with pytest.raises(ValidationError, match="semantic|authority"):
            build_open_question(_candidate(target=target))


def test_persisted_semantic_answers_use_the_same_authority_guard() -> None:
    with pytest.raises(ValidationError, match="authority"):
        SemanticAnswer(
            question_id="q_" + "a" * 24,
            question_digest=EVIDENCE_A,
            target_path="/semantic/certification/tier",
            value=True,
            evidence_refs=("#/unresolved_gaps/0",),
        )


def test_answer_application_rejects_unresolvable_evidence_references() -> None:
    evidence = _evidence()
    questions = build_open_questions(
        evidence_digest=evidence.bundle_digest,
        candidates=(_candidate(reference="#/unresolved_gaps/9"),),
    )
    answers = build_answer_set(
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(
            AnswerSubmission(
                question_id=questions.questions[0].question_id,
                value=5,
            ),
        ),
    )

    with pytest.raises(QuestionError, match="missing unresolved gap"):
        apply_answers(evidence, questions, answers)


def test_answer_domain_is_closed_and_rejects_unbounded_free_text() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        AnswerDomain(kind="string")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="requires only enum_values"):
        AnswerDomain(kind="enum", enum_values=())
    with pytest.raises(ValidationError, match="finite"):
        AnswerDomain(kind="number_range", minimum=0, maximum=float("inf"))

    values = ("review", "reject")
    ordered = tuple(
        value
        for _, value in sorted(
            (sha256_json({"value": value}), value) for value in values
        )
    )
    domain = AnswerDomain(kind="enum", enum_values=ordered)
    assert set(domain.enum_values) == {"review", "reject"}


def test_open_question_slot_cannot_contain_an_answer() -> None:
    question = build_open_question(_candidate())
    document = question.model_dump(mode="json")
    document["answer"] = 7

    with pytest.raises(ValidationError, match="Input should be None"):
        OpenQuestion.model_validate(document)


def test_artifact_rejects_tamper_stale_evidence_and_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    artifact = build_open_questions(
        evidence_digest=EVIDENCE_A,
        candidates=(_candidate(),),
    )
    path = write_open_questions(artifact, tmp_path / "open_questions.json")
    assert load_open_questions(path, evidence_digest=EVIDENCE_A) == artifact

    with pytest.raises(QuestionError, match="stale evidence"):
        load_open_questions(path, evidence_digest=EVIDENCE_B)

    document = artifact.model_dump(mode="json")
    document["questions"][0]["target_path"] = "/semantic/business_rules/other"
    with pytest.raises(ValidationError, match="question digest mismatch"):
        OpenQuestionsArtifact.model_validate(document)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"bfcl-open-questions-v1",'
        f'"evidence_digest":"{EVIDENCE_A}",'
        f'"evidence_digest":"{EVIDENCE_B}",'
        '"questions":[],"artifact_digest":"sha256:'
        + "0" * 64
        + '"}',
        encoding="utf-8",
    )
    with pytest.raises(QuestionError, match="repeats JSON key"):
        load_open_questions(duplicate, evidence_digest=EVIDENCE_A)


def test_question_bounds_and_reference_shape_fail_closed() -> None:
    with pytest.raises(ValidationError, match="between 1 and 16"):
        build_open_question(
            _candidate().model_copy(update={"evidence_refs": ()})
        )
    with pytest.raises(ValidationError, match="JSON pointers"):
        build_open_question(
            _candidate(reference="/identity/content_digest")
        )

    candidates = tuple(
        _candidate(
            target=f"/semantic/business_rules/rule_{index}",
            prompt=f"Choose rule {index}.",
            reference=f"#/unresolved_gaps/{index}",
        )
        for index in range(101)
    )
    with pytest.raises(ValidationError, match="exceeds 100"):
        build_open_questions(evidence_digest=EVIDENCE_A, candidates=candidates)


def test_question_artifact_bytes_are_canonical_json(tmp_path: Path) -> None:
    artifact = build_open_questions(
        evidence_digest=EVIDENCE_A,
        candidates=(_candidate(),),
    )
    path = write_open_questions(artifact, tmp_path / "questions.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["artifact_digest"] == artifact.artifact_digest
    assert path.read_bytes().endswith(b"\n")


def test_answers_create_a_new_evidence_revision_without_mutating_parent(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    original = evidence.model_dump(mode="json")
    questions = build_open_questions(
        evidence_digest=evidence.bundle_digest,
        candidates=(_candidate(),),
    )
    answers = build_answer_set(
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(
            AnswerSubmission(
                question_id=questions.questions[0].question_id,
                value=5,
            ),
        ),
    )

    revision = apply_answers(evidence, questions, answers)

    assert evidence.model_dump(mode="json") == original
    assert revision.evidence.bundle_digest != evidence.bundle_digest
    assert revision.evidence.revision is not None
    assert revision.evidence.revision.root_bundle_digest == evidence.bundle_digest
    assert revision.evidence.revision.parent_bundle_digest == evidence.bundle_digest
    assert revision.evidence.semantic_answers[0].value == 5
    assert revision.record.revised_bundle_digest == revision.evidence.bundle_digest
    revision_root = write_evidence_revision(revision, tmp_path / "revisions")
    assert (revision_root / "evidence.json").exists()
    assert (revision_root / "revision_record.json").exists()
    with pytest.raises(QuestionError, match="already exists"):
        write_evidence_revision(revision, tmp_path / "revisions")


def test_answer_application_rejects_stale_missing_extra_and_invalid_values() -> None:
    evidence = _evidence()
    questions = build_open_questions(
        evidence_digest=evidence.bundle_digest,
        candidates=(_candidate(),),
    )
    question_id = questions.questions[0].question_id

    stale = build_answer_set(
        evidence_digest=EVIDENCE_A,
        question_artifact_digest=questions.artifact_digest,
        answers=(AnswerSubmission(question_id=question_id, value=5),),
    )
    with pytest.raises(QuestionError, match="stale"):
        apply_answers(evidence, questions, stale)

    missing = build_answer_set(
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(),
    )
    with pytest.raises(QuestionError, match="must be exact"):
        apply_answers(evidence, questions, missing)

    extra = build_answer_set(
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(
            AnswerSubmission(question_id=question_id, value=5),
            AnswerSubmission(question_id="q_" + "f" * 24, value=5),
        ),
    )
    with pytest.raises(QuestionError, match="must be exact"):
        apply_answers(evidence, questions, extra)

    wrong_type = build_answer_set(
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(AnswerSubmission(question_id=question_id, value="five"),),
    )
    with pytest.raises(QuestionError, match="integer"):
        apply_answers(evidence, questions, wrong_type)

    outside_range = build_answer_set(
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(AnswerSubmission(question_id=question_id, value=99),),
    )
    with pytest.raises(QuestionError, match="outside its range"):
        apply_answers(evidence, questions, outside_range)

    with pytest.raises(ValidationError, match="sorted with unique"):
        build_answer_set(
            evidence_digest=evidence.bundle_digest,
            question_artifact_digest=questions.artifact_digest,
            answers=(
                AnswerSubmission(question_id=question_id, value=5),
                AnswerSubmission(question_id=question_id, value=6),
            ),
        )


def test_parent_approval_cannot_authorize_answered_revision(tmp_path: Path) -> None:
    evidence = _evidence()
    questions = build_open_questions(
        evidence_digest=evidence.bundle_digest,
        candidates=(_candidate(),),
    )
    answers = build_answer_set(
        evidence_digest=evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(
            AnswerSubmission(
                question_id=questions.questions[0].question_id,
                value=5,
            ),
        ),
    )
    revision = apply_answers(evidence, questions, answers)
    approval_path = write_canonical_json(
        {
            "approval_version": MIGRATION_APPROVAL_VERSION,
            "approved_by": "reviewer@example.test",
            "source_bundle_digest": evidence.bundle_digest,
            "normalized_bundle_digest": evidence.bundle_digest,
            "migration_record_digest": None,
            "acknowledged_warnings": [],
            "acknowledged_findings": [],
            "note": None,
        },
        tmp_path / "parent-approval.json",
    )

    with pytest.raises(EvidenceMigrationError, match="does not match evidence"):
        load_normalized_approval(
            approval_path,
            NormalizedSourceEvidence(
                source_digest=evidence.bundle_digest,
                evidence=revision.evidence,
                migration=None,
            ),
        )


def test_resume_rejects_tampered_and_replayed_answer_artifacts(
    tmp_path: Path,
) -> None:
    parent = _evidence()
    parent_path = write_canonical_json(
        parent.model_dump(mode="json"),
        tmp_path / "parent.json",
    )
    questions = build_open_questions(
        evidence_digest=parent.bundle_digest,
        candidates=(_candidate(),),
    )
    question_path = write_open_questions(questions, tmp_path / "questions.json")
    answers = build_answer_set(
        evidence_digest=parent.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(
            AnswerSubmission(
                question_id=questions.questions[0].question_id,
                value=5,
            ),
        ),
    )
    answer_path = write_answer_set(answers, tmp_path / "answers.json")
    revision = apply_answers(parent, questions, answers)

    verify_answered_revision(
        revision.evidence,
        parent_evidence_path=parent_path,
        open_questions_path=question_path,
        answer_set_path=answer_path,
    )

    tampered = json.loads(answer_path.read_text(encoding="utf-8"))
    tampered["answers"][0]["value"] = 6
    tampered_path = write_canonical_json(tampered, tmp_path / "tampered.json")
    with pytest.raises(QuestionError, match="answer set digest mismatch"):
        load_answer_set(
            tampered_path,
            evidence_digest=parent.bundle_digest,
            question_artifact_digest=questions.artifact_digest,
        )

    other_questions = build_open_questions(
        evidence_digest=parent.bundle_digest,
        candidates=(
            _candidate(
                target="/semantic/business_rules/renewal_limit",
                prompt="How many renewals are allowed?",
            ),
        ),
    )
    other_path = write_open_questions(
        other_questions,
        tmp_path / "other-questions.json",
    )
    with pytest.raises(QuestionError, match="different question artifact"):
        verify_answered_revision(
            revision.evidence,
            parent_evidence_path=parent_path,
            open_questions_path=other_path,
            answer_set_path=answer_path,
        )
