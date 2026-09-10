from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemotron.steps.byob.runtime.authoring_workflow.resume import AuthoringResumeError
from nemotron.steps.byob.runtime.pack_authoring.artifacts import write_canonical_json
from nemotron.steps.byob.runtime.pack_authoring.authorization import ExposureSubject
from nemotron.steps.byob.runtime.pack_authoring.questions import (
    AnswerSubmission,
    build_answer_set,
    build_open_questions,
    write_answer_set,
    write_open_questions,
)
from nemotron.steps.byob.runtime.source_adapters.evidence import (
    UnsignedSourceEvidence,
    build_source_evidence,
    load_source_evidence,
)
from nemotron.steps.byob.scripts import bfcl_author
from tests.steps.byob.test_bfcl_authoring_questions import _candidate, _evidence
from tests.steps.byob.test_bfcl_authoring_revisions import _committed_session

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


class _NoopResumed:
    lease = None

    def __enter__(self) -> _NoopResumed:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _write_head(workspace: Path, session_digest: str, phase: str) -> None:
    write_canonical_json(
        {
            "schema_version": "bfcl-authoring-head-v1",
            "tenant_id": "tenant-a",
            "run_id": "run-a",
            "phase": phase,
            "session_digest": session_digest,
        },
        workspace / "authoring_head.json",
    )


def test_help_lists_commands_and_separate_approval_boundaries() -> None:
    help_text = bfcl_author._parser().format_help()

    for command in (
        "author",
        "resume",
        "answer",
        "authorize",
        "apply-policy",
        "approve",
        "draft",
        "review",
        "release",
        "freeze",
        "publish",
        "purge-cache",
    ):
        assert command in help_text
    assert "Pre-model authorization" in help_text
    assert "final release approval" in help_text
    assert "CI mode never prompts" in help_text


def test_resume_parser_accepts_policy_transition() -> None:
    args = bfcl_author._parser().parse_args(
        [
            "resume",
            "--workspace",
            "/tmp/workspace",
            "--next",
            "apply_policy",
        ]
    )

    assert args.next_command == "apply_policy"


def test_release_signing_refuses_policy_changed_after_intake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "resolved_authoring_config.json").write_text("{}", encoding="utf-8")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("schema_version: bfcl-authoring-policy-v1\n", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "load_resolved_authoring_config",
        lambda _path: SimpleNamespace(
            resolved_paths=SimpleNamespace(policy=SimpleNamespace(value=str(policy_path))),
            inputs=SimpleNamespace(policy_digest=SimpleNamespace(value=DIGEST_A)),
        ),
    )
    monkeypatch.setattr(
        bfcl_author,
        "load_authoring_policy",
        lambda _path: (SimpleNamespace(release=None), DIGEST_B),
    )

    with pytest.raises(bfcl_author.GuidedCliError) as raised:
        bfcl_author._release_signing_arguments(
            SimpleNamespace(
                workspace=tmp_path,
                signing_key=None,
                signing_key_id=None,
                seal_issuer=None,
                seal_public_key=None,
            )
        )

    assert raised.value.code == "authoring_policy_changed"


def test_delegated_verdict_survives_progress_output_and_stays_visible(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _noisy(_module: str, _arguments: list[str]) -> None:
        print("stage prepare: 12 rows")
        print('{"nested": {"depth": 1}}')
        print(json.dumps({"status": "published", "run_id": "run-a"}))

    monkeypatch.setattr(bfcl_author, "_delegate", _noisy)

    document = bfcl_author._delegate_document("publish", [])

    assert document == {"status": "published", "run_id": "run-a"}
    captured = capsys.readouterr()
    assert "stage prepare: 12 rows" in captured.err
    assert captured.out == ""


def test_tee_forwards_each_progress_write_immediately() -> None:
    capture = io.StringIO()
    forward = io.StringIO()
    tee = bfcl_author._TeeTextIO(capture, forward)

    tee.write("stage prepare\n")

    assert capture.getvalue() == "stage prepare\n"
    assert forward.getvalue() == "stage prepare\n"


def test_delegated_document_refuses_output_without_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda _module, _arguments: print("stage prepare: 12 rows"),
    )

    with pytest.raises(bfcl_author.GuidedCliError) as raised:
        bfcl_author._delegate_document("publish", [])

    assert raised.value.code == "delegated_output_invalid"


def test_explicit_release_confirmation_replaces_the_interactive_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = tmp_path / "review_packet.json"
    packet.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "load_review_packet",
        lambda _path: bfcl_author.ReviewPacketV2({"risks": [], "record_digest": DIGEST_A}),
    )
    monkeypatch.setattr(
        bfcl_author,
        "derive_machine_checklist",
        lambda _packet: {name: True for name in bfcl_author.REQUIRED_CHECKLIST_V2},
    )
    monkeypatch.setattr(
        bfcl_author,
        "build_review_approval",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args: pytest.fail("an explicit confirmation must not prompt"),
    )
    gate = SimpleNamespace(
        load_state=lambda _digest: SimpleNamespace(
            bindings=SimpleNamespace(
                review_packet=SimpleNamespace(path="review_packet.json")
            )
        )
    )
    args = SimpleNamespace(
        workspace=tmp_path,
        approved_by="reviewer@example.test",
        confirm_reviewed_content=True,
        ci=False,
        note=None,
        approval_output=None,
    )

    arguments, approval_output = bfcl_author._streamlined_release_approval_arguments(
        args,
        gate,
        SimpleNamespace(verdict=SimpleNamespace(session_digest=DIGEST_B)),
    )

    assert approval_output == (tmp_path / "release_approval.json").resolve()
    assert arguments[:2] == ["--packet", str(packet.resolve())]
    assert "--guided-decision-sources" in arguments
    for name in bfcl_author.REQUIRED_CHECKLIST_V2:
        assert f"--accept-{name.replace('_', '-')}" in arguments


@pytest.mark.parametrize(
    ("marker", "expected_adapter", "expected_module"),
    [
        (
            "backend.py",
            "local_python",
            "nemotron.steps.byob.scripts.build_source_intake",
        ),
        (
            "endpoint_config.yaml",
            "http_package",
            "nemotron.steps.byob.scripts.build_source_intake",
        ),
    ],
)
def test_author_detects_conventional_adapter_and_delegates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    expected_adapter: str,
    expected_module: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / marker).write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate inventory lookup.", encoding="utf-8")
    observed: dict[str, Any] = {}

    def delegate(module: str, arguments: list[str]) -> None:
        observed.update(module=module, arguments=arguments)

    monkeypatch.setattr(bfcl_author, "_delegate", delegate)
    monkeypatch.setattr(bfcl_author, "_commit_intake_session", lambda **_kwargs: None)
    monkeypatch.setenv(
        ("BFCL_ENABLE_LOCAL_PYTHON" if expected_adapter == "local_python" else "BFCL_ENABLE_HTTP_PACKAGE"),
        "1",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "--ci",
            "author",
            "--workspace",
            str(tmp_path / "workspace"),
            "--source",
            source.as_uri(),
            "--brief",
            str(brief),
            "--pack-id",
            "inventory",
            "--pack-version",
            "1.0.0",
            "--held-out-not-applicable-reason",
            "The catalog is public.",
            "--held-out-reviewed-by",
            "reviewer@example.test",
        ],
    )
    bfcl_author.main()

    assert observed["module"] == expected_module
    arguments = observed["arguments"]
    assert arguments[arguments.index("--adapter") + 1] == expected_adapter
    assert arguments[arguments.index("--pack-id") + 1] == "inventory"
    assert arguments[arguments.index("--held-out-not-applicable-reason") + 1] == "The catalog is public."
    assert arguments[arguments.index("--held-out-reviewed-by") + 1] == "reviewer@example.test"
    assert "--resolved-authoring-config" in arguments
    assert (tmp_path / "workspace" / "resolved_authoring_config.json").is_file()
    assert not (tmp_path / "workspace" / ".locks" / "default" / "authoring.lock").read_text(encoding="utf-8")


def test_author_detects_mcp_and_delegates_existing_intake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mcp_intake.yaml"
    source.write_text("intake_version: test\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate MCP tools.", encoding="utf-8")
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(
            module=module,
            arguments=arguments,
        ),
    )
    monkeypatch.setattr(bfcl_author, "_commit_intake_session", lambda **_kwargs: None)
    monkeypatch.setenv("BFCL_ENABLE_MCP_MODE_A", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "author",
            "--workspace",
            str(tmp_path / "workspace"),
            "--source",
            str(source),
            "--brief",
            str(brief),
            "--pack-id",
            "mcp-tools",
            "--pack-version",
            "1.0.0",
            "--held-out-not-applicable-reason",
            "No reserved content in this catalog.",
            "--held-out-reviewed-by",
            "reviewer@example.test",
        ],
    )
    bfcl_author.main()

    assert observed["module"] == "nemotron.steps.byob.scripts.build_mcp_intake"
    assert "--intake" in observed["arguments"]
    assert "--domain-brief" in observed["arguments"]
    assert "--held-out-not-applicable-reason" in observed["arguments"]
    assert "--held-out-reviewed-by" in observed["arguments"]


def test_author_uses_reviewed_held_out_policy_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")
    policy = tmp_path / "authoring-policy.yaml"
    policy.write_text(
        "\n".join(
            (
                "schema_version: bfcl-authoring-policy-v1",
                "adapter_rollout:",
                "  local_python: true",
                "held_out:",
                "  not_applicable_reason: The catalog is public.",
                "  reviewed_by: policy-owner@example.test",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(
            module=module,
            arguments=arguments,
        ),
    )
    monkeypatch.setattr(bfcl_author, "_commit_intake_session", lambda **_kwargs: None)

    _run_cli(
        monkeypatch,
        [
            *_author_argv(tmp_path, source, brief),
            "--policy",
            str(policy),
        ],
    )

    arguments = observed["arguments"]
    assert arguments[arguments.index("--held-out-reviewed-by") + 1] == ("policy-owner@example.test")
    assert arguments[arguments.index("--held-out-not-applicable-reason") + 1] == ("The catalog is public.")


def test_author_refuses_policy_change_between_defaults_and_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("reviewed: true\n", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "load_authoring_policy",
        lambda _path: (SimpleNamespace(held_out=None), DIGEST_A),
    )
    monkeypatch.setattr(
        bfcl_author,
        "resolve_authoring_config",
        lambda **_kwargs: SimpleNamespace(inputs=SimpleNamespace(policy_digest=SimpleNamespace(value=DIGEST_B))),
    )
    args, remainder = bfcl_author._parser().parse_known_args(
        [
            "--ci",
            "author",
            "--workspace",
            str(tmp_path / "workspace"),
            "--source",
            str(source),
            "--brief",
            str(brief),
            "--policy",
            str(policy_path),
            "--pack-id",
            "inventory",
            "--pack-version",
            "1.0.0",
            "--held-out-not-applicable-reason",
            "Public catalog.",
            "--held-out-reviewed-by",
            "reviewer@example.test",
        ]
    )

    with pytest.raises(bfcl_author.GuidedCliError) as raised:
        bfcl_author._run_author(args, remainder)

    assert raised.value.code == "authoring_policy_changed_during_resolution"


def _author_argv(tmp_path: Path, source: Path, brief: Path) -> list[str]:
    return [
        "--ci",
        "author",
        "--workspace",
        str(tmp_path / "workspace"),
        "--source",
        str(source),
        "--brief",
        str(brief),
        "--pack-id",
        "inventory",
        "--pack-version",
        "1.0.0",
    ]


def test_author_refuses_to_start_without_a_held_out_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held-out status is settled before intake, so it cannot be an unnamed passthrough."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda *_args: pytest.fail("intake ran without a held-out decision"),
    )

    with pytest.raises(SystemExit) as exited:
        _run_cli(monkeypatch, _author_argv(tmp_path, source, brief))
    assert exited.value.code == 1

    with pytest.raises(SystemExit) as reviewer_missing:
        _run_cli(
            monkeypatch,
            [
                *_author_argv(tmp_path, source, brief),
                "--held-out-not-applicable-reason",
                "The catalog is public.",
            ],
        )
    assert reviewer_missing.value.code == 1


def test_author_refuses_held_out_content_without_its_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")
    content = tmp_path / "reserved.yaml"
    content.write_text("terms: [secret]\n", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda *_args: pytest.fail("intake ran with unbound held-out content"),
    )

    with pytest.raises(SystemExit) as exited:
        _run_cli(
            monkeypatch,
            [
                *_author_argv(tmp_path, source, brief),
                "--held-out-not-applicable-reason",
                "The catalog is public.",
                "--held-out-reviewed-by",
                "reviewer@example.test",
                "--held-out-content",
                str(content),
            ],
        )

    assert exited.value.code == 1
    assert json.loads(capsys.readouterr().err)["code"] == "held_out_content_unbound"


def test_evidence_approval_is_distinct_and_digest_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace, _, session_digest, paths = _committed_session(
        tmp_path,
        phase="intake_complete",
    )
    _write_head(workspace, session_digest, "intake_complete")
    evidence_digest = load_source_evidence(paths["evidence"]).bundle_digest
    subject = write_canonical_json(
        ExposureSubject(
            evidence_digest=evidence_digest,
            domain_brief_content_digest=DIGEST_A,
            domain_brief_source_digest=DIGEST_A,
            domain_brief_redaction_report_digest=DIGEST_A,
            held_out_decision_digest=DIGEST_A,
            held_out_policy_digest=None,
            held_out_redaction_report_digest=DIGEST_A,
        ).model_dump(mode="json"),
        workspace / "subject.json",
    )
    output = workspace / "evidence_approval.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "authorize",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--subject",
            str(subject),
            "--authorized-by",
            "reviewer@example.test",
        ],
    )
    bfcl_author.main()
    capsys.readouterr()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "approve",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--boundary",
            "evidence",
            "--approved-by",
            "reviewer@example.test",
            "--source-bundle-digest",
            DIGEST_A,
            "--normalized-bundle-digest",
            evidence_digest,
            "--output",
            str(output),
        ],
    )
    bfcl_author.main()

    approval = json.loads(output.read_text(encoding="utf-8"))
    result = json.loads(capsys.readouterr().out)
    assert approval["source_bundle_digest"] == DIGEST_A
    assert approval["normalized_bundle_digest"] == evidence_digest
    assert result["status"] == "evidence_approved"
    assert "cannot substitute for final release approval" in result["note"]


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["bfcl_author.py", *argv])
    bfcl_author.main()


def test_answer_commits_a_revision_that_must_be_reauthorized_and_reapproved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One correction must travel the guided CLI as a new content-addressed revision."""
    workspace, gate, parent_digest, paths = _committed_session(
        tmp_path,
        phase="intake_complete",
        with_approval=False,
    )
    _write_head(workspace, parent_digest, "intake_complete")
    parent_evidence = load_source_evidence(paths["evidence"])
    questions = build_open_questions(
        evidence_digest=parent_evidence.bundle_digest,
        candidates=(_candidate(),),
    )
    answers = build_answer_set(
        evidence_digest=parent_evidence.bundle_digest,
        question_artifact_digest=questions.artifact_digest,
        answers=(
            AnswerSubmission(
                question_id=questions.questions[0].question_id,
                value=5,
            ),
        ),
    )
    questions_path = write_open_questions(questions, workspace / "open_questions.json")
    answers_path = write_answer_set(answers, workspace / "answer_set.json")
    workspace_argv = [
        "--workspace",
        str(workspace),
        "--tenant-id",
        "tenant-a",
        "--run-id",
        "run-a",
    ]

    _run_cli(
        monkeypatch,
        [
            "answer",
            *workspace_argv,
            "--evidence",
            str(paths["evidence"]),
            "--questions",
            str(questions_path),
            "--answers",
            str(answers_path),
        ],
    )
    revised = json.loads(capsys.readouterr().out)

    assert revised["status"] == "evidence_revised"
    revised_digest = revised["evidence_digest"]
    assert revised_digest != parent_evidence.bundle_digest
    revision_root = Path(revised["revision"])
    assert revision_root.parent == (workspace / "revisions").resolve()
    assert revision_root.name == revised_digest.removeprefix("sha256:")
    assert (revision_root / "revision_record.json").is_file()

    head = json.loads((workspace / "authoring_head.json").read_text(encoding="utf-8"))
    assert head["phase"] == "evidence_revised"
    revised_session = head["session_digest"]
    assert revised_session != parent_digest

    # The correction withdraws the right to draft until the boundaries are redone.
    with pytest.raises(AuthoringResumeError) as blocked:
        gate.open(revised_session, command="draft")
    assert blocked.value.code == "resume_command_not_permitted"

    subject = write_canonical_json(
        ExposureSubject(
            evidence_digest=revised_digest,
            domain_brief_content_digest=DIGEST_A,
            domain_brief_source_digest=DIGEST_A,
            domain_brief_redaction_report_digest=DIGEST_A,
            held_out_decision_digest=DIGEST_A,
            held_out_policy_digest=None,
            held_out_redaction_report_digest=DIGEST_A,
        ).model_dump(mode="json"),
        workspace / "revised_subject.json",
    )
    _run_cli(
        monkeypatch,
        [
            "authorize",
            *workspace_argv,
            "--subject",
            str(subject),
            "--authorized-by",
            "reviewer@example.test",
        ],
    )
    capsys.readouterr()
    _run_cli(
        monkeypatch,
        [
            "approve",
            *workspace_argv,
            "--boundary",
            "evidence",
            "--approved-by",
            "reviewer@example.test",
            "--source-bundle-digest",
            DIGEST_A,
            "--normalized-bundle-digest",
            revised_digest,
            "--output",
            str(workspace / "revised_approval.json"),
        ],
    )
    capsys.readouterr()

    head = json.loads((workspace / "authoring_head.json").read_text(encoding="utf-8"))
    assert head["phase"] == "evidence_approved"
    resumed = gate.open(head["session_digest"], command="draft")
    try:
        assert resumed.verdict.permitted_commands == ("draft",)
    finally:
        resumed.lease.release()


def test_pre_model_authorization_is_a_separate_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace, _, session_digest, _ = _committed_session(
        tmp_path,
        phase="intake_complete",
    )
    _write_head(workspace, session_digest, "intake_complete")
    subject_path = write_canonical_json(
        ExposureSubject(
            evidence_digest=DIGEST_A,
            domain_brief_content_digest=DIGEST_A,
            domain_brief_source_digest=DIGEST_A,
            domain_brief_redaction_report_digest=DIGEST_A,
            held_out_decision_digest=DIGEST_A,
            held_out_policy_digest=None,
            held_out_redaction_report_digest=DIGEST_A,
        ).model_dump(mode="json"),
        workspace / "subject.json",
    )
    output = workspace / "authorization.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "authorize",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
            "--subject",
            str(subject_path),
            "--authorized-by",
            "reviewer@example.test",
            "--output",
            str(output),
        ],
    )
    bfcl_author.main()

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "model_exposure_authorized"
    assert result["note"] == "This is not final release approval."
    assert output.is_file()


def test_apply_policy_replaces_both_per_run_pre_model_approvals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_evidence = _evidence()
    unsigned = source_evidence.model_dump(exclude={"bundle_digest"})
    unsigned["unresolved_gaps"] = []
    clean_evidence = build_source_evidence(UnsignedSourceEvidence.model_validate(unsigned))
    workspace, gate, session_digest, paths = _committed_session(
        tmp_path,
        phase="intake_complete",
        evidence=clean_evidence,
        with_approval=False,
    )
    _write_head(workspace, session_digest, "intake_complete")
    evidence = load_source_evidence(paths["evidence"])
    subject = ExposureSubject(
        evidence_digest=evidence.bundle_digest,
        resolved_authoring_config_digest=DIGEST_A,
        domain_brief_content_digest=DIGEST_A,
        domain_brief_source_digest=DIGEST_A,
        domain_brief_redaction_report_digest=DIGEST_A,
        held_out_decision_digest=DIGEST_A,
        held_out_policy_digest=None,
        held_out_redaction_report_digest=DIGEST_A,
    )
    write_canonical_json(
        subject.model_dump(mode="json"),
        workspace / "model_exposure_subject.json",
    )
    write_canonical_json(
        {"advisory": []},
        workspace / "domain_brief_redaction.json",
    )
    write_canonical_json({}, workspace / "held_out_redaction.json")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("reviewed: true\n", encoding="utf-8")
    policy = SimpleNamespace(
        pre_model=SimpleNamespace(
            exposure_authorization="organizational_policy",
            clean_evidence_approval="organizational_policy",
        )
    )
    resolved = SimpleNamespace(
        resolved_paths=SimpleNamespace(policy=SimpleNamespace(value=str(policy_path))),
        inputs=SimpleNamespace(policy_digest=SimpleNamespace(value=DIGEST_A)),
        resolved_authoring_config_digest=DIGEST_A,
    )
    monkeypatch.setattr(bfcl_author, "load_resolved_authoring_config", lambda _path: resolved)
    monkeypatch.setattr(
        bfcl_author,
        "load_authoring_policy",
        lambda _path: (policy, DIGEST_A),
    )
    monkeypatch.setattr(
        bfcl_author.DomainBriefRedactionReport,
        "model_validate_json",
        lambda _document: SimpleNamespace(advisory=()),
    )
    monkeypatch.setattr(
        bfcl_author.HeldOutRedactionReport,
        "model_validate_json",
        lambda _document: SimpleNamespace(),
    )
    monkeypatch.setattr(
        bfcl_author,
        "build_exposure_subject",
        lambda *_args, **_kwargs: subject,
    )

    _run_cli(
        monkeypatch,
        [
            "apply-policy",
            "--workspace",
            str(workspace),
            "--tenant-id",
            "tenant-a",
            "--run-id",
            "run-a",
        ],
    )

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "pre_model_policy_applied"
    assert "No per-run human approval" in result["note"]
    authorization = json.loads((workspace / "exposure_authorization.json").read_text(encoding="utf-8"))
    approval = json.loads((workspace / "evidence_approval.json").read_text(encoding="utf-8"))
    assert authorization["mode"] == "organizational_policy"
    assert authorization["organizational_policy_digest"] == DIGEST_A
    assert approval["mode"] == "organizational_policy"
    assert approval["organizational_policy_digest"] == DIGEST_A
    assert "approved_by" not in approval
    head = json.loads((workspace / "authoring_head.json").read_text(encoding="utf-8"))
    assert head["phase"] == "evidence_approved"
    approved_state = gate.load_state(head["session_digest"])
    assert approved_state.parent_session_digest is not None
    authorized_state = gate.load_state(approved_state.parent_session_digest)
    assert authorized_state.phase == "exposure_authorized"
    assert authorized_state.parent_session_digest == session_digest
    resumed = gate.open(head["session_digest"], command="draft")
    resumed.lease.release()


def test_freeze_delegates_to_adapter_neutral_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    output = tmp_path / "workspace" / "release"
    output.mkdir(parents=True)
    write_canonical_json({}, output / "freeze_manifest.json")
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(
            module=module,
            arguments=arguments,
        ),
    )
    monkeypatch.setattr(
        bfcl_author,
        "_current_session",
        lambda *_args: (object(), _NoopResumed()),
    )
    monkeypatch.setattr(bfcl_author, "_commit_transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "freeze",
            "--workspace",
            str(tmp_path / "workspace"),
            "--freeze-inputs",
            "freeze_inputs.json",
            "--output",
            str(output),
        ],
    )
    bfcl_author.main()

    assert observed["module"] == "nemotron.steps.byob.scripts.freeze_authoring_pack"
    assert observed["arguments"] == [
        "--freeze-inputs",
        "freeze_inputs.json",
        "--output",
        str(output),
    ]


def test_review_dispatches_local_adapter_to_generalized_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    output = tmp_path / "workspace" / "review.json"
    output.parent.mkdir(parents=True)
    write_canonical_json({}, output)
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda module, arguments: observed.update(
            module=module,
            arguments=arguments,
        ),
    )
    monkeypatch.setattr(
        bfcl_author,
        "_current_session",
        lambda *_args: (object(), _NoopResumed()),
    )
    monkeypatch.setattr(bfcl_author, "_commit_transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "review",
            "--workspace",
            str(tmp_path / "workspace"),
            "--adapter-kind",
            "local_python",
            "--pack",
            "candidate",
            "--output",
            str(output),
        ],
    )
    bfcl_author.main()

    assert observed["module"] == "nemotron.steps.byob.scripts.build_authoring_review"
    assert observed["arguments"] == [
        "--adapter-kind",
        "local_python",
        "--pack",
        "candidate",
        "--output",
        str(output),
    ]


def test_ci_mode_never_prompts_for_missing_required_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: pytest.fail("CI mode attempted to prompt"),
    )
    monkeypatch.setattr(sys, "argv", ["bfcl_author.py", "--ci", "author"])

    with pytest.raises(SystemExit) as exited:
        bfcl_author.main()
    assert exited.value.code == 2


def test_author_fails_before_adapter_when_rollout_is_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for variable in (
        "BFCL_ENABLE_LOCAL_PYTHON",
        "BFCL_ENABLE_HTTP_PACKAGE",
        "BFCL_ENABLE_MCP_MODE_A",
        "BFCL_ENABLE_EXPERIMENTAL_MCP",
    ):
        monkeypatch.delenv(variable, raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "backend.py").write_text("# reviewed\n", encoding="utf-8")
    brief = tmp_path / "brief.txt"
    brief.write_text("Evaluate tools.", encoding="utf-8")
    monkeypatch.setattr(
        bfcl_author,
        "_delegate",
        lambda *_args: pytest.fail("disabled adapter was invoked"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bfcl_author.py",
            "--ci",
            "author",
            "--workspace",
            str(tmp_path / "workspace"),
            "--source",
            str(source),
            "--brief",
            str(brief),
            "--pack-id",
            "tools",
            "--pack-version",
            "1.0.0",
            "--held-out-not-applicable-reason",
            "No reserved content.",
            "--held-out-reviewed-by",
            "reviewer@example.test",
        ],
    )

    with pytest.raises(SystemExit) as exited:
        bfcl_author.main()

    result = json.loads(capsys.readouterr().err)
    assert exited.value.code == 1
    assert result["code"] == "adapter_rollout_disabled"
