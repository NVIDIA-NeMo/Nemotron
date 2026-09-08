from __future__ import annotations

import json
import sys
from pathlib import Path
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
from nemotron.steps.byob.runtime.source_adapters.evidence import load_source_evidence
from nemotron.steps.byob.scripts import bfcl_author
from tests.steps.byob.test_bfcl_authoring_questions import _candidate
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
        "approve",
        "draft",
        "review",
        "freeze",
        "publish",
        "purge-cache",
    ):
        assert command in help_text
    assert "Pre-model authorization" in help_text
    assert "final release approval" in help_text
    assert "CI mode never prompts" in help_text


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
        (
            "BFCL_ENABLE_LOCAL_PYTHON"
            if expected_adapter == "local_python"
            else "BFCL_ENABLE_HTTP_PACKAGE"
        ),
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
    assert (
        arguments[arguments.index("--held-out-not-applicable-reason") + 1]
        == "The catalog is public."
    )
    assert (
        arguments[arguments.index("--held-out-reviewed-by") + 1]
        == "reviewer@example.test"
    )
    assert "--resolved-authoring-config" in arguments
    assert (tmp_path / "workspace" / "resolved_authoring_config.json").is_file()
    assert not (tmp_path / "workspace" / ".locks" / "default" / "authoring.lock").read_text(
        encoding="utf-8"
    )


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
    assert exited.value.code == 2

    with pytest.raises(SystemExit) as reviewer_missing:
        _run_cli(
            monkeypatch,
            [
                *_author_argv(tmp_path, source, brief),
                "--held-out-not-applicable-reason",
                "The catalog is public.",
            ],
        )
    assert reviewer_missing.value.code == 2


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
