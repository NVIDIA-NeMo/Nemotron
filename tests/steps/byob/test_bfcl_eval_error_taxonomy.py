"""The error taxonomy is published in every artifact, so drift must fail here.

``eval_report.json`` and ``eval_manifest.json`` carry ``error_taxonomy_hash``.
A consumer reads that hash as a promise about which codes exist and how each one
is attributed, so a code added to the pipeline without a taxonomy entry would let
the hash claim a coverage it no longer has.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, get_args

import pytest
import tomllib

from nemotron.steps.byob.runtime.benchmark_families import bfcl as bfcl_family
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    EpisodeStatus,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    EXECUTABLE_EPISODE_ATTRIBUTION,
    FATAL_EVAL_ERROR_CODES,
    METRIC_NOT_APPLICABLE_CODES,
    REASON_CODE_NAMESPACES,
    TRACE_EPISODE_ATTRIBUTION,
    EvalFailureRecord,
    episode_attribution,
    episode_failure_record,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    ExecutableEpisodeStatus,
)

_FAMILY_ROOT = Path(bfcl_family.__file__).parent
# .../byob/runtime/benchmark_families/bfcl -> .../byob/bfcl/step.toml
_STEP_TOML = _FAMILY_ROOT.parents[2] / "bfcl" / "step.toml"
_TAXONOMY_FILE = "error_taxonomy.py"
_REASON_KEYWORDS = frozenset({"na_reason", "reason", "reason_code"})


def _sources() -> list[tuple[Path, ast.Module]]:
    return [
        (path, ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(_FAMILY_ROOT.rglob("*.py"))
    ]


def _declared_error_codes() -> dict[str, str]:
    """Map every exception class in the family to the code it reports."""
    codes: dict[str, str] = {}
    for _path, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                target = (
                    statement.targets[0]
                    if isinstance(statement, ast.Assign) and statement.targets
                    else statement.target
                    if isinstance(statement, ast.AnnAssign)
                    else None
                )
                value = statement.value if isinstance(statement, ast.Assign | ast.AnnAssign) else None
                if (
                    isinstance(target, ast.Name)
                    and target.id == "code"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    codes[node.name] = value.value
    return codes


def _emitted_reason_codes() -> set[str]:
    """Collect the literal reason codes the family can hand to a report.

    Most are passed by keyword. Metric N/A codes are also selected inline by a
    conditional expression, so every literal in that namespace counts too.
    """
    found: set[str] = set()
    for path, tree in _sources():
        if path.name == _TAXONOMY_FILE:
            continue
        for node in ast.walk(tree):
            keyworded = isinstance(node, ast.keyword) and node.arg in _REASON_KEYWORDS
            value = node.value if isinstance(node, ast.keyword) else node
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            if keyworded or value.value.startswith("metric."):
                found.add(value.value)
    return found


def _raised_class_names() -> set[str]:
    raised: set[str] = set()
    for _path, tree in _sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            called = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(called, ast.Name):
                raised.add(called.id)
    return raised


def test_taxonomy_covers_exactly_the_exception_codes_the_family_declares() -> None:
    declared = set(_declared_error_codes().values())
    assert declared == set(FATAL_EVAL_ERROR_CODES)


def test_every_directly_raised_error_code_has_a_step_registry_recovery() -> None:
    """A code that reaches an operator without recovery text is an unusable report."""
    codes = _declared_error_codes()
    raised = {codes[name] for name in _raised_class_names() if name in codes}
    registry = {
        entry["name"] for entry in tomllib.loads(_STEP_TOML.read_text(encoding="utf-8"))["errors"]
    }
    assert raised <= registry


def test_episode_attribution_covers_every_declared_episode_status() -> None:
    assert set(TRACE_EPISODE_ATTRIBUTION) == set(get_args(EpisodeStatus))
    assert set(EXECUTABLE_EPISODE_ATTRIBUTION) == set(get_args(ExecutableEpisodeStatus))
    for status in get_args(ExecutableEpisodeStatus):
        assert episode_attribution(status, executable=True) in {
            "success",
            "candidate",
            "infrastructure",
        }


def test_trace_statuses_keep_their_attribution_under_executable_evaluation() -> None:
    for status, attribution in TRACE_EPISODE_ATTRIBUTION.items():
        assert EXECUTABLE_EPISODE_ATTRIBUTION[status] == attribution


def test_metric_not_applicable_codes_match_the_ones_scoring_can_emit() -> None:
    emitted = {
        code for code in _emitted_reason_codes() if code.startswith("metric.")
    }
    assert emitted == set(METRIC_NOT_APPLICABLE_CODES)


def test_every_reason_code_namespace_the_family_emits_is_registered() -> None:
    namespaces = {
        code.partition(".")[0]
        for code in _emitted_reason_codes()
        if "." in code and code.replace(".", "").replace("_", "").isalnum()
    }
    assert namespaces
    assert namespaces <= set(REASON_CODE_NAMESPACES)


def test_episode_failure_record_attributes_a_terminal_and_ignores_success() -> None:
    assert episode_failure_record("completed", executable=True) is None
    record = episode_failure_record("oracle_timeout", executable=True)
    assert record is not None
    assert record.as_document() == {
        "layer": "episode",
        "code": "episode.oracle_timeout",
        "attribution": "infrastructure",
        "subject": "episode",
    }
    candidate = episode_failure_record("candidate_mismatch", executable=True)
    assert candidate is not None
    assert candidate.attribution == "candidate"


def test_an_unregistered_episode_status_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="unregistered episode status"):
        episode_attribution("oracle_reset_failed", executable=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"layer": "setup", "code": "not_a_registered_code", "attribution": "fatal_setup"},
        {"layer": "setup", "code": "eval_runner_invalid", "attribution": "infrastructure"},
        {"layer": "gate", "code": "  ", "attribution": "candidate"},
    ],
)
def test_a_failure_record_refuses_an_unattributable_shape(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        EvalFailureRecord(subject="eval.gate", **payload)
