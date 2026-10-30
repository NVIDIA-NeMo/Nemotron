# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The line between proposing a filtering policy and approving one."""

from __future__ import annotations

import inspect

import pytest
import yaml

from nemotron.steps.curate.nemo_curator.runtime import policy

PROFILE_DIGEST = "sha256:" + "a" * 64
CORPUS_FINGERPRINT = "sha256:" + "b" * 64
LANGPACK_CONTENT_HASH = "sha256:" + "c" * 64


def _candidates(**overrides):
    document = policy.build_candidate_policies(
        candidates=[{"signal": "word_count", "bands": []}],
        profile_digest=PROFILE_DIGEST,
        signals_impl_version="curate-runtime-0.1.0",
        corpus={"glob": "./x/*.jsonl", "document_count": 10},
    )
    document.update(overrides)
    return document


def _approved(**overrides):
    document = {
        "schema_version": 1,
        "approved": True,
        "corpus": {"fingerprint": CORPUS_FINGERPRINT, "document_count": 10, "source_field": "source"},
        "signals_impl_version": "curate-runtime-0.1.0",
        "profile_digest": PROFILE_DIGEST,
        "approval": {
            "method": "manual",
            "approver": "someone",
            "date": "2026-08-24",
            "evidence": "reviewed 200 rejected documents",
        },
        "thresholds": [{"signal": "word_count", "min": 30, "max": 5000}],
    }
    document.update(overrides)
    return document


# -- candidates are never approved --------------------------------------------


def test_candidates_are_written_unapproved() -> None:
    assert _candidates()["approved"] is False


def test_approval_is_not_a_parameter_of_the_builder() -> None:
    """A caller must not be able to ask this function for an approved policy."""
    assert "approved" not in inspect.signature(policy.build_candidate_policies).parameters


def test_promote_is_the_only_function_that_can_mark_a_policy_approved() -> None:
    """The invariant, named rather than grepped.

    It used to be enforced by asserting the string ``"approved": True`` appeared
    nowhere in the module. That was a proxy: what matters is that *profiling*
    cannot produce an executable policy, not that the literal is absent.
    :func:`policy.promote` is the deliberate act itself, so it is the one place
    the literal belongs — and this test fails if a second place appears.
    """
    import ast
    from pathlib import Path

    # Scanned across the whole package, not just this module. Scoping it to
    # ``policy`` would let a *new* module — a promotion helper on a flow step,
    # say — write ``approved: True`` while skipping promote()'s checks, and this
    # guard would stay green while the property it names had been lost.
    root = Path(inspect.getfile(policy)).parents[1]
    approving: set[str] = set()
    for source_file in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(source_file.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a syntax error fails elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Dict) and any(
                    isinstance(k, ast.Constant)
                    and k.value == "approved"
                    and isinstance(v, ast.Constant)
                    and v.value is True
                    for k, v in zip(inner.keys, inner.values, strict=True)
                    if k is not None
                ):
                    approving.add(f"{source_file.relative_to(root)}::{node.name}")

    assert approving == {"runtime/policy.py::promote"}, (
        f"functions across steps/curate that can mark a policy approved: {sorted(approving)}; "
        "only runtime/policy.py::promote may, because only it checks the corpus fingerprint, "
        "the bound direction, and that the signal was actually profiled"
    )


def test_the_profiling_path_cannot_produce_an_approved_policy() -> None:
    """The property the string check was really guarding."""
    import inspect as _inspect

    assert "approved" not in _inspect.signature(policy.build_candidate_policies).parameters
    assert (
        policy.build_candidate_policies(
            candidates=[],
            profile_digest=PROFILE_DIGEST,
            signals_impl_version="v",
            corpus={"fingerprint": CORPUS_FINGERPRINT},
        )["approved"]
        is False
    )


def test_writing_a_policy_claiming_approval_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="refusing to write"):
        policy.write_candidate_policies(tmp_path / "p.yaml", _candidates(approved=True))


def test_candidates_carry_the_caveat_in_the_file(tmp_path) -> None:
    """Someone opening the YAML must see it is not a recommendation."""
    path = policy.write_candidate_policies(tmp_path / "p.yaml", _candidates())

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["approved"] is False
    assert "do not establish" in loaded["note"]


def test_candidates_link_back_to_the_profile_they_came_from() -> None:
    assert _candidates()["profile_digest"] == PROFILE_DIGEST


def test_the_digest_is_content_addressed() -> None:
    assert policy.digest({"a": 1, "b": 2}) == policy.digest({"b": 2, "a": 1})
    assert policy.digest({"a": 1}) != policy.digest({"a": 2})


# -- approved policy schema ---------------------------------------------------


def test_a_complete_approved_policy_validates() -> None:
    assert policy.validate_approved_policy(_approved()) == []


def test_an_unapproved_policy_does_not_validate_as_approved() -> None:
    problems = policy.validate_approved_policy(_approved(approved=False))

    assert any("approved must be true" in p for p in problems)


@pytest.mark.parametrize("field", ["corpus", "thresholds", "profile_digest"])
def test_every_required_field_is_enforced(field) -> None:
    document = _approved()
    document.pop(field)

    assert any(field in p for p in policy.validate_approved_policy(document))


def test_an_approval_must_say_how_it_was_reached() -> None:
    problems = policy.validate_approved_policy(_approved(approval={"method": "vibes"}))

    assert any("approval.method" in p for p in problems)


def test_an_approved_policy_must_name_someone_and_cite_evidence() -> None:
    """Checked here as well as in promote(), because a policy can be hand-written.

    Most of the gate is machine-checkable and stays that way -- the corpus
    fingerprint, the profile digest, the scorer version, the direction of every
    bound -- and none of it depends on a name. What a name is for is narrower:
    `approved: true` asserts a person decided, the manifest and the flow report
    repeat that assertion, and an assertion nobody stands behind should not be
    writable. `date` remains optional; it is provenance, not accountability.
    """
    problems = policy.validate_approved_policy(_approved(approval=None))

    assert any("approver" in p for p in problems)
    assert any("evidence" in p for p in problems)
    assert not any("approval.date" in p for p in problems), "date is provenance, not a gate"


def test_an_unapproved_policy_needs_no_signature() -> None:
    """The escape that keeps the requirement honest: not claiming approval costs
    nothing, so there is never a reason to write one nobody stands behind."""
    problems = policy.validate_approved_policy(_approved(approved=False, approval=None))

    assert not any("approver" in p or "evidence" in p for p in problems)


def test_an_approval_that_is_present_must_still_be_a_mapping() -> None:
    problems = policy.validate_approved_policy(_approved(approval="signed by me"))

    assert any("mapping" in p for p in problems)


def test_a_named_method_must_be_one_of_the_known_ones() -> None:
    """Optional, but a typo in a field that IS given should not pass silently."""
    problems = policy.validate_approved_policy(_approved(approval={"method": "vibes"}))

    assert any("approval.method" in p for p in problems)


def test_a_threshold_that_sets_no_bound_is_rejected() -> None:
    problems = policy.validate_approved_policy(_approved(thresholds=[{"signal": "word_count"}]))

    assert any("neither min nor max" in p for p in problems)


def test_a_policy_must_name_the_corpus_it_was_derived_from() -> None:
    """A filtering decision nobody can trace is one nobody can revisit."""
    problems = policy.validate_approved_policy(_approved(corpus={"document_count": 10}))

    assert any("fingerprint" in p for p in problems)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("signals_impl_version", None),
        ("signals_impl_version", ""),
        ("profile_digest", None),
        ("profile_digest", ""),
    ],
)
def test_required_provenance_values_cannot_be_empty(field, value) -> None:
    problems = policy.validate_approved_policy(_approved(**{field: value}))

    assert any(field in problem for problem in problems)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"profile_digest": "sha256:abc"}, "profile_digest"),
        ({"profile_digest": "sha256:" + "g" * 64}, "profile_digest"),
        ({"corpus": {"fingerprint": "sha256:def"}}, "corpus.fingerprint"),
    ],
)
def test_provenance_digests_must_be_full_sha256_values(overrides, expected) -> None:
    problems = policy.validate_approved_policy(_approved(**overrides))

    assert any(expected in problem for problem in problems)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "0.5", True])
def test_threshold_bounds_must_be_finite_numbers(value) -> None:
    problems = policy.validate_approved_policy(
        _approved(thresholds=[{"signal": "unicode_alpha_numeric", "max": value}])
    )

    assert any("finite number" in problem for problem in problems)


def test_interval_bounds_cannot_be_inverted() -> None:
    problems = policy.validate_approved_policy(
        _approved(thresholds=[{"signal": "word_count", "min": 5000, "max": 50}])
    )

    assert any("greater than max" in problem for problem in problems)


def test_a_non_mapping_is_rejected_without_raising() -> None:
    assert policy.validate_approved_policy("not a policy")


# -- the execution gate -------------------------------------------------------


def test_an_approved_policy_passes_the_gate() -> None:
    assert policy.require_approved(_approved()) == []


def test_an_unapproved_policy_is_refused_by_default() -> None:
    with pytest.raises(policy.PolicyNotApprovedError, match="not approved for execution"):
        policy.require_approved(_candidates())


def test_the_override_warns_and_names_what_is_missing() -> None:
    warnings = policy.require_approved(_candidates(), allow_unvalidated=True)

    assert warnings
    assert "allow_unvalidated_policy" in warnings[0]
    assert "approved must be true" in warnings[0]


# -- the schema the producer and consumer must actually share -----------------


def test_an_unknown_signal_name_is_a_schema_problem() -> None:
    """This function exists so producer and consumer check the same thing.

    A name that validates cleanly here and then fails at pipeline construction
    makes that claim false, and moves the failure to the machine that runs the
    job rather than the one that wrote the policy.
    """
    document = _approved(thresholds=[{"signal": "not_a_signal", "max": 0.5}])

    problems = policy.validate_approved_policy(document)

    assert any("unknown signal" in p for p in problems)
    assert any("not_a_signal" in p for p in problems)


def test_a_bound_that_would_invert_the_gate_is_a_schema_problem() -> None:
    """`max:` on a min-direction signal is a valid-looking document that inverts."""
    document = _approved(thresholds=[{"signal": "stopword_ratio", "max": 0.9}])

    problems = policy.validate_approved_policy(document)

    assert any("invert the gate" in p for p in problems)


def test_an_interval_signal_with_one_bound_is_a_schema_problem() -> None:
    document = _approved(thresholds=[{"signal": "word_count", "min": 50}])

    problems = policy.validate_approved_policy(document)

    assert any("gates from both sides" in p for p in problems)


def test_a_correctly_specified_policy_still_validates() -> None:
    document = _approved(
        thresholds=[
            {"signal": "unicode_alpha_numeric", "max": 0.3},
            {"signal": "stopword_ratio", "min": 0.1},
            {"signal": "word_count", "min": 50, "max": 100000},
        ]
    )

    assert policy.validate_approved_policy(document) == []


# -- promotion ----------------------------------------------------------------
#
# The step between curate/profile and curate/nemo_curator that did not exist:
# the profile emits bands, the filter needs min/max, and the corpus fingerprint
# the consumer requires was never produced. Promoting meant hand-writing a
# document and learning the schema by being rejected.


def _candidate_with_bands(**overrides):
    document = policy.build_candidate_policies(
        candidates=[
            {
                "signal": "unicode_alpha_numeric",
                "bands": [{"threshold_low": 0.25, "threshold_high": 0.40}],
                "note": "retention-stable range; not a recommendation",
            },
            {
                "signal": "stopword_ratio",
                "bands": [{"threshold_low": 0.05, "threshold_high": 0.20}],
                "note": "retention-stable range; not a recommendation",
            },
        ],
        profile_digest=PROFILE_DIGEST,
        signals_impl_version="curate-runtime-0.1.0",
        corpus={"glob": "./x/*.jsonl", "document_count": 10, "fingerprint": CORPUS_FINGERPRINT},
        langpack={"language_tag": "vi", "content_hash": LANGPACK_CONTENT_HASH},
    )
    document.update(overrides)
    return document


APPROVAL = {
    "method": "manual",
    "approver": "someone@example.test",
    "date": "2026-08-25",
    "evidence": "reviewed 200 rejected documents",
}


def test_promote_produces_an_executable_policy() -> None:
    document, _ = policy.promote(
        _candidate_with_bands(),
        thresholds=[{"signal": "unicode_alpha_numeric", "max": 0.30}],
        approval=APPROVAL,
    )

    assert document["approved"] is True
    assert policy.validate_approved_policy(document) == []


def test_promote_carries_provenance_forward() -> None:
    """A policy that cannot name its corpus, pack and profile cannot be audited."""
    document, _ = policy.promote(
        _candidate_with_bands(),
        thresholds=[{"signal": "unicode_alpha_numeric", "max": 0.30}],
        approval=APPROVAL,
    )

    assert document["corpus"]["fingerprint"] == CORPUS_FINGERPRINT
    assert document["langpack"]["content_hash"] == LANGPACK_CONTENT_HASH
    assert document["profile_digest"] == PROFILE_DIGEST
    assert document["signals_impl_version"] == "curate-runtime-0.1.0"


def test_promote_cannot_be_asked_for_an_approval_it_was_not_given() -> None:
    """``approval`` is keyword-only with no default, so there is no quiet path."""
    import inspect

    parameter = inspect.signature(policy.promote).parameters["approval"]

    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_promote_refuses_an_approval_nobody_is_named_on() -> None:
    """`approved: true` asserts a person decided.

    Written with nobody named and nothing cited, that assertion is untrue, and
    the manifest and the flow report repeat it downstream. This test used to
    assert the opposite -- that thresholds alone were enough -- which is what let
    the documentation and the code disagree about the same contract.
    """
    with pytest.raises(policy.PolicyNotPromotableError, match="approver"):
        policy.promote(
            _candidate_with_bands(),
            thresholds=[{"signal": "unicode_alpha_numeric", "max": 0.30}],
            approval={},
        )


def test_promote_accepts_an_approval_that_names_someone_and_cites_evidence() -> None:
    document, _ = policy.promote(
        _candidate_with_bands(),
        thresholds=[{"signal": "unicode_alpha_numeric", "max": 0.30}],
        approval={"approver": "reviewer@example.test", "evidence": "read 200 dropped documents"},
    )

    assert document["approved"] is True
    assert document["thresholds"][0]["signal"] == "unicode_alpha_numeric"


def test_promote_refuses_a_signal_the_profile_never_measured() -> None:
    """A threshold for an unprofiled signal is a number with no evidence."""
    with pytest.raises(policy.PolicyNotPromotableError, match="not profiled on this corpus"):
        policy.promote(
            _candidate_with_bands(),
            thresholds=[{"signal": "word_count", "min": 50, "max": 100000}],
            approval=APPROVAL,
        )


def test_promote_refuses_a_bound_that_would_invert_the_gate() -> None:
    with pytest.raises(policy.PolicyNotPromotableError, match="invert the gate"):
        policy.promote(
            _candidate_with_bands(),
            thresholds=[{"signal": "stopword_ratio", "max": 0.9}],
            approval=APPROVAL,
        )


def test_promote_refuses_non_finite_and_inverted_thresholds() -> None:
    with pytest.raises(policy.PolicyNotPromotableError, match="non-finite"):
        policy.promote(
            _candidate_with_bands(),
            thresholds=[{"signal": "unicode_alpha_numeric", "max": float("nan")}],
            approval=APPROVAL,
        )

    candidate = _candidate_with_bands()
    candidate["candidates"].append({"signal": "word_count", "bands": []})
    with pytest.raises(policy.PolicyNotPromotableError, match="greater than max"):
        policy.promote(
            candidate,
            thresholds=[{"signal": "word_count", "min": 5000, "max": 50}],
            approval=APPROVAL,
        )


def test_promote_refuses_to_re_approve_an_approved_policy() -> None:
    """Re-approving would overwrite one approval record with another."""
    document, _ = policy.promote(
        _candidate_with_bands(),
        thresholds=[{"signal": "unicode_alpha_numeric", "max": 0.30}],
        approval=APPROVAL,
    )

    with pytest.raises(policy.PolicyNotPromotableError, match="already approved"):
        policy.promote(document, thresholds=[{"signal": "x", "max": 1}], approval=APPROVAL)


def test_promote_refuses_a_policy_that_gates_nothing() -> None:
    with pytest.raises(policy.PolicyNotPromotableError, match="gates nothing"):
        policy.promote(_candidate_with_bands(), thresholds=[], approval=APPROVAL)


def test_a_threshold_outside_every_measured_band_warns() -> None:
    """Legal, but nobody measured what it removes."""
    _, warnings = policy.promote(
        _candidate_with_bands(),
        thresholds=[{"signal": "unicode_alpha_numeric", "max": 0.95}],
        approval=APPROVAL,
    )

    assert any("outside every retention-stable band" in w for w in warnings)


def test_a_threshold_between_swept_points_warns_that_it_was_not_measured() -> None:
    """The mistake this catches: quoting a neighbouring grid point's retention.

    The registry sweeps Grid(0.0, 1.0, 64), whose points are i/63, so a round
    number like 0.30 is never itself measured — and an approval record citing
    "retains 97.9%" for it is quoting the value at 0.2857 instead.
    """
    _, warnings = policy.promote(
        _candidate_with_bands(),
        thresholds=[{"signal": "unicode_alpha_numeric", "max": 0.30}],
        approval=APPROVAL,
    )

    assert any("not one of the thresholds the profile swept" in w for w in warnings)


def test_a_swept_threshold_inside_a_band_warns_about_nothing() -> None:
    """Otherwise every promotion warns and the warnings stop being read."""
    from nemotron.steps.curate.nemo_curator.runtime import registry as signal_registry

    swept = signal_registry.SIGNALS["unicode_alpha_numeric"].grid.values()
    inside = next(v for v in swept if 0.25 <= v <= 0.40)

    _, warnings = policy.promote(
        _candidate_with_bands(),
        thresholds=[{"signal": "unicode_alpha_numeric", "max": inside}],
        approval=APPROVAL,
    )

    assert warnings == []


def test_a_candidate_without_a_fingerprint_cannot_be_promoted() -> None:
    """Which is what forced curate/profile to start emitting one."""
    candidate = _candidate_with_bands()
    candidate["corpus"] = {"glob": "./x/*.jsonl", "document_count": 10}

    with pytest.raises(policy.PolicyNotPromotableError, match="fingerprint"):
        policy.promote(
            candidate,
            thresholds=[{"signal": "unicode_alpha_numeric", "max": 0.30}],
            approval=APPROVAL,
        )
