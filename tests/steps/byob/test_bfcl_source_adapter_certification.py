from __future__ import annotations

from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json
from nemotron.steps.byob.runtime.source_adapters.certification import (
    PUBLISHED_CERTIFICATION_PROFILES,
    AdapterCertificationReport,
    AdapterProbeObservation,
    AdapterTier,
    CertificationAuthority,
    CertificationError,
    CertificationProbe,
    CertificationProfile,
    CertificationRefusalCode,
    ProbeExecutionRecord,
    ProbeOutcome,
    build_certification_report,
    certification_input_digest,
    certification_profile_for,
    certification_reference,
    derive_attained_tier,
    http_package_reference_profile,
    local_python_reference_profile,
    mcp_reference_profile,
    project_mcp_probe_report,
    project_probe_executions,
    verify_certification_report,
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

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
AUTHORITY = CertificationAuthority(
    key_id="test-root",
    private_key=Ed25519PrivateKey.from_private_bytes(b"\x01" * 32),
)
TRUSTED_KEYS = {AUTHORITY.key_id: AUTHORITY.public_key}


def _descriptor(*, kind: str = "mcp_mode_a") -> AdapterDescriptor:
    return AdapterDescriptor(
        contract_version="bfcl-source-adapter-v1",
        kind=kind,
        implementation_name="bfcl.mcp_mode_a",
        implementation_version="1.0.0",
        capabilities=(
            AdapterCapability.DESCRIBE_STATE,
            AdapterCapability.DESCRIBE_TOOLS,
            AdapterCapability.GET_STATE,
            AdapterCapability.OBSERVE,
            AdapterCapability.PIN_IDENTITY,
            AdapterCapability.RESET_STATE,
        ),
        fixture_access=FixtureAccessPolicy(
            kind=FixtureAccessKind.PUSHED,
            supports_redaction=True,
        ),
        probe_safety=ProbeSafetyPolicy(
            kind=ProbeSafetyKind.RESET_ISOLATED,
            max_calls=20,
            timeout_s=5.0,
        ),
        cleanup=CleanupSemantics(kind=CleanupKind.SESSION, timeout_s=5.0),
    )


def _passing_outcomes() -> tuple[ProbeOutcome, ...]:
    input_digest = certification_input_digest(
        _descriptor(),
        source_identity_digest=SHA_B,
        profile=mcp_reference_profile(),
    )
    return tuple(
        ProbeOutcome(
            probe=probe,
            status="pass",
            input_digest=input_digest,
            evidence_digest=sha256_json({"probe_index": index}),
            evidence={"probe_index": index},
        )
        for index, probe in enumerate(CertificationProbe)
    )


def _replace_outcome(
    outcomes: tuple[ProbeOutcome, ...],
    probe: CertificationProbe,
    **changes: Any,
) -> tuple[ProbeOutcome, ...]:
    return tuple(
        ProbeOutcome.model_validate(
            {
                **item.model_dump(mode="json"),
                **changes,
            }
        )
        if item.probe is probe
        else item
        for item in outcomes
    )


def _mcp_report(
    *,
    overrides: dict[str, tuple[str, str | None]] | None = None,
    missing: set[str] | None = None,
) -> dict[str, Any]:
    changes = overrides or {}
    omitted = missing or set()
    probes = []
    for index in range(1, 12):
        identifier = f"P{index}"
        if identifier in omitted:
            continue
        status, reason = changes.get(identifier, ("pass", None))
        probes.append(
            {
                "id": identifier,
                "requirement": ("conditional" if identifier in {"P7", "P8"} else "required"),
                "status": status,
                "reason": reason,
            }
        )
    return {"probes": probes}


def test_published_profiles_are_complete_bounded_and_transport_specific() -> None:
    assert tuple(PUBLISHED_CERTIFICATION_PROFILES) == (
        "http_package",
        "local_python",
        "mcp_mode_a",
    )
    profiles = (
        http_package_reference_profile(),
        local_python_reference_profile(),
        mcp_reference_profile(),
    )
    assert len({profile.digest for profile in profiles}) == 3
    assert [profile.adapter_kinds for profile in profiles] == [
        ("http_package",),
        ("local_python",),
        ("mcp_mode_a",),
    ]
    assert certification_profile_for("http_package") is profiles[0]
    for profile in profiles:
        assert profile.owner == "bfcl"
        assert sum(item.execution.max_calls for item in profile.probes) <= (profile.max_total_calls)
        assert sum(item.execution.timeout_s for item in profile.probes) <= (profile.max_wall_time_s)
        for requirement in profile.probes:
            assert requirement.execution.executor == "bfcl"
            assert requirement.execution.evidence_issuer == ("bfcl-source-adapter-verifier-v1")
            assert requirement.execution.input_binding == ("bfcl-adapter-probe-input-v1")
            assert requirement.execution.outcome_schema == ("bfcl-adapter-probe-outcome-v1")
            assert requirement.execution.max_calls > 0
            assert requirement.execution.timeout_s > 0
            assert requirement.allowed_failure_reasons

    assert {
        requirement.probe: requirement.execution.cleanup for requirement in local_python_reference_profile().probes
    } == {
        probe: (
            CleanupKind.NONE
            if probe
            in {
                CertificationProbe.IDENTITY_INTEGRITY,
                CertificationProbe.CATALOG_INTEGRITY,
            }
            else CleanupKind.PROCESS
        )
        for probe in CertificationProbe
    }
    assert {
        requirement.probe: requirement.execution.cleanup for requirement in http_package_reference_profile().probes
    } == {probe: CleanupKind.SESSION for probe in CertificationProbe}


def test_per_case_local_probes_are_bounded_by_their_own_call_budget() -> None:
    profile = local_python_reference_profile()
    policies = {requirement.probe: requirement.execution for requirement in profile.probes}

    # These two open one isolated episode per reviewed case, so a source is bounded by
    # the size of its own catalogue rather than by a deadline sized for a single call.
    for probe in (
        CertificationProbe.EXECUTABLE_OBSERVATION,
        CertificationProbe.STRUCTURED_ERROR_SHAPE,
    ):
        assert policies[probe].timeout_s == policies[probe].max_calls * 4.0
        # Cleanup is one operation however many episodes ran.
        assert policies[probe].cleanup_timeout_s == 10.0

    other = set(CertificationProbe) - {
        CertificationProbe.EXECUTABLE_OBSERVATION,
        CertificationProbe.STRUCTURED_ERROR_SHAPE,
    }
    assert {policies[probe].timeout_s for probe in other} == {10.0}


def test_profile_budget_and_registry_fail_closed() -> None:
    document = local_python_reference_profile().model_dump(mode="json")
    document["max_total_calls"] = 1
    with pytest.raises(ValidationError, match="call budgets exceed"):
        CertificationProfile.model_validate(document)

    with pytest.raises(CertificationError, match="no published") as error:
        certification_profile_for("unknown_adapter")
    assert error.value.code == CertificationRefusalCode.PROFILE_MISMATCH.value


def test_adapter_observations_cannot_claim_authority_and_bfcl_binds_outcomes() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AdapterProbeObservation.model_validate(
            {
                "probe": "identity_integrity",
                "status": "pass",
                "evidence": {"observed": True},
                "attained_tier": "A2",
                "input_digest": SHA_A,
                "issuer": "adapter",
            }
        )

    records = tuple(
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=probe,
                status="pass",
                evidence={"probe": probe.value, "observed": True},
            ),
            observed_calls=1,
            elapsed_s=0.1,
            cleanup_status=(
                "not_required"
                if probe
                in {
                    CertificationProbe.IDENTITY_INTEGRITY,
                    CertificationProbe.CATALOG_INTEGRITY,
                }
                else "passed"
            ),
        )
        for probe in CertificationProbe
    )
    outcomes = project_probe_executions(
        local_python_reference_profile(),
        records,
        input_digest=SHA_A,
    )

    assert derive_attained_tier(local_python_reference_profile(), outcomes) is (AdapterTier.A2)
    assert all(outcome.input_digest == SHA_A for outcome in outcomes)
    assert all(outcome.evidence_digest == sha256_json(outcome.evidence) for outcome in outcomes)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"observed_calls": 2}, CertificationRefusalCode.PROBE_UNSAFE),
        ({"cleanup_status": "failed"}, CertificationRefusalCode.CLEANUP_FAILED),
    ],
)
def test_probe_projection_enforces_bfcl_budget_and_cleanup(
    changes: dict[str, Any],
    expected: CertificationRefusalCode,
) -> None:
    records = [
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=probe,
                status="pass",
                evidence={"probe": probe.value},
            ),
            observed_calls=1,
            elapsed_s=0.1,
            cleanup_status=(
                "not_required"
                if probe
                in {
                    CertificationProbe.IDENTITY_INTEGRITY,
                    CertificationProbe.CATALOG_INTEGRITY,
                }
                else "passed"
            ),
        )
        for probe in CertificationProbe
    ]
    records[0] = records[0].model_copy(update=changes)

    outcomes = project_probe_executions(
        local_python_reference_profile(),
        records,
        input_digest=SHA_A,
    )

    assert outcomes[0].status == "fail"
    assert outcomes[0].reason == expected.value
    assert derive_attained_tier(local_python_reference_profile(), outcomes) is (AdapterTier.NONE)


def test_host_elapsed_time_is_diagnostic_not_certification_identity() -> None:
    records = [
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=probe,
                status="pass",
                evidence={"probe": probe.value},
            ),
            observed_calls=1,
            elapsed_s=0.1,
            cleanup_status=(
                "not_required"
                if probe
                in {
                    CertificationProbe.IDENTITY_INTEGRITY,
                    CertificationProbe.CATALOG_INTEGRITY,
                }
                else "passed"
            ),
        )
        for probe in CertificationProbe
    ]
    loaded_host = [record.model_copy(update={"elapsed_s": 999.0}) for record in records]

    expected = project_probe_executions(
        local_python_reference_profile(),
        records,
        input_digest=SHA_A,
    )
    observed = project_probe_executions(
        local_python_reference_profile(),
        loaded_host,
        input_digest=SHA_A,
    )

    assert observed == expected
    assert derive_attained_tier(local_python_reference_profile(), observed) is AdapterTier.A2


def test_missing_or_unprofiled_raw_observation_uses_stable_refusal_codes() -> None:
    records = tuple(
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=probe,
                status="pass",
                evidence={"probe": probe.value},
            ),
            observed_calls=1,
            elapsed_s=0.1,
            cleanup_status="passed",
        )
        for probe in CertificationProbe
        if probe is not CertificationProbe.RESULT_SHAPE_COVERAGE
    )
    outcomes = project_probe_executions(
        http_package_reference_profile(),
        records,
        input_digest=SHA_A,
    )
    missing = outcomes[-1]
    assert missing.reason == CertificationRefusalCode.PROBE_MISSING.value
    assert derive_attained_tier(http_package_reference_profile(), outcomes) is (AdapterTier.A1)

    invalid = (
        ProbeExecutionRecord(
            observation=AdapterProbeObservation(
                probe=CertificationProbe.IDENTITY_INTEGRITY,
                status="fail",
                reason="adapter_says_so",
            ),
            observed_calls=1,
            elapsed_s=0.1,
            cleanup_status="not_required",
        ),
    )
    with pytest.raises(CertificationError, match="unknown failure reason") as error:
        project_probe_executions(
            local_python_reference_profile(),
            invalid,
            input_digest=SHA_A,
        )
    assert error.value.code == CertificationRefusalCode.PROBE_EVIDENCE_INVALID.value


def test_bfcl_builds_and_rebinds_a_deterministic_a2_report() -> None:
    descriptor = _descriptor()
    profile = mcp_reference_profile()

    report = build_certification_report(
        descriptor,
        source_identity_digest=SHA_B,
        profile=profile,
        outcomes=_passing_outcomes(),
        authority=AUTHORITY,
    )
    rebuilt = build_certification_report(
        descriptor,
        source_identity_digest=SHA_B,
        profile=profile,
        outcomes=_passing_outcomes(),
        authority=AUTHORITY,
    )

    assert report == rebuilt
    # Pin the serialized contract independently of the production digest helper.
    # Recomputing the expectation with sha256_json(report) would only prove that
    # the implementation agrees with itself.
    assert (
        report.report_digest
        == "sha256:385ec7d931a0ab6e53f7942e3d2ffd9a4775f4c14c6cdb9fbfee35324eb0f21d"
    )
    assert report.attained_tier is AdapterTier.A2
    verify_certification_report(
        report,
        descriptor=descriptor,
        source_identity_digest=SHA_B,
        profile=profile,
        required_tier=AdapterTier.A2,
        trusted_public_keys=TRUSTED_KEYS,
    )
    reference = certification_reference(report)
    assert reference.report_digest == report.report_digest
    assert reference.descriptor_digest == report.descriptor_digest
    assert reference.attained_tier == "A2"


def test_tier_is_derived_from_evidence_and_cannot_be_raised_by_intent() -> None:
    outcomes = _replace_outcome(
        _passing_outcomes(),
        CertificationProbe.RESET_DETERMINISM,
        status="fail",
        evidence_digest=None,
        evidence=None,
        reason="reset_nondeterministic",
    )
    profile = mcp_reference_profile()
    report = build_certification_report(
        _descriptor(),
        source_identity_digest=SHA_B,
        profile=profile,
        outcomes=outcomes,
        authority=AUTHORITY,
    )

    assert report.attained_tier is AdapterTier.A1
    with pytest.raises(CertificationError, match="below required A2") as error:
        verify_certification_report(
            report,
            descriptor=_descriptor(),
            source_identity_digest=SHA_B,
            profile=profile,
            required_tier=AdapterTier.A2,
            trusted_public_keys=TRUSTED_KEYS,
        )
    assert error.value.code == CertificationRefusalCode.ADAPTER_UNDER_CERTIFIED.value


def test_valid_conditional_not_applicable_can_still_attain_a2() -> None:
    outcomes = _replace_outcome(
        _passing_outcomes(),
        CertificationProbe.STRUCTURED_ERROR_SHAPE,
        status="not_applicable",
        evidence_digest=sha256_json({"applicable": False}),
        evidence={"applicable": False},
        reason="no_structured_error_case",
    )
    outcomes = _replace_outcome(
        outcomes,
        CertificationProbe.CONFIRMATION_SAFETY,
        status="not_applicable",
        evidence_digest=sha256_json({"applicable": False}),
        evidence={"applicable": False},
        reason="no_confirmation_tools",
    )

    assert derive_attained_tier(mcp_reference_profile(), outcomes) is AdapterTier.A2


def test_unapproved_not_applicable_reason_fails_closed() -> None:
    outcomes = _replace_outcome(
        _passing_outcomes(),
        CertificationProbe.CONFIRMATION_SAFETY,
        status="not_applicable",
        evidence_digest=sha256_json({"applicable": False}),
        evidence={"applicable": False},
        reason="adapter_says_so",
    )

    with pytest.raises(CertificationError, match="unapproved not_applicable"):
        derive_attained_tier(mcp_reference_profile(), outcomes)


def test_report_refuses_forged_issuer_tampering_and_input_drift() -> None:
    descriptor = _descriptor()
    profile = mcp_reference_profile()
    report = build_certification_report(
        descriptor,
        source_identity_digest=SHA_B,
        profile=profile,
        outcomes=_passing_outcomes(),
        authority=AUTHORITY,
    )
    document = report.model_dump(mode="json")
    document["issuer"] = "adapter-self-issued"
    with pytest.raises(ValidationError, match="bfcl-source-adapter-verifier-v1"):
        AdapterCertificationReport.model_validate(document)

    document = report.model_dump(mode="json")
    document["source_identity_digest"] = SHA_C
    with pytest.raises(ValidationError, match="report digest mismatch"):
        AdapterCertificationReport.model_validate(document)

    with pytest.raises(CertificationError, match="source_identity_digest"):
        verify_certification_report(
            report,
            descriptor=descriptor,
            source_identity_digest=SHA_C,
            profile=profile,
            required_tier=AdapterTier.A0,
            trusted_public_keys=TRUSTED_KEYS,
        )


def test_recomputed_self_digest_cannot_forge_bfcl_signature() -> None:
    report = build_certification_report(
        _descriptor(),
        source_identity_digest=SHA_B,
        profile=mcp_reference_profile(),
        outcomes=_passing_outcomes(),
        authority=AUTHORITY,
    )
    attacker = CertificationAuthority(
        key_id="attacker",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x09" * 32),
    )
    document = report.model_dump(mode="json")
    document["signing_key_id"] = attacker.key_id
    unsigned = {key: value for key, value in document.items() if key not in {"report_digest", "signature"}}
    document["report_digest"] = sha256_json(unsigned)
    from base64 import b64encode

    document["signature"] = b64encode(attacker.private_key.sign(document["report_digest"].encode("ascii"))).decode(
        "ascii"
    )
    forged = AdapterCertificationReport.model_validate(document)

    with pytest.raises(CertificationError, match="not trusted"):
        verify_certification_report(
            forged,
            descriptor=_descriptor(),
            source_identity_digest=SHA_B,
            profile=mcp_reference_profile(),
            required_tier=AdapterTier.A2,
            trusted_public_keys=TRUSTED_KEYS,
        )


def test_probe_input_evidence_and_descriptor_ceiling_fail_closed() -> None:
    wrong_input = tuple(item.model_copy(update={"input_digest": SHA_A}) for item in _passing_outcomes())
    with pytest.raises(CertificationError, match="do not match the certified"):
        build_certification_report(
            _descriptor(),
            source_identity_digest=SHA_B,
            profile=mcp_reference_profile(),
            outcomes=wrong_input,
            authority=AUTHORITY,
        )

    document = _passing_outcomes()[0].model_dump(mode="json")
    document["evidence"]["probe_index"] = 999
    with pytest.raises(ValidationError, match="evidence digest mismatch"):
        ProbeOutcome.model_validate(document)

    limited = _descriptor().model_copy(
        update={
            "capabilities": (
                AdapterCapability.DESCRIBE_TOOLS,
                AdapterCapability.PIN_IDENTITY,
            ),
            "probe_safety": ProbeSafetyPolicy(
                kind=ProbeSafetyKind.IDENTITY_ONLY,
                max_calls=1,
                timeout_s=1.0,
            ),
        }
    )
    limited_input = certification_input_digest(
        limited,
        source_identity_digest=SHA_B,
        profile=mcp_reference_profile(),
    )
    limited_outcomes = tuple(item.model_copy(update={"input_digest": limited_input}) for item in _passing_outcomes())
    with pytest.raises(CertificationError, match="permit at most A0"):
        build_certification_report(
            limited,
            source_identity_digest=SHA_B,
            profile=mcp_reference_profile(),
            outcomes=limited_outcomes,
            authority=AUTHORITY,
        )


def test_execution_inputs_are_bound_through_every_signed_probe_outcome() -> None:
    descriptor = _descriptor()
    profile = mcp_reference_profile()
    input_digest = certification_input_digest(
        descriptor,
        source_identity_digest=SHA_B,
        profile=profile,
        execution_inputs_digest=SHA_C,
    )
    outcomes = tuple(item.model_copy(update={"input_digest": input_digest}) for item in _passing_outcomes())
    report = build_certification_report(
        descriptor,
        source_identity_digest=SHA_B,
        profile=profile,
        outcomes=outcomes,
        authority=AUTHORITY,
        execution_inputs_digest=SHA_C,
    )
    verify_certification_report(
        report,
        descriptor=descriptor,
        source_identity_digest=SHA_B,
        profile=profile,
        required_tier=AdapterTier.A2,
        trusted_public_keys=TRUSTED_KEYS,
        execution_inputs_digest=SHA_C,
    )

    with pytest.raises(CertificationError, match="probe_input_digest"):
        verify_certification_report(
            report,
            descriptor=descriptor,
            source_identity_digest=SHA_B,
            profile=profile,
            required_tier=AdapterTier.A2,
            trusted_public_keys=TRUSTED_KEYS,
            execution_inputs_digest=SHA_A,
        )


def test_uncertified_report_cannot_be_referenced() -> None:
    outcomes = _replace_outcome(
        _passing_outcomes(),
        CertificationProbe.IDENTITY_INTEGRITY,
        status="fail",
        evidence_digest=None,
        evidence=None,
        reason="identity_drift",
    )
    report = build_certification_report(
        _descriptor(),
        source_identity_digest=SHA_B,
        profile=mcp_reference_profile(),
        outcomes=outcomes,
        authority=AUTHORITY,
    )

    assert report.attained_tier is AdapterTier.NONE
    with pytest.raises(CertificationError, match="uncertified"):
        certification_reference(report)


def test_profile_requires_every_probe_in_canonical_order() -> None:
    document = mcp_reference_profile().model_dump(mode="json")
    document["probes"] = document["probes"][:-1]

    with pytest.raises(ValidationError, match="every generic probe"):
        CertificationProfile.model_validate(document)


def test_profile_cannot_certify_a_different_adapter_kind() -> None:
    with pytest.raises(CertificationError, match="does not allow"):
        build_certification_report(
            _descriptor(kind="http_endpoint"),
            source_identity_digest=SHA_B,
            profile=mcp_reference_profile(),
            outcomes=_passing_outcomes(),
            authority=AUTHORITY,
        )


def test_full_mcp_p1_p11_report_projects_to_a2() -> None:
    outcomes = project_mcp_probe_report(
        _mcp_report(),
        input_digest=SHA_A,
        structured_error_applicable=True,
        confirmation_applicable=True,
    )

    assert derive_attained_tier(mcp_reference_profile(), outcomes) is AdapterTier.A2
    assert [item.probe for item in outcomes] == list(CertificationProbe)


def test_mcp_conditional_probes_map_only_to_profile_owned_reasons() -> None:
    outcomes = project_mcp_probe_report(
        _mcp_report(
            overrides={
                "P7": (
                    "not_applicable",
                    "the pack declares no structured-error validation case",
                ),
                "P8": (
                    "not_applicable",
                    "the pack declares no confirmation-gated tool",
                ),
            }
        ),
        input_digest=SHA_A,
        structured_error_applicable=False,
        confirmation_applicable=False,
    )

    by_probe = {item.probe: item for item in outcomes}
    assert by_probe[CertificationProbe.STRUCTURED_ERROR_SHAPE].reason == ("no_structured_error_case")
    assert by_probe[CertificationProbe.CONFIRMATION_SAFETY].reason == ("no_confirmation_tools")
    assert derive_attained_tier(mcp_reference_profile(), outcomes) is AdapterTier.A2

    with pytest.raises(CertificationError, match="cannot be not_applicable"):
        project_mcp_probe_report(
            _mcp_report(
                overrides={
                    "P8": (
                        "not_applicable",
                        "the pack declares no confirmation-gated tool",
                    )
                }
            ),
            input_digest=SHA_A,
            structured_error_applicable=True,
            confirmation_applicable=True,
        )


def test_missing_or_failed_mcp_probe_cannot_be_relabelled_as_a2() -> None:
    outcomes = project_mcp_probe_report(
        _mcp_report(missing={"P11"}),
        input_digest=SHA_A,
        structured_error_applicable=True,
        confirmation_applicable=True,
    )

    assert derive_attained_tier(mcp_reference_profile(), outcomes) is AdapterTier.A1
    result_shape = next(item for item in outcomes if item.probe is CertificationProbe.RESULT_SHAPE_COVERAGE)
    assert result_shape.status == "fail"
    assert result_shape.reason == "probe_missing"


def test_mcp_projection_rejects_unknown_reordered_and_malformed_probes() -> None:
    unknown = _mcp_report()
    unknown["probes"].append(
        {
            "id": "P12",
            "requirement": "required",
            "status": "pass",
            "reason": None,
        }
    )
    with pytest.raises(CertificationError, match="unknown probe P12"):
        project_mcp_probe_report(
            unknown,
            input_digest=SHA_A,
            structured_error_applicable=True,
            confirmation_applicable=True,
        )

    reordered = _mcp_report()
    reordered["probes"][0], reordered["probes"][1] = (
        reordered["probes"][1],
        reordered["probes"][0],
    )
    with pytest.raises(CertificationError, match="P1 through P11 order"):
        project_mcp_probe_report(
            reordered,
            input_digest=SHA_A,
            structured_error_applicable=True,
            confirmation_applicable=True,
        )

    invalid_not_applicable = _mcp_report(overrides={"P5": ("not_applicable", "source chose not to run it")})
    with pytest.raises(CertificationError, match="required MCP probe P5"):
        project_mcp_probe_report(
            invalid_not_applicable,
            input_digest=SHA_A,
            structured_error_applicable=True,
            confirmation_applicable=True,
        )
