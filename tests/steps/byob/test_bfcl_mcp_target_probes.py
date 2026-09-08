from __future__ import annotations

import json

from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.mcp_target_probes import (
    GATEWAY_CONFORMANCE_REPORT_PATH,
    assess_gateway_timeout_report,
    build_target_probe_report,
    load_gateway_conformance_report,
)


def _checks(*, schema: str = "pass", validation: str = "pass") -> list[dict]:
    return [
        {"id": 3, "status": schema, "failures": []},
        {"id": 5, "status": validation, "failures": []},
        {"id": 6, "status": "pass", "failures": []},
    ]


def _observations() -> dict:
    return {
        "calls_complete": True,
        "state_deltas_complete": True,
        "calls": [
            {"tool": "lookup", "result_class": "success"},
            {"tool": "reserve", "result_class": "success"},
            {
                "tool": "lookup",
                "result_class": "structured_error",
                "error_code": "NOT_FOUND",
            },
        ],
        "state_deltas": [
            {"tool": "lookup", "changed": False},
            {"tool": "reserve", "changed": True},
        ],
    }


def _extras(
    *,
    determinism: str = "pass",
    isolation: str = "pass",
) -> list[dict]:
    return [
        {"id": "D1", "status": determinism, "failures": []},
        {"id": "MP6", "status": isolation, "failures": []},
        {"id": "D2", "status": "pass", "failures": []},
        {"id": "MP9", "status": "pass", "failures": []},
        {"id": "M1", "status": "pass", "failures": []},
    ]


def test_p1_through_p4_require_live_identity_exact_catalog_and_success_per_tool() -> None:
    report = build_target_probe_report(
        checks=_checks(),
        extra_checks=_extras(),
        endpoint_metadata={"content_digest": "sha256:" + "a" * 64},
        observations=_observations(),
        tool_names={"lookup", "reserve"},
        confirmation_tool_names={"reserve"},
        structured_error_declared=True,
    )

    assert [probe["id"] for probe in report["probes"]] == [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "P10",
        "P11",
    ]
    assert all(probe["status"] == "pass" for probe in report["probes"])
    assert all(probe["reason"] is None for probe in report["probes"])


def test_p4_fails_when_any_tool_lacks_success_or_observation_log_is_incomplete() -> None:
    observations = _observations()
    observations["calls_complete"] = False
    observations["calls"] = observations["calls"][:1]

    report = build_target_probe_report(
        checks=_checks(),
        extra_checks=_extras(),
        endpoint_metadata={"content_digest": "sha256:" + "a" * 64},
        observations=observations,
        tool_names={"lookup", "reserve"},
        confirmation_tool_names={"reserve"},
        structured_error_declared=True,
    )
    p4 = report["probes"][3]

    assert p4["status"] == "fail"
    assert "incomplete" in p4["reason"]
    assert "reserve" in p4["reason"]


def test_schema_failure_cannot_be_relabelled_as_discovery_or_executable_pass() -> None:
    report = build_target_probe_report(
        checks=_checks(schema="fail"),
        extra_checks=_extras(),
        endpoint_metadata={"content_digest": "sha256:" + "a" * 64},
        observations=_observations(),
        tool_names={"lookup", "reserve"},
        confirmation_tool_names={"reserve"},
        structured_error_declared=True,
    )

    assert [probe["status"] for probe in report["probes"]] == [
        "fail",
        "fail",
        "fail",
        "fail",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
    ]


def test_p5_is_derived_only_from_fresh_double_replay_check() -> None:
    report = build_target_probe_report(
        checks=_checks(),
        extra_checks=_extras(determinism="fail"),
        endpoint_metadata={"content_digest": "sha256:" + "a" * 64},
        observations=_observations(),
        tool_names={"lookup", "reserve"},
        confirmation_tool_names={"reserve"},
        structured_error_declared=True,
    )

    assert report["probes"][4] == {
        "id": "P5",
        "requirement": "required",
        "status": "fail",
        "reason": "two fresh replays did not produce identical results and final state",
    }


def test_p6_requires_the_interleaved_live_episode_probe() -> None:
    report = build_target_probe_report(
        checks=_checks(),
        extra_checks=_extras(isolation="fail"),
        endpoint_metadata={"content_digest": "sha256:" + "a" * 64},
        observations=_observations(),
        tool_names={"lookup", "reserve"},
        confirmation_tool_names={"reserve"},
        structured_error_declared=True,
    )

    assert report["probes"][5]["status"] == "fail"
    assert "crossed" in report["probes"][5]["reason"]


def test_p9_requires_all_bounded_timeout_and_cleanup_observations(tmp_path) -> None:
    document = {
        "suite": {
            "kind": "gateway",
            "profile_version": "bfcl-mcp-gateway-conformance-v1",
            "p9": {
                "timeout_observed": True,
                "business_call_attempts": 1,
                "episode_poisoned": True,
                "transport_cleanup_completed": True,
                "unknown_commit_state_preserved": True,
            },
        }
    }
    path = tmp_path / GATEWAY_CONFORMANCE_REPORT_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_gateway_conformance_report(tmp_path)
    assert loaded == document
    assert assess_gateway_timeout_report(loaded)["status"] == "pass"

    document["suite"]["p9"]["business_call_attempts"] = 2
    failed = assess_gateway_timeout_report(document)
    assert failed["status"] == "fail"
    assert failed["failures"][0]["field"] == "business_call_attempts"
