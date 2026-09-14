from __future__ import annotations

import json
import shutil
import urllib.error
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    ENDPOINT_CREDENTIAL_POLICY_ENV,
    ENDPOINT_CREDENTIAL_POLICY_VERSION,
    PROTOCOL_VERSION,
    EndpointConfig,
    EndpointCredentialPolicy,
    EndpointIdentity,
    EndpointOracleClient,
    load_endpoint_config,
    resolve_endpoint_headers,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import ProcessWorker

DIGEST = "sha256:" + "a" * 64
IDENTITY = {
    "protocol_version": PROTOCOL_VERSION,
    "oracle_id": "test-oracle",
    "oracle_version": "1.0.0",
    "content_digest": DIGEST,
}


def _write_endpoint_config(tmp_path: Path, *, base_url: str = "https://oracle.example") -> Path:
    path = tmp_path / "endpoint_config.yaml"
    path.write_text(
        f"""
protocol_version: {PROTOCOL_VERSION}
base_url: {base_url}
auth:
  bearer_token_env: BFCL_TEST_TOKEN
  headers:
    X-Tenant: BFCL_TEST_TENANT
expected:
  oracle_id: test-oracle
  oracle_version: 1.0.0
  content_digest: {DIGEST}
max_response_bytes: 4096
""",
        encoding="utf-8",
    )
    return path


def test_endpoint_config_is_https_and_resolves_secret_references(tmp_path: Path) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )

    assert config.expected.content_digest == DIGEST
    assert resolve_endpoint_headers(
        config,
        {"BFCL_TEST_TOKEN": "secret", "BFCL_TEST_TENANT": "tenant-a"},
    ) == {"Authorization": "Bearer secret", "X-Tenant": "tenant-a"}

    with pytest.raises(ValueError, match="HTTPS"):
        load_endpoint_config(
            _write_endpoint_config(tmp_path, base_url="http://oracle.example"),
            allowed_roots=(tmp_path,),
        )


def test_endpoint_config_requires_declared_secret_environment(tmp_path: Path) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )

    with pytest.raises(ValueError, match="BFCL_TEST_TOKEN"):
        resolve_endpoint_headers(config, {})


def _credential_policy(
    *,
    base_url: str = "https://oracle.example",
    references: list[str] | None = None,
) -> EndpointCredentialPolicy:
    return EndpointCredentialPolicy.from_mapping(
        {
            "schema_version": ENDPOINT_CREDENTIAL_POLICY_VERSION,
            "grants": [
                {
                    "base_url": base_url,
                    "credential_references": references or ["BFCL_TEST_TOKEN", "BFCL_TEST_TENANT"],
                }
            ],
        },
        source="test policy",
    )


def test_ambient_credentials_require_an_exact_operator_destination_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )
    monkeypatch.setenv("BFCL_TEST_TOKEN", "secret")
    monkeypatch.setenv("BFCL_TEST_TENANT", "tenant-a")
    monkeypatch.delenv(ENDPOINT_CREDENTIAL_POLICY_ENV, raising=False)

    with pytest.raises(ValueError, match="operator credential policy"):
        resolve_endpoint_headers(config)
    with pytest.raises(ValueError, match="does not authorize"):
        resolve_endpoint_headers(
            config,
            credential_policy=_credential_policy(base_url="https://attacker.example"),
        )
    with pytest.raises(ValueError, match="does not authorize"):
        resolve_endpoint_headers(
            config,
            credential_policy=_credential_policy(references=["NGC_API_KEY"]),
        )

    assert resolve_endpoint_headers(
        config,
        credential_policy=_credential_policy(),
    ) == {"Authorization": "Bearer secret", "X-Tenant": "tenant-a"}


def test_operator_policy_file_binds_ambient_reference_to_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )
    policy = tmp_path / "operator-policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": ENDPOINT_CREDENTIAL_POLICY_VERSION,
                "grants": [
                    {
                        "base_url": config.base_url,
                        "credential_references": [
                            "BFCL_TEST_TOKEN",
                            "BFCL_TEST_TENANT",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(ENDPOINT_CREDENTIAL_POLICY_ENV, str(policy))
    monkeypatch.setenv("BFCL_TEST_TOKEN", "secret")
    monkeypatch.setenv("BFCL_TEST_TENANT", "tenant-a")

    assert resolve_endpoint_headers(config)["Authorization"] == "Bearer secret"


def test_pack_cannot_exfiltrate_an_ambient_operator_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_endpoint_config(
        tmp_path,
        base_url="https://attacker.example/collect",
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["auth"] = {"bearer_token_env": "NGC_API_KEY"}
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    config = load_endpoint_config(path, allowed_roots=(tmp_path,))
    monkeypatch.setenv("NGC_API_KEY", "ambient-operator-secret")

    approved_oracle_only = _credential_policy(
        base_url="https://oracle.example",
        references=["NGC_API_KEY"],
    )
    with pytest.raises(ValueError, match="does not authorize"):
        resolve_endpoint_headers(
            config,
            credential_policy=approved_oracle_only,
        )


class _Response:
    def __init__(self, payload: Any):
        self._raw = json.dumps(payload).encode()
        self.headers = {"Content-Length": str(len(self._raw))}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self._raw[:limit]


class _EmptyResponse:
    headers: dict[str, str] = {}

    def __enter__(self) -> _EmptyResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return b""


class _FakeOpener:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def open(self, request: Any, timeout: float) -> _Response:
        self.requests.append(request)
        path = request.full_url.removeprefix("https://oracle.example")
        if path == "/v1/metadata":
            return _Response(IDENTITY)
        if path == "/v1/tools":
            return _Response({"tools": ["lookup"]})
        if path == "/v1/sessions" and request.method == "POST":
            return _Response({"session_id": "session/1", "oracle": IDENTITY})
        if path == "/v1/sessions/session%2F1/calls":
            return _Response({"value": 7})
        if path == "/v1/sessions/session%2F1/state":
            return _Response({"calls": 1})
        if path == "/v1/sessions/session%2F1" and request.method == "DELETE":
            return _Response({})
        raise AssertionError(f"unexpected request {request.method} {path}")


def test_endpoint_client_runs_an_isolated_session_and_closes_it(tmp_path: Path) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )
    client = EndpointOracleClient(
        config,
        headers={"Authorization": "Bearer secret"},
        timeout_s=1.0,
    )
    opener = _FakeOpener()
    client._opener = opener  # type: ignore[assignment]
    ctx = SimpleNamespace(
        clock=datetime.fromisoformat("2026-03-02T09:00:00+07:00"),
        seed=7,
        timeout_s=1.0,
        task_id="task-1",
        turn_index=2,
    )

    assert client.metadata() == IDENTITY
    assert client.list_tools() == ["lookup"]
    client.reset(ctx=ctx, fixtures={"items": [{"id": "A"}]})
    assert client.call_tool("lookup", {"id": "A"}, ctx=ctx) == {"value": 7}
    assert client.get_state() == {"calls": 1}
    client.close()

    assert opener.requests[-1].method == "DELETE"
    assert opener.requests[0].headers["Authorization"] == "Bearer secret"


def test_endpoint_transport_errors_redact_secret_values(tmp_path: Path) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )
    client = EndpointOracleClient(
        config,
        headers={"Authorization": "Bearer top-secret"},
        timeout_s=1.0,
    )

    class FailingOpener:
        def open(self, request: Any, timeout: float) -> Any:
            raise urllib.error.URLError("top-secret")

    client._opener = FailingOpener()  # type: ignore[assignment]
    with pytest.raises(RuntimeError) as raised:
        client.metadata()
    assert "top-secret" not in str(raised.value)


def test_endpoint_metadata_drift_is_rejected(tmp_path: Path) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )
    client = EndpointOracleClient(config, headers={}, timeout_s=1.0)

    class DriftedOpener:
        def open(self, request: Any, timeout: float) -> _Response:
            return _Response({**IDENTITY, "content_digest": "sha256:" + "b" * 64})

    client._opener = DriftedOpener()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="does not match"):
        client.metadata()


def test_endpoint_close_preserves_session_id_until_deletion_succeeds(tmp_path: Path) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )
    client = EndpointOracleClient(config, headers={}, timeout_s=1.0)
    client.session_id = "retry-me"

    class RetryOpener:
        attempts = 0

        def open(self, request: Any, timeout: float) -> _Response:
            self.attempts += 1
            if self.attempts == 1:
                raise urllib.error.URLError("temporary failure")
            return _Response({})

    opener = RetryOpener()
    client._opener = opener  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        client.close()
    assert client.session_id == "retry-me"

    client.close()
    assert client.session_id is None
    assert opener.attempts == 2


def test_endpoint_close_accepts_204_no_content(tmp_path: Path) -> None:
    config = load_endpoint_config(
        _write_endpoint_config(tmp_path),
        allowed_roots=(tmp_path,),
    )
    client = EndpointOracleClient(config, headers={}, timeout_s=1.0)
    client.session_id = "empty-delete"

    class EmptyDeleteOpener:
        def open(self, request: Any, timeout: float) -> _EmptyResponse:
            assert request.method == "DELETE"
            return _EmptyResponse()

    client._opener = EmptyDeleteOpener()  # type: ignore[assignment]
    client.close()
    assert client.session_id is None


def test_endpoint_rejects_oversized_request_before_transport(tmp_path: Path) -> None:
    config = EndpointConfig(
        path=tmp_path / "endpoint_config.yaml",
        base_url="https://oracle.example",
        expected=EndpointIdentity.from_mapping(IDENTITY, source="test"),
        max_request_bytes=8,
    )
    client = EndpointOracleClient(config, headers={}, timeout_s=1.0)

    class UnusedOpener:
        def open(self, request: Any, timeout: float) -> Any:
            raise AssertionError("oversized request reached the transport")

    client._opener = UnusedOpener()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="max_request_bytes"):
        client._request("POST", "/v1/sessions", {"fixtures": "too large"})


def test_thread_worker_closes_endpoint_session_after_operation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EndpointConfig(
        path=tmp_path / "endpoint_config.yaml",
        base_url="https://oracle.example",
        expected=EndpointIdentity.from_mapping(IDENTITY, source="test"),
    )
    deleted: list[str] = []

    def fake_request(
        self: EndpointOracleClient,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        if method == "POST" and path == "/v1/sessions":
            return {"session_id": "thread-session", "oracle": IDENTITY}
        if method == "POST" and path.endswith("/calls"):
            raise RuntimeError("tool failed")
        if method == "DELETE":
            deleted.append(path)
            return {}
        raise AssertionError(f"unexpected request {method} {path}")

    monkeypatch.setattr(EndpointOracleClient, "_request", fake_request)

    with pytest.raises(RuntimeError, match="tool failed"):
        ProcessWorker(default_timeout_s=2.0, worker="thread").run_episode(
            endpoint_config=config,
            fixtures={},
            clock_iso="2026-03-02T09:00:00+07:00",
            seed=0,
            task_id="thread-cleanup",
            steps=[
                {"op": "reset"},
                {"op": "call_tool", "name": "fail", "arguments": {}},
            ],
            episode_timeout_s=2.0,
        )
    assert deleted == ["/v1/sessions/thread-session"]


def test_process_worker_accepts_endpoint_oracle_and_redacts_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EndpointConfig(
        path=tmp_path / "endpoint_config.yaml",
        base_url="https://127.0.0.1:1",
        expected=EndpointIdentity.from_mapping(IDENTITY, source="test"),
        bearer_token_env="BFCL_TEST_TOKEN",
    )
    monkeypatch.setenv("BFCL_TEST_TOKEN", "process-secret")

    # The endpoint refuses the connection immediately, so this asserts redaction rather
    # than a deadline. The timeouts only bound the test, and they stay well clear of
    # worker startup so a loaded machine cannot turn this into a timeout instead.
    with pytest.raises(RuntimeError) as raised:
        ProcessWorker(default_timeout_s=30.0).run_episode(
            endpoint_config=config,
            fixtures=None,
            clock_iso="2026-03-02T09:00:00+07:00",
            seed=0,
            task_id="endpoint-process",
            steps=[{"op": "metadata"}],
            tool_timeout_s=30.0,
            episode_timeout_s=60.0,
        )
    assert "process-secret" not in str(raised.value)


def test_endpoint_preflight_failure_is_recorded_by_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import pack_loader, pipeline
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import stages as stages_module
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import (
        oracle_validation,
        prepare,
    )

    pipeline._VALIDATED_THIS_PROCESS.clear()
    pack = SimpleNamespace(paths=SimpleNamespace())
    report = {"checks": [{"status": "fail", "failures": [{"reason": "endpoint_unavailable"}]}]}
    config = SimpleNamespace(output_dir=tmp_path, expt_name="preflight")
    monkeypatch.setattr(prepare, "prepare_oracle_pack", lambda _: pack)
    monkeypatch.setattr(pack_loader, "pack_fingerprint", lambda _: "pack-fingerprint")
    monkeypatch.setattr(
        oracle_validation,
        "validation_config_fingerprint",
        lambda _: "config-fingerprint",
    )
    monkeypatch.setattr(oracle_validation, "run_oracle_validation", lambda *_: report)
    monkeypatch.setattr(stages_module, "stage_cache_dir", lambda _: tmp_path)
    monkeypatch.setattr(
        pipeline,
        "_endpoint_metadata",
        lambda *_: (_ for _ in ()).throw(RuntimeError("endpoint unavailable")),
    )

    actual, report_path = pipeline._validate_pack(config)

    assert actual is report
    assert report_path == tmp_path / "oracle_validation_report.json"


def test_endpoint_pack_without_attestation_prepares_but_cannot_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import (
        generate_bfcl,
        prepare_bfcl,
    )

    root = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"
    source_pack = root / "data" / "tiny_oracle_pack"
    pack = tmp_path / "endpoint_pack"
    shutil.copytree(source_pack, pack, ignore=shutil.ignore_patterns("__pycache__"))
    (pack / "backend.py").unlink()
    manifest = yaml.safe_load((pack / "manifest.yaml").read_text(encoding="utf-8"))
    manifest["paths"].pop("backend")
    manifest["paths"]["endpoint"] = "endpoint_config.yaml"
    (pack / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (pack / "endpoint_config.yaml").write_text(
        f"""
protocol_version: {PROTOCOL_VERSION}
base_url: https://oracle.example
expected:
  oracle_id: test-oracle
  oracle_version: 1.0.0
  content_digest: {DIGEST}
""",
        encoding="utf-8",
    )

    config = yaml.safe_load((root / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path / "output")
    config["oracle_pack"]["manifest_path"] = str(pack / "manifest.yaml")
    config["oracle_runtime"]["allowed_roots"] = [str(tmp_path), str(source_pack)]
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    original = ProcessWorker.run_episode

    def fake_endpoint_episode(self: ProcessWorker, **kwargs: Any) -> list[Any]:
        endpoint = kwargs.pop("endpoint_config", None)
        if endpoint is None:
            return original(self, **kwargs)
        steps = list(kwargs.pop("steps"))
        local_steps = [step for step in steps if step["op"] != "metadata"]
        kwargs["backend_path"] = source_pack / "backend.py"
        local_outputs = iter(original(self, steps=local_steps, **kwargs))
        return [IDENTITY if step["op"] == "metadata" else next(local_outputs) for step in steps]

    monkeypatch.setattr(ProcessWorker, "run_episode", fake_endpoint_episode)

    report = json.loads(prepare_bfcl(config_path).read_text(encoding="utf-8"))
    assert report["tier"] == "silver"
    assert report["endpoint_metadata"] == IDENTITY
    assert report["extra_checks"][-1] == {
        "id": "A1",
        "name": "endpoint_conformance",
        "status": "fail",
        "failures": [{"reason": "endpoint_attestation_missing"}],
    }
    with pytest.raises(RuntimeError, match="refuses non-gold pack"):
        generate_bfcl(config_path)
