"""Contract tests for the nemo_evaluator_bundle writer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    nemo_native_adapter,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.nemo_native_adapter import (
    NEMO_NATIVE_FAILURE_FILE,
    NemoNativeAdapterConfig,
    NemoNativeAdapterError,
    install_native_framework,
    launcher_task_entry,
    native_evaluation_result_document,
    native_framework_definition,
    run_nemo_native_adapter,
    verify_native_bundle,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_aggregation import (
    TraceMetricResult,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    NemoEvaluatorBundle,
    NemoEvaluatorRecord,
    export_tree_hash,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ProjectionSource,
    project_benchmark_rows,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NEMO_BUNDLE_FILE,
    NEMO_BUNDLE_FILES,
    NEMO_DATASET_FILE,
    NEMO_DATASET_SCHEMA_FILE,
    NEMO_EVALUATOR_CONFIG_FILE,
    NEMO_EVALUATOR_ROOT,
    NEMO_METADATA_FILE,
    NEMO_SYSTEM_PROMPT_FILE,
    NemoEvaluatorArtifact,
    NemoEvaluatorWriteError,
    bundle_scoring,
    bundle_task_name,
    read_nemo_evaluator_bundle,
    system_prompt_catalog,
    write_nemo_evaluator_bundle,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
    encode_arguments,
)

# Registering a native task, translating metrics, and validating a native result all
# execute against the real evaluator. Those tests are conformance checks for an optional
# integration, so they skip rather than fail where the extra is not installed.
requires_nemo_evaluator = pytest.mark.skipif(
    importlib.util.find_spec("nemo_evaluator") is None,
    reason="native adapter conformance requires the evaluator extra (nemo-evaluator)",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "chuyen_khoan",
            "description": "Chuyển khoản nội bộ.",
            "parameters": {
                "type": "object",
                "properties": {"tai_khoan": {"type": "string"}},
                "required": ["tai_khoan"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "liet_ke_the", "description": "Liệt kê thẻ.", "parameters": None},
    },
]
SYSTEM_PROMPT = "Bạn dùng công cụ khi cần."
TRANSFER: dict[str, Any] = {"tai_khoan": "1"}


def _system(content: str = SYSTEM_PROMPT) -> dict[str, Any]:
    return {"role": "system", "content": content, "tool_calls": None, "tool_call_id": None}


def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content, "tool_calls": None, "tool_call_id": None}


def _assistant_text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": None, "tool_call_id": None}


def _assistant_calls(calls: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": canonical_json(arguments)},
            }
            for index, (name, arguments) in enumerate(calls)
        ],
        "tool_call_id": None,
    }


def _tool_result(call_id: str, payload: Any) -> dict[str, Any]:
    return {"role": "tool", "content": canonical_json(payload), "tool_calls": None, "tool_call_id": call_id}


def _expected(name: str, arguments: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    call = {
        "turn_index": 0,
        "call_group": 0,
        "position_in_group": 0,
        "function_name": name,
        "arguments": encode_arguments(arguments),
    }
    call.update(overrides)
    return call


def _row(task_id: str = "pack__tpl__abcdef", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task_id,
        "template_id": "tpl",
        "variant_index": 0,
        "messages": [
            _system(),
            _user("Chuyển giúp tôi 1 đồng."),
            _assistant_calls([("chuyen_khoan", TRANSFER)]),
            _tool_result("call_0", {"trang_thai": "đã chuyển"}),
            _assistant_text("Đã chuyển xong."),
        ],
        "tools": canonical_json(TOOLS),
        "expected_tool_calls": [_expected("chuyen_khoan", TRANSFER)],
        "success_assertions": ["assert_transfer_committed"],
        "fixture_refs": ['["accounts","1"]'],
        "intent": "chuyen_khoan",
        "category": "payments",
        "difficulty": "easy",
        "required_tools": ["chuyen_khoan"],
        "required_tools_fingerprint": canonical_json(["chuyen_khoan"]),
        "tools_present": ["chuyen_khoan", "liet_ke_the"],
        "turn_policy": "single_turn",
        "is_multi_turn": False,
        "num_tool_calls": 1,
        "call_order": "strict",
        "call_order_prefix": None,
        "system_prompt_id": "sp-1",
        "tier": "gold",
        "gold_eligible": True,
        "validated_by": ["schema", "replay", "assertions"],
        "pack_id": "pack",
        "pack_version": "1.0.0",
        "seed": 7,
        "paraphrase_model": None,
        "paraphrase_model_canonical": None,
        "held_out_hit": False,
        "src": "pack:tpl",
        "metadata": canonical_json(
            {
                "language": "vi",
                "expt_name": "expt",
                "base_task_id": None,
                "surface_source": "template",
                "profile_hash": None,
            }
        ),
    }
    row.update(overrides)
    return row


def _parallel_row(task_id: str = "pack__tpl__parallel") -> dict[str, Any]:
    cards: dict[str, Any] = {}
    return _row(
        task_id,
        turn_policy="multi_tool",
        num_tool_calls=2,
        category="cards",
        difficulty="hard",
        required_tools=["chuyen_khoan", "liet_ke_the"],
        required_tools_fingerprint=canonical_json(["chuyen_khoan", "liet_ke_the"]),
        messages=[
            _system(),
            _user("Chuyển tiền và liệt kê thẻ."),
            _assistant_calls([("chuyen_khoan", TRANSFER), ("liet_ke_the", cards)]),
            _tool_result("call_0", {"trang_thai": "ok"}),
            _tool_result("call_1", {"the": []}),
            _assistant_text("Xong."),
        ],
        expected_tool_calls=[
            _expected("chuyen_khoan", TRANSFER, position_in_group=0),
            _expected("liet_ke_the", cards, position_in_group=1),
        ],
    )


def _multi_turn_row(task_id: str = "pack__tpl__multiturn") -> dict[str, Any]:
    return _row(
        task_id,
        turn_policy="missing_slot",
        is_multi_turn=True,
        call_order="any",
        messages=[
            _system(),
            _user("Chuyển tiền giúp tôi."),
            _assistant_text("Chuyển tới tài khoản nào?"),
            _user("Tài khoản 1."),
            _assistant_calls([("chuyen_khoan", TRANSFER)]),
            _tool_result("call_0", {"trang_thai": "ok"}),
            _assistant_text("Đã chuyển."),
        ],
        expected_tool_calls=[_expected("chuyen_khoan", TRANSFER, turn_index=1)],
    )


def _callless_row(task_id: str = "pack__tpl__nocall", prompt: str = SYSTEM_PROMPT) -> dict[str, Any]:
    return _row(
        task_id,
        turn_policy="irrelevance",
        num_tool_calls=0,
        call_order="any",
        category=None,
        difficulty=None,
        required_tools=[],
        required_tools_fingerprint=canonical_json([]),
        success_assertions=[],
        messages=[
            _system(prompt),
            _user("Hôm nay trời thế nào?"),
            _assistant_text("Tôi chỉ hỗ trợ nghiệp vụ ngân hàng."),
        ],
        expected_tool_calls=[],
        validated_by=["schema", "replay"],
    )


def _projection(rows: list[dict[str, Any]] | None = None) -> CanonicalExportProjection:
    rows = rows if rows is not None else [_row()]
    return project_benchmark_rows(
        rows,
        source=ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=len(rows)),
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset(root: Path) -> list[dict[str, Any]]:
    text = (root / NEMO_DATASET_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_the_writer_produces_the_whole_bundle(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection([_row("t1"), _parallel_row("t2")]), tmp_path)

    assert artifact.format == "nemo_evaluator_bundle"
    assert artifact.root == NEMO_EVALUATOR_ROOT
    assert artifact.rows == 2
    assert set(artifact.files) == set(NEMO_BUNDLE_FILES)
    for name in NEMO_BUNDLE_FILES:
        assert (tmp_path / NEMO_EVALUATOR_ROOT / name).is_file()
    assert artifact.bundle_file == f"{NEMO_EVALUATOR_ROOT}/{NEMO_BUNDLE_FILE}"


def _native_adapter(
    tmp_path: Path,
    artifact: NemoEvaluatorArtifact,
) -> NemoNativeAdapterConfig:
    return NemoNativeAdapterConfig(
        bundle_root=(tmp_path / artifact.root).resolve(),
        bundle_content_hash=artifact.content_hash,
        eval_config_path=(tmp_path / "eval.yaml").resolve(),
        candidate_alias="candidate",
        native_output_dir=(tmp_path / "native-results").resolve(),
    )


@requires_nemo_evaluator
def test_the_native_adapter_verifies_and_registers_the_immutable_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = write_nemo_evaluator_bundle(
        _projection([_row("t1"), _parallel_row("t2")]),
        tmp_path,
    )
    adapter = _native_adapter(tmp_path, artifact)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter.load_eval_config",
        lambda path: SimpleNamespace(
            candidate=lambda alias: SimpleNamespace(
                api=SimpleNamespace(api_key_env="CANDIDATE_API_KEY")
            )
        ),
    )

    verified = verify_native_bundle(adapter)
    fdf = native_framework_definition(
        adapter,
        adapter_config_path="/adapter/native.yaml",
    )
    task = launcher_task_entry(
        adapter,
        adapter_config_path="/adapter/native.yaml",
        container="nvcr.io/example/evaluator@sha256:" + "0" * 64,
    )

    assert verified.task_ids == ("t1", "t2")
    assert verified.content_hash == artifact.content_hash
    assert fdf["evaluations"][0]["defaults"]["config"]["required_capabilities"] == [
        "tools"
    ]
    assert (
        fdf["defaults"]["target"]["api_endpoint"]["adapter_config"]["mode"]
        == "client"
    )
    assert "config.params.extra.runtime_bundle_root" in fdf["defaults"]["command"]
    assert task["name"].endswith(f".{verified.descriptor.task_name}")
    assert task["endpoint_type"] == "chat"
    assert task["nemo_evaluator_config"]["config"]["params"]["extra"][
        "runtime_bundle_root"
    ] == "/datasets/bfcl"
    assert task["env_vars"] == {"CANDIDATE_API_KEY": "host:CANDIDATE_API_KEY"}


def test_native_paths_are_disjoint_and_cli_overrides_are_revalidated(
    tmp_path: Path,
) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    adapter = _native_adapter(tmp_path, artifact)
    payload = adapter.model_dump(mode="python")
    payload["native_output_dir"] = adapter.bundle_root / "results"

    with pytest.raises(ValidationError, match="contain or be contained"):
        NemoNativeAdapterConfig.model_validate(payload)


def test_native_framework_build_is_immutable_and_does_not_register_globally(
    tmp_path: Path,
) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    adapter = _native_adapter(tmp_path, artifact)
    install_root = tmp_path / "framework-build"

    package = install_native_framework(
        adapter,
        adapter_config_path="/adapter/native.yaml",
        install_dir=install_root,
    )
    assert (package / "pyproject.toml").is_file()
    assert not tuple(install_root.rglob("*.pth"))

    install_native_framework(
        adapter,
        adapter_config_path="/adapter/native.yaml",
        install_dir=install_root,
    )
    (package / "nemo_evaluator" / package.name / "framework.yml").write_text(
        "changed",
        encoding="utf-8",
    )
    with pytest.raises(NemoNativeAdapterError, match="immutable evidence"):
        install_native_framework(
            adapter,
            adapter_config_path="/adapter/native.yaml",
            install_dir=install_root,
        )


def test_the_native_adapter_refuses_one_changed_or_extra_bundle_file(
    tmp_path: Path,
) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    adapter = _native_adapter(tmp_path, artifact)
    metadata = tmp_path / artifact.root / NEMO_METADATA_FILE
    metadata.write_text(metadata.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(NemoNativeAdapterError, match="tree hash"):
        verify_native_bundle(adapter)

    metadata.write_text(
        json.dumps(_read_json(metadata), sort_keys=True),
        encoding="utf-8",
    )
    (tmp_path / artifact.root / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(NemoNativeAdapterError, match="bundle files differ"):
        verify_native_bundle(adapter)


def test_native_run_maps_setup_failures_to_immutable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    adapter = _native_adapter(tmp_path, artifact)
    (adapter.bundle_root / "unexpected.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter.verify_nemo_runtime",
        lambda config, require_launcher: {"nemo-evaluator": "0.2.8"},
    )

    with pytest.raises(NemoNativeAdapterError, match="bundle files differ"):
        run_nemo_native_adapter(adapter)

    failure = _read_json(adapter.native_output_dir / NEMO_NATIVE_FAILURE_FILE)
    assert failure["error_code"] == "eval_nemo_adapter_invalid"
    assert failure["error_type"] == "NemoNativeAdapterError"
    assert failure["adapter_config_hash"] == adapter.config_hash


@requires_nemo_evaluator
def test_native_metric_translation_preserves_counts_and_omits_na() -> None:
    aggregate = type(
        "Aggregate",
        (),
        {
            "metrics": (
                TraceMetricResult(
                    metric="tool_selection_pass_rate",
                    numerator=1,
                    denominator=2,
                    value=0.5,
                ),
                TraceMetricResult(
                    metric="arguments_pass_rate",
                    numerator=0,
                    denominator=0,
                    value=None,
                    not_applicable_reason="metric.gate_not_applicable",
                ),
            )
        },
    )()

    document = native_evaluation_result_document(aggregate, task_name="bfcl_pack")  # type: ignore[arg-type]
    scores = document["tasks"]["bfcl_pack"]["metrics"]["pass@1"]["scores"]

    assert scores["tool_selection_pass_rate"]["value"] == 0.5
    assert scores["tool_selection_pass_rate"]["stats"]["count"] == 2
    assert scores["tool_selection_pass_rate"]["stats"]["sum"] == 1.0
    assert "arguments_pass_rate" not in scores


def test_native_authorization_rejects_a_different_execution_dataset_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = write_nemo_evaluator_bundle(_projection([_row("t1")]), tmp_path)
    adapter = _native_adapter(tmp_path, artifact).model_copy(
        update={"target_binding": "exact"}
    )
    candidate = SimpleNamespace(
        alias="candidate",
        api=SimpleNamespace(base_url="https://candidate.example/v1"),
        model="model",
    )
    eval_config = SimpleNamespace(
        candidates=(candidate,),
        outputs=SimpleNamespace(output_dir=(tmp_path / "bfcl-results").resolve()),
        candidate=lambda alias: candidate,
    )
    authorization = SimpleNamespace(
        eval_run_id="eval-run",
        source=SimpleNamespace(
            evaluation_benchmark=SimpleNamespace(
                content_hash="sha256:" + "0" * 64
            )
        ),
        projection=_projection([_row("t2")]),
        plan=SimpleNamespace(evaluation_task_ids=lambda alias: ("t1",)),
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter.verify_nemo_runtime",
        lambda config, require_launcher: {"nemo-evaluator": "0.2.8"},
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter.load_eval_config",
        lambda path: eval_config,
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter.authorize_bfcl_eval",
        lambda config, eval_run_id, probe_oracle: authorization,
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter._run_authorized_eval",
        lambda *args, **kwargs: pytest.fail("candidate execution was reached"),
    )

    with pytest.raises(NemoNativeAdapterError, match="task order differs"):
        run_nemo_native_adapter(adapter)


@requires_nemo_evaluator
def test_native_run_maps_the_bfcl_result_and_publishes_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = _projection([_row("t1")])
    artifact = write_nemo_evaluator_bundle(projection, tmp_path)
    adapter = _native_adapter(tmp_path, artifact).model_copy(
        update={"target_binding": "exact", "probe_oracle": False}
    )
    candidate = SimpleNamespace(
        alias="candidate",
        api=SimpleNamespace(base_url="https://candidate.example/v1"),
        model="model",
    )
    eval_config = SimpleNamespace(
        candidates=(candidate,),
        outputs=SimpleNamespace(output_dir=(tmp_path / "bfcl-results").resolve()),
        candidate=lambda alias: candidate,
    )
    aggregate = SimpleNamespace(
        candidate_alias="candidate",
        scope="trace",
        aggregate_hash="sha256:" + "2" * 64,
        eval_config_hash="sha256:" + "4" * 64,
        plan_identity="sha256:" + "5" * 64,
        source_verification_identity="sha256:" + "6" * 64,
        metrics=(
            TraceMetricResult(
                    metric="tool_selection_pass_rate",
                numerator=1,
                denominator=1,
                value=1.0,
            ),
            TraceMetricResult(
                metric="arguments_pass_rate",
                numerator=0,
                denominator=0,
                value=None,
                not_applicable_reason="metric.gate_not_applicable",
            ),
        ),
    )
    bfcl_result = SimpleNamespace(
        eval_run_id="eval-run",
        plan=SimpleNamespace(
            candidate_aliases=("candidate",),
            evaluation_task_ids=lambda alias: ("t1",),
        ),
        source=SimpleNamespace(
            evaluation_benchmark=SimpleNamespace(
                content_hash="sha256:" + "0" * 64,
            )
        ),
        candidate_scores=(aggregate,),
        artifacts=SimpleNamespace(
            report_path=tmp_path / "bfcl-results" / "eval_report.json",
            report_hash="sha256:" + "3" * 64,
        ),
    )
    authorization = SimpleNamespace(
        eval_run_id="eval-run",
        source=bfcl_result.source,
        projection=projection,
        plan=bfcl_result.plan,
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter.verify_nemo_runtime",
        lambda config, require_launcher: {"nemo-evaluator": "0.2.8"},
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter.load_eval_config",
        lambda path: eval_config,
    )
    probe_decisions: list[bool] = []

    def authorize(config, eval_run_id, probe_oracle):  # type: ignore[no-untyped-def]
        probe_decisions.append(probe_oracle)
        return authorization

    def execute(config, authorized, probe_oracle):  # type: ignore[no-untyped-def]
        probe_decisions.append(probe_oracle)
        return bfcl_result

    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter.authorize_bfcl_eval",
        authorize,
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval."
        "nemo_native_adapter._run_authorized_eval",
        execute,
    )

    result = run_nemo_native_adapter(
        adapter,
        target_url="https://candidate.example/v1/",
        target_model_id="model",
    )
    native = _read_json(result.result_path)
    manifest = _read_json(result.manifest_path)

    score = native["tasks"]["pack"]["metrics"]["pass@1"]["scores"]["tool_selection"]
    assert manifest["served_identity"]["model"] == "model"
    assert manifest["served_identity"]["base_url"] == "https://candidate.example/v1"
    assert manifest["served_identity"]["canonical_model_identity"]
    assert score["value"] == 1.0
    assert manifest["aggregate_hash"] == aggregate.aggregate_hash
    assert manifest["omitted_not_applicable_metrics"] == {
        "arguments_pass_rate": "metric.gate_not_applicable"
    }
    assert result.result_hash == (
        "sha256:" + hashlib.sha256(result.result_path.read_bytes()).hexdigest()
    )
    assert probe_decisions == [False, False]


def test_native_run_refuses_a_runtime_probe_override(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    adapter = _native_adapter(tmp_path, artifact)

    with pytest.raises(NemoNativeAdapterError, match="differs from.*adapter config"):
        run_nemo_native_adapter(adapter, probe_oracle=False)


def test_launcher_binding_refuses_a_different_served_model_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    adapter = _native_adapter(tmp_path, artifact)
    payload = {
        "candidates": [
            {
                "alias": "candidate",
                "model": "original-route",
                "api": {"base_url": "https://original.example/v1"},
                "model_identity": {"canonical_id": "weights"},
            }
        ]
    }
    config = SimpleNamespace(model_dump=lambda mode: payload)
    class RuntimeConfig:
        @classmethod
        def model_validate(cls, value: dict[str, Any]) -> dict[str, Any]:
            return value

    monkeypatch.setattr(nemo_native_adapter, "BfclEvalConfig", RuntimeConfig)

    with pytest.raises(NemoNativeAdapterError, match="served target"):
        nemo_native_adapter._runtime_eval_config(
            adapter,
            config,  # type: ignore[arg-type]
            target_url="http://managed-endpoint/v1",
            target_model_id="launcher-served-name",
        )


def test_launcher_binding_may_rebind_url_for_the_same_served_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    adapter = _native_adapter(tmp_path, artifact)
    payload = {
        "candidates": [
            {
                "alias": "candidate",
                "provider": "nvidia",
                "model": "original-route",
                "api": {"base_url": "https://original.example/v1"},
                "model_identity": {
                    "source": "nvidia",
                    "model": "original-route",
                    "revision": None,
                    "weights_digest": None,
                },
            }
        ]
    }
    config = SimpleNamespace(model_dump=lambda mode: payload)

    class RuntimeConfig:
        @classmethod
        def model_validate(cls, value: dict[str, Any]) -> dict[str, Any]:
            return value

    monkeypatch.setattr(nemo_native_adapter, "BfclEvalConfig", RuntimeConfig)

    result = nemo_native_adapter._runtime_eval_config(
        adapter,
        config,  # type: ignore[arg-type]
        target_url="http://managed-endpoint/v1",
        target_model_id="original-route",
    )

    candidate = result["candidates"][0]
    assert candidate["model"] == "original-route"
    assert candidate["api"]["base_url"] == "http://managed-endpoint/v1"
    assert candidate["model_identity"]["model"] == "original-route"


def test_launcher_binding_does_not_use_a_pin_to_excuse_served_model_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pin names weights, so a serving name that happens to match it changes nothing."""
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    adapter = _native_adapter(tmp_path, artifact)
    pinned = {
        "source": "huggingface",
        "model": "original-route",
        "revision": "3f9a1c5e7b2d4068a1c3e5b7d9f0a2c4e6b8d0f2",
        "weights_digest": None,
    }
    payload = {
        "candidates": [
            {
                "alias": "candidate",
                "provider": "nvidia",
                "model": "original-route",
                "api": {"base_url": "https://original.example/v1"},
                "model_identity": dict(pinned),
            }
        ]
    }
    config = SimpleNamespace(model_dump=lambda mode: payload)

    class RuntimeConfig:
        @classmethod
        def model_validate(cls, value: dict[str, Any]) -> dict[str, Any]:
            return value

    monkeypatch.setattr(nemo_native_adapter, "BfclEvalConfig", RuntimeConfig)

    with pytest.raises(NemoNativeAdapterError, match="served target"):
        nemo_native_adapter._runtime_eval_config(
            adapter,
            config,  # type: ignore[arg-type]
            target_url="http://managed-endpoint/v1",
            target_model_id="launcher-served-name",
        )


def test_the_descriptor_names_every_other_file_and_pins_the_dataset(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection([_row("t1"), _row("t2")]), tmp_path)
    root = tmp_path / NEMO_EVALUATOR_ROOT

    bundle = NemoEvaluatorBundle.model_validate(_read_json(root / NEMO_BUNDLE_FILE))

    assert bundle.dataset_file == NEMO_DATASET_FILE
    assert bundle.dataset_schema_file == NEMO_DATASET_SCHEMA_FILE
    assert bundle.metadata_file == NEMO_METADATA_FILE
    assert bundle.evaluator_config_file == NEMO_EVALUATOR_CONFIG_FILE
    assert bundle.system_prompt_file == NEMO_SYSTEM_PROMPT_FILE
    assert bundle.record_count == 2
    assert bundle.dataset_content_hash == export_tree_hash(root, (NEMO_DATASET_FILE,))
    assert artifact.content_hash != bundle.dataset_content_hash


def test_the_descriptor_cites_the_benchmark_it_was_derived_from(tmp_path: Path) -> None:
    projection = _projection()

    write_nemo_evaluator_bundle(projection, tmp_path)

    bundle = NemoEvaluatorBundle.model_validate(_read_json(tmp_path / NEMO_EVALUATOR_ROOT / NEMO_BUNDLE_FILE))
    assert bundle.source.benchmark_file == "benchmark.parquet"
    assert bundle.source.benchmark_content_hash == projection.source.content_hash
    assert bundle.source.pack_id == "pack"
    assert bundle.source.pack_version == "1.0.0"
    assert bundle.source.expt_name == "expt"


def test_the_dataset_keeps_publication_order_and_one_record_per_task(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection([_row("t2"), _row("t1")]), tmp_path)

    records = _dataset(tmp_path / NEMO_EVALUATOR_ROOT)

    assert [record["task_id"] for record in records] == ["t2", "t1"]


def test_a_dataset_record_replays_the_recorded_tool_results(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection(), tmp_path)

    (record,) = _dataset(tmp_path / NEMO_EVALUATOR_ROOT)

    # Gold assistant output is reference-only; only the answer-free seed is model
    # input, and tool results are released by the adapter after a matching call.
    assert [message["role"] for message in record["seed_messages"]] == ["system", "user"]
    assert all(message["role"] not in {"assistant", "tool"} for message in record["seed_messages"])
    tool_messages = record["replay_steps"][0]["tool_results"]
    assert json.loads(tool_messages[0]["content"]) == {"trang_thai": "đã chuyển"}
    assert record["replay_steps"][0]["expected_call_indexes"] == [0]
    assert record["reference_trace"][2]["role"] == "assistant"
    assert record["expected_tool_calls"][0]["arguments"] == TRANSFER
    assert record["success_assertions"] == ["assert_transfer_committed"]
    assert record["call_order"] == "strict"
    assert record["turn_policy"] == "single_turn"


def test_nested_list_and_null_arguments_survive_the_dataset_record(tmp_path: Path) -> None:
    arguments = {
        "tai_khoan": "1",
        "ghi_chu": None,
        "tags": ["ưu tiên", 2, False],
        "options": {"nested": {"enabled": True}},
    }
    row = _row()
    row["messages"][2] = _assistant_calls([("chuyen_khoan", arguments)])
    row["expected_tool_calls"][0]["arguments"] = encode_arguments(arguments)

    write_nemo_evaluator_bundle(_projection([row]), tmp_path)

    (record,) = _dataset(tmp_path / NEMO_EVALUATOR_ROOT)
    assert record["expected_tool_calls"][0]["arguments"] == arguments
    assert record["reference_trace"][2]["tool_calls"][0]["function"]["arguments"] == canonical_json(
        arguments
    )


def test_the_dataset_schema_is_derived_from_the_record_model(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection(), tmp_path)

    schema = _read_json(tmp_path / NEMO_EVALUATOR_ROOT / NEMO_DATASET_SCHEMA_FILE)

    assert schema == NemoEvaluatorRecord.model_json_schema()
    (record,) = _dataset(tmp_path / NEMO_EVALUATOR_ROOT)
    assert set(record) <= set(schema["properties"])


def test_the_system_prompt_catalog_resolves_every_record(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection([_row("t1"), _multi_turn_row("t2")]), tmp_path)
    root = tmp_path / NEMO_EVALUATOR_ROOT

    catalog = _read_json(root / NEMO_SYSTEM_PROMPT_FILE)

    assert catalog == {"sp-1": SYSTEM_PROMPT}
    for record in _dataset(root):
        system = [message for message in record["seed_messages"] if message["role"] == "system"]
        assert catalog["sp-1"] == system[0]["content"]


def test_one_prompt_id_naming_two_prompts_is_refused() -> None:
    projection = _projection([_row("t1"), _callless_row("t2", prompt="Một lời nhắc khác.")])

    with pytest.raises(NemoEvaluatorWriteError, match="names two different prompts"):
        system_prompt_catalog(projection)


def test_a_pack_that_renders_no_system_prompt_still_exports(tmp_path: Path) -> None:
    # The export contract allows a row without a system message, and the record
    # carries its own conversation, so the catalog omits the id rather than
    # mapping it to empty text.
    row = _callless_row("t1")
    row["messages"] = [_user("Hôm nay trời thế nào?"), _assistant_text("Tôi chỉ hỗ trợ nghiệp vụ ngân hàng.")]

    artifact = write_nemo_evaluator_bundle(_projection([row]), tmp_path)

    catalog = _read_json(tmp_path / NEMO_EVALUATOR_ROOT / NEMO_SYSTEM_PROMPT_FILE)
    assert catalog == {}
    bundle, records = read_nemo_evaluator_bundle(tmp_path, artifact)
    assert bundle.record_count == 1
    assert records[0]["seed_messages"][0]["role"] == "user"


def test_one_prompt_id_that_sometimes_names_a_prompt_is_refused() -> None:
    promptless = _callless_row("t2")
    promptless["messages"] = [_user("Hôm nay trời thế nào?"), _assistant_text("Tôi chỉ hỗ trợ nghiệp vụ.")]

    for rows in ([_row("t1"), promptless], [promptless, _row("t1")]):
        with pytest.raises(NemoEvaluatorWriteError, match="a prompt for some tasks and none for others"):
            system_prompt_catalog(_projection(list(rows)))


def test_a_task_that_expects_no_call_is_still_exported(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection([_row("t1"), _callless_row("t2")]), tmp_path)
    root = tmp_path / NEMO_EVALUATOR_ROOT

    records = _dataset(root)

    assert records[1]["expected_tool_calls"] == []
    assert records[1]["category"] is None
    metadata = _read_json(root / NEMO_METADATA_FILE)
    assert metadata["counts"]["tasks_without_expected_calls"] == 1
    assert metadata["counts"]["tasks_with_assertions"] == 1


def test_ordering_is_only_scored_when_a_task_expects_several_calls(tmp_path: Path) -> None:
    single = bundle_scoring(_projection([_row()]))
    parallel = bundle_scoring(_projection([_parallel_row()]))

    assert single.metrics == ("tool_selection", "arguments")
    assert parallel.metrics == ("tool_selection", "arguments", "call_ordering")


def test_scoring_never_claims_metrics_the_bundle_cannot_support(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection([_row("t1"), _parallel_row("t2")]), tmp_path)

    bundle = NemoEvaluatorBundle.model_validate(_read_json(tmp_path / NEMO_EVALUATOR_ROOT / NEMO_BUNDLE_FILE))

    # Both need the pack's tools re-executed against oracle state, which no bundle
    # file provides; claiming them would invite scoring against a stale snapshot.
    assert "results" not in bundle.scoring.metrics
    assert "task_success" not in bundle.scoring.metrics
    assert bundle.scoring.argument_match == "canonical_json_exact"


def test_the_declared_call_order_policies_are_the_ones_the_rows_carry(tmp_path: Path) -> None:
    prefix = _parallel_row("t3")
    prefix["call_order"] = "prefix"
    prefix["call_order_prefix"] = 1
    write_nemo_evaluator_bundle(_projection([_row("t1"), _multi_turn_row("t2"), prefix]), tmp_path)

    bundle = NemoEvaluatorBundle.model_validate(_read_json(tmp_path / NEMO_EVALUATOR_ROOT / NEMO_BUNDLE_FILE))

    assert bundle.scoring.call_order_policies == ("any", "prefix", "strict")


def test_the_evaluator_config_pins_the_prompt_source_to_the_records(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection(), tmp_path)

    config = yaml.safe_load((tmp_path / NEMO_EVALUATOR_ROOT / NEMO_EVALUATOR_CONFIG_FILE).read_text(encoding="utf-8"))

    assert config["task"]["dataset"] == NEMO_DATASET_FILE
    assert config["task"]["type"] == "function_calling"
    assert config["kind"] == "nemotron_byob_function_calling_input"
    assert config["execution"]["direct_launcher_config"] is False
    assert config["execution"]["requires_registered_environment"] is True
    assert config["prompt"]["source"] == "seed_messages"
    assert config["prompt"]["system_prompts"] == NEMO_SYSTEM_PROMPT_FILE
    assert config["prompt"]["reference_trace_is_model_input"] is False
    assert config["interaction"]["steps_field"] == "replay_steps"
    assert config["interaction"]["release_tool_results_after_expected_call_match"] is True
    assert config["scoring"]["metrics"] == ["tool_selection", "arguments"]
    assert config["reporting"]["languages"] == ["vi"]


def test_the_metadata_describes_the_shape_without_scoring_anything(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection([_row("t1"), _parallel_row("t2")]), tmp_path)

    metadata = _read_json(tmp_path / NEMO_EVALUATOR_ROOT / NEMO_METADATA_FILE)

    assert metadata["records"] == 2
    assert metadata["task_name"] == "pack"
    assert metadata["counts"]["expected_tool_calls"] == 3
    assert metadata["counts"]["tasks_without_expected_calls"] == 0
    assert metadata["counts"]["tasks_with_assertions"] == 2
    assert metadata["categories"] == ["cards", "payments"]
    assert metadata["difficulties"] == ["easy", "hard"]
    assert metadata["projection"]["expt_name"] == "expt"
    assert metadata["projection"]["parallel_call_rows"] == 1
    assert metadata["projection"]["source"]["file"] == "benchmark.parquet"


@pytest.mark.parametrize(
    "pack_id,prefix",
    [
        ("pack", "pack"),
        ("Banking VN", "banking_vn"),
        ("bank/vn:v2", "bank_vn_v2"),
        ("__pack__", "pack"),
        ("1pack", "1pack"),
        ("银行工具", "pack"),
    ],
)
def test_a_bundle_task_name_is_derived_deterministically(pack_id: str, prefix: str) -> None:
    expected = prefix
    if prefix != pack_id.lower():
        expected += f"-{hashlib.sha256(pack_id.encode('utf-8')).hexdigest()[:12]}"
    assert bundle_task_name(pack_id) == expected
    assert bundle_task_name(pack_id) == expected


def test_normalized_task_names_that_would_otherwise_collide_are_distinct() -> None:
    assert bundle_task_name("bank/vn") != bundle_task_name("bank:vn")


def test_the_normalized_task_name_keeps_the_verbatim_pack_id_beside_it(tmp_path: Path) -> None:
    rows = [_row(pack_id="Banking VN", src="Banking VN:tpl")]

    write_nemo_evaluator_bundle(_projection(rows), tmp_path)

    metadata = _read_json(tmp_path / NEMO_EVALUATOR_ROOT / NEMO_METADATA_FILE)
    assert metadata["task_name"].startswith("banking_vn-")
    assert metadata["projection"]["pack_id"] == "Banking VN"


def test_vietnamese_text_is_written_readable_rather_than_escaped(tmp_path: Path) -> None:
    write_nemo_evaluator_bundle(_projection(), tmp_path)
    root = tmp_path / NEMO_EVALUATOR_ROOT

    for name in (NEMO_DATASET_FILE, NEMO_SYSTEM_PROMPT_FILE):
        text = (root / name).read_text(encoding="utf-8")
        assert "\\u" not in text
    assert SYSTEM_PROMPT in (root / NEMO_SYSTEM_PROMPT_FILE).read_text(encoding="utf-8")


def test_two_writes_of_one_projection_produce_the_same_bytes(tmp_path: Path) -> None:
    rows = [_row("t1"), _parallel_row("t2"), _multi_turn_row("t3")]
    first = write_nemo_evaluator_bundle(_projection(rows), tmp_path / "first")
    second = write_nemo_evaluator_bundle(_projection(rows), tmp_path / "second")

    assert first.content_hash == second.content_hash
    for name in NEMO_BUNDLE_FILES:
        left = (tmp_path / "first" / NEMO_EVALUATOR_ROOT / name).read_bytes()
        assert left == (tmp_path / "second" / NEMO_EVALUATOR_ROOT / name).read_bytes()
        assert left.endswith(b"\n")


def test_the_tree_hash_notices_any_bundle_file_changing(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    root = tmp_path / NEMO_EVALUATOR_ROOT

    for name in NEMO_BUNDLE_FILES:
        path = root / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        assert export_tree_hash(root, NEMO_BUNDLE_FILES) != artifact.content_hash
        path.write_bytes(original)
    assert export_tree_hash(root, NEMO_BUNDLE_FILES) == artifact.content_hash


def test_nothing_is_written_when_the_projection_cannot_be_encoded(tmp_path: Path) -> None:
    conflicting = _callless_row("t2", prompt="Một lời nhắc khác.")

    with pytest.raises(NemoEvaluatorWriteError, match="names two different prompts"):
        write_nemo_evaluator_bundle(_projection([_row("t1"), conflicting]), tmp_path)

    # Five of six files and no way to tell which is missing is worse than none.
    assert not (tmp_path / NEMO_EVALUATOR_ROOT).exists()


def test_a_file_left_by_an_earlier_write_does_not_ship_inside_the_bundle(tmp_path: Path) -> None:
    root = tmp_path / NEMO_EVALUATOR_ROOT
    root.mkdir(parents=True)
    (root / "leftover.jsonl").write_text("{}\n", encoding="utf-8")

    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)

    # The tree hash covers the six declared names only, so an unlisted file would
    # travel with the bundle unnoticed.
    assert not (root / "leftover.jsonl").exists()
    assert sorted(path.name for path in root.iterdir()) == sorted(NEMO_BUNDLE_FILES)
    assert artifact.content_hash == export_tree_hash(root, NEMO_BUNDLE_FILES)


def test_an_edited_config_or_metadata_file_is_refused_on_read_back(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    root = tmp_path / NEMO_EVALUATOR_ROOT

    # The descriptor pins only the dataset, so these would otherwise read back as
    # though the writer had produced them.
    for name in (NEMO_EVALUATOR_CONFIG_FILE, NEMO_METADATA_FILE, NEMO_DATASET_SCHEMA_FILE):
        path = root / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(NemoEvaluatorWriteError, match="no longer matches the tree hash"):
            read_nemo_evaluator_bundle(tmp_path, artifact)
        path.write_bytes(original)
    read_nemo_evaluator_bundle(tmp_path, artifact)


def test_the_bundle_reads_back_as_it_was_written(tmp_path: Path) -> None:
    rows = [_row("t1"), _multi_turn_row("t2")]
    artifact = write_nemo_evaluator_bundle(_projection(rows), tmp_path)

    bundle, records = read_nemo_evaluator_bundle(tmp_path, artifact)

    assert bundle.record_count == 2
    assert [record["task_id"] for record in records] == ["t1", "t2"]


def test_a_moved_bundle_keeps_its_digest(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    moved = tmp_path / "archive"
    moved.mkdir()
    (tmp_path / NEMO_EVALUATOR_ROOT).rename(moved / NEMO_EVALUATOR_ROOT.split("/")[-1])

    # The digest covers bundle-relative names, so archiving the directory elsewhere
    # does not invalidate the descriptor that travels inside it.
    assert export_tree_hash(moved / "nemo_evaluator_bundle", NEMO_BUNDLE_FILES) == artifact.content_hash


def test_a_missing_bundle_file_is_refused(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    (tmp_path / NEMO_EVALUATOR_ROOT / NEMO_METADATA_FILE).unlink()

    with pytest.raises(NemoEvaluatorWriteError, match=f"missing {NEMO_METADATA_FILE}"):
        read_nemo_evaluator_bundle(tmp_path, artifact)


def test_a_dataset_edited_after_the_descriptor_pinned_it_is_refused(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    path = tmp_path / NEMO_EVALUATOR_ROOT / NEMO_DATASET_FILE
    path.write_text(path.read_text(encoding="utf-8").replace("chuyen_khoan", "liet_ke_the"), encoding="utf-8")

    with pytest.raises(NemoEvaluatorWriteError, match="changed after the descriptor pinned its hash"):
        read_nemo_evaluator_bundle(tmp_path, artifact)


def test_a_dropped_dataset_line_contradicts_the_descriptor(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection([_row("t1"), _row("t2")]), tmp_path)
    path = tmp_path / NEMO_EVALUATOR_ROOT / NEMO_DATASET_FILE
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")

    with pytest.raises(NemoEvaluatorWriteError, match="but the descriptor claims 2"):
        read_nemo_evaluator_bundle(tmp_path, artifact)


def test_a_malformed_dataset_line_is_refused(tmp_path: Path) -> None:
    artifact = write_nemo_evaluator_bundle(_projection(), tmp_path)
    path = tmp_path / NEMO_EVALUATOR_ROOT / NEMO_DATASET_FILE
    original = path.read_text(encoding="utf-8")

    path.write_text(original + "\n", encoding="utf-8")
    with pytest.raises(NemoEvaluatorWriteError, match="is blank"):
        read_nemo_evaluator_bundle(tmp_path, artifact)

    path.write_text(original + "{not json\n", encoding="utf-8")
    with pytest.raises(NemoEvaluatorWriteError, match="not valid JSON"):
        read_nemo_evaluator_bundle(tmp_path, artifact)

    path.write_text(original + "[1]\n", encoding="utf-8")
    with pytest.raises(NemoEvaluatorWriteError, match="not a JSON object"):
        read_nemo_evaluator_bundle(tmp_path, artifact)


def test_an_artifact_must_describe_the_benchmark_it_projected() -> None:
    source = ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=2)

    with pytest.raises(ValidationError, match="every row of the benchmark"):
        NemoEvaluatorArtifact(rows=1, content_hash="sha256:" + "1" * 64, source=source)


def test_an_incomplete_bundle_is_not_a_bundle() -> None:
    source = ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=1)

    with pytest.raises(ValidationError, match="canonical files exactly once"):
        NemoEvaluatorArtifact(
            rows=1,
            content_hash="sha256:" + "1" * 64,
            source=source,
            files=(NEMO_BUNDLE_FILE, NEMO_DATASET_FILE),
        )


def test_an_artifact_refuses_duplicate_or_reordered_files() -> None:
    source = ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=1)
    for files in (
        NEMO_BUNDLE_FILES + (NEMO_BUNDLE_FILE,),
        tuple(reversed(NEMO_BUNDLE_FILES)),
    ):
        with pytest.raises(ValidationError, match="canonical files exactly once"):
            NemoEvaluatorArtifact(
                rows=1,
                content_hash="sha256:" + "1" * 64,
                source=source,
                files=files,
            )


@pytest.mark.parametrize("root", ["../nemo", "/tmp/nemo", r"exports\nemo"])
def test_an_artifact_refuses_a_root_that_can_escape_the_run_directory(root: str) -> None:
    source = ProjectionSource(file="benchmark.parquet", content_hash="sha256:" + "0" * 64, rows=1)

    with pytest.raises(ValidationError, match="relative|POSIX path|inside"):
        NemoEvaluatorArtifact(
            rows=1,
            content_hash="sha256:" + "1" * 64,
            source=source,
            root=root,
        )
