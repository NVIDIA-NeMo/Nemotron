"""Run with the pinned NeMo RL environment; exercise the actual worker initializer."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_rl.models.generation.vllm.vllm_worker")

from nemo_rl.models.generation.vllm import vllm_worker

from nemotron.recipes.parse.training import native_vllm


@pytest.fixture
def native_config(monkeypatch):
    monkeypatch.setattr(
        native_vllm.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(architectures=["RadioMBartPrefixForCausalLM"]),
    )
    monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "EXISTING")
    return {
        "model_name": "probe",
        "vllm_cfg": {
            "async_engine": False,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
            "precision": "bfloat16",
        },
    }


def test_native_worker_preserves_upstream_module_and_environment_forwarding(native_config):
    original = vllm_worker._apply_vllm_patches
    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    worker._init_config(native_config, None, 0.5, 42, ["CUSTOMER_TASK_PATH"])
    assert worker.llm is None  # Non-owner initialization requires no GPU.
    assert vllm_worker._apply_vllm_patches is original
    assert {"EXISTING", "CUSTOMER_TASK_PATH", "RAY_ENABLE_UV_RUN_RUNTIME_ENV"} <= set(
        os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"].split(",")
    )


@pytest.mark.parametrize("package", ["vllm", "transformers"])
def test_native_worker_rejects_unvalidated_version(native_config, monkeypatch, package):
    versions = {"vllm": "0.33.0", "transformers": "5.16.1", package: "999"}
    monkeypatch.setattr(native_vllm, "version", versions.__getitem__)
    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    with pytest.raises(RuntimeError, match=f"requires {package}"):
        worker._init_config(native_config, None, 0.5, 42, None)


def test_native_worker_rejects_another_architecture(native_config, monkeypatch):
    monkeypatch.setattr(
        native_vllm.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(architectures=["OtherModel"])
    )
    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    with pytest.raises(ValueError, match="RadioMBartPrefix"):
        worker._init_config(native_config, None, 0.5, 42, None)


@pytest.mark.parametrize(
    "setting,value",
    [("async_engine", True), ("tensor_parallel_size", 2), ("pipeline_parallel_size", 2), ("expert_parallel_size", 2)],
)
def test_native_worker_rejects_unvalidated_parallelism(native_config, setting, value):
    native_config["vllm_cfg"][setting] = value
    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    with pytest.raises(ValueError, match="single-GPU"):
        worker._init_config(native_config, None, 0.5, 42, None)


def test_native_worker_refuses_changed_upstream_code():
    def changed():
        raise AssertionError("must never run")

    with pytest.raises(RuntimeError, match="initializer changed"):
        native_vllm._native_initializer(changed)


def test_native_worker_rejects_quantized_generation(native_config):
    native_config["quant_cfg"] = {"enabled": True}
    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    with pytest.raises(ValueError, match="quantized generation"):
        worker._init_config(native_config, None, 0.5, 42, None)


@pytest.mark.parametrize("feature", ["nemotron_h_fp32_lm_head", "require_moe_routed_experts_capture"])
def test_native_worker_refuses_to_silently_drop_patch_features(feature):
    with pytest.raises(ValueError, match="patch features"):
        native_vllm._forward_environment(**{feature: True})


@pytest.mark.parametrize("owner,colocated", [(True, True), (False, True), (True, False)])
def test_native_shutdown_wakes_all_tags_before_parent_cleanup(monkeypatch, owner, colocated):
    calls = []
    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    worker.cfg = {"colocated": {"enabled": colocated}}
    worker.llm = SimpleNamespace(wake_up=lambda *a, **kw: calls.append(("wake", a, kw))) if owner else None
    monkeypatch.setattr(vllm_worker.VllmGenerationWorkerImpl, "shutdown", lambda self: calls.append("cleanup") or True)

    assert worker.shutdown() is True
    assert calls == ([("wake", (), {}), "cleanup"] if owner and colocated else ["cleanup"])


@pytest.mark.parametrize("failure", ["wake", "cleanup"])
def test_native_shutdown_cleans_up_and_preserves_errors(monkeypatch, failure):
    calls = []

    def action(name):
        calls.append(name)
        if name == failure:
            raise RuntimeError(name)
        return True

    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    worker.cfg = {"colocated": {"enabled": True}}
    worker.llm = SimpleNamespace(wake_up=lambda: action("wake"))
    monkeypatch.setattr(vllm_worker.VllmGenerationWorkerImpl, "shutdown", lambda self: action("cleanup"))
    with pytest.raises(RuntimeError, match=failure):
        worker.shutdown()
    assert calls == ["wake", "cleanup"]


def test_native_shutdown_raises_when_parent_reports_cleanup_failure(monkeypatch):
    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    worker.cfg = {"colocated": {"enabled": True}}
    calls = []
    worker.llm = SimpleNamespace(wake_up=lambda: calls.append("wake"))
    monkeypatch.setattr(vllm_worker.VllmGenerationWorkerImpl, "shutdown", lambda self: False)

    with pytest.raises(RuntimeError, match="Native vLLM worker cleanup failed"):
        worker.shutdown()
    assert calls == ["wake"]


def test_native_shutdown_preserves_wake_error_when_parent_returns_false(monkeypatch):
    def wake():
        raise ValueError("wake failed")

    worker = object.__new__(native_vllm.NativeParseGenerationWorkerImpl)
    worker.cfg = {"colocated": {"enabled": True}}
    worker.llm = SimpleNamespace(wake_up=wake)
    monkeypatch.setattr(vllm_worker.VllmGenerationWorkerImpl, "shutdown", lambda self: False)

    with pytest.raises(ValueError, match="wake failed"):
        worker.shutdown()
