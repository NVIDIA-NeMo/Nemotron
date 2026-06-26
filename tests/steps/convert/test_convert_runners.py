# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf

from nemotron.steps._runners import convert


def _install_fake_megatron_bridge(monkeypatch: pytest.MonkeyPatch, auto_bridge: type) -> None:
    megatron_mod = ModuleType("megatron")
    megatron_mod.__path__ = []
    bridge_mod = ModuleType("megatron.bridge")
    bridge_mod.AutoBridge = auto_bridge
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge_mod)


def test_convert_steps_have_default_configs(steps_root: Path) -> None:
    for step_id in ("hf_to_megatron", "megatron_to_hf", "merge_lora"):
        config_path = steps_root / "convert" / step_id / "config" / "default.yaml"
        assert config_path.exists(), f"{config_path} is required for nemotron steps run --dry-run"
        assert isinstance(OmegaConf.to_container(OmegaConf.load(config_path), resolve=False), dict)


@pytest.mark.parametrize("step_id", ["hf_to_megatron", "megatron_to_hf"])
def test_distributed_defaults_are_configured(steps_root: Path, step_id: str) -> None:
    config_path = steps_root / "convert" / step_id / "config" / "default.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)

    assert "runtime" not in cfg
    assert cfg["run"]["env"]["container_image"] == "nvcr.io/nvidia/nemo:26.04"
    assert cfg["distributed"] is True
    assert cfg["ep"] == "${oc.env:NEMOTRON_CONVERT_EP,8}"


@pytest.mark.parametrize("step_id", ["hf_to_megatron", "megatron_to_hf"])
def test_specialized_convert_config_overrides_runtime(steps_root: Path, step_id: str) -> None:
    config_path = steps_root / "convert" / step_id / "config" / "tiny.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)

    env = cfg["run"]["env"]

    assert env["container_image"] == "nvcr.io/nvidia/nemo:26.02.nemotron_3_super"
    assert env["nodes"] == "${oc.env:NEMOTRON_CONVERT_NNODES,8}"
    assert "NVIDIA-NeMo/Megatron-Bridge.git@main" in env["mounts"][0]
    assert cfg["script"]["path"] == "/opt/Megatron-Bridge-main/examples/conversion/convert_checkpoints_multi_gpu.py"
    if step_id == "megatron_to_hf":
        assert cfg["duplicate_pp_tensor_policy"] == "first"
    assert cfg["tp"] == "${oc.env:NEMOTRON_CONVERT_TP,8}"
    assert cfg["pp"] == "${oc.env:NEMOTRON_CONVERT_PP,4}"
    assert cfg["ep"] == "${oc.env:NEMOTRON_CONVERT_EP,8}"


def test_hf_to_megatron_forwards_autobridge_args(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeAutoBridge:
        @staticmethod
        def import_ckpt(**kwargs):
            calls.append(kwargs)

    _install_fake_megatron_bridge(monkeypatch, FakeAutoBridge)

    convert.import_hf_to_megatron(
        {
            "hf_model_id": "hf-source",
            "megatron_path": "/tmp/megatron",
            "device_map": "auto",
            "trust_remote_code": True,
        }
    )

    assert calls == [
        {
            "hf_model_id": "hf-source",
            "megatron_path": "/tmp/megatron",
            "device_map": "auto",
            "trust_remote_code": True,
        }
    ]


def test_hf_to_megatron_prefers_torch_dtype_over_deprecated_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    sentinel_dtype = object()

    class FakeAutoBridge:
        @staticmethod
        def import_ckpt(**kwargs):
            calls.append(kwargs)

    _install_fake_megatron_bridge(monkeypatch, FakeAutoBridge)
    monkeypatch.setattr(convert, "_torch_dtype", lambda name: sentinel_dtype if name == "float16" else name)

    convert.import_hf_to_megatron(
        {
            "hf_model_id": "hf-source",
            "megatron_path": "/tmp/megatron",
            "dtype": "bfloat16",
            "torch_dtype": "float16",
        }
    )

    assert calls[0]["torch_dtype"] is sentinel_dtype


def test_megatron_to_hf_prefers_hf_pretrained_export(monkeypatch: pytest.MonkeyPatch) -> None:
    from_hf_pretrained_calls: list[tuple] = []
    export_calls: list[dict] = []

    class FakeBridge:
        def export_ckpt(self, **kwargs):
            export_calls.append(kwargs)

    class FakeAutoBridge:
        @staticmethod
        def from_hf_pretrained(*args, **kwargs):
            from_hf_pretrained_calls.append((args, kwargs))
            return FakeBridge()

    _install_fake_megatron_bridge(monkeypatch, FakeAutoBridge)

    convert.export_megatron_to_hf(
        megatron_path="/tmp/megatron/iter_0000001",
        hf_model_id="hf-config",
        hf_path="/tmp/hf",
        trust_remote_code=True,
        show_progress=False,
        strict=False,
    )

    assert from_hf_pretrained_calls == [(("hf-config",), {"trust_remote_code": True})]
    assert export_calls == [
        {
            "megatron_path": "/tmp/megatron/iter_0000001",
            "hf_path": "/tmp/hf",
            "show_progress": False,
            "strict": False,
        }
    ]


def test_megatron_to_hf_falls_back_to_auto_config_export(monkeypatch: pytest.MonkeyPatch) -> None:
    from_auto_config_calls: list[tuple] = []

    class FakeBridge:
        def export_ckpt(self, **_kwargs):
            pass

    class FakeAutoBridge:
        @staticmethod
        def from_auto_config(*args, **kwargs):
            from_auto_config_calls.append((args, kwargs))
            return FakeBridge()

    _install_fake_megatron_bridge(monkeypatch, FakeAutoBridge)

    convert.export_megatron_to_hf(
        megatron_path="/tmp/megatron/iter_0000001",
        hf_model_id="hf-config",
        hf_path="/tmp/hf",
        trust_remote_code=True,
    )

    assert from_auto_config_calls == [(("/tmp/megatron/iter_0000001", "hf-config"), {"trust_remote_code": True})]


def test_megatron_lora_merge_command_uses_cpu_script_by_default() -> None:
    cmd = convert.build_megatron_lora_merge_command(
        {
            "upstream_script": "/opt/Megatron-Bridge/examples/peft/merge_lora.py",
            "lora_checkpoint": "/tmp/lora",
            "hf_model_id": "hf-config",
            "base_megatron_path": "/tmp/base-megatron",
            "cpu": True,
            "tp": 1,
            "pp": 1,
            "ep": 1,
        },
        merged_megatron_path="/tmp/merged-megatron",
    )

    assert cmd[:2] == [sys.executable, "/opt/Megatron-Bridge/examples/peft/merge_lora.py"]
    assert "--lora-checkpoint" in cmd
    assert "/tmp/lora" in cmd
    assert "--pretrained" in cmd
    assert "/tmp/base-megatron" in cmd
    assert "--cpu" in cmd


def test_merge_backend_auto_uses_base_path_shape() -> None:
    assert convert._resolve_merge_backend({"backend": "auto", "base_hf_path": "/tmp/base-hf"}) == "hf_peft"
    assert (
        convert._resolve_merge_backend({"backend": "auto", "base_megatron_path": "/tmp/base-megatron"})
        == "megatron_bridge"
    )


@pytest.mark.parametrize("backend", ["hf", "peft", "megatron", "mbridge"])
def test_merge_backend_hidden_aliases_are_rejected(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setattr(convert, "load_convert_config", lambda _default_config: {"backend": backend})

    with pytest.raises(ValueError, match="auto, hf_peft, megatron_bridge"):
        convert.run_merge_lora(Path("unused.yaml"))


def test_hf_peft_adapter_path_resolves_nested_latest_checkpoint(tmp_path: Path) -> None:
    old_adapter = tmp_path / "global_step5" / "model"
    new_adapter = tmp_path / "global_step10" / "model"
    old_adapter.mkdir(parents=True)
    new_adapter.mkdir(parents=True)
    (old_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (new_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert convert._resolve_hf_peft_adapter_path(str(tmp_path)) == str(new_adapter)


def test_hf_peft_adapter_path_error_points_to_adapter_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="adapter_config.json"):
        convert._resolve_hf_peft_adapter_path(str(tmp_path))


def test_missing_required_config_value_is_clear() -> None:
    with pytest.raises(ValueError, match="hf_model_id"):
        convert.import_hf_to_megatron({"megatron_path": "/tmp/megatron"})


def test_distributed_import_command_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    cmd = convert.build_distributed_conversion_command(
        "import",
        {
            "hf_model_id": "hf-source",
            "megatron_path": "/tmp/megatron",
            "torch_dtype": "bfloat16",
            "trust_remote_code": True,
            "tp": 2,
            "pp": 1,
            "ep": 4,
            "etp": 1,
            "torchrun": {"nproc_per_node": 8},
        },
    )

    assert cmd[:2] == ["torchrun", "--nproc_per_node=8"]
    assert cmd[2] == sys.executable
    assert cmd[3].endswith("/examples/conversion/convert_checkpoints_multi_gpu.py")
    assert cmd[cmd.index("--hf-model") + 1] == "hf-source"
    assert cmd[cmd.index("--megatron-path") + 1] == "/tmp/megatron"
    assert cmd[cmd.index("--tp") + 1] == "2"
    assert cmd[cmd.index("--ep") + 1] == "4"
    assert "--trust-remote-code" in cmd


def test_distributed_command_reuses_existing_world(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")

    cmd = convert.build_distributed_conversion_command(
        "import",
        {
            "hf_model_id": "hf-source",
            "megatron_path": "/tmp/megatron",
            "ep": 8,
            "torchrun": {"nproc_per_node": 8},
        },
    )

    assert cmd[0] == sys.executable
    assert cmd[1].endswith("/examples/conversion/convert_checkpoints_multi_gpu.py")
    assert "torchrun" not in cmd
    assert "--distributed-timeout-minutes" not in cmd


def test_distributed_env_preserves_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/src")

    env = convert._distributed_conversion_env({})

    assert env["PYTHONPATH"] == "/existing/src"


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (
            {"tp": 1, "pp": 1, "ep": 1, "etp": 1, "torchrun": {"nproc_per_node": 8}},
            "no model or expert parallelism",
        ),
        (
            {"tp": 8, "pp": 4, "ep": 1, "etp": 1, "torchrun": {"nproc_per_node": 8}},
            r"tp\*pp\*cp",
        ),
        (
            {"tp": 1, "pp": 4, "ep": 8, "etp": 1, "torchrun": {"nproc_per_node": 16}},
            r"etp\*ep\*pp",
        ),
    ],
)
def test_distributed_command_rejects_invalid_parallelism(
    monkeypatch: pytest.MonkeyPatch,
    cfg: dict,
    match: str,
) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    with pytest.raises(ValueError, match=match):
        convert.build_distributed_conversion_command(
            "export",
            {
                "hf_model_id": "hf-config",
                "megatron_path": "/tmp/megatron",
                "hf_path": "/tmp/hf",
                **cfg,
            },
        )


def test_distributed_command_uses_configured_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    cmd = convert.build_distributed_conversion_command(
        "export",
        {
            "hf_model_id": "hf-config",
            "megatron_path": "/tmp/megatron",
            "hf_path": "/tmp/hf",
            "tp": 8,
            "pp": 4,
            "ep": 8,
            "etp": 1,
            "run": {"env": {"nodes": 8, "nprocs_per_node": 8}},
            "torchrun": {"nproc_per_node": 8},
        },
    )

    assert "--nnodes=8" in cmd
    assert cmd[cmd.index("--pp") + 1] == "4"


def test_distributed_command_can_wrap_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    cmd = convert.build_distributed_conversion_command(
        "export",
        {
            "hf_model_id": "hf-config",
            "megatron_path": "/tmp/megatron",
            "hf_path": "/tmp/hf",
            "duplicate_pp_tensor_policy": "first",
            "tp": 8,
            "pp": 4,
            "ep": 8,
            "script": {"path": "/opt/Megatron-Bridge-main/examples/conversion/convert_checkpoints_multi_gpu.py"},
            "torchrun": {"nproc_per_node": 8, "nnodes": 8},
        },
    )

    assert cmd[:3] == ["torchrun", "--nproc_per_node=8", "--nnodes=8"]
    python_index = cmd.index(sys.executable)
    assert cmd[python_index : python_index + 2] == [sys.executable, "-c"]
    assert "MegatronParamMapping.broadcast_from_pp_rank" in cmd[python_index + 2]
    assert (
        convert._distributed_converter_script_path(cmd)
        == "/opt/Megatron-Bridge-main/examples/conversion/convert_checkpoints_multi_gpu.py"
    )


def test_distributed_export_command_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    cmd = convert.build_distributed_conversion_command(
        "export",
        {
            "hf_model_id": "hf-config",
            "megatron_path": "/tmp/megatron/iter_0000001",
            "hf_path": "/tmp/hf",
            "show_progress": False,
            "strict": False,
            "distributed_save": True,
            "save_every_n_ranks": 2,
            "ep": 4,
            "torchrun": {"nproc_per_node": 4},
        },
    )

    assert cmd[:2] == ["torchrun", "--nproc_per_node=4"]
    assert "export" in cmd
    assert cmd[cmd.index("--hf-path") + 1] == "/tmp/hf"
    assert "--no-progress" in cmd
    assert "--not-strict" in cmd
    assert "--distributed-save" in cmd
    assert cmd[cmd.index("--save-every-n-ranks") + 1] == "2"


def test_distributed_command_respects_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    cmd = convert.build_distributed_conversion_command(
        "export",
        {
            "hf_model_id": "hf-config",
            "megatron_path": "/tmp/megatron",
            "hf_path": "/tmp/hf",
            "trust_remote_code": True,
            "distributed_save": True,
            "save_every_n_ranks": 1,
            "tp": 8,
            "pp": 4,
            "ep": 8,
            "etp": 1,
            "script": {
                "path": "/opt/Megatron-Bridge/examples/conversion/convert_checkpoints.py",
                "supports_trust_remote_code": False,
                "supports_export_torch_dtype": False,
                "supports_distributed_save": False,
                "supports_save_every_n_ranks": False,
            },
            "torchrun": {"nproc_per_node": 8, "nnodes": 8},
        },
    )

    assert "--torch-dtype" not in cmd
    assert "--trust-remote-code" not in cmd
    assert "--distributed-save" not in cmd
    assert "--save-every-n-ranks" not in cmd
    assert cmd[cmd.index("--tp") + 1] == "8"
    assert cmd[cmd.index("--pp") + 1] == "4"
    assert cmd[cmd.index("--ep") + 1] == "8"
    assert cmd[cmd.index("--etp") + 1] == "1"


def test_runner_uses_distributed_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []
    cfg = {
        "distributed": True,
        "hf_model_id": "hf-config",
        "megatron_path": "/tmp/megatron",
        "hf_path": "/tmp/hf",
    }
    monkeypatch.setattr(convert, "load_convert_config", lambda _default_config: cfg)
    monkeypatch.setattr(convert, "exec_distributed_conversion", lambda direction, config: calls.append((direction, dict(config))))

    convert.run_megatron_to_hf(Path("unused.yaml"))

    assert calls == [("export", cfg)]
