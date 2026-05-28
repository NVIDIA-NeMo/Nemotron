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
