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

"""Shared runners for checkpoint conversion steps."""

from __future__ import annotations

import inspect
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemotron.kit.train_script import apply_hydra_overrides, load_omegaconf_yaml, parse_config_and_overrides

_PEFT_ADAPTER_CONFIG = "adapter_config.json"


def load_convert_config(default_config: Path) -> dict[str, Any]:
    """Load the step YAML and CLI overrides as a plain resolved mapping."""
    config_path, cli_overrides = parse_config_and_overrides(default_config=default_config)
    raw = apply_hydra_overrides(load_omegaconf_yaml(config_path), cli_overrides)
    cfg = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping")
    return cfg


def run_hf_to_megatron(default_config: Path) -> None:
    cfg = load_convert_config(default_config)
    import_hf_to_megatron(cfg)


def run_megatron_to_hf(default_config: Path) -> None:
    cfg = load_convert_config(default_config)
    export_megatron_to_hf(
        megatron_path=_required_str(cfg, "megatron_path"),
        hf_model_id=_required_str(cfg, "hf_model_id"),
        hf_path=_required_str(cfg, "hf_path"),
        trust_remote_code=_as_bool(cfg.get("trust_remote_code", True)),
        show_progress=_as_bool(cfg.get("show_progress", True)),
        strict=_as_bool(cfg.get("strict", True)),
    )


def run_merge_lora(default_config: Path) -> None:
    cfg = load_convert_config(default_config)
    backend = _resolve_merge_backend(cfg)

    if backend == "hf_peft":
        merge_hf_peft_lora(cfg)
        return

    if backend == "megatron_bridge":
        merge_megatron_bridge_lora(cfg)
        return

    raise ValueError("merge_lora backend must be one of: auto, hf_peft, megatron_bridge")


def import_hf_to_megatron(cfg: Mapping[str, Any]) -> None:
    dtype = cfg.get("torch_dtype") or cfg.get("dtype")
    kwargs = {
        "hf_model_id": _required_str(cfg, "hf_model_id"),
        "megatron_path": _required_str(cfg, "megatron_path"),
    }
    if dtype:
        kwargs["torch_dtype"] = _torch_dtype(str(dtype))
    if cfg.get("device_map"):
        kwargs["device_map"] = str(cfg["device_map"])
    if cfg.get("trust_remote_code") is not None:
        kwargs["trust_remote_code"] = _as_bool(cfg["trust_remote_code"])

    from megatron.bridge import AutoBridge

    print(f"Importing HF checkpoint {kwargs['hf_model_id']} -> {kwargs['megatron_path']}", flush=True)
    _call_with_supported_kwargs(AutoBridge.import_ckpt, **kwargs)


def export_megatron_to_hf(
    *,
    megatron_path: str,
    hf_model_id: str,
    hf_path: str,
    trust_remote_code: bool = True,
    show_progress: bool = True,
    strict: bool = True,
) -> None:
    from megatron.bridge import AutoBridge

    print(f"Exporting Megatron checkpoint {megatron_path} -> {hf_path}", flush=True)
    bridge = _autobridge_for_hf_export(
        AutoBridge,
        megatron_path=megatron_path,
        hf_model_id=hf_model_id,
        trust_remote_code=trust_remote_code,
    )
    _call_with_supported_kwargs(
        bridge.export_ckpt,
        megatron_path=megatron_path,
        hf_path=hf_path,
        show_progress=show_progress,
        strict=strict,
    )


def merge_hf_peft_lora(cfg: Mapping[str, Any]) -> None:
    """Merge a HuggingFace PEFT adapter into its HF base and save a standalone checkpoint."""
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "convert/merge_lora backend=hf_peft requires the optional 'peft' package. "
            "Install PEFT in the runtime image or use backend=megatron_bridge for Megatron-Bridge adapters."
        ) from exc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_hf_path = _required_str(cfg, "base_hf_path")
    requested_lora_checkpoint = _required_str(cfg, "lora_checkpoint")
    lora_checkpoint = _resolve_hf_peft_adapter_path(requested_lora_checkpoint)
    output_hf_path = Path(_required_str(cfg, "output_hf_path"))

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": _as_bool(cfg.get("trust_remote_code", True)),
        "low_cpu_mem_usage": _as_bool(cfg.get("low_cpu_mem_usage", True)),
    }
    if cfg.get("device_map"):
        model_kwargs["device_map"] = str(cfg["device_map"])
    if cfg.get("torch_dtype"):
        dtype = str(cfg["torch_dtype"])
        model_kwargs["torch_dtype"] = dtype if dtype == "auto" else _torch_dtype(dtype)

    if lora_checkpoint != requested_lora_checkpoint:
        print(f"Resolved HF PEFT adapter {requested_lora_checkpoint} -> {lora_checkpoint}", flush=True)
    print(f"Merging HF PEFT adapter {lora_checkpoint} into {base_hf_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_hf_path, **model_kwargs)
    merged_model = PeftModel.from_pretrained(model, lora_checkpoint).merge_and_unload()
    output_hf_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_hf_path, safe_serialization=_as_bool(cfg.get("safe_serialization", True)))

    if _as_bool(cfg.get("save_tokenizer", True)):
        tokenizer = AutoTokenizer.from_pretrained(base_hf_path, trust_remote_code=model_kwargs["trust_remote_code"])
        tokenizer.save_pretrained(output_hf_path)
    print(f"Merged HF checkpoint written to {output_hf_path}", flush=True)


def merge_megatron_bridge_lora(cfg: Mapping[str, Any]) -> None:
    """Merge a Megatron-Bridge adapter, then optionally export the merged checkpoint to HF."""
    merged_megatron_path = _required_str(cfg, "output_megatron_path")
    command = build_megatron_lora_merge_command(cfg, merged_megatron_path=merged_megatron_path)

    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)

    if not _as_bool(cfg.get("export_hf", True)):
        return

    export_megatron_to_hf(
        megatron_path=merged_megatron_path,
        hf_model_id=_required_str(cfg, "hf_model_id", fallback_keys=("hf_model_path", "base_hf_path")),
        hf_path=_required_str(cfg, "output_hf_path"),
        trust_remote_code=_as_bool(cfg.get("trust_remote_code", True)),
        show_progress=_as_bool(cfg.get("show_progress", True)),
        strict=_as_bool(cfg.get("strict", True)),
    )


def build_megatron_lora_merge_command(
    cfg: Mapping[str, Any],
    *,
    merged_megatron_path: str,
) -> list[str]:
    script = str(cfg.get("upstream_script") or "/opt/Megatron-Bridge/examples/peft/merge_lora.py")
    cpu = _as_bool(cfg.get("cpu", True))
    tp = int(cfg.get("tp", 1))
    pp = int(cfg.get("pp", 1))
    ep = int(cfg.get("ep", 1))

    if cpu or (os.environ.get("WORLD_SIZE") and os.environ.get("RANK")):
        command = [sys.executable, script]
    else:
        nproc = int(cfg.get("nproc_per_node") or cfg.get("gpus_per_node") or tp)
        command = ["torchrun", f"--nproc_per_node={nproc}", script]

    command.extend(
        [
            "--lora-checkpoint",
            _required_str(cfg, "lora_checkpoint"),
            "--hf-model-path",
            _required_str(cfg, "hf_model_path", fallback_keys=("hf_model_id", "base_hf_path")),
            "--output",
            merged_megatron_path,
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
        ]
    )
    pretrained = cfg.get("pretrained") or cfg.get("base_megatron_path")
    if pretrained:
        command.extend(["--pretrained", str(pretrained)])
    if cpu:
        command.append("--cpu")
    if _as_bool(cfg.get("debug", False)):
        command.append("--debug")
    return command


def _resolve_merge_backend(cfg: Mapping[str, Any]) -> str:
    backend = str(cfg.get("backend") or "auto").lower()
    if backend != "auto":
        return backend
    if cfg.get("base_megatron_path") or cfg.get("pretrained"):
        return "megatron_bridge"
    return "hf_peft"


def _resolve_hf_peft_adapter_path(lora_checkpoint: str) -> str:
    checkpoint_path = Path(lora_checkpoint)
    if (checkpoint_path / _PEFT_ADAPTER_CONFIG).is_file():
        return str(checkpoint_path)

    if checkpoint_path.is_dir():
        configs = sorted(
            checkpoint_path.rglob(_PEFT_ADAPTER_CONFIG),
            key=lambda path: _adapter_config_sort_key(path, checkpoint_path),
        )
        if configs:
            return str(configs[-1].parent)

    raise ValueError(
        f"Can't find {_PEFT_ADAPTER_CONFIG!r} at or below {lora_checkpoint!r}. "
        "Set lora_checkpoint to the adapter directory containing adapter_config.json. "
        f"To inspect candidates, run: find {shlex.quote(lora_checkpoint)} -name {_PEFT_ADAPTER_CONFIG} -print"
    )


def _adapter_config_sort_key(config_path: Path, root_path: Path) -> tuple[int, float, str]:
    try:
        relative_parent = str(config_path.parent.relative_to(root_path))
    except ValueError:
        relative_parent = str(config_path.parent)

    numbers = [int(match) for match in re.findall(r"\d+", relative_parent)]
    checkpoint_step = max(numbers, default=-1)
    try:
        mtime = config_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return checkpoint_step, mtime, str(config_path.parent)


def _call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a Megatron-Bridge helper while tolerating older signatures."""
    return _call_with_supported_args(func, **kwargs)


def _call_with_supported_args(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a helper with positional args and only supported keyword args."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    parameters = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return func(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(*args, **supported)


def _autobridge_for_hf_export(
    auto_bridge: type,
    *,
    megatron_path: str,
    hf_model_id: str,
    trust_remote_code: bool,
) -> Any:
    """Create an AutoBridge instance using the best API for Megatron -> HF export."""
    from_hf_pretrained = getattr(auto_bridge, "from_hf_pretrained", None)
    if from_hf_pretrained is not None:
        return _call_with_supported_args(
            from_hf_pretrained,
            hf_model_id,
            trust_remote_code=trust_remote_code,
        )

    from_auto_config = getattr(auto_bridge, "from_auto_config", None)
    if from_auto_config is not None:
        return _call_with_supported_args(
            from_auto_config,
            megatron_path,
            hf_model_id,
            trust_remote_code=trust_remote_code,
        )

    from_hf_config = getattr(auto_bridge, "from_hf_config", None)
    if from_hf_config is not None:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)
        return _call_with_supported_args(from_hf_config, hf_config, trust_remote_code=trust_remote_code)

    raise AttributeError(
        "AutoBridge does not provide from_hf_pretrained, from_auto_config, or from_hf_config; "
        "cannot construct a Megatron -> HF export bridge"
    )


def _torch_dtype(name: str) -> Any:
    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype {name!r}; choose one of {sorted(dtype_map)}")
    return dtype_map[name]


def _required_str(cfg: Mapping[str, Any], key: str, *, fallback_keys: tuple[str, ...] = ()) -> str:
    for candidate in (key, *fallback_keys):
        value = cfg.get(candidate)
        if value not in (None, ""):
            return str(value)
    names = ", ".join((key, *fallback_keys))
    raise ValueError(f"Missing required config value: {names}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)
