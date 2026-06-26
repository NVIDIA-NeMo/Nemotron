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
import textwrap
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemotron.kit.train_script import apply_hydra_overrides, load_omegaconf_yaml, parse_config_and_overrides

_PEFT_ADAPTER_CONFIG = "adapter_config.json"
_DEFAULT_DISTRIBUTED_CONVERTER_SCRIPT = "/opt/Megatron-Bridge/examples/conversion/convert_checkpoints_multi_gpu.py"
# TODO: Remove this compatibility shim once the Super3 conversion image ships
# Megatron-Bridge's duplicate pipeline-stage tensor handling.
_DUPLICATE_PP_TENSOR_BOOTSTRAP = textwrap.dedent(
    """
    import inspect
    import runpy
    import sys

    import torch

    script = sys.argv[1]
    script_args = sys.argv[2:]

    def _broadcast_from_first_pp_rank(self, tensor, cache_key=None):
        if self.pp_size == 1:
            return tensor

        if cache_key is not None and cache_key in self._tensor_spec_output_cache:
            tensor_spec_output = self._tensor_spec_output_cache[cache_key]
        else:
            if tensor is not None:
                tensor_spec = (
                    tensor.shape,
                    tensor.dtype,
                    getattr(tensor, "tensor_model_parallel", None),
                    getattr(tensor, "partition_dim", None),
                )
            else:
                tensor_spec = None

            tensor_spec_output = [None] * self.pp_size
            torch.distributed.all_gather_object(tensor_spec_output, tensor_spec, group=self.pp_group)
            if cache_key is not None:
                self._tensor_spec_output_cache[cache_key] = tensor_spec_output

        target_tensor_spec = None
        src_rank = None
        for rank, spec in enumerate(tensor_spec_output):
            if spec is not None:
                target_tensor_spec = spec
                src_rank = rank
                break

        if target_tensor_spec is None:
            raise ValueError(
                "Object must exist on at least one PP rank. "
                f"megatron_param={self.megatron_param}, hf_param={self.hf_param}, cache_key={cache_key}"
            )

        if tensor is None:
            shape, dtype, tensor_parallel, partition_dim = target_tensor_spec
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if tensor_parallel is not None:
                tensor.tensor_model_parallel = tensor_parallel
            if partition_dim is not None:
                tensor.partition_dim = partition_dim

        global_src = torch.distributed.get_global_rank(group=self.pp_group, group_rank=src_rank)
        torch.distributed.broadcast(tensor, src=global_src, group=self.pp_group)
        return tensor

    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    existing = getattr(MegatronParamMapping, "broadcast_from_pp_rank", None)
    if existing is None:
        raise RuntimeError("MegatronParamMapping.broadcast_from_pp_rank is missing; cannot apply PP duplicate policy.")
    parameters = inspect.signature(existing).parameters
    if "tensor" not in parameters or "cache_key" not in parameters:
        raise RuntimeError(
            "MegatronParamMapping.broadcast_from_pp_rank has an unexpected signature; "
            "disable duplicate_pp_tensor_policy or update the compatibility shim."
        )

    MegatronParamMapping.broadcast_from_pp_rank = _broadcast_from_first_pp_rank
    sys.argv = [script, *script_args]
    runpy.run_path(script, run_name="__main__")
    """
).strip()


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
    if _use_distributed_conversion(cfg):
        exec_distributed_conversion("import", cfg)
        return
    if _skip_direct_conversion_on_nonzero_rank():
        return
    import_hf_to_megatron(cfg)


def run_megatron_to_hf(default_config: Path) -> None:
    cfg = load_convert_config(default_config)
    if _use_distributed_conversion(cfg):
        exec_distributed_conversion("export", cfg)
        return
    if _skip_direct_conversion_on_nonzero_rank():
        return
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


def exec_distributed_conversion(direction: str, cfg: Mapping[str, Any]) -> None:
    """Replace the current process with the multi-GPU conversion command."""
    command = build_distributed_conversion_command(direction, cfg)
    _ensure_distributed_converter_script(command)
    env = _distributed_conversion_env(cfg)
    print(f"$ {shlex.join(command)}", flush=True)
    os.execvpe(command[0], command, env)


def build_distributed_conversion_command(direction: str, cfg: Mapping[str, Any]) -> list[str]:
    """Build the torchrun command for the container-provided Megatron-Bridge converter."""
    if direction not in {"import", "export"}:
        raise ValueError("direction must be 'import' or 'export'")

    _validate_distributed_parallelism(cfg)
    command = _distributed_converter_invocation(cfg)
    if _in_torchrun_world():
        return [*command, *_distributed_conversion_args(direction, cfg)]

    torchrun = _optional_mapping(cfg.get("torchrun"), "torchrun")
    run_env = _run_env_mapping(cfg)
    nproc = _configured_torchrun_nproc(cfg)
    torchrun_cmd = ["torchrun", f"--nproc_per_node={nproc}"]
    for key in ("nnodes", "node_rank", "master_addr", "master_port"):
        run_env_key = "nodes" if key == "nnodes" else key
        value = torchrun.get(key, cfg.get(key, run_env.get(run_env_key)))
        if value is not None:
            torchrun_cmd.append(f"--{key}={value}")
    return [*torchrun_cmd, *command, *_distributed_conversion_args(direction, cfg)]


def _distributed_converter_invocation(cfg: Mapping[str, Any]) -> list[str]:
    script = (
        cfg.get("distributed_script")
        or cfg.get("upstream_script")
        or _optional_mapping(cfg.get("script"), "script").get("path")
        or _DEFAULT_DISTRIBUTED_CONVERTER_SCRIPT
    )
    if str(cfg.get("duplicate_pp_tensor_policy") or "").lower() == "first":
        return [sys.executable, "-c", _DUPLICATE_PP_TENSOR_BOOTSTRAP, str(script)]
    return [sys.executable, str(script)]


def _distributed_conversion_env(cfg: Mapping[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONFAULTHANDLER", "1")
    return env


def _ensure_distributed_converter_script(command: list[str]) -> None:
    script = _distributed_converter_script_path(command)
    if script is None:
        return
    if Path(script).is_file():
        return
    raise FileNotFoundError(
        f"Distributed conversion script not found: {script}. "
        "Use a NeMo image that ships convert_checkpoints_multi_gpu.py, "
        "or set script.path/distributed_script to a valid converter path."
    )


def _distributed_converter_script_path(command: list[str]) -> str | None:
    try:
        python_index = command.index(sys.executable)
    except ValueError:
        return None
    if python_index + 1 >= len(command):
        return None
    if command[python_index + 1] == "-c":
        if python_index + 3 >= len(command):
            return None
        return command[python_index + 3]
    script = str(command[python_index + 1])
    if script.startswith("-"):
        return None
    return script


def _distributed_conversion_args(direction: str, cfg: Mapping[str, Any]) -> list[str]:
    args = [direction]
    hf_model = _required_str(cfg, "hf_model_id", fallback_keys=("hf_model",))
    args.extend(["--hf-model", hf_model])

    if direction == "import":
        args.extend(["--megatron-path", _required_str(cfg, "megatron_path")])
    else:
        args.extend(
            [
                "--megatron-path",
                _required_str(cfg, "megatron_path"),
                "--hf-path",
                _required_str(cfg, "hf_path"),
            ]
        )

    for key in ("tp", "pp", "ep", "etp"):
        args.extend([f"--{key}", str(int(cfg.get(key, 1)))])

    dtype = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    dtype_capability = "import_torch_dtype" if direction == "import" else "export_torch_dtype"
    if _script_supports(cfg, dtype_capability, default=_script_supports(cfg, "torch_dtype", default=True)):
        args.extend(["--torch-dtype", str(dtype)])

    if _as_bool(cfg.get("trust_remote_code", True)) and _script_supports(cfg, "trust_remote_code", default=True):
        args.append("--trust-remote-code")

    # Some newer upstream converters accept this flag, but the 26.04 container
    # script does not. Only forward it when explicitly configured.
    timeout = cfg.get("distributed_timeout_minutes")
    if timeout not in (None, ""):
        args.extend(["--distributed-timeout-minutes", str(int(timeout))])

    if direction == "export":
        if not _as_bool(cfg.get("show_progress", True)):
            args.append("--no-progress")
        if not _as_bool(cfg.get("strict", True)):
            args.append("--not-strict")
        if _as_bool(cfg.get("distributed_save", True)) and _script_supports(cfg, "distributed_save", default=True):
            args.append("--distributed-save")
        save_every_n_ranks = cfg.get("save_every_n_ranks")
        if save_every_n_ranks is not None and _script_supports(cfg, "save_every_n_ranks", default=True):
            args.extend(["--save-every-n-ranks", str(int(save_every_n_ranks))])

    args.extend(str(item) for item in (cfg.get("extra_args") or []))
    args.extend(str(item) for item in (cfg.get("distributed_extra_args") or []))
    return args


def _script_supports(cfg: Mapping[str, Any], capability: str, *, default: bool) -> bool:
    script = _optional_mapping(cfg.get("script"), "script")
    direct_key = f"supports_{capability}"
    if direct_key in script:
        return _as_bool(script[direct_key])
    capabilities = script.get("supports")
    if isinstance(capabilities, Mapping) and capability in capabilities:
        return _as_bool(capabilities[capability])
    return default


def _use_distributed_conversion(cfg: Mapping[str, Any]) -> bool:
    setting = cfg.get("distributed", cfg.get("use_distributed", False))
    if isinstance(setting, str):
        normalized = setting.lower()
        if normalized == "auto":
            return _in_torchrun_world() or _configured_torchrun_nproc(cfg) > 1 or _parallelism_size(cfg) > 1
        return normalized in {"1", "true", "yes", "on"}
    if setting is None:
        return False
    return bool(setting)


def _configured_torchrun_nproc(cfg: Mapping[str, Any]) -> int:
    torchrun = _optional_mapping(cfg.get("torchrun"), "torchrun")
    run_env = _run_env_mapping(cfg)
    value = torchrun.get(
        "nproc_per_node",
        cfg.get("nproc_per_node", run_env.get("nprocs_per_node", run_env.get("gpus_per_node", 1))),
    )
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _configured_conversion_world_size(cfg: Mapping[str, Any]) -> int:
    if os.environ.get("WORLD_SIZE"):
        try:
            return int(os.environ["WORLD_SIZE"])
        except ValueError:
            return 1

    torchrun = _optional_mapping(cfg.get("torchrun"), "torchrun")
    run_env = _run_env_mapping(cfg)
    nnodes = torchrun.get("nnodes", cfg.get("nnodes", run_env.get("nodes", 1)))
    try:
        return int(nnodes) * _configured_torchrun_nproc(cfg)
    except (TypeError, ValueError):
        return _configured_torchrun_nproc(cfg)


def _run_env_mapping(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    run = _optional_mapping(cfg.get("run"), "run")
    return _optional_mapping(run.get("env"), "run.env")


def _parallelism_size(cfg: Mapping[str, Any]) -> int:
    return _dense_parallelism_size(cfg) * _expert_parallelism_size(cfg)


def _dense_parallelism_size(cfg: Mapping[str, Any]) -> int:
    return _parallelism_product(cfg, ("tp", "pp", "cp"), fallback_keys={"cp": ("context_parallel_size",)})


def _expert_parallelism_size(cfg: Mapping[str, Any]) -> int:
    return _parallelism_product(
        cfg,
        ("etp", "ep", "pp"),
        fallback_keys={"etp": ("expert_tensor_parallel_size",), "ep": ("expert_model_parallel_size",)},
    )


def _parallelism_product(
    cfg: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    fallback_keys: Mapping[str, tuple[str, ...]] | None = None,
) -> int:
    size = 1
    fallback_keys = fallback_keys or {}
    for key in keys:
        value = cfg.get(key)
        if value is None:
            for fallback_key in fallback_keys.get(key, ()):
                value = cfg.get(fallback_key)
                if value is not None:
                    break
        if value is None:
            value = 1
        try:
            size *= int(value)
        except (TypeError, ValueError):
            return 1
    return size


def _validate_distributed_parallelism(cfg: Mapping[str, Any]) -> None:
    world_size = _configured_conversion_world_size(cfg)
    dense_size = _dense_parallelism_size(cfg)
    expert_size = _expert_parallelism_size(cfg)

    if world_size <= 1:
        return
    if dense_size <= 1 and expert_size <= 1:
        raise ValueError(
            "distributed=true is launching multiple ranks, but no model or expert parallelism was configured. "
            "Set at least one of tp, pp, cp, ep, or etp above 1 so conversion actually shards the model "
            "(for Nemotron MoE checkpoints the common default is tp=1 pp=1 ep=8 etp=1)."
        )
    if world_size % dense_size != 0:
        raise ValueError(
            f"distributed conversion world size ({world_size}) must be divisible by tp*pp*cp ({dense_size}). "
            "Set the conversion rank count and checkpoint parallelism to compatible values."
        )
    if expert_size > 1 and world_size % expert_size != 0:
        raise ValueError(
            f"distributed conversion world size ({world_size}) must be divisible by etp*ep*pp ({expert_size}) "
            "for expert-parallel checkpoints."
        )


def _in_torchrun_world() -> bool:
    return bool(os.environ.get("WORLD_SIZE") and os.environ.get("RANK"))


def _skip_direct_conversion_on_nonzero_rank() -> bool:
    if not _in_torchrun_world():
        return False
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError:
        return False
    if rank == 0:
        return False
    print(f"distributed=false; rank {rank} is idle for single-process conversion", flush=True)
    return True


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


def _optional_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping when set")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)
