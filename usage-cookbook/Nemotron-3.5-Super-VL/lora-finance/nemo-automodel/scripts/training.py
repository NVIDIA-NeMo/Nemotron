"""Generate task YAMLs, train isolated LoRA adapters and audit their updates."""
from __future__ import annotations


# Build finance config
import argparse
import copy
import json
from pathlib import Path

import yaml

from data import TASK_DATASETS, task_dataset_paths
from model import MODEL_ID, MODEL_REVISION, inspect_snapshot
from environment import PROFILES, detect_profile


OFFICIAL_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
OFFICIAL_CONFIG = Path(
    "examples/vlm_finetune/nemotron_3_5_super_vl/nemotron_3_5_super_vl_cord_v2_peft.yaml"
)


def apply_gpu_config(cfg, workspace, profile, num_gpus):
    """Apply a native AutoModel YAML overlay, then resolve the detected mesh size."""
    if profile not in PROFILES or num_gpus not in PROFILES[profile]["gpu_counts"]:
        raise ValueError(f"Unsupported GPU profile/count: {profile}/{num_gpus}")
    path = workspace / PROFILES[profile]["config_file"]
    overlay = yaml.safe_load(path.read_text())
    if not isinstance(overlay, dict) or set(overlay) != {"model", "step_scheduler", "distributed", "optimizer"}:
        raise ValueError(f"Unexpected GPU overlay structure: {path}")
    if overlay["model"] != {"torch_dtype": "torch.bfloat16"}:
        raise ValueError("Hardware overlays must not override model identity or checkpoint paths")
    if overlay["step_scheduler"] != {"global_batch_size": 8, "local_batch_size": 1}:
        raise ValueError("Changing batch size requires reviewing training-step calculations")

    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict):
                merge(target.setdefault(key, {}), value)
            else:
                target[key] = value

    result = copy.deepcopy(cfg)
    merge(result, overlay)
    result["distributed"]["ep_size"] = num_gpus
    return result, path


def build_config(
    automodel_root: Path,
    workspace: Path,
    output: Path,
    model_id: str,
    model_revision: str,
    model_path: Path | None = None,
    data_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    max_steps: int = 2400,
    validation_every: int = 300,
    checkpoint_every: int = 300,
    warmup_steps: int = 100,
    mtp_layers: int = 0,
    validation_samples_per_task: int = 32,
    capability: str | None = None,
    num_gpus: int = 8,
    gpu_profile: str = "auto",
) -> None:
    if model_id != MODEL_ID or model_revision != MODEL_REVISION:
        raise ValueError("Only the manifest-pinned BF16 checkpoint is allowed")
    if model_path is None:
        raise ValueError("Provide a complete local BF16 snapshot; implicit model downloads are forbidden")
    snapshot_audit = inspect_snapshot(model_path)
    if mtp_layers != 0:
        raise ValueError("MTP-head training is not enabled in this three-adapter recipe")
    source = automodel_root / OFFICIAL_CONFIG
    if not source.is_file():
        raise FileNotFoundError(source)
    cfg = yaml.safe_load(source.read_text())
    if cfg["recipe"] != "FinetuneRecipeForVLM":
        raise ValueError("Unexpected upstream recipe")
    if cfg["model"]["pretrained_model_name_or_path"] != OFFICIAL_MODEL:
        raise ValueError("Upstream model changed; compatibility must be reviewed")
    if cfg["peft"]["dim"] != 64 or cfg["peft"]["alpha"] != 128:
        raise ValueError("Upstream LoRA topology changed; compatibility must be reviewed")
    if gpu_profile == "auto":
        hardware = detect_profile(source="torch")
        if hardware["num_gpus"] != num_gpus:
            raise ValueError("GPU count differs from detected hardware")
        gpu_profile = hardware["profile"]
    cfg, gpu_config_path = apply_gpu_config(cfg, workspace, gpu_profile, num_gpus)

    cfg["step_scheduler"] = {
        "global_batch_size": 8,
        "local_batch_size": 1,
        "ckpt_every_steps": checkpoint_every,
        "val_every_steps": validation_every,
        "max_steps": max_steps,
    }
    model_source = str(model_path.resolve()) if model_path else model_id
    cfg["model"]["pretrained_model_name_or_path"] = model_source
    cfg["model"]["revision"] = model_revision
    cfg["model"]["num_nextn_predict_layers"] = mtp_layers
    cfg["processor"]["pretrained_model_name_or_path"] = model_source
    cfg["processor"]["revision"] = model_revision
    resolved_data = (data_dir or (workspace / "data" / "finance_core_tatdqa_bizbench_v2")).resolve()
    resolved_task_data = list(task_dataset_paths(resolved_data.parent).values())
    checkpoint_name = f"finance_{capability}_lora_bf16_20261001" if capability else "finance_core_peft_8gpu_v2"
    resolved_checkpoint = (checkpoint_dir or (workspace / "checkpoints" / checkpoint_name)).resolve()
    cfg["checkpoint"]["checkpoint_dir"] = str(resolved_checkpoint)
    cfg["checkpoint"]["save_consolidated"] = False
    if num_gpus not in (4, 8):
        raise ValueError(f"Configured H100/GB200 profiles require 4 or 8 GPUs, got {num_gpus}; selection is not a memory-fit guarantee")
    if capability:
        if capability not in TASK_DATASETS:
            raise ValueError(f"Unknown capability: {capability}")
        task_path = task_dataset_paths(resolved_data.parent)[capability]
        cfg["dataset"] = {
            "_target_": "data.make_finance_single_task_dataset",
            "path_or_dataset": str(task_path),
            "capability": capability,
            "split": "train",
        }
        cfg["validation_dataset"] = {
            "_target_": "data.make_finance_single_task_validation_dataset",
            "path_or_dataset": str(task_path),
            "capability": capability,
            "split": "validation",
            "samples_per_task": validation_samples_per_task,
        }
    else:
        cfg["dataset"] = {
            "_target_": "data.make_finance_task_mixture_dataset",
            "dataset_paths": [str(path) for path in resolved_task_data],
            "split": "train",
            "task_balanced": True,
        }
        cfg["validation_dataset"] = {
            "_target_": "data.make_finance_task_validation_dataset",
            "dataset_paths": [str(path) for path in resolved_task_data],
            "split": "validation",
            "samples_per_task": validation_samples_per_task,
        }
    cfg["dataloader"]["shuffle"] = True
    cfg["validation_dataloader"]["shuffle"] = False
    # NemotronV3's no-cache training path passes this weight directly into the
    # fused Mamba kernel, bypassing the module forward where LoRA is applied.
    # Exclude the otherwise dead adapters instead of saving permanently-zero B tensors.
    mamba_out_proj = "*.mixer.out_proj"
    if mamba_out_proj not in cfg["peft"]["exclude_modules"]:
        cfg["peft"]["exclude_modules"].append(mamba_out_proj)
    cfg["lr_scheduler"] = {"lr_decay_style": "cosine", "lr_warmup_steps": warmup_steps, "min_lr": 1e-5}
    cfg["clip_grad_norm"] = {"max_norm": 1.0}
    run_name = f"nemotron_3_5_super_vl_finance_{capability}_lora" if capability else "nemotron_3_5_super_vl_finance_core_peft"
    cfg["wandb"].update({"enable": False, "name": run_name})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# BF16 LoRA configuration generated by the notebook; runtime validation remains required.\n"
        "# Source snapshot is pinned and must not be quantized or converted.\n"
        + f"# GPU profile: {gpu_profile}; overlay: {PROFILES[gpu_profile]['config_file']}; devices: {num_gpus}\n"
        + yaml.safe_dump(cfg, sort_keys=False)
    )
    snapshot_audit["hardware_profile"] = {"profile": gpu_profile, "config_file": str(gpu_config_path), "num_gpus": num_gpus}
    output.with_suffix(".model-contract.json").write_text(json.dumps(snapshot_audit, indent=2) + "\n")


def config_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automodel-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
    )
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Exact local snapshot directory; avoids AutoModel's unpinned internal snapshot prefetch.",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--validation-every", type=int, default=300)
    parser.add_argument("--checkpoint-every", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--mtp-layers", type=int, choices=(0, 1), default=0)
    parser.add_argument("--validation-samples-per-task", type=int, default=32)
    parser.add_argument("--capability", choices=tuple(TASK_DATASETS))
    parser.add_argument("--num-gpus", type=int, choices=(4, 8), default=8)
    parser.add_argument("--gpu-profile", choices=("auto", "H100", "GB200"), default="auto")
    args = parser.parse_args()
    build_config(
        args.automodel_root.resolve(), args.workspace.resolve(), args.output.resolve(),
        args.model_id, args.model_revision, args.model_path,
        data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps, validation_every=args.validation_every,
        checkpoint_every=args.checkpoint_every, warmup_steps=args.warmup_steps,
        mtp_layers=args.mtp_layers,
        validation_samples_per_task=args.validation_samples_per_task,
        capability=args.capability,
        num_gpus=args.num_gpus,
        gpu_profile=args.gpu_profile,
    )
    print(args.output.resolve())


# Audit mtp adapter
import argparse
import json
from pathlib import Path



def audit_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-mtp", action="store_true")
    args = parser.parse_args()

    import torch
    from safetensors import safe_open

    files = sorted(args.adapter.resolve().glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No safetensors files under {args.adapter}")

    total = mtp = mtp_lora = 0
    b_stats: list[dict] = []
    all_b_tensors = all_b_nonzero = 0
    examples: list[str] = []
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                total += 1
                lower = key.lower()
                if "mtp" in lower:
                    mtp += 1
                if "mtp" in lower and "lora_" in lower:
                    mtp_lora += 1
                    if len(examples) < 20:
                        examples.append(key)
                    if "lora_b" in lower:
                        tensor = handle.get_tensor(key)
                        nonzero = int(torch.count_nonzero(tensor).item())
                        b_stats.append(
                            {
                                "key": key,
                                "elements": tensor.numel(),
                                "nonzero_elements": nonzero,
                                "l2_norm": float(tensor.float().norm().item()),
                            }
                        )
                if "lora_b" in lower:
                    all_b_tensors += 1
                    if int(torch.count_nonzero(handle.get_tensor(key)).item()) > 0:
                        all_b_nonzero += 1

    report = {
        "adapter": str(args.adapter.resolve()),
        "files": [str(path) for path in files],
        "total_tensors": total,
        "mtp_tensors": mtp,
        "mtp_lora_tensors": mtp_lora,
        "mtp_lora_B_tensors": len(b_stats),
        "mtp_lora_B_nonzero_tensors": sum(item["nonzero_elements"] > 0 for item in b_stats),
        "lora_B_tensors": all_b_tensors,
        "lora_B_nonzero_tensors": all_b_nonzero,
        "mtp_lora_examples": examples,
        "mtp_lora_B_stats": b_stats,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")

    if not all_b_tensors or all_b_nonzero != all_b_tensors:
        raise SystemExit("At least one LoRA B tensor is missing or remains zero")
    if args.expect_mtp:
        if mtp_lora == 0 or not b_stats:
            raise SystemExit("MTP LoRA tensors were not found")
        if not all(item["nonzero_elements"] > 0 and item["l2_norm"] > 0 for item in b_stats):
            raise SystemExit("At least one MTP LoRA B tensor remains zero")


# Run finance task lora
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from data import TASK_DATASETS, task_dataset_paths


from model import MODEL_ID, MODEL_REVISION, MODEL_LOCAL_DIR, inspect_snapshot, require_runtime_support
from environment import detect_profile

# Notebook task defaults. None retains the dataset-derived one-epoch budget.
# Each task resolves these when its notebook-launched subprocess starts; an
# already-running trainer keeps the configuration it loaded at startup.
DEFAULT_TASK_STEP_LIMITS = {"visual_ocr": None, "reasoning": None, "code_generation": None}


def resolve_max_steps(capability: str, train_rows: int, global_batch_size: int,
                      override: int | None = None) -> int:
    selected = override if override is not None else DEFAULT_TASK_STEP_LIMITS[capability]
    if train_rows <= 0 or global_batch_size <= 0 or (selected is not None and selected <= 0):
        raise ValueError("Training rows, batch size and explicit step limits must be positive")
    return selected if selected is not None else math.ceil(train_rows / global_batch_size)


def adapter_exists(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "LOWEST_VAL" / "model" / "adapter_model.safetensors").is_file()


def project_script(workspace: Path, name: str) -> Path:
    """Support both the historical flat workspace and the GitLab scripts/ layout."""
    for candidate in (workspace / "scripts" / name, workspace / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Project script not found: {name}")


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            lower = line.casefold()
            if any(marker in lower for marker in (
                "step ", "loss", "validation", "checkpoint", "saving", "error", "traceback", "out of memory",
            )):
                print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_stage(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    status_path: Path,
) -> None:
    if status_path.is_file() and status_path.read_text().strip() == "exit_0":
        print(f"{name}: already complete; reusing {status_path}")
        return
    if status_path.is_file() and status_path.read_text().strip() == "running":
        raise RuntimeError(f"{name} is already marked running: {status_path}")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("running\n")
    try:
        run_logged(command, cwd=cwd, env=env, log_path=log_path)
    except subprocess.CalledProcessError as error:
        status_path.write_text(f"exit_{error.returncode}\n")
        raise
    else:
        status_path.write_text("exit_0\n")


def task_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability", choices=tuple(TASK_DATASETS), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--automodel-root", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=os.environ.get("NEMO_LORA_MODEL_PATH"),
        help="User-downloaded Hugging Face snapshot; defaults to $NEMO_LORA_MODEL_PATH or artifacts/model-bf16.",
    )
    parser.add_argument("--stage", choices=("config", "train", "audit", "all"), default="all")
    parser.add_argument("--num-gpus", type=int, choices=(4, 8))
    parser.add_argument("--gpu-profile", choices=("auto", "H100", "GB200"), default="auto")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int,
                        help="Override the dataset-derived one-epoch step budget for this task")
    parser.add_argument("--validation-every", type=int)
    parser.add_argument("--validation-samples-per-task", type=int, default=32)
    args = parser.parse_args()
    if args.model_id != MODEL_ID or args.model_revision != MODEL_REVISION:
        parser.error("Only the manifest-pinned BF16 checkpoint is allowed")
    if args.stage in {"train", "all"}:
        require_runtime_support("training")

    capability = args.capability
    hardware = detect_profile(source="torch", requested=args.gpu_profile)
    num_gpus = hardware["num_gpus"]
    if args.num_gpus is not None and args.num_gpus != num_gpus:
        parser.error("--num-gpus conflicts with the detected CUDA device count")
    os.environ["TORCH_CUDA_ARCH_LIST"] = hardware["torch_cuda_arch_list"]
    workspace = args.workspace.expanduser().resolve()
    automodel_root = (args.automodel_root or workspace / "Automodel").expanduser().resolve()
    python = automodel_root / ".venv" / "bin" / "python"
    automodel = automodel_root / ".venv" / "bin" / "automodel"
    model_path = args.model_path or (workspace / MODEL_LOCAL_DIR)
    model_path = model_path.expanduser().resolve()
    data_path = task_dataset_paths(workspace / "data")[capability]
    manifest_path = data_path / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["capability"] != capability:
        raise AssertionError(f"Capability mismatch in {manifest_path}")
    if not python.is_file() or not automodel.is_file():
        raise FileNotFoundError(f"AutoModel environment is incomplete under {automodel_root}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Pinned model snapshot is incomplete: {model_path}")
    inspect_snapshot(model_path)

    train_rows = int(manifest["counts"]["train"])
    max_steps = resolve_max_steps(capability, train_rows, args.global_batch_size, args.max_steps)
    validation_every = args.validation_every or max(50, max_steps // 4)
    warmup_steps = max(1, round(max_steps * 0.05))
    stem = f"finance_{capability}_lora_bf16_20261001"
    config = workspace / "configs" / f"{stem}.yaml"
    checkpoint_dir = workspace / "checkpoints" / stem
    logs = workspace / "logs"
    training_status = logs / f"{stem}_training.status"
    if args.stage in {"train", "all"}:
        state = training_status.read_text().strip() if training_status.exists() else "not_started"
        if state == "running":
            raise RuntimeError(f"Training is already marked running; inspect before retrying: {training_status}")
        if checkpoint_dir.exists() and (state != "exit_0" or not adapter_exists(checkpoint_dir)):
            raise FileExistsError(f"Incomplete training artifacts require operator inspection: {checkpoint_dir}")

    build_command = [
        str(python), str(project_script(workspace, "training.py")), "config",
        "--automodel-root", str(automodel_root),
        "--workspace", str(workspace),
        "--output", str(config),
        "--model-id", args.model_id,
        "--model-revision", args.model_revision,
        "--model-path", str(model_path),
        "--data-dir", str(workspace / "data" / "finance_core_tatdqa_bizbench_v2"),
        "--checkpoint-dir", str(checkpoint_dir),
        "--max-steps", str(max_steps),
        "--validation-every", str(validation_every),
        "--checkpoint-every", str(validation_every),
        "--warmup-steps", str(warmup_steps),
        "--mtp-layers", "0",
        "--validation-samples-per-task", str(args.validation_samples_per_task),
        "--capability", capability,
        "--num-gpus", str(num_gpus),
        "--gpu-profile", hardware["profile"],
    ]
    subprocess.run(build_command, cwd=workspace, check=True)
    generated = json.loads(json.dumps({
        "capability": capability,
        "dataset": str(data_path),
        "train_rows": train_rows,
        "max_steps": max_steps,
        "validation_every": validation_every,
        "warmup_steps": warmup_steps,
        "num_gpus": num_gpus,
        "hardware": hardware,
        "config": str(config),
        "checkpoint_dir": str(checkpoint_dir),
    }))
    print(json.dumps(generated, indent=2), flush=True)
    if args.stage == "config":
        return

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(workspace / "hf-cache"))
    python_paths = [str(workspace / "scripts"), str(workspace)]
    env["PYTHONPATH"] = os.pathsep.join(python_paths + [env.get("PYTHONPATH", "")])
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_OFFLINE", "1")
    launch_train = [str(automodel), str(config), "--nproc-per-node", str(num_gpus)]

    if args.stage in {"train", "all"}:
        if checkpoint_dir.exists() and not adapter_exists(checkpoint_dir):
            raise FileExistsError(f"Refusing to reuse incomplete training directory: {checkpoint_dir}")
        run_stage(
            f"{capability} training", launch_train,
            cwd=automodel_root, env=env,
            log_path=logs / f"{stem}_training.log",
            status_path=logs / f"{stem}_training.status",
        )
    if args.stage in {"audit", "train", "all"}:
        adapter = checkpoint_dir / "LOWEST_VAL" / "model"
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Best adapter missing: {adapter}")
        subprocess.run([
            str(python), str(project_script(workspace, "training.py")), "audit", str(adapter),
            "--output", str(logs / f"{stem}_final_adapter_audit.json"),
        ], check=True, env=env)


def main():
    import sys
    actions = {'config': config_main, 'audit': audit_main, 'task': task_main}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Commands: " + ", ".join(actions))
        return
    command = sys.argv.pop(1)
    if command not in actions:
        raise SystemExit("Unknown command: " + command)
    actions[command]()


if __name__ == "__main__":
    main()
