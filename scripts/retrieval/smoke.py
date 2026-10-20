#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run real Nemotron 3.5 mining, save/resume, and evaluation with frozen stages.

From a Linux source checkout with uv, Python 3.12, and an available NVIDIA GPU:
    python scripts/retrieval/smoke.py --output /scratch/retrieval-smoke --gpu 0

Optional: --serve checks vLLM using Docker; --rerank also checks native 8B
save/resume on --rerank-gpus 0,1. The unchanged production checkpoint writer is
used. Provision the documented checkpoint RAM/disk budget before the 8B phase.
The runner installs the existing stage lockfiles, downloads revision-pinned
public models, and retains logs/checkpoints/results. It never deletes outputs.
This is an execution check on fictional data, not SDG or model-quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "src/nemotron/recipes"
EMBED_MODEL = "nvidia/Nemotron-3.5-Embed-1B-BF16"
EMBED_REVISION = "c0f2e8ab5367f5090e6eed608f607bafe76d707b"
RERANK_MODEL = "nvidia/Nemotron-3.5-Rerank-8B-BF16"
RERANK_REVISION = "39aa01525864f736e63f1d5742fc4cf59e754abd"


def uv_command(stage: str, *arguments: str) -> list[str]:
    project = RECIPES / stage
    config = tomllib.loads((project / "pyproject.toml").read_text())
    extras = ["--extra", "vl"] if "vl" in config["project"].get("optional-dependencies", {}) else []
    return ["uv", "run", "--frozen", "--python", "3.12", "--project", str(project), *extras, "python", *arguments]


def stage_command(
    stage: str,
    script: str,
    overrides: dict,
    *,
    processes: int = 0,
    runtime_stage: str | None = None,
) -> list[str]:
    arguments = []
    if processes:
        arguments += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(processes)]
    arguments += [str(RECIPES / stage / script), "--config", str(RECIPES / stage / "config/nemotron-3.5.yaml")]
    arguments += [
        f"{key}={str(value).lower() if isinstance(value, bool) else 'null' if value is None else value}"
        for key, value in overrides.items()
    ]
    return uv_command(runtime_stage or stage, *arguments)


def run(command: list[str], log: Path, gpu: str) -> None:
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    # Only the checked-out source is added. All third-party packages come from
    # the selected stage's frozen environment.
    environment.update(PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=gpu, WANDB_MODE="disabled")
    print(f"\n{shlex.join(command)}\nLog: {log}", flush=True)
    with log.open("w") as stream:
        stream.write(f"CUDA_VISIBLE_DEVICES={gpu}\n{shlex.join(command)}\n\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def checkpoint_command(stage: str, checkpoint: Path, expected: int, result: Path) -> list[str]:
    code = """import json, pathlib, sys, torch
root, expected, output = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
latest = (root / 'LATEST').resolve(strict=True)
state = torch.load(latest / 'step_scheduler.pt', map_location='cpu', weights_only=True)
assert state['step'] == expected, state
assert (latest / 'model/consolidated/config.json').is_file(), latest
assert any((latest / 'optim').rglob('*.distcp')), 'Full optimizer checkpoint missing'
output.write_text(json.dumps({'checkpoint': str(latest), 'step': state['step']}, indent=2) + '\\n')
"""
    return uv_command(stage, "-c", code, str(checkpoint), str(expected), str(result))


def check_metrics(path: Path, names: tuple[str, ...]) -> None:
    result = json.loads(path.read_text())
    for name in names:
        for metric in ("NDCG", "MAP", "Recall", "Precision"):
            values = result[name][metric]
            if not values or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in values.values()
            ):
                raise ValueError(f"Missing or nonfinite {name}/{metric} metrics in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory on a filesystem with ample space"
    )
    parser.add_argument("--gpu", default="0", help="One host GPU index or UUID for embedding")
    parser.add_argument("--serve", action="store_true", help="Also deploy/evaluate vLLM; requires Docker GPU access")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--rerank", action="store_true", help="Also train/save/resume the native 8B reranker")
    parser.add_argument("--rerank-gpus", default="0,1", help="Host GPU indices/UUIDs for optional reranking")
    args = parser.parse_args()
    if not args.gpu.strip() or "," in args.gpu:
        parser.error("--gpu must select exactly one embedding GPU")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    rerank_devices = args.rerank_gpus.split(",")
    if args.rerank and (
        not all(rerank_devices) or len(set(rerank_devices)) != len(rerank_devices) or 12 % len(rerank_devices)
    ):
        parser.error("--rerank-gpus must name distinct GPUs with a count that divides the 12-example fixture")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    embed_stage = "embed/stage2_finetune"
    prep_stage = "embed/stage1_data_prep"
    eval_stage = "embed/stage3_eval"
    fixture = output / "fixture"
    manifest = {
        "embedding_model": EMBED_MODEL,
        "embedding_revision": EMBED_REVISION,
        "reranking_model": RERANK_MODEL if args.rerank else None,
        "reranking_revision": RERANK_REVISION if args.rerank else None,
        "fixture": "original fictional mixed-modality documents; execution coverage only",
        "stage_locks": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in RECIPES.glob("*/stage*/uv.lock")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run(
        uv_command(prep_stage, str(ROOT / "scripts/retrieval/smoke_fixture.py"), str(fixture)),
        logs / "fixture.log",
        args.gpu,
    )
    snapshot_code = (
        "from huggingface_hub import snapshot_download; import pathlib,sys; "
        "pathlib.Path(sys.argv[3]).write_text(snapshot_download(sys.argv[1], revision=sys.argv[2]))"
    )
    model_path = output / "embedding-model.txt"
    run(
        uv_command(prep_stage, "-c", snapshot_code, EMBED_MODEL, EMBED_REVISION, str(model_path)),
        logs / "embedding-download.log",
        args.gpu,
    )
    model = model_path.read_text()
    shared = {"base_model": model, "image_longest_edge": 224, "query_max_length": 64, "passage_max_length": 256}
    run(
        stage_command(
            prep_stage,
            "data_prep.py",
            {
                **shared,
                "artifact_root": output / "embed",
                "sdg_input_path": None,
                "retrieval_view": None,
                "train_input_file": fixture / "train.json",
                "hard_negatives_to_mine": 3,
                "hard_neg_margin": 1.0,
            },
        ),
        logs / "mining.log",
        args.gpu,
    )
    training = output / "embed/stage1_data_prep/train_mined.automodel_unrolled.json"
    for step in (2, 4):
        run(
            stage_command(
                embed_stage,
                "train.py",
                {
                    **shared,
                    "artifact_root": output / "embed",
                    "train_data_path": training,
                    "num_epochs": None,
                    "max_steps": step,
                    "global_batch_size": 3,
                    "local_batch_size": 1,
                    "train_n_passages": 2,
                    "lr_warmup_steps": 0,
                    "checkpoint_every_steps": 2,
                    "val_every_steps": 2,
                },
                processes=1,
            ),
            logs / f"embedding-train-{step}.log",
            args.gpu,
        )
        run(
            checkpoint_command(
                embed_stage,
                output / "embed/stage2_finetune/checkpoints",
                step,
                output / f"embedding-checkpoint-{step}.json",
            ),
            logs / f"embedding-checkpoint-{step}.log",
            args.gpu,
        )
    if "Loading checkpoint from" not in (logs / "embedding-train-4.log").read_text():
        raise RuntimeError("Second embedding invocation did not restore a checkpoint")
    evaluation = {
        **shared,
        "artifact_root": output / "embed",
        "sdg_input_path": None,
        "retrieval_view": None,
        "eval_data_path": fixture / "beir",
        "image_root": fixture,
        "eval_base": True,
        "eval_finetuned": True,
        "eval_nim": False,
    }
    run(stage_command(eval_stage, "eval.py", evaluation), logs / "embedding-eval.log", args.gpu)
    check_metrics(output / "embed/stage3_eval/eval_results.json", ("base", "finetuned"))
    if args.rerank:
        rerank_stage = "rerank/stage2_finetune"
        gpu_count = len(rerank_devices)
        for step in (1, 2):
            run(
                stage_command(
                    rerank_stage,
                    "train.py",
                    {
                        "artifact_root": output / "rerank",
                        "base_model": RERANK_MODEL,
                        "model_revision": RERANK_REVISION,
                        # One full-fixture batch per epoch makes epochs 1 and 2 exact
                        # optimizer-step targets through the supported rerank interface.
                        "train_data_path": training,
                        "num_epochs": step,
                        "global_batch_size": 12,
                        "local_batch_size": 1,
                        "train_n_passages": 2,
                        "lr_warmup_steps": 0,
                        "checkpoint_every_steps": 1,
                        "val_every_steps": 1,
                    },
                    processes=gpu_count,
                ),
                logs / f"rerank-train-{step}.log",
                args.rerank_gpus,
            )
            run(
                checkpoint_command(
                    rerank_stage,
                    output / "rerank/stage2_finetune/checkpoints",
                    step,
                    output / f"rerank-checkpoint-{step}.json",
                ),
                logs / f"rerank-checkpoint-{step}.log",
                args.rerank_gpus,
            )
        if "Loading checkpoint from" not in (logs / "rerank-train-2.log").read_text():
            raise RuntimeError("Second reranking invocation did not restore a checkpoint")
    if args.serve:
        name = f"nemotron-smoke-{os.getpid()}"
        # A unique container name makes cleanup specific to this invocation.
        deploy_stage = "embed/stage5_deploy"
        # Deployment has no lockfile. Its lightweight imports are already in
        # the frozen evaluation environment, so reuse that environment.
        command = stage_command(
            deploy_stage,
            "deploy.py",
            {
                "artifact_root": output / "embed",
                "gpus": f"device={args.gpu}",
                "host_port": args.port,
                "container_name": name,
                "detach": True,
                "model_dir": output / "embed/stage2_finetune/checkpoints/LATEST/model/consolidated",
                "nim_model": EMBED_MODEL,
            },
            runtime_stage=eval_stage,
        )
        try:
            run(command, logs / "deploy.log", args.gpu)
            run(
                stage_command(
                    eval_stage,
                    "eval.py",
                    {
                        **evaluation,
                        "eval_base": False,
                        # Native evaluation completed before vLLM reserved this GPU.
                        "eval_finetuned": False,
                        "eval_nim": True,
                        "embedding_api_backend": "vllm",
                        "nim_model": EMBED_MODEL,
                        "nim_url": f"http://localhost:{args.port}",
                        "output_dir": output / "served",
                    },
                ),
                logs / "served-eval.log",
                args.gpu,
            )
            check_metrics(output / "served/eval_results.json", ("vllm",))
        finally:
            subprocess.run(["docker", "rm", "-f", name], check=False)
    (output / "SUCCESS").write_text("Execution checks completed; no model-quality claim.\n")
    print(f"Smoke checks passed. Logs, checkpoints, and results: {output}")


if __name__ == "__main__":
    main()
