"""Direct Docker + torchrun workflow used by the full-SFT script."""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import yaml
from assets import download, sha256, write_json
from prepare import prepare

HERE = Path(__file__).resolve().parent
# Container mount targets created by this launcher; neither is a required host path.
CONTAINER_WORKSPACE = Path("/workspace")
CONTAINER_HELPERS = Path("/tutorial") / HERE.name


class Pipeline:
    def __init__(self, settings, mode, smoke_steps=None):
        self.settings = Path(settings).resolve()
        self.config = yaml.safe_load(os.path.expandvars(self.settings.read_text()))
        # Match the notebook's optional model overrides; settings carry the GA defaults.
        for key, variable in (("repo_id", "SUPER_VL_MODEL_ID"), ("revision", "SUPER_VL_MODEL_REVISION")):
            if variable in os.environ:
                self.config["model"][key] = os.environ[variable]
        self.mode = mode
        if mode not in ("lora", "sft"):
            raise ValueError(mode)
        self.smoke_steps = smoke_steps
        if smoke_steps is not None and not 1 <= smoke_steps <= 20:
            raise ValueError("smoke_steps must be between 1 and 20")
        workspace = Path(self.config["workspace"]).expanduser()
        self.workspace = (workspace if workspace.is_absolute() else self.settings.parent / workspace).resolve()
        if "," in str(self.workspace):
            raise ValueError("Docker bind-mount paths cannot contain commas")
        run_id = self.config["run_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
            raise ValueError("run_id must use letters, digits, underscores, or hyphens")
        self.run = self.workspace / "runs" / run_id
        self.container_run = str(CONTAINER_WORKSPACE / "runs" / run_id)
        self.container_assets = CONTAINER_WORKSPACE / "assets"
        self.nodes = 1 if mode == "lora" else 4
        self.world_size = self.nodes * 4
        self.ready_timeout = int(self.config.get("ready_timeout_seconds", 21600))
        self.shared_fs_timeout = int(self.config.get("shared_fs_timeout_seconds", 1800))
        self.peer_completion_timeout = int(self.config.get("peer_completion_timeout_seconds", 3600))
        if min(self.ready_timeout, self.shared_fs_timeout, self.peer_completion_timeout) <= 0:
            raise ValueError("Workflow timeouts must be positive integer seconds")
        source_hashes = {
            str(path.relative_to(HERE.parent)): sha256(path)
            for directory in (HERE, HERE.parent / f"super_vl_rxncaption_{mode}")
            for path in directory.iterdir()
            if path.suffix in (".py", ".yaml")
        }
        payload = dict(config=self.config, mode=mode, smoke_steps=smoke_steps, sources=source_hashes)
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def identity(self):
        return dict(fingerprint=self.fingerprint, mode=self.mode, smoke_steps=self.smoke_steps)

    def initialize(self):
        self.run.mkdir(parents=True, exist_ok=True)
        state = self.run / "run.json"
        if state.exists():
            if json.loads(state.read_text()) != self.identity():
                raise ValueError("Run settings or tutorial source changed; choose a fresh run_id")
        else:
            write_json(state, self.identity())

    def recipe(self, steps):
        path = HERE.parent / f"super_vl_rxncaption_{self.mode}" / f"{self.mode}_config.yaml"
        config = yaml.safe_load(path.read_text())
        for name in ("model", "processor"):
            config[name]["pretrained_model_name_or_path"] = str(self.container_assets / "model")
        for name in ("dataset", "validation_dataset"):
            config[name]["path_or_dataset"] = str(self.container_assets / "data")
        config["checkpoint"]["checkpoint_dir"] = self.container_run + "/checkpoints"
        schedule = config["step_scheduler"]
        schedule["max_steps"] = schedule["ckpt_every_steps"] = steps
        config["lr_scheduler"]["lr_decay_steps"] = steps
        if self.smoke_steps:
            schedule["val_every_steps"] = steps
        return config

    def docker(self, stage, arguments, gpu=False, extra_env=None):
        image = self.config["eval_image" if stage == "eval" else "train_image"]
        python = self.config.get("eval_python" if stage == "eval" else "train_python", "python3")
        command = [
            "docker",
            "run",
            "--rm",
            "--network=host",
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--mount",
            f"type=bind,source={self.workspace},target={CONTAINER_WORKSPACE}",
            "--mount",
            f"type=bind,source={HERE.parent},target={CONTAINER_HELPERS.parent},readonly",
            "--workdir",
            str(CONTAINER_WORKSPACE),
            "--entrypoint",
            python,
        ]
        if gpu:
            command += ["--gpus", "all"]
        if self.config.get("cpu_memory_nodes"):
            command += ["--cpuset-mems", str(self.config["cpu_memory_nodes"])]
        cache = CONTAINER_WORKSPACE / "cache"
        env = dict(
            PYTHONPATH=f"{self.container_assets / 'Automodel'}:{CONTAINER_HELPERS}",
            AUTOMODEL_SOURCE=str(self.container_assets / "Automodel"),
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            RUN_DIR=self.container_run,
            TUTORIAL_MODE=self.mode,
            HF_HOME=str(cache / "huggingface"),
            XDG_CACHE_HOME=str(cache),
            HOME=str(cache / "home"),
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="8",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            VLLM_NO_USAGE_STATS="1",
        )
        env.update(self.config.get("runtime_env", {}))
        env.update(extra_env or {})
        for key, value in env.items():
            if not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("Invalid environment variable name")
            command += ["--env", f"{key}={value}"]
        return command + [image, *map(str, arguments)]

    def training_command(self, node_rank=0, master_addr="127.0.0.1", master_port=29500):
        if not 0 <= node_rank < self.nodes or not 1024 <= master_port <= 65535:
            raise ValueError("Invalid node rank or rendezvous port")
        if self.nodes > 1 and master_addr in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Full SFT needs --master-addr reachable from all four nodes")
        return self.docker(
            "train",
            [
                "-m",
                "torch.distributed.run",
                f"--nnodes={self.nodes}",
                "--nproc-per-node=4",
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                f"--master-port={master_port}",
                "--max-restarts=0",
                str(CONTAINER_HELPERS / "train.py"),
                self.container_run + "/train.yaml",
            ],
            gpu=True,
        )

    def logged(self, name, command):
        logs = self.run / "logs"
        logs.mkdir(exist_ok=True)
        print(f"{name}: {logs / (name + '.log')}", flush=True)
        # argv is passed directly; no shell interpolation of settings or file paths.
        with (logs / (name + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    def download(self):
        download(self.config, self.workspace)

    def prepare(self):
        self.initialize()
        (self.workspace / "cache/home").mkdir(parents=True, exist_ok=True)
        if (self.run / "ready.json").exists():
            self.verify_ready()
            return json.loads((self.run / "prepared.json").read_text())
        report = prepare(self.workspace, self.run)
        steps = self.smoke_steps or report["full_steps"]
        (self.run / "train.yaml").write_text(yaml.safe_dump(self.recipe(steps), sort_keys=False))
        write_json(
            self.run / "ready.json",
            dict(**self.identity(), steps=steps, recipe_sha256=sha256(self.run / "train.yaml")),
        )
        return report

    def verify_ready(self):
        ready = json.loads((self.run / "ready.json").read_text())
        if ready["fingerprint"] != self.fingerprint or ready["recipe_sha256"] != sha256(self.run / "train.yaml"):
            raise ValueError(
                "Prepared run differs from this invocation; use identical settings and source on all nodes"
            )
        return ready

    def wait_for(self, paths, timeout=10800):
        deadline = time.monotonic() + timeout
        while not all(path.is_file() for path in paths):
            failures = list(self.run.glob("failed*.json"))
            if failures:
                raise RuntimeError(f"Pipeline failed; inspect {failures[0]}")
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for the other nodes; check their logs and the shared workspace")
            time.sleep(2)

    def shared_workspace_barrier(self, node_rank):
        """Prove that every node sees and can write the same run directory."""
        marker = self.run / f"workspace_visible_{node_rank}.json"
        write_json(marker, dict(fingerprint=self.fingerprint, node_rank=node_rank))
        expected = [self.run / f"workspace_visible_{rank}.json" for rank in range(self.nodes)]
        try:
            self.wait_for(expected, timeout=self.shared_fs_timeout)
        except TimeoutError as error:
            raise TimeoutError(
                "Shared-workspace preflight failed. Confirm that every node uses the same "
                "absolute workspace and that all node commands were launched."
            ) from error
        for rank, path in enumerate(expected):
            value = json.loads(path.read_text())
            if value != dict(fingerprint=self.fingerprint, node_rank=rank):
                raise ValueError(f"Shared-workspace marker differs on node {rank}: {path}")

    def train(self, node_rank=0, master_addr="127.0.0.1", master_port=29500):
        self.verify_ready()
        try:
            self.shared_workspace_barrier(node_rank)
            started = self.run / f"train_started_{node_rank}.json"
            # Repeating a training cell must never overwrite a completed or partial checkpoint.
            with started.open("x") as stream:
                json.dump(self.identity(), stream)
            self.logged(f"train_node{node_rank}", self.training_command(node_rank, master_addr, master_port))
            write_json(self.run / f"train_complete_{node_rank}.json", self.identity())
            if node_rank == 0:
                self.wait_for(
                    [self.run / f"train_complete_{rank}.json" for rank in range(self.nodes)],
                    timeout=self.peer_completion_timeout,
                )
                write_json(self.run / "training_complete.json", self.identity())
        except BaseException as error:
            write_json(
                self.run / f"failed_node{node_rank}.json",
                {"error_type": type(error).__name__, "detail": "Training failed; inspect this node log."},
            )
            raise

    def export(self):
        self.verify_ready()
        if json.loads((self.run / "training_complete.json").read_text()) != self.identity():
            raise ValueError("Training did not complete for this run")
        self.logged(
            "export",
            self.docker(
                "export",
                [
                    CONTAINER_HELPERS / "export.py",
                    "--run-dir",
                    self.container_run,
                    "--base-model",
                    self.container_assets / "model",
                    "--mode",
                    self.mode,
                    "--world-size",
                    str(self.world_size),
                ],
            ),
        )
        return json.loads((self.run / "export.json").read_text())

    def evaluate(self):
        self.verify_ready()
        if not (self.run / "export.json").is_file():
            raise ValueError("Export the checkpoint before evaluation")
        self.logged(
            "eval",
            self.docker(
                "eval",
                [
                    CONTAINER_HELPERS / "evaluate.py",
                    "--run-dir",
                    self.container_run,
                    "--benchmark",
                    self.container_assets / "benchmark",
                    "--port",
                    str(self.config.get("eval_port", 12520)),
                ],
                gpu=True,
                extra_env={"CUDA_VISIBLE_DEVICES": "0,1"},
            ),
        )
        return json.loads((self.run / "evaluation/metrics.json").read_text())

    def all(self, node_rank=0, master_addr="127.0.0.1", master_port=29500):
        self.training_command(node_rank, master_addr, master_port)  # Validate before downloading.
        try:
            if node_rank == 0:
                self.initialize()
                self.download()
                self.prepare()
            else:
                self.wait_for([self.run / "ready.json"], timeout=self.ready_timeout)
            self.train(node_rank, master_addr, master_port)
            if node_rank == 0:
                self.export()
                print(json.dumps(self.evaluate(), indent=2))
        except BaseException as error:
            write_json(
                self.run / f"failed_node{node_rank}.json",
                {"error_type": type(error).__name__, "detail": "Stage failed; inspect the local traceback and logs."},
            )
            raise


def main(mode):
    parser = argparse.ArgumentParser(
        description="Parse chemical reactions with NVIDIA Nemotron 3.5 Super VL: full SFT"
    )
    parser.add_argument("stage", choices=["plan", "download", "prepare", "train", "export", "eval", "all"])
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()
    flow = Pipeline(args.settings, mode, args.smoke_steps)
    topology = (args.node_rank, args.master_addr, args.master_port)
    if args.stage == "plan":
        print(shlex.join(flow.training_command(*topology)))
        print("Stages: download -> prepare -> train -> export -> eval")
    elif args.stage in ("all", "train"):
        getattr(flow, args.stage)(*topology)
    else:
        result = getattr(flow, "evaluate" if args.stage == "eval" else args.stage)()
        if result is not None:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit("Use run_sft.py in the neighboring super_vl_rxncaption_sft directory.")
