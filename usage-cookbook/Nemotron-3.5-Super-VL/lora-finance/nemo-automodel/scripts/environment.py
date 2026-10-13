"""Host credentials, GPU selection, source pinning and required preflight."""
from __future__ import annotations


# Hardware profile
import argparse
import csv
import json
import os
import platform
from pathlib import Path
import subprocess


PROFILES = {
    "H100": {"cuda_arch": "9.0", "gpu_counts": (4, 8), "config_file": "configs/automodel_h100.yaml"},
    "GB200": {"cuda_arch": "10.0", "gpu_counts": (4, 8), "config_file": "configs/automodel_gb200.yaml"},
}


def select_profile(devices, *, cpu_arch=None, requested="auto"):
    if not devices:
        raise ValueError("No visible CUDA GPUs; use the GPU allocation and NeMo AutoModel kernel")
    families = set()
    for device in devices:
        name = device["name"].upper()
        matches = [key for key in PROFILES if key in name]
        if len(matches) != 1 or device.get("mig", False):
            raise ValueError(f"Unsupported GPU or MIG instance: {device['name']}")
        families.add(matches[0])
    if len(families) != 1:
        raise ValueError("Mixed GPU families are not supported in a single training run")
    family = families.pop()
    if requested not in ("auto", family):
        raise ValueError(f"Requested {requested}, but detected {family}; refusing a mismatched profile")
    spec = PROFILES[family]
    count = len(devices)
    if count not in spec["gpu_counts"]:
        raise ValueError(f"{family} profile supports 4 or 8 GPUs; detected {count}. Other counts need validation")
    for device in devices:
        if str(device["compute_capability"]) != spec["cuda_arch"]:
            raise ValueError(f"Unexpected compute capability for {device['name']}: {device['compute_capability']}")
    cpu_arch = cpu_arch or platform.machine()
    if cpu_arch not in ("x86_64", "aarch64"):
        raise ValueError(f"Unsupported CPU architecture: {cpu_arch}")
    warnings = ["Profile selection is not a memory-fit or end-to-end training validation."]
    if family == "H100" and count == 4:
        warnings.append("Four H100s may have limited BF16 memory headroom; validate actual peak allocation.")
    return {
        "profile": family, "cpu_arch": cpu_arch, "num_gpus": count,
        "config_file": spec["config_file"],
        "torch_cuda_arch_list": spec["cuda_arch"],
        "distributed": {"tp_size": 1, "cp_size": 1, "pp_size": 1, "ep_size": count},
        "global_batch_size": 8, "local_batch_size": 1,
        "activation_checkpointing": True, "devices": devices, "warnings": warnings,
    }


def detect_profile(*, source="torch", requested="auto"):
    if source == "torch":
        import torch
        devices = []
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            devices.append({"name": prop.name, "memory_mib": prop.total_memory // 2**20,
                            "compute_capability": f"{prop.major}.{prop.minor}",
                            "mig": "MIG" in prop.name.upper()})
    elif source == "smi":
        inventory = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        if "MIG " in inventory:
            raise ValueError("MIG instances detected; this workflow requires full allocated GPUs")
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,name,memory.total,compute_cap", "--format=csv,noheader,nounits"
        ], text=True)
        rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            tokens = [token.strip() for token in visible.split(",") if token.strip()]
            selected = []
            for token in tokens:
                matches = ([rows[int(token)]] if token.isdigit() and int(token) < len(rows)
                           else [row for row in rows if row[0] == token])
                if len(matches) != 1:
                    raise ValueError("Cannot resolve CUDA_VISIBLE_DEVICES against host inventory")
                selected.extend(matches)
            if len({row[0] for row in selected}) != len(selected):
                raise ValueError("Duplicate CUDA_VISIBLE_DEVICES entries")
            rows = selected
        devices = [{"name": name, "memory_mib": int(float(memory)), "compute_capability": cap}
                   for _, name, memory, cap in rows]
    else:
        raise ValueError(f"Unknown inventory source: {source}")
    result = select_profile(devices, requested=requested)
    override = os.environ.get("NEMO_LORA_NUM_GPUS")
    if override is not None and int(override) != result["num_gpus"]:
        raise ValueError("NEMO_LORA_NUM_GPUS does not match the visible CUDA device count")
    return result


def docker_gpu_flags(mode="auto"):
    """Prefer installed CDI specs; keep legacy support without changing the host.

    Both paths expose all GPUs, just as the existing --gpus all workflow does.
    Use only on an allocation authorized for all Docker-visible GPUs.
    """
    if mode not in {"auto", "cdi", "legacy"}:
        raise ValueError("NEMO_GPU_INJECTION must be auto, cdi or legacy")
    if mode != "legacy":
        try:
            result = subprocess.run(["nvidia-ctk", "cdi", "list"], capture_output=True,
                                    text=True, timeout=10)
            ready = result.returncode == 0 and "nvidia.com/gpu=all" in result.stdout.splitlines()
        except (OSError, subprocess.TimeoutExpired):
            ready = False
        if ready:
            return ["--runtime=nvidia", "--env", "NVIDIA_VISIBLE_DEVICES=nvidia.com/gpu=all"]
        if mode == "cdi":
            raise ValueError("CDI requested but no installed nvidia.com/gpu=all specification is available")
    return ["--gpus", "all"]


def hardware_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("smi", "torch"), default="smi")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--field", choices=("torch_cuda_arch_list",))
    parser.add_argument("--docker-gpu-flags", action="store_true")
    args = parser.parse_args()
    if args.docker_gpu_flags:
        print("\n".join(docker_gpu_flags(os.environ.get("NEMO_GPU_INJECTION", "auto"))))
        return
    result = detect_profile(source=args.source)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(result[args.field] if args.field else rendered, end="\n" if args.field else "")


# Hf credentials
import argparse
import os
from pathlib import Path
import shlex
import stat
import sys


def read_private(path):
    path = Path(path)
    # O_NOFOLLOW and fstat avoid reading a substituted symlink between checks.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Credential file must be a regular file owned by the current user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("Credential file permissions are too broad; run chmod 600 on it")
        return handle.read()


def parse_dotenv(text):
    token = None
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if key.strip() != "HF_TOKEN":
            continue  # Do not import arbitrary environment variables from this file.
        if not separator or token is not None:
            raise ValueError(f"Malformed or duplicate HF_TOKEN entry on line {number}")
        try:
            pieces = shlex.split(value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"Invalid quoting on HF_TOKEN line {number}") from None
        if len(pieces) > 1:
            raise ValueError(f"HF_TOKEN must be one value on line {number}")
        token = pieces[0] if pieces else ""
    return token


def get_token(workspace, environ=None):
    environ = os.environ if environ is None else environ
    token = environ.get("HF_TOKEN", "").strip()
    source = "environment"
    if not token:
        path = Path(workspace) / ".env"
        if not path.exists():
            raise ValueError("Run setup.sh to create a private .env, then add your HF_TOKEN")
        token = (parse_dotenv(read_private(path)) or "").strip()
        source = ".env"
    if not token or any(char.isspace() for char in token):
        raise ValueError("HF_TOKEN is empty or contains whitespace; enter a Hugging Face read token")
    return token, source


def import_legacy(workspace):
    """One-time migration of this project's private token file; never overwrite .env."""
    workspace = Path(workspace)
    token = read_private(workspace / ".secrets/huggingface_token").strip()
    if not token or any(char.isspace() for char in token):
        raise ValueError("Existing credential is empty or malformed")
    fd = os.open(workspace / ".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("# Private Hugging Face credential; never commit this file.\nHF_TOKEN=" + shlex.quote(token) + "\n")
    print("Created private .env from the existing credential; original private file preserved")


def init_private(workspace):
    """Create an empty private template exclusively; preserve existing files/symlinks."""
    try:
        fd = os.open(Path(workspace) / ".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return
    with os.fdopen(fd, "w") as handle:
        handle.write("# Private Hugging Face read token. Never commit this file.\nHF_TOKEN=\n")
    print("Created private .env; add your authorized Hugging Face read token there")


def credentials_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--import-legacy", action="store_true")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.init:
            init_private(args.workspace)
            return
        if args.import_legacy:
            import_legacy(args.workspace)
            return
        token, source = get_token(args.workspace)
        if args.check:
            print(f"Hugging Face credential available via {source} (value hidden; access not tested)")
            return
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("Provide a command or --check")
        env = dict(os.environ, HF_TOKEN=token, NEMO_HF_AUTH_LOADED="1")
        os.execvpe(command[0], command, env)
    except (ValueError, OSError):
        # Do not serialize exception details which could include credential input.
        print("Hugging Face token needed. Add HF_TOKEN=your_read_token to the project .env file, run chmod 600 .env, then rerun bash setup.sh. An exported HF_TOKEN also works. Existing .env files are never overwritten.", file=sys.stderr)
        raise SystemExit(2) from None


# Resolve automodel source
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess

OFFICIAL_REPO = "https://github.com/NVIDIA-NeMo/Automodel.git"


def git(root, *args):
    return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-c", f"safe.directory={root}", "-C", str(root), *args], text=True).strip()


def resolve(workspace, root, *, repo=OFFICIAL_REPO):
    workspace, root = Path(workspace).resolve(), Path(root).resolve()
    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    receipt = logs / "automodel-source.json"
    with (logs / "automodel-source.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            record = json.loads(receipt.read_text())
            sha = record["commit"]
            if record["repo"] != repo or record["ref"] != "main" or not re.fullmatch(r"[0-9a-f]{40}", sha):
                raise ValueError("AutoModel source receipt does not match the requested official main source")
            if git(root, "rev-parse", "HEAD") != sha or git(root, "remote", "get-url", "origin") != repo:
                raise ValueError("Existing AutoModel checkout differs from its recorded source; use a new workspace")
            return record
        if (root / ".venv").exists():
            raise ValueError("Existing environment has no main-branch receipt; use a new workspace, do not upgrade in place")
        if not root.exists():
            subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--filter=blob:none", "--no-checkout", repo, str(root)], check=True)
            fresh = True
        else:
            fresh = False
            if not (root / ".git").exists():
                raise ValueError(f"Refusing to overwrite existing non-Git directory: {root}")
        if git(root, "remote", "get-url", "origin") != repo:
            raise ValueError("AutoModel origin differs from requested source")
        if not fresh and git(root, "status", "--porcelain"):
            raise ValueError("Refusing to switch a dirty AutoModel checkout")
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", f"safe.directory={root}", "-C", str(root), "fetch", "origin", "main"], check=True)
        sha = git(root, "rev-parse", "FETCH_HEAD")
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", f"safe.directory={root}", "-C", str(root), "checkout", "--detach", sha], check=True)
        record = {"repo": repo, "ref": "main", "commit": sha,
                  "policy": "latest main at first setup; reuse this exact commit for the run"}
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        return record


def verify(workspace, root):
    record = json.loads((Path(workspace) / "logs" / "automodel-source.json").read_text())
    if record["repo"] != OFFICIAL_REPO or record["ref"] != "main":
        raise ValueError("Expected NVIDIA-NeMo/Automodel main source")
    if git(root, "rev-parse", "HEAD") != record["commit"] or git(root, "remote", "get-url", "origin") != OFFICIAL_REPO:
        raise ValueError("AutoModel checkout changed since setup")
    return record


def source_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(resolve(args.workspace, args.root), indent=2))


# Preflight
import argparse
import os
from pathlib import Path
import shutil

from model import MODEL_LOCAL_DIR, inspect_snapshot


def check_source(repo):
    required = ('setup.sh', 'scripts/environment.py', 'scripts/model.py', 'scripts/data.py', 'scripts/training.py', 'scripts/evaluation.py', 'scripts/notebook.py', 'scripts/bootstrap.sh', 'configs/automodel_gb200.yaml', 'configs/finance_three_adapter_router.yaml', 'configs/finance_three_adapter_dynamo.yaml', 'configs/automodel_h100.yaml', 'dynamo/start_dynamo_after_training.sh', 'dynamo/sitecustomize.py', 'manifests/environment.json', 'manifests/datasets/sources.json', 'notebooks/nemotron_3_5_super_vl_finance_three_adapters.ipynb')
    missing = [name for name in required if not (repo / name).is_file()]
    if missing:
        raise ValueError("Incomplete project: " + ", ".join(missing))


def check_assets(repo):
    paths = {}
    for key, default in (("MODEL_PATH", MODEL_LOCAL_DIR),
                         ("TAT_SOURCE_DIR", "data/sources/tat-dqa"),
                         ("BIZ_SOURCE_DIR", "data/sources/bizbench")):
        path = Path(os.environ.get(f"NEMO_LORA_{key}_HOST", repo / default)).expanduser().resolve()
        if not path.is_relative_to(repo):
            raise ValueError("Assets must be inside the repository bind mount")
        paths[key] = path
    inspect_snapshot(paths["MODEL_PATH"])
    tat_files = ("tatdqa_dataset_train.json", "tatdqa_dataset_dev.json", "tatdqa_dataset_test_gold.json",
                 "tatdqa_docs_train.zip", "tatdqa_docs_dev.zip", "tatdqa_docs_test.zip")
    if not all((paths["TAT_SOURCE_DIR"] / name).is_file() for name in tat_files):
        raise ValueError("TAT-DQA download is incomplete")
    biz = paths["BIZ_SOURCE_DIR"]
    if not (all((biz / name).is_file() for name in ("bizbench_train.parquet", "bizbench_test.parquet"))
            or (len(list((biz / "data").glob("train-*.parquet"))) == 1
                and len(list((biz / "data").glob("test-*.parquet"))) == 1)):
        raise ValueError("BizBench download is incomplete")
    if shutil.disk_usage(repo).free < 80 * 2**30:
        raise ValueError("At least 80 GiB of workspace space must remain after downloads")


def preflight_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--assets", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    check_source(repo)
    if args.assets:
        check_assets(repo)
    print("Required source files verified" + ("; assets and workspace space verified" if args.assets else ""))


def main():
    import sys
    actions = {'hardware': hardware_main, 'credentials': credentials_main, 'source': source_main, 'preflight': preflight_main}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Commands: " + ", ".join(actions))
        return
    command = sys.argv.pop(1)
    if command not in actions:
        raise SystemExit("Unknown command: " + command)
    actions[command]()


if __name__ == "__main__":
    main()
