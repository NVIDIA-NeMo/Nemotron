"""Notebook streaming, run guards and the fixed-action local serving bridge."""
from __future__ import annotations


# Notebook workflow
import fcntl
import json
import os
from pathlib import Path
import subprocess


def run_visible(command, *, cwd=None):
    """Forward child output to Jupyter's output stream and propagate failure."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [str(part) for part in command], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    except BaseException:
        # Do not silently leave a trainer running after Interrupt Kernel.
        import signal
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"Process group {process.pid} still exists; inspect it before retrying.", flush=True)
        raise
    finally:
        process.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, command)


def require_adapters(workspace):
    workspace = Path(workspace)
    result = {}
    for name in ("visual_ocr", "reasoning", "code_generation"):
        stem = f"finance_{name}_lora_bf16_20261001"
        status = workspace / "logs" / f"{stem}_training.status"
        adapter = workspace / "checkpoints" / stem / "LOWEST_VAL" / "model"
        if not status.is_file() or status.read_text().strip() != "exit_0":
            raise RuntimeError(f"Complete the {name} training cell first: {status}")
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(adapter)
        result[name] = str(adapter)
    print(json.dumps(result, indent=2))
    return result


def training_guard(workspace):
    """Hold across a notebook's training cell; prevent two kernels training together."""
    path = Path(workspace) / "logs" / "notebook-training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        serving = path.with_name("dynamo-serving.lock").open("a")
        try:
            fcntl.flock(serving, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            serving.close()
            raise RuntimeError("Dynamo is using this workspace. Do not retrain until serving is explicitly stopped.")
        serving.close()
    except BaseException:
        handle.close()
        raise
    return handle


# Serving bridge
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def request_start(workspace, timeout=1800):
    control = Path(workspace) / ".notebook-control"
    heartbeat = control / "heartbeat"
    if not heartbeat.exists() or time.time() - heartbeat.stat().st_mtime > 15:
        raise RuntimeError("Host serving bridge unavailable; rerun setup.sh on the GPU host.")
    with (control / "client.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request_id = uuid.uuid4().hex
        write_json(control / "request.json", {"id": request_id, "action": "start"})
        deadline = time.monotonic() + timeout
        offset = 0
        last_notice = 0
        while time.monotonic() < deadline:
            state_path = control / "response.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get("id") == request_id:
                output = control / "launch.log"
                if output.exists():
                    with output.open() as stream:
                        stream.seek(offset)
                        print(stream.read(), end="", flush=True)
                        offset = stream.tell()
                if state.get("status") == "done":
                    if state["returncode"]:
                        raise RuntimeError(f"Dynamo startup failed (exit {state['returncode']}); inspect logs.")
                    print("Dynamo launch accepted; the next operation waits for adapter readiness.")
                    return
            if time.monotonic() - last_notice > 30:
                print("Waiting for host Dynamo launch (image pull may take several minutes)...", flush=True)
                last_notice = time.monotonic()
            time.sleep(1)
        raise TimeoutError("Dynamo launch timed out; inspect the host bridge and serving logs before retrying.")


def serve(workspace):
    workspace = Path(workspace).resolve()
    control = workspace / ".notebook-control"
    control.mkdir(mode=0o700, exist_ok=True)
    os.chmod(control, 0o700)
    with (control / "server.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        # Never replay a request from a prior setup/session after a host restart.
        request_path = control / "request.json"
        previous = json.loads(request_path.read_text()).get("id") if request_path.exists() else None
        while True:
            (control / "heartbeat").touch()
            request = json.loads(request_path.read_text()) if request_path.exists() else {}
            if request.get("id") and request.get("id") != previous:
                previous = request["id"]
                if request != {"id": previous, "action": "start"}:
                    write_json(control / "response.json", {"id": previous, "status": "done", "returncode": 2})
                    continue
                with (control / "launch.log").open("w") as log:
                    write_json(control / "response.json", {"id": previous, "status": "running"})
                    env = os.environ.copy()
                    env["NEMO_LORA_WORKSPACE_HOST"] = str(workspace)
                    process = subprocess.Popen(
                        ["bash", str(workspace / "dynamo" / "start_dynamo_after_training.sh")],
                        cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT,
                    )
                    while process.poll() is None:
                        (control / "heartbeat").touch()
                        time.sleep(1)
                    write_json(control / "response.json", {"id": previous, "status": "done", "returncode": process.returncode})
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    serve(args.workspace)
