"""Serve this run's export, infer all benchmark images, and compute paper F1 metrics."""

import argparse
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import metrics
from assets import sha256, write_json
from convert_predictions import convert
from prompt import PROMPT_AFTER, PROMPT_BEFORE
from rxncaption_dataset_adapter import _clean_target


def normalize_response(response):
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        value, _, _ = _clean_target(cleaned, remove_empty_text=False)
        return value, False
    except (ValueError, KeyError, TypeError):
        return "[]", True


def score(ground_truth, converted):
    gold = {row["file_name"]: row for row in json.loads(ground_truth.read_text())["images"]}
    predictions = {row["file_name"]: row for row in json.loads(converted.read_text())["images"]}
    if set(gold) != set(predictions):
        raise ValueError("Every benchmark image must be included in scoring")
    result = {}
    for name, function, mol_only in (
        ("SoftMatch", metrics._soft_match, True),
        ("HybridMatch", metrics._hybrid_match, False),
    ):
        tp = fp = fn = 0
        for filename in sorted(gold):
            a, b, c, _ = metrics.evaluate_image(gold[filename], predictions[filename], function, mol_only)
            tp, fp, fn = tp + a, fp + b, fn + c
        result[name] = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=100 * tp / (tp + fp) if tp + fp else 0,
            recall=100 * tp / (tp + fn) if tp + fn else 0,
            f1=200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        )
    return result


def predict(image, endpoint):
    encoded = base64.b64encode(image.read_bytes()).decode()
    payload = dict(
        model="model",
        temperature=0,
        max_tokens=2048,
        chat_template_kwargs={"enable_thinking": False},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_BEFORE},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    {"type": "text", "text": PROMPT_AFTER},
                ],
            }
        ],
    )
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as reply:
        content = json.load(reply)["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("Server returned no answer content")
    return content


def serving_view(model, output):
    output.mkdir(exist_ok=False)
    for path in model.iterdir():
        if path.is_file() and path.suffix in (".json", ".py", ".txt", ".jinja"):
            shutil.copyfile(path, output / path.name)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if Path(name).name != name:
            raise ValueError("Invalid shard name")
        (output / name).symlink_to((model / name).resolve())
    config = json.loads((output / "config.json").read_text())
    config["rxn_vision_final_layernorm"] = False
    write_json(output / "config.json", config)
    tokenizer_path = output / "tokenizer_config.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    tokenizer["tokenizer_class"] = "PreTrainedTokenizerFast"
    write_json(tokenizer_path, tokenizer)


def evaluate(run, benchmark, port):
    report = json.loads((run / "export.json").read_text())
    model = Path(report["model_path"])
    if report["status"] != "verified" or (model / ".export_incomplete").exists():
        raise ValueError("Model export is incomplete")
    prepared = json.loads((run / "prepared.json").read_text())
    for name, digest in prepared["benchmark_labels_sha256"].items():
        if sha256(benchmark / name) != digest:
            raise ValueError(f"Benchmark labels changed after preparation: {name}")
    names = sorted(prepared["benchmark_sha256"])
    for name in names:
        if sha256(benchmark / "images/gt_bivp" / name) != prepared["benchmark_sha256"][name]:
            raise ValueError(f"Benchmark image changed after preparation: {name}")
    output = run / "evaluation"
    output.mkdir(exist_ok=False)
    view = output / "serving_model"
    serving_view(model, view)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(view),
        "--served-model-name",
        "model",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "65536",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--no-enable-prefix-caching",
        "--reasoning-parser",
        "nemotron_v3",
    ]
    endpoint = f"http://127.0.0.1:{port}"
    with (output / "server.log").open("w") as log:
        environment = {k: v for k, v in os.environ.items() if k != "VLLM_API_KEY"}
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment
        )
        try:
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited; inspect evaluation/server.log")
                try:
                    with urllib.request.urlopen(endpoint + "/v1/models", timeout=2) as reply:
                        available = json.load(reply)["data"]
                    if any(record["id"] == "model" for record in available):
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("vLLM startup exceeded 20 minutes")
                time.sleep(5)
            raw, normalized, invalid, errors = {}, {}, [], []
            for number, name in enumerate(names, 1):
                try:
                    response = predict(benchmark / "images/gt_bivp" / name, endpoint)
                except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as error:
                    errors.append(name)
                    response = f"ERROR: {type(error).__name__}"
                value, bad = normalize_response(response)
                if bad:
                    invalid.append(name)
                raw[name] = {"response": response}
                normalized[name] = {"response": value}
                write_json(output / "raw.json", raw)
                print(f"Evaluated {number}/{len(names)}", flush=True)
            write_json(output / "normalized.json", normalized)
            ground_truth = benchmark / "ground_truth/ground_truth.json"
            convert(
                output / "normalized.json",
                benchmark / "mapdicts/mapdict_gt.json",
                ground_truth,
                output / "converted.json",
            )
            result = dict(
                images=len(names),
                inference_errors=errors,
                invalid_json_images=invalid,
                smoke_test=report["smoke_test"],
                metrics=score(ground_truth, output / "converted.json"),
            )
            write_json(output / "metrics.json", result)
            print(json.dumps(result, indent=2))
            if errors:
                raise RuntimeError(f"{len(errors)} inference requests failed; this evaluation is not successful")
            return result
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--port", type=int, default=12520)
    args = parser.parse_args()
    evaluate(args.run_dir, args.benchmark, args.port)
