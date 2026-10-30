"""Task routing, shuffled test inference and finance scoring."""
from __future__ import annotations


# Finance metrics
import re
import unicodedata
from collections import defaultdict


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:!?")


def number_and_scale(value: object) -> tuple[float | None, str]:
    text = normalize(value).replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    match = re.fullmatch(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)\s*(%|percent|thousand|million|billion)?",
        text,
    )
    if not match:
        return None, ""
    scale = match.group(2) or ""
    if scale == "%":
        scale = "percent"
    return float(match.group(1)), scale


def relaxed(prediction: str, reference: str, tolerance: float) -> bool:
    (left, left_scale), (right, right_scale) = number_and_scale(prediction), number_and_scale(reference)
    if left is not None and right is not None:
        if right_scale and left_scale != right_scale:
            return False
        return abs(left - right) <= max(1e-6, abs(right) * tolerance)
    return normalize(prediction) == normalize(reference)


def answer_tokens(value: object) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(value))


def token_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    predicted, expected = Counter(answer_tokens(prediction)), Counter(answer_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted and not expected:
        return 1.0
    if not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def unordered_spans(value: object) -> list[str]:
    return sorted(
        normalize(part) for part in re.split(r"\s*(?:\||##|;)\s*", str(value)) if normalize(part)
    )


def task_score(task: str, prediction: str, reference: str, tolerance: float) -> tuple[bool, float]:
    if task.endswith("multi-span"):
        matched = unordered_spans(prediction) == unordered_spans(reference)
        return matched, token_f1(prediction, reference)
    if task.endswith("span"):
        score = token_f1(prediction, reference)
        return normalize(prediction) == normalize(reference), score
    matched = relaxed(prediction, reference, tolerance)
    return matched, float(matched)

def metrics(rows: list[dict], expected: int, tolerance: float) -> dict:
    def summarize(items: list[dict]) -> dict:
        total = len(items)
        return {
            "samples": total,
            "exact_accuracy": sum(row["exact"] for row in items) / total if total else 0.0,
            "relaxed_accuracy": sum(row["relaxed"] for row in items) / total if total else 0.0,
            "closed_trace_rate": sum(row["trace_closed"] for row in items) / total if total else 0.0,
            "python_syntax_rate": (
                sum(row["program_syntax_valid"] for row in items if row["program_syntax_valid"] is not None)
                / sum(row["program_syntax_valid"] is not None for row in items)
                if any(row["program_syntax_valid"] is not None for row in items) else None
            ),
            "task_accuracy": sum(row["task_correct"] for row in items) / total if total else 0.0,
            "answer_f1": sum(row["answer_f1"] for row in items) / total if total else 0.0,
        }

    grouped: dict[str, list[dict]] = defaultdict(list)
    grouped_capabilities: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
        grouped_capabilities[row["capability"]].append(row)
    result = summarize(rows)
    result.update({
        "expected_samples": expected,
        "evaluated_samples": len(rows),
        "coverage": len(rows) / expected if expected else 0.0,
        "numeric_relative_tolerance": tolerance,
        "hit_max_new_tokens": sum(row["hit_max_new_tokens"] for row in rows),
        "by_capability": {
            name: summarize(items) for name, items in sorted(grouped_capabilities.items())
        },
        "by_task": {name: summarize(items) for name, items in sorted(grouped.items())},
    })
    return result


# Finance three adapter router
import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from data import TASK_DATASETS, load_finance_task_split, task_dataset_paths


from model import MODEL_ID, MODEL_REVISION


def read_route_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_rows(test, samples_per_task: int) -> list[dict]:
    counts: Counter[str] = Counter()
    selected = []
    for row in test:
        task = str(row["task"])
        if samples_per_task and counts[task] >= samples_per_task:
            continue
        counts[task] += 1
        selected.append(dict(row))
    return selected


def route_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--router-config", type=Path,
        help="Portable routing contract; defaults to configs/finance_three_adapter_router.yaml.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-task", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--output-name", default="finance_three_adapter_routed_test_bf16_20261001")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    for capability in TASK_DATASETS:
        parser.add_argument(
            f"--{capability.replace('_', '-')}-adapter",
            type=Path,
            dest=f"{capability}_adapter",
        )
    return parser.parse_args()


def route_main() -> None:
    args = route_args()
    workspace = args.workspace.expanduser().resolve()
    paths = task_dataset_paths(workspace / "data")
    router_config_path = (
        args.router_config or workspace / "configs" / "finance_three_adapter_router.yaml"
    ).expanduser().resolve()
    router_config = yaml.safe_load(router_config_path.read_text())
    if set(router_config["routes"]) != set(TASK_DATASETS):
        raise ValueError(f"Router capabilities do not match task datasets: {router_config_path}")
    for capability, spec in TASK_DATASETS.items():
        route = router_config["routes"][capability]
        if set(route["tasks"]) != set(spec["tasks"]):
            raise ValueError(f"Router task mismatch for {capability}")
        if (workspace / route["dataset"]).resolve() != paths[capability]:
            raise ValueError(f"Router dataset mismatch for {capability}")
    test = load_finance_task_split(list(paths.values()), "test")
    selected = select_rows(test, args.samples_per_task)
    for row in selected:
        if str(row.get("reasoning") or "").strip() or str(row.get("program") or "").strip():
            raise AssertionError(f"Test trace leakage in {row['sample_id']}")
        if str(row["target"]).strip() != str(row["answer"]).strip():
            raise AssertionError(f"Test target is not answer-only in {row['sample_id']}")

    adapters = {
        capability: (
            getattr(args, f"{capability}_adapter")
            or workspace / router_config["routes"][capability]["adapter"]
        ).expanduser().resolve()
        for capability in TASK_DATASETS
    }
    indices = list(range(len(selected)))
    random.Random(args.seed).shuffle(indices)
    output_dir = workspace / "evaluation" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "route_manifest.jsonl"
    manifest_rows = []
    for shuffled_index, original_index in enumerate(indices):
        row = selected[original_index]
        capability = str(row["capability"])
        if capability not in adapters:
            raise ValueError(f"No adapter route for capability {capability!r}")
        manifest_rows.append({
            "shuffled_index": shuffled_index,
            "source_index": original_index,
            "sample_id": str(row["sample_id"]),
            "task": str(row["task"]),
            "capability": capability,
            "adapter": str(adapters[capability]),
            "route_basis": "explicit Task tag (reference answer and test trace are not used)",
        })
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows)
    )
    plan = {
        "seed": args.seed,
        "samples": len(manifest_rows),
        "samples_per_task": args.samples_per_task,
        "route_mode": "deterministic explicit-task-tag routing",
        "execution_mode": "Dynamo dynamic LoRA in shuffled request order",
        "counts_by_capability": dict(Counter(row["capability"] for row in manifest_rows)),
        "adapters": {name: str(path) for name, path in adapters.items()},
        "router_config": str(router_config_path),
        "manifest": str(manifest_path),
    }
    (output_dir / "route_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    if any(getattr(args, f"{capability}_adapter") for capability in TASK_DATASETS):
        raise ValueError("Configure custom adapters in the Dynamo YAML and load them before evaluation")
    command = [
        sys.executable, str(Path(__file__).with_name("evaluation.py")), "evaluate",
        "--workspace", str(workspace), "--seed", str(args.seed),
        "--output-name", args.output_name,
        "--max-new-tokens", str(args.max_new_tokens),
        "--tolerance", str(args.tolerance),
    ]
    if args.samples_per_task:
        command.extend(["--samples-per-task", str(args.samples_per_task)])
    if args.resume:
        command.append("--resume")
    subprocess.run(command, cwd=workspace, check=True)


# Evaluate finance dynamo
import argparse
import ast
import base64
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from data import build_user_prompt, load_finance_task_split, task_dataset_paths


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def split_raw_completion(raw: str) -> tuple[str, str]:
    """Return final answer and reasoning from Nemotron parser output.

    Dynamo's Nemotron reasoning parser may remove the opening ``<think>`` token
    while leaving ``</think>`` in ``message.content``. Split on the closing
    delimiter even when the opening delimiter is absent.
    """
    text = str(raw or "").strip()
    if "</think>" not in text:
        return text, ""
    reasoning, answer = text.split("</think>", 1)
    if "<think>" in reasoning:
        reasoning = reasoning.split("<think>", 1)[1]
    return answer.strip(), reasoning.strip()


def completion_text(message: dict) -> tuple[str, str, str]:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    raw = f"<think>\n{reasoning}\n</think>\n{content}" if reasoning else str(content)
    prediction, parsed_reasoning = split_raw_completion(raw)
    return prediction, parsed_reasoning, raw.strip()


def post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode()
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url, data=body, method="POST", headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (OSError, urllib.error.HTTPError) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                detail = error.read().decode(errors="replace")
                raise RuntimeError(f"Dynamo request rejected: HTTP {error.code}: {detail}") from error
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("Dynamo request failed after retries") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-task", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-name", default="finance_three_adapter_dynamo_test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Bounded simultaneous requests; 1 runs serially (default: 8)")
    return parser.parse_args()


def evaluate_main() -> None:
    args = parse_args()
    if not 1 <= args.concurrency <= 32:
        raise ValueError("--concurrency must be between 1 and 32")
    workspace = args.workspace.resolve()
    config_path = (args.config or workspace / "configs" / "finance_three_adapter_dynamo.yaml").resolve()
    config = yaml.safe_load(config_path.read_text())
    output_dir = workspace / "evaluation" / args.output_name
    plan_command = [
        sys.executable, str(Path(__file__).with_name("evaluation.py")), "route",
        "--workspace", str(workspace), "--plan-only", "--seed", str(args.seed),
        "--output-name", args.output_name,
    ]
    if args.samples_per_task:
        plan_command.extend(["--samples-per-task", str(args.samples_per_task)])
    subprocess.run(plan_command, cwd=workspace, check=True)
    route_manifest = read_jsonl(output_dir / "route_manifest.jsonl")

    test = load_finance_task_split(list(task_dataset_paths(workspace / "data").values()), "test")
    by_id = {str(row["sample_id"]): dict(row) for row in test}
    if len(by_id) != len(test):
        raise ValueError("Test sample IDs are not unique")
    route_models = {name: route["model"] for name, route in config["routes"].items()}
    frontend_base = os.environ.get(
        "DYNAMO_FRONTEND_URL", f"http://127.0.0.1:{config['runtime']['frontend_port']}",
    ).rstrip("/")
    frontend = frontend_base + "/v1/chat/completions"
    predictions_path = output_dir / "routed_predictions.jsonl"
    metrics_path = output_dir / "routed_metrics.json"
    stored_predictions = (
        read_jsonl(predictions_path)
        if args.resume and predictions_path.is_file()
        else []
    )
    if stored_predictions:
        for result in stored_predictions:
            row = by_id[str(result["sample_id"])]
            prediction, reasoning = split_raw_completion(result.get("raw_completion") or "")
            syntax_valid = None
            if row["task_family"] == "program_synthesis":
                try:
                    ast.parse(reasoning)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False
            scored, answer_f1 = task_score(
                row["task"], prediction, row["answer"], args.tolerance,
            )
            result.update({
                "predicted_answer": prediction,
                "reasoning": reasoning,
                "exact": normalize(prediction) == normalize(row["answer"]),
                "relaxed": relaxed(prediction, row["answer"], args.tolerance),
                "task_correct": scored,
                "answer_f1": answer_f1,
                "trace_closed": bool(reasoning) or "</think>" in str(result.get("raw_completion") or ""),
                "program_syntax_valid": syntax_valid,
            })
        predictions_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stored_predictions)
        )
    completed = {str(row["sample_id"]): row for row in stored_predictions}
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {predictions_path}; use --resume")

    mode = "a" if completed else "w"
    with predictions_path.open(mode) as output:
        def evaluate_route(item):
            ordinal, route = item
            sample_id = str(route["sample_id"])
            row = by_id[sample_id]
            if str(row.get("reasoning") or "").strip() or str(row.get("program") or "").strip():
                raise AssertionError(f"Test trace leakage in {sample_id}")
            capability = str(row["capability"])
            prompt = build_user_prompt(
                row["task"], row["task_family"], row["question"], row.get("context") or "",
            )
            if row.get("image_path"):
                content: object = [
                    {"type": "image_url", "image_url": {"url": image_data_url(workspace / row["image_path"])}},
                    {"type": "text", "text": prompt},
                ]
            else:
                content = prompt
            task_limit = min(
                args.max_new_tokens,
                256 if row["task"].endswith(("arithmetic", "count")) else 512,
            )
            payload = {
                "model": route_models[capability],
                "messages": [{"role": "user", "content": content}],
                "max_tokens": task_limit,
                "temperature": 0,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            started = time.perf_counter()
            response = post_json(frontend, payload, args.timeout)
            latency = time.perf_counter() - started
            message = response["choices"][0]["message"]
            prediction, reasoning, raw = completion_text(message)
            syntax_valid = None
            if row["task_family"] == "program_synthesis":
                try:
                    ast.parse(reasoning)
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False
            scored, answer_f1 = task_score(row["task"], prediction, row["answer"], args.tolerance)
            usage = response.get("usage") or {}
            result = {
                "shuffled_index": route["shuffled_index"], "sample_id": sample_id,
                "source_dataset": row["source_dataset"], "capability": capability,
                "task": row["task"], "question": row["question"], "reference_answer": row["answer"],
                "raw_completion": raw, "predicted_answer": prediction, "reasoning": reasoning,
                "routed_model": route_models[capability],
                "exact": normalize(prediction) == normalize(row["answer"]),
                "relaxed": relaxed(prediction, row["answer"], args.tolerance),
                "task_correct": scored, "answer_f1": answer_f1,
                "trace_closed": bool(reasoning) or "</think>" in raw,
                "program_syntax_valid": syntax_valid,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "generated_tokens": int(usage.get("completion_tokens") or 0),
                "hit_max_new_tokens": int(usage.get("completion_tokens") or 0) >= task_limit,
                "latency_seconds": round(latency, 4),
            }
            return ordinal, result

        pending = [(ordinal, route) for ordinal, route in enumerate(route_manifest, 1)
                   if str(route["sample_id"]) not in completed]
        print(f"Evaluating {len(pending)} pending samples with concurrency={args.concurrency}", flush=True)
        # Bound both requests and queued work. Persist in seeded route order;
        # only this thread writes results, so resume files cannot interleave.
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for start in range(0, len(pending), args.concurrency):
                for ordinal, result in executor.map(evaluate_route, pending[start:start + args.concurrency]):
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    completed[result["sample_id"]] = result
                    current = metrics(list(completed.values()), len(route_manifest), args.tolerance)
                    current.update({"route_backend": "NVIDIA Dynamo vLLM dynamic LoRA", "route_config": str(config_path),
                                    "request_concurrency": args.concurrency})
                    metrics_path.write_text(json.dumps(current, indent=2) + "\n")
                    print(f"{ordinal}/{len(route_manifest)} {result['capability']} {result['task']} "
                          f"correct={result['task_correct']} {result['latency_seconds']:.1f}s", flush=True)

    report = metrics(list(completed.values()), len(route_manifest), args.tolerance)
    report.update({
        "route_backend": "NVIDIA Dynamo vLLM dynamic LoRA",
        "route_config": str(config_path), "seed": args.seed,
        "models": route_models,
        "request_concurrency": args.concurrency,
    })
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    import sys
    actions = {'route': route_main, 'evaluate': evaluate_main}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Commands: " + ", ".join(actions))
        return
    command = sys.argv.pop(1)
    if command not in actions:
        raise SystemExit("Unknown command: " + command)
    actions[command]()


if __name__ == "__main__":
    main()
