#!/usr/bin/env python3
"""Prepare Hugging Face or JSONL DAPO-Math rows for NeMo Gym.

The Ultra NeMo Gym guide uses ``NemoGymDataset``. That dataset expects one
JSON object per line, and each object is passed through to NeMo Gym as the
environment input. This helper converts DAPO-Math-17K style rows into the
``math_with_judge_simple_agent`` row shape used by the baseline recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROMPT_CANDIDATES = (
    "prompt",
    "question",
    "problem",
    "input",
    "instruction",
    "query",
    "text",
)
SCHEMA_CANDIDATES = (
    "schema_str",
    "schema",
    "json_schema",
    "output_schema",
    "response_schema",
    "schema_json",
)
ANSWER_CANDIDATES = (
    "expected_answer",
    "ground_truth",
    "answer",
    "final_answer",
    "target",
)
PASSTHROUGH_KEYS = (
    "schema_str",
    "schema_type",
    "schema_fields_count",
    "question",
    "expected_answer",
    "ability",
    "data_source",
    "reward_model",
    "extra_info",
    "pass_rate",
    "pass_rate_total",
    "pass_rate_passed",
    "dataset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HF/local DAPO rows into NeMo Gym JSONL."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "HF dataset name, load_from_disk directory, or local .json/.jsonl file. "
            "For local files, no datasets dependency is required."
        ),
    )
    parser.add_argument("--config", default=None, help="Optional HF dataset config.")
    parser.add_argument("--split", default="train", help="HF dataset split.")
    parser.add_argument(
        "--data-files",
        default=None,
        help="Optional data_files value for datasets.load_dataset.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache dir.")
    parser.add_argument(
        "--token",
        default=None,
        help="HF token or true to let datasets use the logged-in token.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output NeMo Gym JSONL path. Required unless --inspect is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum source rows to convert before optional repeat-to.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of source rows to skip before conversion.",
    )
    parser.add_argument(
        "--repeat-to",
        type=int,
        default=None,
        help="Repeat converted rows cyclically until this many rows are written.",
    )
    parser.add_argument(
        "--prompt-field",
        default=None,
        help=f"Prompt field. Defaults to first present of {PROMPT_CANDIDATES}.",
    )
    parser.add_argument(
        "--schema-field",
        default=None,
        help=f"Schema field. Defaults to first present of {SCHEMA_CANDIDATES}.",
    )
    parser.add_argument(
        "--answer-field",
        default=None,
        help=(
            "Expected-answer field. Defaults to reward_model.ground_truth or first "
            f"present of {ANSWER_CANDIDATES}."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Value for the output row's dataset field. Defaults to --dataset.",
    )
    parser.add_argument(
        "--agent-name",
        default="math_with_judge_simple_agent",
        help="NeMo Gym responses_api agent name.",
    )
    parser.add_argument(
        "--agent-type",
        default="responses_api_agents",
        help="NeMo Gym agent_ref type.",
    )
    parser.add_argument("--schema-type", default="json", help="Schema type metadata.")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message prepended to string prompts.",
    )
    parser.add_argument(
        "--copy-field",
        action="append",
        default=[],
        help="Extra source field to copy to the output row. May be repeated.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a row is missing prompt/answer fields instead of skipping it.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the first row's top-level keys and exit without writing.",
    )
    return parser.parse_args()


def load_local_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_no}: expected JSON object")
                yield row
        return

    if path.suffix == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            for row in obj:
                if not isinstance(row, dict):
                    raise TypeError(f"{path}: expected list of JSON objects")
                yield row
            return
        if isinstance(obj, dict):
            for key in ("data", "train", "rows", "examples"):
                value = obj.get(key)
                if isinstance(value, list):
                    for row in value:
                        if not isinstance(row, dict):
                            raise TypeError(f"{path}:{key}: expected JSON objects")
                        yield row
                    return
        raise TypeError(f"{path}: expected JSONL, a JSON list, or a dict with rows")

    raise ValueError(f"Unsupported local input suffix: {path.suffix}")


def load_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    path = Path(args.dataset)
    if path.is_file():
        return load_local_rows(path)

    if path.is_dir():
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise RuntimeError(
                "datasets is required for load_from_disk directories"
            ) from exc
        dataset = load_from_disk(str(path))
        if isinstance(dataset, dict):
            dataset = dataset[args.split]
        return iter(dataset)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for HF dataset names. Install it or pass a local JSONL."
        ) from exc

    kwargs: dict[str, Any] = {
        "split": args.split,
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config:
        kwargs["name"] = args.config
    if args.data_files:
        kwargs["data_files"] = args.data_files
    if args.token:
        kwargs["token"] = True if args.token.lower() == "true" else args.token
    dataset = load_dataset(args.dataset, **{k: v for k, v in kwargs.items() if v is not None})
    return iter(dataset)


def get_field(row: dict[str, Any], explicit: str | None, candidates: tuple[str, ...]) -> Any:
    if explicit:
        return row.get(explicit)
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def get_answer(row: dict[str, Any], explicit: str | None) -> Any:
    if explicit:
        return row.get(explicit)
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") not in (None, ""):
        return reward_model["ground_truth"]
    return get_field(row, None, ANSWER_CANDIDATES)


def to_schema_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def schema_fields_count(schema_str: str | None) -> str:
    if not schema_str:
        return ""
    try:
        schema = json.loads(schema_str)
    except json.JSONDecodeError:
        return ""
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            return str(len(props))
        required = schema.get("required")
        if isinstance(required, list):
            return str(len(required))
    return ""


def normalize_input(prompt: Any, system_prompt: str | None) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        messages = []
        for message in prompt:
            if not isinstance(message, dict):
                raise TypeError("message prompts must be a list of objects")
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            messages.append({"role": role, "content": str(content)})
        return messages

    if isinstance(prompt, dict) and "input" in prompt:
        return normalize_input(prompt["input"], system_prompt)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": str(prompt)})
    return messages


def first_user_content(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[0].get("content", "") if messages else ""


def convert_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    row_index: int,
) -> dict[str, Any] | None:
    if "responses_create_params" in row and "agent_ref" in row:
        output = dict(row)
        output.setdefault("dataset", args.dataset_name or args.dataset)
        return output

    prompt = get_field(row, args.prompt_field, PROMPT_CANDIDATES)
    answer = get_answer(row, args.answer_field)

    if prompt in (None, "") or answer in (None, ""):
        message = (
            f"row {row_index}: missing "
            f"{'prompt' if prompt in (None, '') else ''}"
            f"{' and ' if prompt in (None, '') and answer in (None, '') else ''}"
            f"{'expected_answer' if answer in (None, '') else ''}"
        )
        if args.strict:
            raise ValueError(message)
        print(f"Skipping {message}", file=sys.stderr)
        return None

    messages = normalize_input(prompt, args.system_prompt)
    output: dict[str, Any] = {
        "responses_create_params": {
            "input": messages,
        },
        "question": str(row.get("question") or first_user_content(messages)),
        "expected_answer": str(answer),
        "agent_ref": {
            "type": args.agent_type,
            "name": args.agent_name,
        },
        "dataset": args.dataset_name or args.dataset,
    }

    schema = to_schema_str(get_field(row, args.schema_field, SCHEMA_CANDIDATES))
    if schema:
        output["schema_str"] = schema
        output["schema_type"] = args.schema_type
        output["schema_fields_count"] = schema_fields_count(schema)

    for key in PASSTHROUGH_KEYS:
        if key in row and key not in output:
            output[key] = row[key]
    for key in args.copy_field:
        if key in row:
            output[key] = row[key]
    return output


def main() -> int:
    args = parse_args()
    if not args.inspect and not args.output:
        raise SystemExit("--output is required unless --inspect is set")

    rows_iter = load_rows(args)

    if args.inspect:
        first = next(rows_iter, None)
        if first is None:
            print("dataset is empty")
            return 1
        print("Top-level keys:")
        for key in sorted(first):
            print(f"- {key}: {type(first[key]).__name__}")
        return 0

    converted: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows_iter, start=1):
        if row_index <= args.skip:
            continue
        if args.limit is not None and len(converted) >= args.limit:
            break
        output = convert_row(dict(row), args, row_index)
        if output is not None:
            converted.append(output)

    if not converted:
        raise RuntimeError("no rows converted")

    if args.repeat_to is not None:
        base = list(converted)
        while len(converted) < args.repeat_to:
            converted.append(base[len(converted) % len(base)])
        converted = converted[: args.repeat_to]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")

    print(f"Wrote {len(converted)} rows to {output_path}")
    print("First row summary:")
    first = converted[0]
    print(json.dumps({
        "agent_ref": first.get("agent_ref"),
        "dataset": first.get("dataset"),
        "has_expected_answer": "expected_answer" in first,
        "has_responses_create_params": "responses_create_params" in first,
        "question_preview": str(first.get("question", ""))[:80],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
