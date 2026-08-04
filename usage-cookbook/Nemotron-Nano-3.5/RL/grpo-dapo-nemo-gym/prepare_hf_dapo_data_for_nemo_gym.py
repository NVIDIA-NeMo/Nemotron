#!/usr/bin/env python3
"""Convert DAPO-Math/AIME Hugging Face or local JSONL rows for NeMo Gym."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat-to", type=int)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def local_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(f"{path}:{line_number}: expected an object")
                    yield row
        return
    obj = json.loads(path.read_text())
    rows = obj if isinstance(obj, list) else obj.get("data", obj.get("rows"))
    if not isinstance(rows, list):
        raise TypeError(f"{path}: expected a JSON list or a data/rows list")
    yield from rows


def load_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    path = Path(args.dataset)
    if path.is_file():
        return local_rows(path)
    from datasets import load_dataset

    return iter(
        load_dataset(
            args.dataset,
            split=args.split,
            cache_dir=args.cache_dir,
        )
    )


def answer_from(row: dict[str, Any]) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        answer = reward_model.get("ground_truth")
        if answer not in (None, ""):
            return answer
    for key in ("expected_answer", "ground_truth", "answer", "target"):
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def prompt_from(row: dict[str, Any]) -> Any:
    for key in ("prompt", "question", "problem", "input"):
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def messages_from(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        return [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
            for message in prompt
        ]
    if isinstance(prompt, dict) and "input" in prompt:
        return messages_from(prompt["input"])
    return [{"role": "user", "content": str(prompt)}]


def convert(row: dict[str, Any], strict: bool, index: int) -> dict[str, Any] | None:
    if "responses_create_params" in row and "agent_ref" in row:
        return row
    prompt, answer = prompt_from(row), answer_from(row)
    if prompt in (None, "") or answer in (None, ""):
        if strict:
            raise ValueError(f"row {index}: missing prompt or expected answer")
        return None
    messages = messages_from(prompt)
    question = next(
        (message["content"] for message in messages if message["role"] == "user"),
        messages[0]["content"],
    )
    return {
        "responses_create_params": {"input": messages},
        "question": str(row.get("question") or question),
        "expected_answer": str(answer),
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "math_with_judge_simple_agent",
        },
        "dataset": str(row.get("data_source") or row.get("dataset") or "dapo-math"),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    first: dict[str, Any] | None = None
    repeat_base: list[dict[str, Any]] = []
    with output.open("w") as stream:
        for index, row in enumerate(load_rows(args), 1):
            if index <= args.skip:
                continue
            item = convert(dict(row), args.strict, index)
            if item is None:
                continue
            if first is None:
                first = item
            if args.repeat_to:
                repeat_base.append(item)
            stream.write(
                json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n"
            )
            count += 1
            if args.limit is not None and count >= args.limit:
                break
            if args.repeat_to is not None and count >= args.repeat_to:
                break
        if args.repeat_to and repeat_base:
            while count < args.repeat_to:
                item = repeat_base[count % len(repeat_base)]
                stream.write(
                    json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n"
                )
                count += 1
    if first is None:
        output.unlink(missing_ok=True)
        raise RuntimeError("no rows converted")
    print(f"Wrote {count} rows to {output}")
    print(json.dumps(first, indent=2)[:1200])


if __name__ == "__main__":
    main()
