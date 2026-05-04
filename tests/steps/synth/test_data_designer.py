# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/synth/data_designer``.

Also validates the declarative column-spec shape that step.py translates into
the upstream ``DataDesignerConfigBuilder`` API. We don't import data_designer
itself (heavy runtime dep), only ensure the YAML keys are well-formed.
"""

import json
import re
from pathlib import Path

import yaml

from nemotron.steps.synth.data_designer.step import project_records, records_from_designer_result

from .._step_helpers import assert_step_static, step_dir

VALID_COLUMN_TYPES = {"category", "seed", "llm_text", "llm_structured", "llm_judge"}
LLM_COLUMN_TYPES = {"llm_text", "llm_structured", "llm_judge"}

STEP = step_dir(__file__, "synth", "data_designer")
REPO_ROOT = STEP.parents[4]


def _config_paths() -> list[Path]:
    return sorted((STEP / "config").glob("*.yaml"))


def _load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(data, dict), f"{path}: YAML must be a mapping"
    return data


def test_synth_data_designer_static() -> None:
    assert_step_static(
        STEP,
        expected_name="steps/synth/data_designer",
        expected_launch="python",
        expected_default_config="default",
    )


def _load_columns(path: Path) -> list[dict]:
    data = _load_config(path)
    cols = data.get("columns", [])
    assert isinstance(cols, list), f"{path}: 'columns' must be a list"
    return cols


def _seed_fields(config: dict) -> set[str]:
    seed = config.get("seed_dataset") or {}
    fields = seed.get("fields") or []
    return set(fields)


def _declared_fields(config: dict) -> set[str]:
    return _seed_fields(config) | {col["name"] for col in config.get("columns") or []}


def test_columns_use_supported_types() -> None:
    for path in _config_paths():
        for col in _load_columns(path):
            assert col["type"] in VALID_COLUMN_TYPES, f"unknown column type {col['type']!r} in {path.name}"


def test_seed_columns_reference_declared_seed_fields() -> None:
    """Every ``type: seed`` column must name a field supplied by the seed dataset."""
    for path in _config_paths():
        seed_fields = _seed_fields(_load_config(path))
        for col in _load_columns(path):
            if col["type"] == "seed":
                assert col["name"] in seed_fields, (
                    f"{path.name}: seed column {col['name']!r} is not listed in seed_dataset.fields"
                )


def test_seed_dataset_paths_and_fields_are_valid() -> None:
    for path in _config_paths():
        cfg = _load_config(path)
        seed = cfg.get("seed_dataset")
        if not seed:
            continue

        fields = seed.get("fields")
        assert fields, f"{path.name}: seed_dataset must declare non-empty fields"
        assert all(isinstance(field, str) and field for field in fields), (
            f"{path.name}: seed_dataset.fields must be non-empty strings"
        )

        raw_path = seed.get("path")
        assert raw_path, f"{path.name}: seed_dataset.path is required"
        if raw_path.startswith("${oc.env:PWD}/"):
            seed_path = REPO_ROOT / raw_path.removeprefix("${oc.env:PWD}/")
            assert seed_path.exists(), f"{path.name}: seed dataset does not exist: {seed_path}"

            with seed_path.open(encoding="utf-8") as f:
                first_record = json.loads(next(line for line in f if line.strip()))
            missing = set(fields) - set(first_record)
            assert not missing, f"{path.name}: seed file {seed_path.name} missing fields {sorted(missing)}"


def test_llm_columns_reference_declared_model_aliases() -> None:
    for path in _config_paths():
        cfg = _load_config(path)
        aliases = {model["alias"] for model in cfg.get("models") or []}
        assert aliases, f"{path.name}: at least one model alias must be declared"
        for col in _load_columns(path):
            if col["type"] in LLM_COLUMN_TYPES:
                alias = col.get("model_alias", "nvidia-text")
                assert alias in aliases, f"{path.name}: column {col['name']!r} references unknown model {alias!r}"


def test_structured_llm_columns_have_output_format() -> None:
    for path in _config_paths():
        for col in _load_columns(path):
            if col["type"] in {"llm_structured", "llm_judge"}:
                assert isinstance(col.get("output_format"), dict), (
                    f"{path.name}: column {col['name']!r} must declare output_format"
                )


def test_llm_text_columns_reference_existing_columns_in_prompts() -> None:
    """Light Jinja-reference check: ``{{ <name> }}`` must point at a column
    declared earlier in the same pipeline OR be supplied implicitly by the
    seed dataset (Designer auto-adds those columns at compile time).
    """
    placeholder = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
    for path in _config_paths():
        cfg = _load_config(path)
        cols = _load_columns(path)
        seen: set[str] = _seed_fields(cfg)
        for col in cols:
            prompt = col.get("prompt") or ""
            for ref in placeholder.findall(prompt):
                assert ref in seen, (
                    f"{path.name}: column {col['name']!r} prompt references "
                    f"{ref!r} which is not declared earlier and not provided "
                    f"by the seed dataset"
                )
            seen.add(col["name"])


def test_output_projection_references_declared_fields() -> None:
    for path in _config_paths():
        cfg = _load_config(path)
        projection = cfg.get("output_projection") or {}
        if not projection:
            continue
        declared_fields = _declared_fields(cfg)
        kind = projection.get("type")

        if kind == "openai_messages":
            fields = {
                projection.get("user_field", "user_query"),
                projection.get("assistant_field", "assistant_response"),
                *(projection.get("metadata_fields") or []),
            }
        elif kind == "structured_messages":
            fields = {
                projection.get("source_field", "conversation"),
                *(projection.get("metadata_fields") or []),
            }
        elif kind == "dpo_preference":
            fields = {
                projection.get("prompt_field", "prompt"),
                projection.get("response_a_field", "response_a"),
                projection.get("response_b_field", "response_b"),
                projection.get("judge_field", "judge"),
            }
        else:
            raise AssertionError(f"{path.name}: unknown or missing output_projection.type {kind!r}")

        missing = fields - declared_fields
        assert not missing, f"{path.name}: output_projection references undeclared fields {sorted(missing)}"


def test_openai_messages_projection() -> None:
    records = [
        {
            "persona": "teacher",
            "topic": "fractions",
            "user_query": "Can you explain fractions?",
            "assistant_response": "Fractions are parts of a whole.",
        }
    ]

    assert project_records(
        records,
        {
            "type": "openai_messages",
            "metadata_fields": ["persona", "topic"],
        },
    ) == [
        {
            "messages": [
                {"role": "user", "content": "Can you explain fractions?"},
                {"role": "assistant", "content": "Fractions are parts of a whole."},
            ],
            "persona": "teacher",
            "topic": "fractions",
        }
    ]


def test_records_from_dataset_creation_result() -> None:
    class Frame:
        def to_dict(self, orient: str) -> list[dict]:
            assert orient == "records"
            return [{"topic": "math"}]

    class Result:
        def load_dataset(self) -> Frame:
            return Frame()

    assert records_from_designer_result(Result()) == [{"topic": "math"}]


def test_records_from_preview_result() -> None:
    class Frame:
        def to_dict(self, orient: str) -> list[dict]:
            assert orient == "records"
            return [{"topic": "science"}]

    class Result:
        dataset = Frame()

    assert records_from_designer_result(Result()) == [{"topic": "science"}]


def test_records_from_hf_dataset_like_result() -> None:
    class Frame:
        def to_dict(self, orient: str) -> list[dict]:
            assert orient == "records"
            return [{"topic": "history"}]

    class Dataset:
        def to_pandas(self) -> Frame:
            return Frame()

    class Result:
        dataset = Dataset()

    assert records_from_designer_result(Result()) == [{"topic": "history"}]


def test_structured_messages_projection() -> None:
    records = [
        {
            "customer_name": "Priya",
            "issue": "late delivery",
            "conversation": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "description": "Look up an order.",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "messages": [
                    {"role": "system", "content": "You are a support agent."},
                    {"role": "user", "content": "Where is my order?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_lookup_001",
                                "type": "function",
                                "function": {
                                    "name": "lookup_order",
                                    "arguments": '{"order_id":"ORD-10492"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_lookup_001",
                        "name": "lookup_order",
                        "content": '{"status":"delayed"}',
                    },
                ],
            },
        }
    ]

    assert project_records(
        records,
        {
            "type": "structured_messages",
            "metadata_fields": ["customer_name", "issue"],
        },
    ) == [
        {
            "tools": records[0]["conversation"]["tools"],
            "messages": records[0]["conversation"]["messages"],
            "customer_name": "Priya",
            "issue": "late delivery",
        }
    ]


def test_dpo_preference_projection() -> None:
    records = [
        {
            "prompt": "Solve 2+2",
            "response_a": "4",
            "response_b": "5",
            "judge": {"winner": "A"},
        }
    ]

    assert project_records(records, {"type": "dpo_preference"}) == [
        {
            "prompt": "Solve 2+2",
            "chosen": "4",
            "rejected": "5",
        }
    ]
