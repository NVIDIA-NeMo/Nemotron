# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/synth/data_designer``.

Also validates the declarative column-spec shape that step.py translates into
the upstream ``DataDesignerConfigBuilder`` API. We don't import data_designer
itself (heavy runtime dep), only ensure the YAML keys are well-formed.
"""

from pathlib import Path

import yaml

from nemotron.steps.synth.data_designer.step import project_records

from .._step_helpers import assert_step_static, step_dir

VALID_COLUMN_TYPES = {"category", "seed", "llm_text", "llm_structured", "llm_judge"}

STEP = step_dir(__file__, "synth", "data_designer")


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


def test_columns_use_supported_types() -> None:
    for path in _config_paths():
        for col in _load_columns(path):
            assert col["type"] in VALID_COLUMN_TYPES, f"unknown column type {col['type']!r} in {path.name}"


def test_seed_columns_reference_seed_column() -> None:
    """Every ``type: seed`` column must specify ``seed_column``."""
    for path in _config_paths():
        for col in _load_columns(path):
            if col["type"] == "seed":
                assert "seed_column" in col, f"{path.name}: seed column {col['name']!r} missing 'seed_column'"


def test_llm_text_columns_reference_existing_columns_in_prompts() -> None:
    """Light Jinja-reference check: ``{{ <name> }}`` must point at a column
    declared earlier in the same pipeline OR be supplied implicitly by the
    seed dataset (Designer auto-adds those columns at compile time).
    """
    import re

    placeholder = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
    fallback_seed_fields = {
        "default.yaml": {"topic"},
        "rl_pref.yaml": {"prompt"},
    }
    for path in _config_paths():
        cfg = _load_config(path)
        cols = _load_columns(path)
        seed_fields = set((cfg.get("seed_dataset") or {}).get("fields") or fallback_seed_fields.get(path.name, set()))
        seen: set[str] = set(seed_fields)
        for col in cols:
            prompt = col.get("prompt") or ""
            for ref in placeholder.findall(prompt):
                assert ref in seen, (
                    f"{path.name}: column {col['name']!r} prompt references "
                    f"{ref!r} which is not declared earlier and not provided "
                    f"by the seed dataset"
                )
            seen.add(col["name"])


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
