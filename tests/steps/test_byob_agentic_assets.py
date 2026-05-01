# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""BYOB-only checks for the agentic benchmark skill package."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
STEPS_ROOT = REPO_ROOT / "src" / "nemotron" / "steps"
BYOB_ROOT = STEPS_ROOT / "byob"


def _load_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"Missing YAML frontmatter in {path}"
    data = yaml.safe_load(match.group(1)) or {}
    assert isinstance(data, dict), f"Frontmatter in {path} must be a mapping"
    return data


def test_byob_skill_assets_exist() -> None:
    expected = [
        "SKILL.md",
        "step.toml",
        "adapter.py",
        "step.py",
        "scripts/run.py",
        "scripts/runtime.py",
        "scripts/validate.py",
        "config/default.yaml",
        "config/tiny.yaml",
        "config/translate.yaml",
        "references/STEP.md",
        "references/guide.md",
        "references/benchmark-schema.md",
        "references/new-family-checklist.md",
        "references/quality-and-filtering.md",
        "patterns/index.yaml",
        "patterns/create-byob-mcq-from-domain-corpus.md",
        "patterns/translate-byob-mcq-benchmark.md",
        "patterns/add-new-benchmark-family.md",
        "eval/golden_cases.yaml",
        "eval/skill_cases.yaml",
    ]
    for rel_path in expected:
        assert (BYOB_ROOT / rel_path).exists(), f"Missing BYOB asset: {rel_path}"


def test_byob_skill_frontmatter_is_valid() -> None:
    frontmatter = _load_frontmatter(BYOB_ROOT / "SKILL.md")
    assert frontmatter["name"] == "byob"
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", frontmatter["name"])
    assert "benchmark" in frontmatter["description"].lower()
    assert "gsm8k" in frontmatter["description"].lower()
    assert frontmatter["when_to_use"]
    assert len(frontmatter["description"] + frontmatter["when_to_use"]) <= 1536


def test_byob_pattern_index_points_to_real_files() -> None:
    index_data = yaml.safe_load((BYOB_ROOT / "patterns" / "index.yaml").read_text(encoding="utf-8"))
    assert index_data["patterns"]
    for pattern in index_data["patterns"]:
        pattern_path = BYOB_ROOT / "patterns" / f"{pattern['id']}.md"
        assert pattern_path.exists(), f"Missing BYOB pattern file: {pattern_path}"


def test_byob_step_manifest_references_byob_files() -> None:
    manifest_path = BYOB_ROOT / "step.toml"
    with manifest_path.open("rb") as handle:
        data = tomllib.load(handle)

    assert data["step"]["id"] == "byob"
    assert data["step"]["category"] == "byob"

    reference = data["reference"]
    for raw_path in reference.values():
        assert (REPO_ROOT / raw_path).exists(), f"Missing BYOB reference path: {raw_path}"


def test_byob_imports_are_lightweight() -> None:
    from nemotron.steps.byob import (
        flatten_mcq_records,
        format_mcq_for_metrics,
        list_family_names,
        restore_mcq_records,
    )

    assert callable(flatten_mcq_records)
    assert callable(restore_mcq_records)
    assert callable(format_mcq_for_metrics)
    assert list_family_names() == ("mcq",)


def test_byob_adapter_round_trip() -> None:
    from nemotron.steps.byob import flatten_mcq_records, restore_mcq_records

    source_records = [
        {
            "question_id": "mcq-1",
            "question": "What is grouped-query attention?",
            "options": {"A": "A decoder attention variant", "B": "A tokenizer", "C": "A dataset"},
            "answer": "A",
        }
    ]

    staged_rows, index = flatten_mcq_records(source_records)
    assert staged_rows == [
        {"text": "What is grouped-query attention?"},
        {"text": "A decoder attention variant"},
        {"text": "A tokenizer"},
        {"text": "A dataset"},
    ]

    translated_rows = [
        {"translated_text": "समूहित-क्वेरी अटेंशन क्या है?", "faith_avg": 4.0},
        {"translated_text": "एक डिकोडर अटेंशन वैरिएंट", "faith_avg": 4.5},
        {"translated_text": "एक टोकनाइज़र", "faith_avg": 5.0},
        {"translated_text": "एक डेटासेट", "faith_avg": 4.5},
    ]
    restored = restore_mcq_records(source_records, index, translated_rows, target_lang="hi-IN")

    assert restored[0]["answer"] == "A"
    assert restored[0]["question"] == "समूहित-क्वेरी अटेंशन क्या है?"
    assert restored[0]["options"]["A"] == "एक डिकोडर अटेंशन वैरिएंट"
    assert restored[0]["translation_metadata"]["target_lang"] == "hi-IN"
    assert restored[0]["faith_avg"] == 4.5
