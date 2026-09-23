# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused offline tests for the Persona MCQ step and reusable plugin."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tomllib
import yaml
from omegaconf import OmegaConf
from typer.testing import CliRunner

from nemo_runspec.cli_context import GlobalContext
from nemo_runspec.config import build_job_config, extract_train_config
from nemotron.cli.bin.nemotron import app
from nemotron.steps.sdg.persona_mcq.runtime.answers import (
    answer_prompt,
    parse_answer_letter,
    split_visible_reasoning,
)
from nemotron.steps.sdg.persona_mcq.runtime.io import write_jsonl
from nemotron.steps.sdg.persona_mcq.runtime.languages import script_fraction
from nemotron.steps.sdg.persona_mcq.runtime.lexical import deduplicate, strip_latin_gloss
from nemotron.steps.sdg.persona_mcq.runtime.pipeline import STAGES, PersonaMCQPipeline, validate_config
from nemotron.steps.sdg.persona_mcq.runtime.semantic import deduplicate_embeddings, greedy_keep_indices
from nemotron.steps.sdg.persona_mcq.runtime.sft import (
    build_sft_records,
    prepare_answer_seed,
    sample_aligned_datasets,
    vote,
)
from nemotron.steps.sdg.plugins.persona_mcq.parsing import parse_question

from .._step_helpers import assert_step_static, step_dir

STEP = step_dir(__file__, "sdg", "persona_mcq")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _tiny_config() -> dict:
    return yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))


def _conversation(question: str, choices: list[str]) -> dict:
    return {
        "conversation": json.dumps(
            {
                "messages": [],
                "metadata": {
                    "difficulty": "hard",
                    "facet": "geography",
                    "parsed_question": {"question": question, "choices": choices, "success": True},
                },
            }
        )
    }


def _answer(query_id: str, model: str, letter: str = "A") -> dict:
    return {
        "query_id": query_id,
        "question": "Which river is longest?",
        "choices": ["Ganga", "Yamuna", "Kaveri", "Godavari"],
        "answer_model": model,
        "answer": f"The evidence supports Ganga.\nAnswer: {letter}",
        "reasoning": "Compare the known river lengths.",
        "parsed_letter": letter,
        "finish_reason": "stop",
        "difficulty": "hard",
        "topic": "geography",
        "facet": "geography",
    }


def test_step_static() -> None:
    assert_step_static(
        STEP,
        expected_name="steps/sdg/persona_mcq",
        expected_launch="python",
        expected_default_config="default",
    )


def test_cli_lists_persona_mcq_in_sdg_catalog() -> None:
    result = CliRunner().invoke(app, ["steps", "list", "--category", "sdg", "--json"])

    assert result.exit_code == 0, result.output
    catalog = json.loads(result.output)
    step_ids = {step["id"] for step in catalog}
    assert "sdg/persona_mcq" in step_ids
    assert "sdg/qasynth" not in step_ids


def test_cli_show_resolves_persona_mcq() -> None:
    result = CliRunner().invoke(app, ["steps", "show", "sdg/persona_mcq"])

    assert result.exit_code == 0, result.output
    assert "sdg/persona_mcq" in result.output
    assert "default" in result.output


def test_cli_train_config_preserves_pipeline_controls() -> None:
    raw = OmegaConf.load(STEP / "config" / "tiny.yaml")
    context = GlobalContext(config="tiny")
    job = build_job_config(
        raw,
        context,
        "steps/sdg/persona_mcq",
        str(STEP / "step.py"),
        [],
    )
    train = extract_train_config(job)

    assert train.pipeline.experiment_name == "persona-mcq-smoke"
    assert list(train.pipeline.stages) == ["all"]


def test_persona_mcq_uses_shared_data_sdg_extra() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]

    assert "persona-mcq-sdg" not in extras
    assert any(requirement.startswith("data-designer>=0.5.9,<0.6") for requirement in extras["data-sdg"])
    assert any(requirement.startswith("sentence-transformers") for requirement in extras["data-sdg"])
    assert any(requirement.startswith("torch") for requirement in extras["data-sdg"])


def test_persona_mcq_entry_point_is_installed() -> None:
    pytest.importorskip("data_designer")
    matches = {point.name: point for point in entry_points(group="data_designer.plugins")}
    assert "persona-mcq" in matches
    assert "qasynth-mcq" not in matches
    plugin = matches["persona-mcq"].load()
    assert plugin.name == "persona-mcq"


def test_persona_mcq_registers_without_installed_entry_point() -> None:
    pytest.importorskip("data_designer")
    code = """
from data_designer.plugins import registry as registry_module

registry_module.PluginRegistry.reset()
registry_module.entry_points = lambda **kwargs: ()

from nemotron.steps.sdg.plugins.persona_mcq.plugin import ensure_registered

ensure_registered()

from data_designer.config.column_types import DataDesignerColumnType
from data_designer.engine.column_generators.registry import create_default_column_generator_registry

column_type = DataDesignerColumnType("persona-mcq")
registry = create_default_column_generator_registry()
assert registry.get_task_type(column_type).__name__ == "PersonaMCQGenerator"
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_parser_requires_exactly_four_distinct_options() -> None:
    valid = "<question>Which river?</question><options>A) Ganga\nB) Yamuna\nC) Kaveri\nD) Godavari</options>"
    assert parse_question(valid)["success"] is True
    assert parse_question(valid.replace("D) Godavari", "C) Godavari"))["success"] is False
    assert parse_question(valid.replace("D) Godavari", "D) Ganga"))["success"] is False


def test_question_prompt_uses_the_persona_geography() -> None:
    pytest.importorskip("data_designer")
    from nemotron.steps.sdg.plugins.persona_mcq.generator import PersonaMCQGenerator

    config = SimpleNamespace(
        facet_weights={"arts_persona": 1.0},
        difficulty_weights={"easy": 1.0},
        language="English",
        num_options=4,
    )
    prompt, metadata = PersonaMCQGenerator._build_prompt(
        {"state": "California", "city": "Oakland", "arts_persona": "West Coast jazz"},
        config,
        random.Random(42),
    )

    assert metadata["region"] == "California"
    assert 'region "State/Region: California; City: Oakland"' in prompt
    assert "India" not in prompt


def test_contextual_taxonomy_is_not_country_specific() -> None:
    from nemotron.steps.sdg.plugins.persona_mcq.taxonomy import CONTEXTUAL_FACETS

    taxonomy = json.dumps(CONTEXTUAL_FACETS, ensure_ascii=False)
    for country_specific_term in ("Kisan", "MUDRA", "PMFBY", "PMJJBY", "UPI", "Ayurveda"):
        assert country_specific_term not in taxonomy


@pytest.mark.parametrize("new_shape", [True, False])
def test_model_facade_response_compatibility(new_shape: bool) -> None:
    pytest.importorskip("data_designer")
    from nemotron.steps.sdg.plugins.persona_mcq.llm import completion_text

    message = SimpleNamespace(content="authored", role="assistant")
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice], raw={"choices": [choice]})
    if not new_shape:
        response.choices = []

    class Facade:
        def completion(self, messages, **kwargs):
            assert len(messages) == 2
            assert kwargs["allow_multiple_choices"] is False
            return response

    assert completion_text(Facade(), system_prompt="system", user_prompt="user") == "authored"


def test_lexical_filters_and_deduplicates() -> None:
    choices = ["गंगा", "यमुना", "कावेरी", "गोदावरी"]
    records = [
        _conversation("भारत की सबसे लंबी नदी कौन सी है?", choices),
        _conversation("भारत की सबसे लंबी नदी कौन सी है?", choices),
        _conversation("Which river is longest?", ["Ganga", "Yamuna", "Kaveri", "Godavari"]),
        _conversation("यह mostly English question with one Hindi token?", choices),
    ]
    output, stats = deduplicate(
        records,
        language="hindi",
        script_pattern="[ऀ-ॿ]",
        min_script_fraction=0.5,
        max_script_fraction=1.0,
        strip_latin_glosses=True,
        source_model="oss",
        permutations=16,
        bands=4,
    )
    assert len(output) == 1
    assert stats["drop_exact"] == 1
    assert stats["drop_wrong_language"] == 2
    assert strip_latin_gloss("गेंदा (Marigold)", "[ऀ-ॿ]", enabled=True) == "गेंदा"


def test_malayalam_language_filter_and_gloss_removal() -> None:
    records = [
        _conversation(
            "കേരളത്തിലെ ഏറ്റവും നീളമുള്ള നദി ഏതാണ്?",
            ["പെരിയാർ", "പമ്പ", "ചാലക്കുടി", "കബനി"],
        ),
        _conversation("Which river is longest?", ["Periyar", "Pamba", "Chalakudy", "Kabani"]),
    ]
    output, stats = deduplicate(
        records,
        language="malayalam",
        script_pattern="[ഀ-ൿ]",
        min_script_fraction=0.5,
        max_script_fraction=1.0,
        strip_latin_glosses=True,
        source_model="oss",
        permutations=16,
        bands=4,
    )

    assert len(output) == 1
    assert stats["drop_wrong_language"] == 1
    assert strip_latin_gloss("ചെണ്ട (Drum)", "[ഀ-ൿ]", enabled=True) == "ചെണ്ട"


def test_semantic_greedy_does_not_chain() -> None:
    # A~B and B~C, while A and C are not duplicates. Greedy with this seed keeps
    # two representatives rather than collapsing the transitive component.
    pairs = [(0, 1), (1, 2)]
    assert len(greedy_keep_indices(pairs, 3, seed=0)) == 2
    records = [{"question": str(index)} for index in range(3)]
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    output, stats = deduplicate_embeddings(
        records,
        embeddings,
        threshold=0.75,
        method="greedy",
        seed=0,
    )
    assert len(output) == 2
    assert stats["accepted"] == 2


def test_answer_seed_shuffle_is_stable() -> None:
    record = {
        "query_id": "a" * 64,
        "question": "Question",
        "choices": ["A", "B", "C", "D"],
        "language": "english",
        "metadata": {"source_model": "oss"},
    }
    prepared = prepare_answer_seed([record], 42)
    assert prepared == prepare_answer_seed([record], 99)
    assert prepared[0]["query_id"] == "7e455300a746e992c8b068df"
    assert prepared[0]["shuffle_permutation"] == [3, 1, 2, 0]
    assert sorted(prepared[0]["choices"]) == record["choices"]


def test_answer_parser_and_vote() -> None:
    assert parse_answer_letter("Reasoning\n**Answer: $C$**") == "C"
    assert parse_answer_letter("उत्तर: (B)", "उत्तर", ["उत्तर है"]) == "B"
    assert parse_answer_letter("उत्तर है: (B)", "उत्तर", ["उत्तर है"]) == "B"
    assert parse_answer_letter("ഉത്തരം: D", "ഉത്തരം") == "D"
    assert parse_answer_letter("Answer: A\nCorrection\nAnswer: C") == "C"
    assert vote(["A", "A", "A"], "unanimous") == "A"
    assert vote(["A", "A", "B"], "majority") == "A"
    assert vote(["A", "A", "A", "B"], "majority") == "A"
    assert vote(["A", "A", "B", "B"], "majority") is None
    assert vote(["A", "A", "A", "B", "C"], "majority") == "A"
    assert vote(["A", "A", "B", "C", "D"], "majority") is None
    assert vote(["A", "A", "B"], "unanimous") is None


def test_provider_english_reasoning_precedes_visible_target_language_rationale() -> None:
    rationale, answer = split_visible_reasoning(
        "കേരളത്തിലെ ഏറ്റവും നീളമുള്ള നദി പെരിയാറാണ്.\nഉത്തരം: A",
        "ഉത്തരം",
    )
    assert rationale == "കേരളത്തിലെ ഏറ്റവും നീളമുള്ള നദി പെരിയാറാണ്."
    assert answer == "ഉത്തരം: A"

    query_id = "e" * 64
    answers = {
        model: [
            {
                **_answer(query_id, model),
                "question": "കേരളത്തിലെ ഏറ്റവും നീളമുള്ള നദി ഏതാണ്?",
                "choices": ["പെരിയാർ", "പമ്പ", "ചാലക്കുടി", "കബനി"],
                "answer": "കേരളത്തിലെ ഏറ്റവും നീളമുള്ള നദി പെരിയാറാണ്.\nഉത്തരം: A",
                "reasoning": "The provider emitted private reasoning in English.",
            }
        ]
        for model in ("qwen", "oss", "gemma")
    }
    rows, stats = build_sft_records(
        answers,
        response_model="qwen",
        language="malayalam",
        language_config=_tiny_config()["languages"]["malayalam"],
        reasoning_config=_tiny_config()["sft"]["reasoning"],
        agreement="unanimous",
    )

    assert stats == {"kept": 1}
    assert rows[0]["messages"][-1] == {
        "role": "assistant",
        "reasoning_content": "The provider emitted private reasoning in English.",
        "content": "ഉത്തരം: A",
    }


def test_target_language_reasoning_is_rejected_when_english_is_required() -> None:
    query_id = "d" * 64
    answers = {
        model: [
            {
                **_answer(query_id, model),
                "answer": "കേരളത്തിലെ ഏറ്റവും നീളമുള്ള നദി പെരിയാറാണ്.\nഉത്തരം: A",
                "reasoning": "കേരളത്തിലെ ഏറ്റവും നീളമുള്ള നദി പെരിയാറാണ്.",
            }
        ]
        for model in ("qwen", "oss", "gemma")
    }

    rows, stats = build_sft_records(
        answers,
        response_model="qwen",
        language="malayalam",
        language_config=_tiny_config()["languages"]["malayalam"],
        reasoning_config=_tiny_config()["sft"]["reasoning"],
        agreement="unanimous",
    )

    assert rows == []
    assert stats == {"kept": 0, "reasoning_language_impurity": 1}


def test_sft_quality_gates_and_aligned_sampling() -> None:
    query_id = "f" * 64
    answers = {model: [_answer(query_id, model)] for model in ("qwen", "oss", "gemma")}
    rows, stats = build_sft_records(
        answers,
        response_model="oss",
        language="english",
        language_config=_tiny_config()["languages"]["english"],
        reasoning_config=_tiny_config()["sft"]["reasoning"],
        agreement="unanimous",
    )
    assert stats["kept"] == 1
    datasets = {teacher: {"english": rows} for teacher in answers}
    sampled, summary = sample_aligned_datasets(
        datasets,
        sample_per_language=1,
        seed=42,
        reasoning_off_fraction=1.0,
        answer_variant="stripped",
    )
    assert summary["selected_by_language"] == {"english": 1}
    assert summary["reasoning_off_by_language"] == {"english": 1}
    assert all("reasoning_content" not in data[0]["messages"][-1] for data in sampled.values())
    assert all(data[0]["messages"][-1]["content"] == "Answer: A" for data in sampled.values())


def test_reasoning_off_fraction_is_exact_per_language() -> None:
    teachers = ("qwen", "oss", "gemma")
    languages = ("english", "hindi", "malayalam")
    datasets = {
        teacher: {
            language: [
                {
                    "messages": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": f"{language} question {index}"},
                        {
                            "role": "assistant",
                            "reasoning_content": f"{language} reasoning {index}",
                            "content": "Answer: A",
                        },
                    ],
                    "metadata": {
                        "query_id": f"{language}-{index}",
                        "language": language,
                        "voted_letter": "A",
                    },
                }
                for index in range(10)
            ]
            for language in languages
        }
        for teacher in teachers
    }

    sampled, summary = sample_aligned_datasets(
        datasets,
        sample_per_language=10,
        seed=42,
        reasoning_off_fraction=0.1,
        answer_variant="full",
    )

    assert summary["reasoning_off_by_language"] == {language: 1 for language in languages}
    for rows in sampled.values():
        counts = {
            language: sum(
                row["metadata"]["language"] == language and row["metadata"]["reasoning_mode"] == "off" for row in rows
            )
            for language in languages
        }
        assert counts == {language: 1 for language in languages}


def test_malayalam_prompt_and_script_quality() -> None:
    config = _tiny_config()
    prompt = answer_prompt(
        {"question": "ചോദ്യം?", "choices": ["ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്"]},
        config["languages"]["malayalam"],
        config["sft"]["reasoning"],
    )

    assert "ഉത്തരം: $LETTER" in prompt
    assert "reasoning step by step in English" in prompt
    assert script_fraction("ഇത് മലയാളം ഉത്തരമാണ്", "[ഀ-ൿ]") == 1.0
    assert script_fraction("This is English", "[ഀ-ൿ]") == 0.0


def test_sample_writes_downstream_training_handoff(tmp_path) -> None:
    config = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    config["pipeline"].update(output_root=str(tmp_path), experiment_name="handoff")
    pipeline = PersonaMCQPipeline(config)
    query_ids = {"english": "a" * 64, "hindi": "b" * 64, "malayalam": "c" * 64}
    for teacher in config["sft"]["response_teachers"]:
        for language, query_id in query_ids.items():
            row = {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "reasoning_content": "reasoning", "content": "Answer: A"},
                ],
                "metadata": {"query_id": query_id, "language": language, "voted_letter": "A"},
            }
            write_jsonl(pipeline.root / "sft" / teacher / f"{language}.jsonl", [row])

    pipeline._sample()

    for teacher in config["sft"]["response_teachers"]:
        training_jsonl = pipeline.root / "training" / teacher / "train.jsonl"
        blend_path = pipeline.root / "training" / teacher / "blend.json"
        assert training_jsonl.is_file()
        assert len(training_jsonl.read_text(encoding="utf-8").splitlines()) == 3
        blend = json.loads(blend_path.read_text(encoding="utf-8"))
        assert blend == {
            "datasets": [
                {
                    "name": f"persona-mcq-{teacher}",
                    "path": str(training_jsonl.resolve()),
                    "weight": 1.0,
                }
            ]
        }
        for view_name, expected_languages in {
            "english_hindi": {"english", "hindi"},
            "english_malayalam": {"english", "malayalam"},
        }.items():
            view_root = pipeline.root / "training" / teacher / view_name
            view_records = [json.loads(line) for line in (view_root / "train.jsonl").read_text().splitlines()]
            assert len(view_records) == 2
            assert {record["metadata"]["language"] for record in view_records} == expected_languages
            view_blend = json.loads((view_root / "blend.json").read_text(encoding="utf-8"))
            assert view_blend["datasets"][0]["path"] == str((view_root / "train.jsonl").resolve())
        assert not (pipeline.root / "final" / f"{teacher}.jsonl").exists()


def test_shipped_configs_validate() -> None:
    for path in sorted((STEP / "config").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_config(config)
        assert set(config["languages"]) == {"english", "hindi", "malayalam"}


def test_personas_is_the_first_pipeline_stage() -> None:
    assert STAGES[0] == "personas"


def test_persona_stage_skips_cached_assets_without_ngc_key(monkeypatch, tmp_path) -> None:
    config = _tiny_config()
    config["pipeline"].update(output_root=str(tmp_path), experiment_name="cached-personas")
    assets = tmp_path / "managed-assets"
    datasets = assets / "datasets"
    datasets.mkdir(parents=True)
    for locale in ("en_IN", "hi_Deva_IN"):
        (datasets / f"{locale}.parquet").write_bytes(b"parquet")
    monkeypatch.setenv("DATA_DESIGNER_MANAGED_ASSETS_PATH", str(assets))
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    pipeline = PersonaMCQPipeline(config)
    pipeline._personas()

    assert pipeline.summary["stages"]["personas"]["cached_locales"] == ["en_IN", "hi_Deva_IN"]
    assert pipeline.summary["stages"]["personas"]["downloaded_locales"] == []


def test_persona_stage_requests_ngc_key_for_missing_assets(monkeypatch, tmp_path) -> None:
    config = _tiny_config()
    config["pipeline"].update(output_root=str(tmp_path), experiment_name="missing-personas")
    monkeypatch.setenv("DATA_DESIGNER_MANAGED_ASSETS_PATH", str(tmp_path / "managed-assets"))
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Export NGC_API_KEY"):
        PersonaMCQPipeline(config)._personas()


def test_persona_stage_downloads_only_missing_configured_locales(monkeypatch, tmp_path) -> None:
    pytest.importorskip("data_designer")
    from data_designer.cli.services.download_service import DownloadService

    config = _tiny_config()
    config["pipeline"].update(output_root=str(tmp_path), experiment_name="download-personas")
    assets = tmp_path / "managed-assets"
    datasets = assets / "datasets"
    datasets.mkdir(parents=True)
    (datasets / "en_IN.parquet").write_bytes(b"cached")
    observed: dict[str, object] = {}

    def fake_download(service, locale):
        observed["locale"] = locale
        observed["ngc_config"] = (Path(os.environ["HOME"]) / ".ngc" / "config").read_text(encoding="utf-8")
        (service.managed_assets_dir / f"{locale}.parquet").write_bytes(b"downloaded")

    monkeypatch.setenv("DATA_DESIGNER_HOME", str(tmp_path / "data-designer"))
    monkeypatch.setenv("DATA_DESIGNER_MANAGED_ASSETS_PATH", str(assets))
    monkeypatch.setenv("NGC_API_KEY", "secret-value")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/ngc" if name == "ngc" else None)
    monkeypatch.setattr(DownloadService, "download_persona_dataset", fake_download)

    original_home = os.environ.get("HOME")
    pipeline = PersonaMCQPipeline(config)
    pipeline._personas()

    assert observed == {
        "locale": "hi_Deva_IN",
        "ngc_config": "[CURRENT]\nformat_type = ascii\norg = no-org\nteam = no-team\nace = no-ace\n",
    }
    assert os.environ.get("HOME") == original_home
    assert pipeline.summary["stages"]["personas"]["downloaded_locales"] == ["hi_Deva_IN"]
    assert not (tmp_path / "data-designer" / ".ngc" / "config").exists()


def test_language_script_contract_is_config_driven() -> None:
    config = _tiny_config()
    config["languages"] = {
        "tamil_custom_key": {
            "display_name": "Tamil",
            "locale": "en_IN",
            "answer_label": "பதில்",
            "script_pattern": "[஀-௿]",
            "question_script_fraction": {"min": 0.5, "max": 1.0},
            "answer_script_fraction": {"min": 0.7, "max": 1.0},
            "strip_latin_glosses": True,
        }
    }

    validate_config(config)
    prompt = answer_prompt(
        {"question": "கேள்வி?", "choices": ["ஒன்று", "இரண்டு", "மூன்று", "நான்கு"]},
        config["languages"]["tamil_custom_key"],
        config["sft"]["reasoning"],
    )

    assert "written in Tamil" in prompt
    assert "reasoning step by step in English" in prompt
    assert "பதில்: $LETTER" in prompt
    assert parse_answer_letter("பதில்: C", "பதில்") == "C"
    assert script_fraction("தமிழ்", "[஀-௿]") == 1.0


@pytest.mark.parametrize(
    ("target", "profile"),
    [
        ("lepton", "lepton_sdg_persona_mcq"),
        ("slurm", "slurm_sdg_persona_mcq"),
        ("dgxcloud", "dgxcloud_sdg_persona_mcq"),
    ],
)
def test_environment_templates_wire_persona_mcq(target: str, profile: str) -> None:
    path = REPO_ROOT / "src" / "nemotron" / "steps" / "env" / "env_toml" / "config" / f"{target}.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    sections = config["sections"]

    assert profile in config["checks"]["required_profiles"]
    assert f"{profile}_tiny" in config["checks"]["required_profiles"]
    assert sections[profile]["gpus_per_node"] == 1
    assert sections[f"{profile}_tiny"]["gpus_per_node"] == 0
    assert "data-designer>=0.5.9,<0.6" in sections[profile]["startup_commands"][0]
    assert "sentence-transformers>=5.0.0,<6.0.0" in sections[profile]["startup_commands"][0]
    assert "torchvision>=0.25,<0.26" in sections[profile]["startup_commands"][1]
    if target == "lepton":
        assert any("ngccli_linux.zip" in command for command in sections[profile]["startup_commands"])
    assert set(sections[profile]["env_vars"]) >= {
        "NEMOTRON_RUN_DIR",
        "DATA_DESIGNER_HOME",
        "DATA_DESIGNER_MANAGED_ASSETS_PATH",
        "QWEN_API_BASE",
        "OSS_API_BASE",
        "GEMMA_API_BASE",
    }
    # Credentials must reach the worker as platform secret references, never as
    # env_vars, whose values are stored verbatim in the submitted job spec. Only
    # Lepton offers secrets; the other backends still pass them through env_vars.
    credentials = {"NVIDIA_API_KEY", "NGC_API_KEY"}
    carrier = "secret_vars" if target == "lepton" else "env_vars"
    assert credentials <= set(sections[profile][carrier])
    if target == "lepton":
        assert not (credentials & set(sections[profile]["env_vars"]))


def test_cli_stage_list_string_is_normalized(tmp_path) -> None:
    config = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    config["pipeline"].update(output_root=str(tmp_path), stages="[answers,build_sft,sample]")
    pipeline = PersonaMCQPipeline(config)
    assert pipeline.config["pipeline"]["stages"] == ["answers", "build_sft", "sample"]


def test_stage_selection_does_not_change_experiment_identity(tmp_path) -> None:
    base = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    base["pipeline"].update(output_root=str(tmp_path), experiment_name="resume-test", stages=["questions"])
    PersonaMCQPipeline(base)._prepare_experiment()

    resumed = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    resumed["pipeline"].update(
        output_root=str(tmp_path),
        experiment_name="resume-test",
        stages="[answers,build_sft]",
        resume=False,
    )
    PersonaMCQPipeline(resumed)._prepare_experiment()

    resumed["question_generation"]["num_records"] += 1
    with pytest.raises(ValueError, match="different config"):
        PersonaMCQPipeline(resumed)._prepare_experiment()


def test_resume_preserves_existing_stage_summary(tmp_path) -> None:
    config = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    config["pipeline"].update(output_root=str(tmp_path), experiment_name="resume-summary")
    first = PersonaMCQPipeline(config)
    first._prepare_experiment()
    first.summary["stages"]["questions"] = {"oss/english": 5}
    first._write_summary()

    resumed_config = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    resumed_config["pipeline"].update(
        output_root=str(tmp_path),
        experiment_name="resume-summary",
        stages=["answers"],
        resume=True,
    )
    resumed = PersonaMCQPipeline(resumed_config)
    resumed._prepare_experiment()

    assert resumed.summary["stages"]["questions"] == {"oss/english": 5}


def test_explicit_overwrite_removes_stale_artifacts(tmp_path) -> None:
    config = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    config["pipeline"].update(output_root=str(tmp_path), experiment_name="overwrite-clean")
    first = PersonaMCQPipeline(config)
    first._prepare_experiment()
    stale = first.root / "training" / "removed-teacher" / "train.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")
    first.summary["stages"]["sample"] = {"stale": True}
    first._write_summary()

    overwrite_config = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    overwrite_config["pipeline"].update(
        output_root=str(tmp_path),
        experiment_name="overwrite-clean",
        overwrite=True,
    )
    overwritten = PersonaMCQPipeline(overwrite_config)
    overwritten._prepare_experiment()

    assert not stale.exists()
    assert overwritten.summary == {"stages": {}}
    assert (overwritten.root / "run.json").is_file()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("sft", "agreement"), "plurality", "sft.agreement"),
        (("sampling", "per_language"), 0, "sampling.per_language"),
        (("sampling", "reasoning_off_fraction"), 1.1, "sampling.reasoning_off_fraction"),
        (("sampling", "answer_variant"), "verbose", "sampling.answer_variant"),
    ],
)
def test_config_rejects_invalid_voting_and_sampling(path, value, message) -> None:
    config = yaml.safe_load((STEP / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    config[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        validate_config(config)
