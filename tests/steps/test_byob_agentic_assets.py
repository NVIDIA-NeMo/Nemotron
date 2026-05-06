# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""BYOB-only checks for the agentic benchmark skill package."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
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
        "runtime/benchmark_families/base.py",
        "runtime/benchmark_families/registry.py",
        "runtime/benchmark_families/mcq/family.py",
        "runtime/benchmark_families/mcq/pipeline.py",
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

    assert not (BYOB_ROOT / "runtime" / "pipeline.py").exists()


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
    for raw_reference in reference.values():
        raw_paths = raw_reference if isinstance(raw_reference, list) else [raw_reference]
        for raw_path in raw_paths:
            assert (REPO_ROOT / raw_path).exists(), f"Missing BYOB reference path: {raw_path}"


def test_byob_runtime_dependencies_are_optional() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)

    base_dependencies = "\n".join(data["project"]["dependencies"])
    byob_dependencies = data["project"]["optional-dependencies"]["byob"]
    byob_text = "\n".join(byob_dependencies)

    for package_name in (
        "data-designer",
        "nemo-curator",
        "sentence-transformers",
        "sacrebleu",
        "bcp47",
        "cuml-cu12",
    ):
        assert package_name not in base_dependencies
        assert package_name in byob_text

    data_designer_requirements = [
        requirement for requirement in byob_dependencies if requirement.startswith("data-designer")
    ]
    assert data_designer_requirements == [
        "data-designer==0.5.5; python_version>='3.11' and python_version<'3.14'"
    ]

    curator_requirements = [requirement for requirement in byob_dependencies if requirement.startswith("nemo-curator")]
    assert curator_requirements == [
        "nemo-curator @ git+https://github.com/NVIDIA-NeMo/Curator.git@main ; "
        "python_version>='3.11' and python_version<'3.14'"
    ]


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


def test_byob_root_cli_lists_families() -> None:
    from typer.testing import CliRunner

    from nemotron.cli.bin.nemotron import app

    result = CliRunner().invoke(app, ["byob", "--list-families"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["mcq"]


def test_byob_use_case_notebook_uses_current_step_structure() -> None:
    example_dir = REPO_ROOT / "use-case-examples" / "build-your-own-benchmark"
    notebook_path = example_dir / "build_mcq_benchmark.ipynb"

    assert (example_dir / "README.md").exists()
    assert notebook_path.exists()

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook_text = "\n".join(
        "".join(cell.get("source", [])) for cell in data.get("cells", []) if isinstance(cell, dict)
    )

    assert "src/nemotron/steps/byob/config/default.yaml" in notebook_text
    assert "nemotron byob" in notebook_text
    assert "benchmark/byob" not in notebook_text
    assert "nemotron.steps.byob.config" not in notebook_text


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
        {"translated_text": "समूहित-क्वेरी अटेंशन क्या है?", "translation_time": 0.1},
        {"translated_text": "एक डिकोडर अटेंशन वैरिएंट", "translation_time": 0.1},
        {"translated_text": "एक टोकनाइज़र", "translation_time": 0.1},
        {"translated_text": "एक डेटासेट", "translation_time": 0.1},
    ]
    restored = restore_mcq_records(source_records, index, translated_rows, target_lang="hi-IN")

    assert restored[0]["answer"] == "A"
    assert restored[0]["question"] == "समूहित-क्वेरी अटेंशन क्या है?"
    assert restored[0]["options"]["A"] == "एक डिकोडर अटेंशन वैरिएंट"
    assert restored[0]["translation_metadata"]["target_lang"] == "hi-IN"
    assert restored[0]["translation_time"] == pytest.approx(0.4)


def test_byob_translate_config_uses_curator_without_mode_selector() -> None:
    from nemotron.steps.byob.runtime.config import ByobTranslationConfig

    config = ByobTranslationConfig.from_yaml(str(BYOB_ROOT / "config" / "translate.yaml"))

    assert "mode" not in config.translation_model_config
    assert config.translation_model_config["backend_type"] == "llm"
    assert "enable_faith_eval" not in config.translation_model_config.get("stage", {})


def test_byob_translate_config_rejects_faith_eval(tmp_path: Path) -> None:
    from nemotron.steps.byob.runtime.config import ByobTranslationConfig

    config_path = tmp_path / "translate.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "expt_name": "unit",
                "dataset_path": "benchmark.parquet",
                "output_dir": "outputs",
                "source_language": "en-US",
                "target_language": "hi-IN",
                "translation_model_config": {
                    "params": {},
                    "stage": {"enable_faith_eval": True},
                },
                "backtranslation_quality_metrics": [{"type": "chrf", "threshold": 50}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="FAITH"):
        ByobTranslationConfig.from_yaml(str(config_path))


def test_byob_translation_has_no_data_designer_fallback() -> None:
    translation_root = BYOB_ROOT / "runtime" / "translation"

    assert not (translation_root / "llm.py").exists()
    assert "data_designer" not in (translation_root / "translation.py").read_text(encoding="utf-8")
    assert "data_designer" not in (translation_root / "translate.py").read_text(encoding="utf-8")


def test_byob_agent_assets_document_current_curator_namespaces() -> None:
    customize_root = REPO_ROOT / "skills" / "nemotron-customize"
    context_path = (
        customize_root
        / "context"
        / "byob-benchmark-curator-translation.txt"
    )
    agent_text = "\n".join(
        [
            (BYOB_ROOT / "SKILL.md").read_text(encoding="utf-8"),
            (BYOB_ROOT / "step.toml").read_text(encoding="utf-8"),
            (BYOB_ROOT / "references" / "STEP.md").read_text(encoding="utf-8"),
            (BYOB_ROOT / "references" / "guide.md").read_text(encoding="utf-8"),
            (BYOB_ROOT / "references" / "quality-and-filtering.md").read_text(encoding="utf-8"),
            (BYOB_ROOT / "patterns" / "create-byob-mcq-from-domain-corpus.md").read_text(encoding="utf-8"),
            (customize_root / "SKILL.md").read_text(encoding="utf-8"),
            (customize_root / "context" / "index.toml").read_text(encoding="utf-8"),
            context_path.read_text(encoding="utf-8"),
        ]
    )

    assert "Curator semantic deduplication" in agent_text
    assert "nemo_curator.backends.ray_data" in agent_text
    assert "nemo_curator.backends.ray_actor_pool" in agent_text
    assert "nemo_curator.stages.deduplication.semantic" in agent_text
    assert "nemo_curator.stages.text.experimental.translation" in agent_text
    assert "TextQualityMetricStage" in agent_text
    assert "runtime/benchmark_families/mcq/pipeline.py" in agent_text
    assert "scripts/runtime.py" in agent_text


def test_byob_translation_uses_curator_experimental_namespace() -> None:
    translation_root = BYOB_ROOT / "runtime" / "translation"
    runtime_text = "\n".join(
        [
            (translation_root / "translate.py").read_text(encoding="utf-8"),
            (translation_root / "quality_metrics.py").read_text(encoding="utf-8"),
        ]
    )

    assert "nemo_curator.stages.text.experimental.translation" in runtime_text
    assert "nemo_curator.stages.text.translation" not in runtime_text


def test_byob_uses_current_curator_backend_namespace() -> None:
    deduplication_text = (BYOB_ROOT / "runtime" / "deduplication.py").read_text(encoding="utf-8")

    assert "nemo_curator.backends.ray_data" in deduplication_text
    assert "nemo_curator.backends.ray_actor_pool" in deduplication_text
    assert "EmbeddingCreatorStage" in deduplication_text
    assert "RayActorPoolExecutor" in deduplication_text
    assert "RayDataExecutor" in deduplication_text
    assert "nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow" in deduplication_text


def test_byob_translation_pipeline_uses_curator_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from nemotron.steps.byob.runtime.config import ByobTranslationConfig
    from nemotron.steps.byob.runtime.translation import translate
    from nemotron.steps.byob.runtime.translation.translation import TranslationPipeline

    stage_calls = []
    client_calls = []

    class FakeBatch:
        def __init__(self, task_id: str, dataset_name: str, data: pd.DataFrame):
            self.task_id = task_id
            self.dataset_name = dataset_name
            self.data = data

        def to_pandas(self) -> pd.DataFrame:
            return self.data

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            client_calls.append(kwargs)

    class FakeStage:
        name = "SegmentTranslationStage"

        def setup(self) -> None:
            pass

        def teardown(self) -> None:
            pass

        def process(self, batch: FakeBatch) -> FakeBatch:
            df = batch.to_pandas().copy()
            df["translation"] = df["text"].map(lambda text: f"{text} [translated]")
            return FakeBatch(batch.task_id, batch.dataset_name, df)

    class FakeTranslationStage:
        def __init__(self, **kwargs):
            stage_calls.append(kwargs)

        def decompose(self):
            return [FakeStage()]

    monkeypatch.setenv("NGC_API_KEY", "test-key")
    monkeypatch.setattr(
        translate,
        "_load_curator_symbols",
        lambda: translate._CuratorSymbols(
            async_openai_client=FakeOpenAIClient,
            document_batch=FakeBatch,
            generation_config=FakeGenerationConfig,
            translation_stage=FakeTranslationStage,
        ),
    )

    config = ByobTranslationConfig(
        expt_name="unit",
        dataset_path="unused.parquet",
        output_dir="unused",
        source_language="en-US",
        target_language="hi-IN",
        translation_model_config={
            "backend_type": "llm",
            "params": {
                "model": "openai/gpt-oss-120b",
                "provider": "nvidia",
                "inference_parameters": {"max_tokens": 128, "max_parallel_requests": 2, "temperature": 0.0},
            },
            "stage": {"output_mode": "both"},
            "segment_stage": {"health_check": False, "max_concurrent_requests": 2},
        },
        backtranslation_quality_metrics=[],
    )
    dataframe = pd.DataFrame(
        [
            {
                "translation_id": "tq#1",
                "question_id": "1",
                "text": "What is inflation?",
                "type": "question",
                "source_language_code": "en-US",
                "target_language_code": "hi-IN",
            },
            {
                "translation_id": "tc#1#0",
                "question_id": "1",
                "text": "A price-level increase",
                "type": "choice",
                "source_language_code": "en-US",
                "target_language_code": "hi-IN",
            },
        ]
    )

    translated = TranslationPipeline(config=config).translate(dataframe)

    assert translated["translation"].tolist() == [
        "What is inflation? [translated]",
        "A price-level increase [translated]",
    ]
    assert len(stage_calls) == 1
    assert stage_calls[0]["source_lang"] == "en"
    assert stage_calls[0]["target_lang"] == "hi"
    assert client_calls[0]["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert client_calls[0]["max_concurrent_requests"] == 2
