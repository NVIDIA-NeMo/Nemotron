# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Directory selectors are usable without the producer in the base CLI environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig


def test_workflow_defaults_and_multimodal_environment_overrides(monkeypatch):
    monkeypatch.delenv("NEMOTRON_3_5_SDG_GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("NEMOTRON_3_5_SDG_JUDGE_MODEL", raising=False)
    multimodal = {"sdg_workflow": "multimodal", "source_directory": {"path": "documents"}}
    default = SDGConfig(**multimodal)
    assert default.sdg_generator_model == default.sdg_judge_model == "nvidia/nemotron-3.5-super-vl-120b-a12b"

    monkeypatch.setenv("NEMOTRON_3_5_SDG_GENERATOR_MODEL", "operator/generator")
    monkeypatch.setenv("NEMOTRON_3_5_SDG_JUDGE_MODEL", "operator/judge")
    configured = SDGConfig(**multimodal)
    assert configured.sdg_generator_model == "operator/generator"
    assert configured.sdg_judge_model == "operator/judge"
    text = SDGConfig()
    assert text.sdg_generator_model is text.sdg_judge_model is None
    assert text.qa_generation_model == text.quality_judge_model == "nvidia/nemotron-3-ultra-550b-a55b"

    explicit = SDGConfig(**multimodal, sdg_generator_model="custom/generator", sdg_judge_model="custom/judge")
    assert explicit.sdg_generator_model == "custom/generator"
    assert explicit.sdg_judge_model == "custom/judge"
    # Saved configurations retain their selected models when the environment changes.
    restored = SDGConfig.model_validate_json(default.model_dump_json())
    assert restored.sdg_generator_model == restored.sdg_judge_model == default.sdg_judge_model


def test_text_model_settings_do_not_select_multimodal_models(monkeypatch):
    monkeypatch.setenv("NEMOTRON_3_5_SDG_GENERATOR_MODEL", "operator/generator")
    monkeypatch.setenv("NEMOTRON_3_5_SDG_JUDGE_MODEL", "operator/judge")
    cfg = SDGConfig(
        sdg_workflow="multimodal",
        sources_file="sources.jsonl",
        qa_generation_model="custom/text-generator",
        quality_judge_model="custom/text-judge",
    )
    assert cfg.sdg_generator_model == "operator/generator"
    assert cfg.sdg_judge_model == "operator/judge"
    assert cfg.qa_generation_model == "custom/text-generator"
    assert cfg.quality_judge_model == "custom/text-judge"


@pytest.mark.parametrize("field", ["sdg_generator_model", "sdg_judge_model"])
def test_multimodal_model_fields_are_rejected_for_text_qa(field):
    with pytest.raises(ValidationError, match="require sdg_workflow='multimodal'"):
        SDGConfig(**{field: "custom/model"})


@pytest.mark.parametrize(
    "inputs",
    [{}, {"sources_file": "sources.jsonl", "source_directory": {"path": "documents"}}],
)
def test_multimodal_requires_exactly_one_source_selector(inputs):
    with pytest.raises(ValidationError, match="exactly one"):
        SDGConfig(sdg_workflow="multimodal", **inputs)


def test_directory_selector_requires_multimodal_workflow():
    with pytest.raises(ValidationError, match="require sdg_workflow='multimodal'"):
        SDGConfig(source_directory={"path": "documents"})


@pytest.mark.parametrize("source", [{}, {"path": ""}, {"path": "   "}])
def test_directory_selector_requires_an_explicit_nonempty_path(source):
    with pytest.raises(ValidationError):
        SDGConfig(sdg_workflow="multimodal", source_directory=source)


def test_directory_config_remains_serializable_without_importing_producer():
    code = """
import importlib.abc
import json
import sys

class RejectProducer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith('data_designer_retrieval_sdg'):
            raise AssertionError('The base CLI must not import the SDG producer')

sys.meta_path.insert(0, RejectProducer())
from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig
cfg = SDGConfig(sdg_workflow='multimodal',
                source_directory={'path': 'documents', 'max_tokens_per_chunk': 512})
print(cfg.model_dump_json())
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    cfg = SDGConfig.model_validate_json(result.stdout)
    assert cfg.source_directory.path == Path("documents")
    assert cfg.source_directory.model_dump()["max_tokens_per_chunk"] == 512
