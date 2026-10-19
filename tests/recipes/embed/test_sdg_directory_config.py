# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Directory selectors are usable without the producer in the base CLI environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig


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
