# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real directory preparation and portable export with scripted model responses."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("data_designer_retrieval_sdg.multimodal")

import pyarrow.parquet as pq
from data_designer_retrieval_sdg.multimodal.models import (
    ContextSummary,
    Localization,
    QueryBatch,
    QueryMetadata,
    QuerySlot,
    RelevanceJudgment,
    SelfSufficiencyJudgment,
    Support,
)
from PIL import Image

from nemotron.recipes.embed.sdg_manifest import resolve_bundle_view, resolve_portable_training_input
from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig, run_sdg


class ScriptedDirectoryInference:
    """Replace only model generation; source preparation and all exports are real."""

    def __init__(self, root, config):
        pass

    def generate(self, requests):
        return [self._response(request) for request in requests]

    @staticmethod
    def _response(request):
        if request.schema is ContextSummary:
            return ContextSummary(summary="Maintenance evidence", visual_evidence="")
        if issubclass(request.schema, QueryBatch):
            source = json.loads(request.text.splitlines()[-1])["sources"][0]
            text = bool(source["text"])
            return QueryBatch(
                queries=[
                    QuerySlot(
                        slot=0,
                        query=(
                            f"What pressure is specified by {source['unit_id']}?"
                            if text
                            else f"Which color appears in {source['unit_id']}?"
                        ),
                        evidence_modality="text" if text else "image",
                    ),
                    QuerySlot(slot=1, query=None, evidence_modality="none"),
                    QuerySlot(slot=2, query=None, evidence_modality="none"),
                ]
            )
        if request.schema is QueryMetadata:
            return QueryMetadata(
                has_answer=False,
                actual_query_type="numerical",
                actual_query_format="question",
                classification_reasoning="Authored fixture judgment",
            )
        if request.schema is SelfSufficiencyJudgment:
            return SelfSufficiencyJudgment(self_sufficiency=5, reasoning="Authored fixture judgment")
        if request.schema is RelevanceJudgment:
            return RelevanceJudgment(relevance=5, reasoning="Authored fixture judgment")
        assert request.schema is Localization
        return Localization(
            supports=[
                Support(
                    unit_id=source["unit_id"],
                    grade=2,
                    modality="text" if source["text"] else "image",
                    quote=source["text"],
                    visual_evidence="" if source["text"] else "A white square",
                    contribution="Authored fixture evidence",
                )
                for source in json.loads(request.text.splitlines()[-1])["sources"]
            ]
        )


def _model_options(root: Path) -> dict:
    """Use an offline tokenizer for producer request-budget validation."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    encoder = Tokenizer(
        models.BPE(
            vocab={char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, merges=[]
        )
    )
    encoder.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer_file = root / "tokenizer.json"
    encoder.save(str(tokenizer_file))
    return {
        "tokenizer_file": str(tokenizer_file),
        "context_window_tokens": 2_000_000,
        "image_tokens_per_image": 100,
        "request_overhead_tokens": 32,
    }


def test_directory_export_handoff_views_resume_and_relocation(tmp_path, monkeypatch):
    directory = tmp_path / "documents"
    directory.mkdir()
    (directory / "manual.txt").write_text("Valve A opens at 20 bar. Valve B opens at 30 bar.", encoding="utf-8")
    (directory / "figure.md").write_text("Valve C opens at 40 bar.", encoding="utf-8")
    Image.new("RGB", (24, 24), "white").save(directory / "figure.png")
    options = _model_options(tmp_path)
    cfg = SDGConfig(
        corpus_id="directory-fixture",
        sdg_workflow="multimodal",
        source_directory={"path": directory, "sentences_per_chunk": 1, "max_tokens_per_chunk": None},
        export_train_ratio=1,
        output_dir=tmp_path / "stage0_sdg",
        sdg_options={"judge_summaries": False, "require_verbatim_quotes": True},
        qa_generation_model="operator/generator",
        quality_judge_model="operator/judge",
        nvidia_api_base_url="https://example.invalid/v1",
        sdg_credential_env="TEST_SDG_KEY",
        sdg_generator_options=options,
        sdg_judge_options=options,
    )
    monkeypatch.setenv("TEST_SDG_KEY", "placeholder")
    monkeypatch.setattr(
        "data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", ScriptedDirectoryInference
    )
    output = run_sdg(cfg)
    handoff = cfg.output_dir / "generation_result.json"
    bundle = output.parent
    assert len(output.read_text().splitlines()) == 4
    frozen_sources = cfg.output_dir / "multimodal/inputs/sources.jsonl"
    sources = [json.loads(line) for line in frozen_sources.read_text().splitlines()]
    assert len(sources) == 4
    assert {source["document_id"] for source in sources} == {"manual.txt", "figure.md", "figure.png"}
    assert not next(source for source in sources if source["document_id"] == "figure.png")["text"]
    assert not next(source for source in sources if source["document_id"] == "figure.md")["images"]
    assert (cfg.output_dir / "multimodal/inputs/source_manifest.json").is_file()
    assert (cfg.output_dir / "multimodal/inputs/sources.jsonl").is_file()

    for view, corpus_count in (("mixed", 4), ("text", 3), ("image_and_text", 1)):
        paths = resolve_bundle_view(bundle / "run_manifest.json", view=view)
        training = json.loads(resolve_portable_training_input(handoff, view).read_text())
        corpus = pq.read_table(sorted(paths.train_corpus.glob("*.parquet"))).to_pylist()
        assert len(corpus) == corpus_count
        assert len(training["data"]) == corpus_count
        assert all(row["query_group_id"] and row["neg_doc"] == [] for row in training["data"])
        assert all(positive["score"] == 2 for row in training["data"] for positive in row["pos_doc"])

    def unexpected_inference(*args, **kwargs):
        pytest.fail("A completed, verified resume must not create an inference client")

    monkeypatch.setattr("data_designer_retrieval_sdg.multimodal.workflow.DataDesignerInference", unexpected_inference)
    before = handoff.read_bytes()
    assert run_sdg(cfg.model_copy(update={"resume": "always"})) == output
    assert handoff.read_bytes() == before

    relocated = tmp_path / "relocated-stage0"
    shutil.copytree(cfg.output_dir, relocated)
    shutil.rmtree(directory)
    shutil.rmtree(cfg.output_dir)
    relocated_handoff = relocated / "generation_result.json"
    training_file = resolve_portable_training_input(relocated_handoff, "mixed")
    assert training_file.is_relative_to(relocated)
    paths = resolve_bundle_view(relocated / "multimodal/bundle/run_manifest.json", view="mixed")
    corpus = pq.read_table(sorted(paths.train_corpus.glob("*.parquet"))).to_pylist()
    assert len(corpus) == 4
    image_documents = [doc for doc in corpus if doc.get("image")]
    assert len(image_documents) == 1
    assert image_documents[0]["image"].startswith(b"\x89PNG")
