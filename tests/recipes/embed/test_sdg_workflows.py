"""Recipe workflow selection and preservation of the public Text QA pipeline."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nemo_runspec.config import load_pydantic_config
from nemotron.recipes.embed.sdg_manifest import resolve_generation_input
from nemotron.recipes.embed.stage0_sdg.data_prep import STAGE_PATH, SDGConfig, run_sdg
from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig
from nemotron.recipes.embed.stage1_data_prep.plugin_adapter import execute_conversion


@pytest.mark.parametrize("selector", ["text_qa", "multimodal"])
def test_workflow_profiles_load(selector, tmp_path):
    profile = "nemotron-3.5" if selector == "multimodal" else "nemotron-3"
    overrides = []
    if selector == "multimodal":
        overrides = [
            f"sources_file={tmp_path / 'sources.jsonl'}",
            "sdg_generator_model=fixture/generator",
            "sdg_judge_model=fixture/judge",
        ]
    cfg = load_pydantic_config(STAGE_PATH / f"config/{profile}.yaml", overrides, SDGConfig)
    assert cfg.sdg_workflow == selector


class _GeneratedTextFixture:
    """Supply controlled model outputs after the real producer reads the corpus."""

    def __init__(self, frame, dataset_path, dataset_name):
        self.frame = frame
        self.artifact_storage = SimpleNamespace(
            resolved_dataset_name=dataset_name,
            base_dataset_path=dataset_path,
        )

    def export(self, path, *, format):
        assert format == "jsonl"
        self.frame.to_json(path, orient="records", lines=True, force_ascii=False)

    def count_records(self):
        return len(self.frame)


@pytest.mark.parametrize("profile", ["nemotron-3", "llama"])
def test_text_profiles_read_directories_and_convert_public_generation_output(tmp_path, monkeypatch, profile):
    """Keep real producer orchestration, chunking, manifests and conversion; replace inference only."""
    pytest.importorskip("data_designer_retrieval_sdg")
    from data_designer.engine.secret_resolver import PlaintextResolver
    from data_designer_retrieval_sdg.text_qa import generation
    from data_designer_retrieval_sdg.text_qa.seed_reader import DocumentChunkerSeedReader

    corpus = tmp_path / "corpus"
    (corpus / "nested").mkdir(parents=True)
    for index in range(10):
        parent = corpus if index < 5 else corpus / "nested"
        suffix = ".txt" if index < 5 else ".md"
        (parent / f"device-{index}{suffix}").write_text(
            f"Device {index} operates at {index + 20} bar. Its relief valve protects the supply line."
        )
    (corpus / "ignored.bin").write_text("This unsupported file must not become a training document.")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    seen = {}

    def create_fixture(self, builder, *, num_records, dataset_name, resume):
        reader = DocumentChunkerSeedReader()
        reader.attach(builder.get_seed_config().source, PlaintextResolver())
        frame = reader._get_output_dataframe()
        assert num_records == len(frame) == 10
        seen["source_ids"] = set(frame["source_id"])
        frame["deduplicated_qa_pairs"] = [
            [
                {
                    "question": f"What operating pressure applies to device {index}?",
                    "segment_ids": [row["chunks"][0]["chunk_id"]],
                }
            ]
            for index, row in frame.iterrows()
        ]
        frame["qa_evaluations"] = [{"evaluations": [{"overall": {"score": 9.0}}]}] * len(frame)
        return _GeneratedTextFixture(frame, tmp_path / "artifacts" / dataset_name, dataset_name)

    monkeypatch.setattr(generation.DataDesigner, "create", create_fixture)
    cfg = load_pydantic_config(
        STAGE_PATH / f"config/{profile}.yaml",
        [f"artifact_root={tmp_path / 'run'}", f"corpus_dir={corpus}", "dataset_name=text-regression"],
        SDGConfig,
    )
    assert cfg.sdg_workflow == "text_qa"
    generated = run_sdg(cfg)
    handoff = cfg.output_dir / "generation_result.json"
    assert resolve_generation_input(handoff) == generated.resolve()
    assert "nested/device-9.md" in seen["source_ids"]
    assert "ignored.bin" not in seen["source_ids"]
    assert all(row["chunks"] for row in map(json.loads, generated.read_text().splitlines()))
    assert json.loads(handoff.read_text()).get("portable_bundle_manifest") is None

    prep = load_pydantic_config(
        STAGE_PATH.parent / f"stage1_data_prep/config/{profile}.yaml",
        [f"artifact_root={tmp_path / 'run'}"],
        DataPrepConfig,
    )
    converted = execute_conversion(prep)
    assert converted.training_examples == 8
    assert converted.evaluation_queries == 2
    assert converted.train_file.is_file()
    assert (converted.evaluation_dir / "corpus.jsonl").is_file()
