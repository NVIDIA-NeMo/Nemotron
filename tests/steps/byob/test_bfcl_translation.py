"""Truth-preserving BFCL translation and localization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    translation_preserved_projection,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    project_published_benchmark,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl
from nemotron.steps.byob.runtime.benchmark_families.bfcl.translation import (
    BFCLTranslationError,
    translate_bfcl,
)

BYOB_ROOT = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"


def _source(tmp_path: Path) -> Path:
    value = yaml.safe_load((BYOB_ROOT / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    value["output_dir"] = str(tmp_path / "generation")
    path = tmp_path / "generate.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    benchmark = generate_bfcl(path)
    return benchmark.parent / "run_manifest.json"


def _translation_config(tmp_path: Path, manifest: Path) -> Path:
    value = {
        "family": "bfcl",
        "stage": "translate",
        "config_status": "resolved",
        "expt_name": "localized_vi",
        "source_run_manifest": str(manifest),
        "output_dir": str(tmp_path / "translations"),
        "source_language": "en",
        "target_language": "vi",
        "translate_tool_descriptions": True,
        "remove_low_quality": False,
        "translation_model_config": {
            "backend_type": "llm",
            "params": {
                "provider": "test",
                "model": "translator-v1",
                "canonical_id": "translator-v1",
                "source": "test-registry",
                "revision": "a" * 40,
            },
        },
        "backtranslation_quality_metrics": [{"type": "chrf", "threshold": 0}],
    }
    path = tmp_path / "translate.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


class _FakeTranslationPipeline:
    calls: list[tuple[str, str]] = []

    def __init__(self, config: Any):
        self.config = config

    def translate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        source = str(dataframe["source_language_code"].iloc[0])
        target = str(dataframe["target_language_code"].iloc[0])
        self.calls.append((source, target))
        output = dataframe.copy()
        prefix = "Bản dịch: " if target == "vi" else "Backtranslation: "
        output["translation"] = output["text"].map(lambda text: prefix + text)
        return output


def _fake_quality(
    dataframe: pd.DataFrame,
    _config: Any,
    **_kwargs: Any,
) -> pd.DataFrame:
    output = dataframe.copy()
    output["score_chrf"] = 100.0
    output["score_chrf_passed"] = True
    output["is_quality_metric_passed"] = True
    return output


def test_translation_config_paths_resolve_from_byob_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    byob_root = tmp_path / "byob"
    byob_root.mkdir()
    config = _translation_config(tmp_path, tmp_path / "unused-run-manifest.json")
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["source_run_manifest"] = "publications/run_manifest.json"
    value["output_dir"] = "translations"
    config.write_text(yaml.safe_dump(value), encoding="utf-8")

    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.setattr(translation, "BYOB_ROOT", byob_root)
    monkeypatch.chdir(unrelated_cwd)

    loaded = translation._load_config(config)

    assert loaded.source_manifest == byob_root / "publications" / "run_manifest.json"
    assert loaded.output_dir == byob_root / "translations" / "localized_vi"


def test_translation_preserves_truth_and_writes_content_addressed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    manifest = _source(tmp_path)
    source_document = manifest.read_bytes()
    source_benchmark = manifest.parent / "benchmark.parquet"
    source_hash = translation._hash_file(source_benchmark)
    config = _translation_config(tmp_path, manifest)
    _FakeTranslationPipeline.calls = []
    monkeypatch.setattr(translation, "TranslationPipeline", _FakeTranslationPipeline)
    monkeypatch.setattr(translation, "evaluate_text_quality_metrics", _fake_quality)

    localized_path = translate_bfcl(config)

    assert _FakeTranslationPipeline.calls == [("en", "vi"), ("vi", "en")]
    assert manifest.read_bytes() == source_document
    assert translation._hash_file(source_benchmark) == source_hash
    source = project_published_benchmark(source_benchmark, expected_content_hash=source_hash)
    localized = project_published_benchmark(
        localized_path,
        expected_content_hash=translation._hash_file(localized_path),
    )
    assert localized.task_ids == source.task_ids
    for original, translated in zip(source.rows, localized.rows, strict=True):
        assert translation_preserved_projection(translated) == (translation_preserved_projection(original))
        assert translated.metadata["language"] == "vi"
        assert any(
            message.content and message.content.startswith("Bản dịch: ")
            for message in translated.messages
            if message.role in {"system", "user"}
        )
        assert [tool["function"].get("description") for tool in translated.tools] != [
            tool["function"].get("description") for tool in original.tools
        ]

    translation_manifest = localized_path.parent / "translation_manifest.json"
    document = json.loads(translation_manifest.read_text(encoding="utf-8"))
    identity = document.pop("translation_id")
    assert identity == translation._hash_json(document)
    assert document["source_run_manifest_content_hash"] == translation._hash_file(manifest)
    assert document["benchmark"]["content_hash"] == translation._hash_file(localized_path)
    assert document["contamination"]["scope"] == "all_translated_rows"
    assert document["model"]["canonical_id"] == "translator-v1"
    assert document["protected_tokens"]["occurrences"] > 0


def test_translation_fails_closed_when_backend_loses_a_protected_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    manifest = _source(tmp_path)
    config = _translation_config(tmp_path, manifest)

    class DropsToken(_FakeTranslationPipeline):
        def translate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            output = super().translate(dataframe)
            output["translation"] = output["translation"].str.replace(
                r"__BFCL_PROTECTED_[0-9]{6}__",
                "",
                n=1,
                regex=True,
            )
            return output

    monkeypatch.setattr(translation, "TranslationPipeline", DropsToken)
    monkeypatch.setattr(translation, "evaluate_text_quality_metrics", _fake_quality)

    with pytest.raises(BFCLTranslationError, match="protected token"):
        translate_bfcl(config)

    assert not (tmp_path / "translations" / "localized_vi").exists()


def test_translation_rejects_identity_localization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    manifest = _source(tmp_path)
    config = _translation_config(tmp_path, manifest)

    class IdentityPipeline:
        def __init__(self, _config: Any):
            pass

        def translate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            output = dataframe.copy()
            output["translation"] = output["text"]
            return output

    monkeypatch.setattr(translation, "TranslationPipeline", IdentityPipeline)

    with pytest.raises(BFCLTranslationError, match="language-change gate"):
        translate_bfcl(config)


def test_translation_recomputes_quality_verdicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    manifest = _source(tmp_path)
    config = _translation_config(tmp_path, manifest)

    def inconsistent_quality(
        dataframe: pd.DataFrame,
        _config: Any,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        output = dataframe.copy()
        output["score_chrf"] = -1.0
        output["score_chrf_passed"] = True
        output["is_quality_metric_passed"] = True
        return output

    monkeypatch.setattr(translation, "TranslationPipeline", _FakeTranslationPipeline)
    monkeypatch.setattr(translation, "evaluate_text_quality_metrics", inconsistent_quality)

    with pytest.raises(BFCLTranslationError, match="inconsistent chrf pass/fail"):
        translate_bfcl(config)


def test_translation_normalizes_locale_and_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    manifest = _source(tmp_path)
    config = _translation_config(tmp_path, manifest)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["source_language"] = "en-US"
    document["target_language"] = "vi_VN"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    monkeypatch.setattr(translation, "TranslationPipeline", _FakeTranslationPipeline)
    monkeypatch.setattr(translation, "evaluate_text_quality_metrics", _fake_quality)

    localized_path = translate_bfcl(config)

    assert localized_path.name == "benchmark.vi-vn.parquet"
    localized = project_published_benchmark(
        localized_path,
        expected_content_hash=translation._hash_file(localized_path),
    )
    assert {row.metadata["language"] for row in localized.rows} == {"vi-VN"}


def test_translation_fails_closed_when_backend_injects_a_protected_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    manifest = _source(tmp_path)
    config = _translation_config(tmp_path, manifest)
    source_projection = project_published_benchmark(
        manifest.parent / "benchmark.parquet",
        expected_content_hash=translation._hash_file(manifest.parent / "benchmark.parquet"),
    )
    injected_token = str(source_projection.rows[0].tools[0]["function"]["name"])

    class InjectsToken(_FakeTranslationPipeline):
        def translate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            output = super().translate(dataframe)
            output["translation"] = output["translation"] + f" {injected_token}"
            return output

    monkeypatch.setattr(translation, "TranslationPipeline", InjectsToken)
    monkeypatch.setattr(translation, "evaluate_text_quality_metrics", _fake_quality)

    with pytest.raises(BFCLTranslationError, match="introduced or duplicated protected value"):
        translate_bfcl(config)


def test_translation_rejects_protected_token_injected_by_backtranslation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    manifest = _source(tmp_path)
    config = _translation_config(tmp_path, manifest)
    source_projection = project_published_benchmark(
        manifest.parent / "benchmark.parquet",
        expected_content_hash=translation._hash_file(manifest.parent / "benchmark.parquet"),
    )
    injected_token = str(source_projection.rows[0].tools[0]["function"]["name"])

    class InjectsDuringBacktranslation(_FakeTranslationPipeline):
        def translate(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            output = super().translate(dataframe)
            if str(dataframe["target_language_code"].iloc[0]) == "en":
                output["translation"] = output["translation"] + f" {injected_token}"
            return output

    monkeypatch.setattr(translation, "TranslationPipeline", InjectsDuringBacktranslation)
    monkeypatch.setattr(translation, "evaluate_text_quality_metrics", _fake_quality)

    with pytest.raises(
        BFCLTranslationError,
        match="backtranslation.*introduced or duplicated protected value",
    ):
        translate_bfcl(config)


def test_nested_schema_literals_are_protected() -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import translation

    tokens = translation._schema_tokens(
        {
            "$defs": {
                "request": {
                    "type": "object",
                    "properties": {
                        "enabled": {"const": True},
                        "fallback": {"default": None},
                    },
                }
            },
            "properties": {
                "mode": {"enum": ["strict"]},
            },
        }
    )

    assert {"request", "enabled", "true", "fallback", "null", "mode", "strict"} <= tokens


def test_translation_refuses_a_bare_benchmark_input(tmp_path: Path) -> None:
    manifest = _source(tmp_path)
    config = _translation_config(tmp_path, manifest)
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["dataset_path"] = str(manifest.parent / "benchmark.parquet")
    del value["source_run_manifest"]
    config.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(BFCLTranslationError, match="refuses dataset_path"):
        translate_bfcl(config)


def test_translation_rejects_string_boolean_configuration(tmp_path: Path) -> None:
    config = _translation_config(tmp_path, _source(tmp_path))
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["translate_tool_descriptions"] = "false"
    config.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(BFCLTranslationError, match="translate_tool_descriptions must be true or false"):
        translate_bfcl(config)
