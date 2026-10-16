from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nemotron.steps.byob.runtime.config import ByobConfig, ByobTranslationConfig


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_byob_mcq_config(tmp_path: Path) -> Path:
    config_path = (
        Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "mcq" / "config" / "tiny.yaml"
    )
    config_data = _load_yaml(config_path)
    input_dir = tmp_path / "input"
    (input_dir / "maths").mkdir(parents=True)
    (input_dir / "maths" / "tiny.txt").write_text("tiny source document\n", encoding="utf-8")
    config_data["input_dir"] = str(input_dir)
    config_data["output_dir"] = str(tmp_path / "output")
    temp_config = tmp_path / "tiny.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return temp_config


def test_checked_in_tiny_config_validates(tmp_path: Path) -> None:
    temp_config = _write_byob_mcq_config(tmp_path)

    config = ByobConfig.from_yaml(str(temp_config))

    assert config.do_coverage_check is False
    assert config.semantic_deduplication_config["enabled"] is False
    assert config.semantic_outlier_detection_config["enabled"] is False


def test_missing_hf_dataset_raises_value_error(tmp_path: Path) -> None:
    temp_config = _write_byob_mcq_config(tmp_path)
    config_data = _load_yaml(temp_config)
    config_data.pop("hf_dataset")
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Field `hf_dataset` is required"):
        ByobConfig.from_yaml(str(temp_config))


def test_invalid_easiness_threshold_raises_value_error(tmp_path: Path) -> None:
    temp_config = _write_byob_mcq_config(tmp_path)
    config_data = _load_yaml(temp_config)
    config_data["easiness_threshold"] = 1.0
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Field `easiness_threshold` must be between 0 and 1"):
        ByobConfig.from_yaml(str(temp_config))


def test_translation_faith_eval_raises_value_error(tmp_path: Path) -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "nemotron"
        / "steps"
        / "byob"
        / "mcq"
        / "config"
        / "translate.yaml"
    )
    config_data = _load_yaml(config_path)
    config_data["translation_model_config"]["stage"]["enable_faith_eval"] = True
    temp_config = tmp_path / "translate.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError, match="FAITH evaluation is not part of this flow"):
        ByobTranslationConfig.from_yaml(str(temp_config))
