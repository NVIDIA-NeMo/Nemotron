from __future__ import annotations

from pathlib import Path

import yaml

from nemotron.steps.byob.runtime.config import ByobConfig


def test_checked_in_tiny_config_validates(tmp_path: Path) -> None:
    config_path = (
        Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob" / "mcq" / "config" / "tiny.yaml"
    )
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_dir = tmp_path / "input"
    (input_dir / "maths").mkdir(parents=True)
    (input_dir / "maths" / "tiny.txt").write_text("tiny source document\n", encoding="utf-8")
    config_data["input_dir"] = str(input_dir)
    config_data["output_dir"] = str(tmp_path / "output")
    temp_config = tmp_path / "tiny.yaml"
    temp_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = ByobConfig.from_yaml(str(temp_config))

    assert config.do_coverage_check is False
    assert config.semantic_deduplication_config["enabled"] is False
    assert config.semantic_outlier_detection_config["enabled"] is False
