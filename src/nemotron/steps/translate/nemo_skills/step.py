#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/translate/nemo_skills"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Thin Speaker translation wrapper; full drivers: `speaker/src/speaker/driver/translate/`."""
from __future__ import annotations
import argparse, subprocess, sys, tempfile; from pathlib import Path; import yaml
DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); cfg = yaml.safe_load(parser.parse_args().config.read_text()); server = cfg["server"]
    translate_cfg = {"model_type": "llm", "server": {"type": "openai", "model": server["translation_model"], "address": server.get("address") or "http://nim-llm:8000/v1"}, "translation": {"source_lang": cfg["source_language"].lower(), "target_lang": cfg["target_language"].lower(), "fields_to_translate": cfg["translation"]["fields_to_translate"], "translation_key": cfg["translation"]["translation_key"], "prompt_config": cfg["translation"]["prompt_config"]}, "processing": cfg["processing"], "inference": cfg["inference"]}; faith_cfg = {"server": {"type": "openai", "model": server["faith_model"], "address": server.get("address") or "http://nim-llm:8000/v1"}, "faith_eval": {"source_lang": cfg["source_language"].lower(), "target_lang": cfg["target_language"].lower(), "fields_to_evaluate": cfg["faith_eval"]["fields_to_evaluate"], "translation_key": cfg["translation"]["translation_key"], "scores_key": cfg["faith_eval"]["scores_key"], "prompt_config": cfg["faith_eval"]["prompt_config"]}, "processing": {"num_chunks": cfg["processing"]["num_chunks"], "max_concurrent_requests": cfg["processing"]["max_concurrent_requests"]}, "inference": cfg["faith_inference"]}
    with tempfile.TemporaryDirectory() as tmpdir:
        translate_path, faith_path = Path(tmpdir) / "translate.yaml", Path(tmpdir) / "faith.yaml"; translate_path.write_text(yaml.safe_dump(translate_cfg, sort_keys=False)); faith_path.write_text(yaml.safe_dump(faith_cfg, sort_keys=False))
        subprocess.run([sys.executable, "-m", "speaker.driver.translate.pipeline_translate", "--config", str(translate_path), "--data-directory", cfg["data_directory"], "--input-file", cfg["input_file"]], check=True); subprocess.run([sys.executable, "-m", "speaker.driver.translate.pipeline_faith_eval", "--config", str(faith_path), "--data-directory", cfg["data_directory"]], check=True)
if __name__ == "__main__":
    main()
