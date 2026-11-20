from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from nemo_runspec.cli_context import GlobalContext
from nemo_runspec.config.loader import build_job_config, find_config_file, load_config


@pytest.mark.parametrize("filename", ["custom.json", "./custom.json", "custom.config"])
def test_find_existing_config_file_before_bundled_profile(tmp_path, monkeypatch, filename) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / filename
    config.write_text('{"model": {"name": "customer"}}')
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / f"{config.name}.yaml").write_text("model:\n  name: bundled\n")

    discovered = find_config_file(filename, profiles)

    assert discovered.resolve() == config
    assert load_config(discovered).model.name == "customer"


def test_find_bundled_profile_when_name_is_a_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "default").mkdir()
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    expected = profiles / "default.yaml"
    expected.write_text("model: bundled\n")

    assert find_config_file("default", profiles) == expected


def test_load_config_merges_simple_defaults_file(tmp_path) -> None:
    (tmp_path / "default.yaml").write_text(
        "model:\n  name: base\ntrain:\n  iters: 100\n",
        encoding="utf-8",
    )
    tiny = tmp_path / "tiny.yaml"
    tiny.write_text(
        'defaults: default.yaml\ntrain:\n  iters: 5\n',
        encoding="utf-8",
    )

    cfg = load_config(tiny)

    assert cfg.model.name == "base"
    assert cfg.train.iters == 5
    assert "defaults" not in cfg


def test_build_job_config_preserves_yaml_owned_image_and_resources() -> None:
    train_config = OmegaConf.create(
        {
            "run": {
                "env": {
                    "container_image": "config-image",
                    "nodes": 8,
                    "gpus_per_node": 8,
                    "env_vars": {"FROM_CONFIG": "1"},
                }
            }
        }
    )
    env_profile = OmegaConf.create(
        {
            "container_image": "profile-image",
            "nodes": 1,
            "gpus_per_node": 4,
            "partition": "site-partition",
            "env_vars": {"FROM_PROFILE": "1"},
        }
    )

    job_config = build_job_config(
        train_config,
        GlobalContext(config="default", batch="site"),
        "steps/example",
        "step.py",
        [],
        env_profile=env_profile,
    )

    env = job_config.run.env
    assert env.container_image == "config-image"
    assert env.nodes == 8
    assert env.gpus_per_node == 8
    assert env.partition == "site-partition"
    assert env.env_vars.FROM_CONFIG == "1"
    assert env.env_vars.FROM_PROFILE == "1"
