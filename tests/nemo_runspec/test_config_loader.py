from __future__ import annotations

from omegaconf import OmegaConf

from nemo_runspec.cli_context import GlobalContext
from nemo_runspec.config.loader import build_job_config, load_config


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
