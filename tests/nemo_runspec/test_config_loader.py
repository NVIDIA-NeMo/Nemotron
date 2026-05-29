from __future__ import annotations

from nemo_runspec.config.loader import load_config


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
