from omegaconf import OmegaConf

from nemotron.kit.recipe_loader import extract_recipe_config


def test_extract_recipe_config_defaults_when_missing_recipe():
    cfg = OmegaConf.create({"x": 1})
    target, kwargs = extract_recipe_config(cfg, default_target="a.b.c")
    assert target == "a.b.c"
    assert kwargs == {}


def test_extract_recipe_config_reads_target_and_kwargs():
    cfg = OmegaConf.create(
        {
            "recipe": {
                "_target_": "m.n.func",
                "alpha": 1,
                "beta": "x",
            }
        }
    )
    target, kwargs = extract_recipe_config(cfg, default_target="a.b.c")
    assert target == "m.n.func"
    assert kwargs == {"alpha": 1, "beta": "x"}
