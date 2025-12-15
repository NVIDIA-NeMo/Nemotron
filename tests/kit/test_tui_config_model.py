from __future__ import annotations

from dataclasses import dataclass, field

from nemotron.kit.tui.models.config_model import ConfigModel


@dataclass
class Inner:
    b: int = 1
    flag: bool = False


@dataclass
class Cfg:
    a: int = 1
    inner: Inner = field(default_factory=Inner)
    items: list[int] = field(default_factory=lambda: [1, 2])


def test_config_model_set_and_diff() -> None:
    model = ConfigModel.from_dataclass(Cfg)
    assert model.diff_paths() == set()

    model.set_value("a", "2")
    assert model.get("a") == 2
    assert "a" in model.diff_paths()

    model.set_value("inner.flag", "true")
    assert model.get("inner.flag") is True
    assert "inner.flag" in model.diff_paths()


def test_config_model_yaml_roundtrip() -> None:
    model = ConfigModel.from_dataclass(Cfg)
    model.replace_from_yaml("a: 5\ninner:\n  b: 7\nitems: [3, 4]\n")
    assert model.get("a") == 5
    assert model.get("inner.b") == 7
    assert model.get("items.0") == 3

    obj = model.build_object()
    assert isinstance(obj, Cfg)
    assert obj.a == 5
    assert obj.inner.b == 7
    assert obj.items == [3, 4]
