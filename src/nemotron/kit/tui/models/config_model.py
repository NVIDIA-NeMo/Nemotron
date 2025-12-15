from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


def _parse_cli_value(text: str) -> Any:
    """Best-effort parse for editor inputs.

    Uses YAML scalar parsing to get bool/int/float/list/dict when possible.
    Falls back to raw string.
    """

    if text.strip() == "":
        return ""
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


class ConfigModel:
    """OmegaConf-backed config model for the TUI editor."""

    def __init__(self, *, base_cfg: Any, cfg: Any) -> None:
        self.base_cfg = base_cfg
        self.cfg = cfg

    @classmethod
    def from_dataclass(cls, config_type: type) -> ConfigModel:
        base_obj = config_type()
        if not is_dataclass(base_obj):
            raise TypeError("ConfigModel requires a dataclass config type")

        base_cfg = OmegaConf.structured(base_obj)
        cfg = OmegaConf.structured(config_type())
        return cls(base_cfg=base_cfg, cfg=cfg)

    @classmethod
    def from_omegaconf(cls, base_cfg: DictConfig, cfg: DictConfig | None = None) -> ConfigModel:
        if cfg is None:
            cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
        return cls(base_cfg=base_cfg, cfg=cfg)

    @classmethod
    def from_yaml_path(cls, path: Path) -> ConfigModel:
        base_cfg = OmegaConf.load(path)
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))
        return cls(base_cfg=base_cfg, cfg=cfg)

    def reset_from_yaml_path(self, path: Path) -> None:
        base_cfg = OmegaConf.load(path)
        self.base_cfg = base_cfg
        self.cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))

    def get(self, path: str) -> Any:
        if not path:
            return self.cfg
        return OmegaConf.select(self.cfg, path)

    def set_value(self, path: str, value_text: str) -> None:
        value = _parse_cli_value(value_text)
        OmegaConf.update(self.cfg, path, value, merge=False)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self.cfg, resolve=True)

    def replace_from_yaml(self, yaml_text: str) -> None:
        data = yaml.safe_load(yaml_text) or {}
        incoming = OmegaConf.create(data)
        merged = OmegaConf.merge(self.base_cfg, incoming)
        self.cfg = merged

    def build_object(self) -> Any:
        return OmegaConf.to_object(self.cfg)

    def as_dictconfig(self) -> DictConfig:
        return OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))

    def diff_paths(self) -> set[str]:
        base = OmegaConf.to_container(self.base_cfg, resolve=True)
        cur = OmegaConf.to_container(self.cfg, resolve=True)

        diffs: set[str] = set()
        sentinel = object()

        def walk(b: Any, c: Any, p: str) -> None:
            if isinstance(b, dict) and isinstance(c, dict):
                for k in sorted(set(b.keys()) | set(c.keys())):
                    bp = b.get(k, sentinel)
                    cp = c.get(k, sentinel)
                    path = f"{p}.{k}" if p else str(k)
                    if bp is sentinel or cp is sentinel:
                        diffs.add(path)
                        continue
                    walk(bp, cp, path)
                return

            if isinstance(b, list) and isinstance(c, list):
                n = max(len(b), len(c))
                for i in range(n):
                    bp = b[i] if i < len(b) else sentinel
                    cp = c[i] if i < len(c) else sentinel
                    path = f"{p}.{i}" if p else str(i)
                    if bp is sentinel or cp is sentinel:
                        diffs.add(path)
                        continue
                    walk(bp, cp, path)
                return

            if b != c and p:
                diffs.add(p)

        walk(base, cur, "")
        return diffs

    def diff_prefixes(self) -> set[str]:
        prefixes: set[str] = set()
        for path in self.diff_paths():
            parts = path.split(".")
            for i in range(1, len(parts) + 1):
                prefixes.add(".".join(parts[:i]))
        return prefixes
