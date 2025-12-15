from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.message import Message
from textual.widgets import Input, Tree

from nemotron.kit.tui.models.config_model import ConfigModel


def _scalar_preview(value: Any, *, max_len: int = 40) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return ""
    s = str(value)
    s = s.replace("\n", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


class ConfigTree(Container):
    class Selected(Message):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    DEFAULT_CSS = """\
    ConfigTree { height: 1fr; }
    #cfg-tree-search { height: 3; }
    #cfg-tree { height: 1fr; border: round $border; }
    """

    def __init__(self, model: ConfigModel) -> None:
        super().__init__()
        self._model = model
        self._search: str = ""
        self._node_by_path: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search", id="cfg-tree-search")
        yield Tree("config", id="cfg-tree")

    def on_mount(self) -> None:
        self.refresh_tree(select_root=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "cfg-tree-search":
            return
        self._search = event.value.strip().lower()
        self.refresh_tree()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:  # type: ignore[name-defined]
        path = event.node.data
        if isinstance(path, str):
            self.post_message(self.Selected(path))

    def refresh_tree(self, *, select_root: bool = False) -> None:
        tree = self.query_one("#cfg-tree", Tree)
        tree.clear()
        self._node_by_path = {"": tree.root}

        overrides = self._model.diff_prefixes()
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(self._model.cfg, resolve=True)

        def add(node: Any, value: Any, prefix: str) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    child_path = f"{prefix}.{k}" if prefix else str(k)
                    label = self._label(str(k), v, child_path, overrides)
                    child = node.add(label, data=child_path)
                    self._node_by_path[child_path] = child
                    add(child, v, child_path)
                return
            if isinstance(value, list):
                for i, v in enumerate(value):
                    child_path = f"{prefix}.{i}" if prefix else str(i)
                    label = self._label(f"[{i}]", v, child_path, overrides)
                    child = node.add(label, data=child_path)
                    self._node_by_path[child_path] = child
                    add(child, v, child_path)

        add(tree.root, data, "")
        tree.root.expand()
        if select_root:
            tree.select_node(tree.root)

    def select_path(self, path: str) -> None:
        tree = self.query_one("#cfg-tree", Tree)
        node = self._node_by_path.get(path)
        if node is None:
            return
        tree.select_node(node)

    def _label(self, key: str, value: Any, path: str, overrides: set[str]) -> Text:
        text = Text(key)

        if path in overrides:
            text.stylize("bold #76b900")

        preview = _scalar_preview(value)
        if preview:
            text.append(f": {preview}", style="dim")

        if self._search and self._search in path.lower():
            text.stylize("reverse")

        return text
