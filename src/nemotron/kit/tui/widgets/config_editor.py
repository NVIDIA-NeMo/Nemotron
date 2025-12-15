from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input

from nemotron.kit.tui.models.config_model import ConfigModel
from nemotron.kit.tui.widgets.config_tree import ConfigTree
from nemotron.kit.tui.widgets.node_editor import NodeEditor


class ConfigEditor(Container):
    class Changed(Message):
        def __init__(self) -> None:
            super().__init__()

    BINDINGS = [
        Binding("ctrl+f", "focus_search", show=False),
    ]

    DEFAULT_CSS = """\
    ConfigEditor { height: 1fr; }
    #cfg-left { width: 40; }
    #cfg-right { width: 1fr; }
    #cfg-right NodeEditor { height: 1fr; }
    """

    def __init__(self, model: ConfigModel) -> None:
        super().__init__()
        self._model = model
        self._tree: ConfigTree | None = None
        self._node: NodeEditor | None = None

    @property
    def model(self) -> ConfigModel:
        return self._model

    @classmethod
    def from_dataclass(cls, config_type: type) -> ConfigEditor:
        return cls(ConfigModel.from_dataclass(config_type))

    @classmethod
    def from_yaml(cls, path: Path) -> ConfigEditor:
        return cls(ConfigModel.from_yaml_path(path))

    def load_yaml(self, path: Path) -> None:
        self._model.reset_from_yaml_path(path)
        self.refresh_from_model(select_root=True)

    def refresh_from_model(self, *, select_root: bool = False) -> None:
        if self._tree is not None:
            self._tree.refresh_tree(select_root=select_root)
        if self._node is not None:
            self._node.set_path(self._node.current_path)

    def action_focus_search(self) -> None:
        try:
            self.query_one("#cfg-tree-search", Input).focus()
        except Exception:
            return

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Container(id="cfg-left"):
                self._tree = ConfigTree(self._model)
                yield self._tree
            with Vertical(id="cfg-right"):
                self._node = NodeEditor(self._model)
                yield self._node

    def on_mount(self) -> None:
        if self._node is not None:
            self._node.set_path("")
        if self._tree is not None:
            self._tree.select_path("")

    def on_config_tree_selected(self, message: ConfigTree.Selected) -> None:
        if self._node is not None:
            self._node.set_path(message.path)

    def on_node_editor_changed(self, _message: NodeEditor.Changed) -> None:
        if self._tree is not None:
            self._tree.refresh_tree()
        self.post_message(self.Changed())

    def build_config_instance(self) -> Any:
        return self._model.build_object()

    def build_config_dictconfig(self):
        return self._model.as_dictconfig()
