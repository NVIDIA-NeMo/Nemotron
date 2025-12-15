from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Input, Static

from nemotron.kit.tui.models.config_model import ConfigModel


class NodeEditor(Container):
    class Changed(Message):
        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    DEFAULT_CSS = """\
    NodeEditor { height: 1fr; }
    #node-title { padding: 0 1; }
    #node-error { color: $error; padding: 0 1; height: 1; }

    #node-scalar { height: auto; }
    #node-scalar Input { width: 1fr; }

    #node-table { height: 1fr; }
    #node-table-table { height: 1fr; border: round $border; }
    #node-table-controls { height: 3; }
    #node-table-controls Input { width: 1fr; }

    .hidden { display: none; }
    """

    def __init__(self, model: ConfigModel) -> None:
        super().__init__()
        self._model = model
        self._path: str = ""
        self._selected_child: str | None = None
        self._mode: str = "dict"

    @property
    def current_path(self) -> str:
        return self._path

    def compose(self) -> ComposeResult:
        yield Static("", id="node-title")

        with Container(id="node-scalar", classes="hidden"):
            with Horizontal(id="node-scalar-controls"):
                yield Input(placeholder="Value", id="node-scalar-input")
                yield Button("Apply", id="node-scalar-apply")

        with Vertical(id="node-table"):
            yield DataTable(id="node-table-table")
            with Horizontal(id="node-table-controls"):
                yield Input(placeholder="Value", id="node-table-input")
                yield Button("Apply", id="node-table-apply")

        yield Static("", id="node-error")

    def set_path(self, path: str) -> None:
        self._path = path
        self._selected_child = None
        self._render_node()

    def _render_node(self) -> None:
        title = self.query_one("#node-title", Static)
        title.update(self._path or "config")
        self._set_error("")

        value = self._model.get(self._path)
        if isinstance(value, (dict, DictConfig, list, ListConfig)):
            self._mode = "list" if isinstance(value, (list, ListConfig)) else "dict"
            self._show_table(value)
        else:
            self._mode = "scalar"
            self._show_scalar(value)

    def _show_scalar(self, value: Any) -> None:
        self.query_one("#node-scalar", Container).remove_class("hidden")
        self.query_one("#node-table", Vertical).add_class("hidden")
        inp = self.query_one("#node-scalar-input", Input)
        inp.value = "" if value is None else str(value)

    def _show_table(self, value: Any) -> None:
        self.query_one("#node-scalar", Container).add_class("hidden")
        self.query_one("#node-table", Vertical).remove_class("hidden")
        table = self.query_one("#node-table-table", DataTable)
        table.clear(columns=True)
        table.add_columns(("key", "key"), ("value", "value"))
        table.cursor_type = "row"

        rows: list[tuple[str, str]] = []
        if isinstance(value, (dict, DictConfig)):
            for k, v in value.items():
                rows.append((str(k), "" if v is None else str(v)))
        else:
            for i, v in enumerate(value):
                rows.append((str(i), "" if v is None else str(v)))
        table.add_rows(rows)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:  # type: ignore[name-defined]
        if event.data_table.id != "node-table-table":
            return
        table = event.data_table
        row_key = event.row_key
        key_text = table.get_cell(row_key, "key")
        val_text = table.get_cell(row_key, "value")
        self._selected_child = "" if key_text is None else str(key_text)
        self.query_one("#node-table-input", Input).value = "" if val_text is None else str(val_text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "node-scalar-apply":
            self._apply_scalar()
        elif event.button.id == "node-table-apply":
            self._apply_child()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "node-scalar-input":
            self._apply_scalar()
        elif event.input.id == "node-table-input":
            self._apply_child()

    def _apply_scalar(self) -> None:
        inp = self.query_one("#node-scalar-input", Input)
        try:
            self._model.set_value(self._path, inp.value)
        except Exception as e:
            self._set_error(str(e))
            return
        self._set_error("")
        self.post_message(self.Changed(self._path))

    def _apply_child(self) -> None:
        if self._selected_child is None:
            self._set_error("Select a row to edit")
            return

        inp = self.query_one("#node-table-input", Input)
        child_path = f"{self._path}.{self._selected_child}" if self._path else self._selected_child
        try:
            self._model.set_value(child_path, inp.value)
        except Exception as e:
            self._set_error(str(e))
            return

        self._set_error("")
        self._render_node()
        self.post_message(self.Changed(child_path))

    def _set_error(self, text: str) -> None:
        self.query_one("#node-error", Static).update(text)
