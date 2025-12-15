from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static, TextArea

from nemotron.kit.tui.models.config_model import ConfigModel


class YamlEditor(Container):
    class Changed(Message):
        def __init__(self) -> None:
            super().__init__()

    DEFAULT_CSS = """\
    YamlEditor { height: 1fr; }
    #yaml-controls { height: 3; }
    #yaml-controls Button { width: 12; }
    #yaml-status { height: 1; color: $error; }
    #yaml-area { height: 1fr; border: round $border; }
    """

    def __init__(
        self,
        model: ConfigModel,
        *,
        start_editing: bool = False,
        show_line_numbers: bool = True,
    ) -> None:
        super().__init__()
        self._model = model
        self._editing: bool = start_editing
        self._show_line_numbers = show_line_numbers

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="yaml-controls"):
                yield Button("Preview" if self._editing else "Edit", id="yaml-toggle")
                yield Button("Apply", id="yaml-apply")
            yield Static("", id="yaml-status")
            yield TextArea(
                "",
                id="yaml-area",
                language="yaml",
                read_only=not self._editing,
                soft_wrap=False,
                show_line_numbers=self._show_line_numbers,
            )

    def on_mount(self) -> None:
        self.sync_from_model(force=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "yaml-toggle":
            self.toggle_edit()
        elif event.button.id == "yaml-apply":
            self.apply_yaml()

    def toggle_edit(self) -> None:
        self._editing = not self._editing
        area = self.query_one("#yaml-area", TextArea)
        area.read_only = not self._editing
        self.query_one("#yaml-toggle", Button).label = "Preview" if self._editing else "Edit"
        if not self._editing:
            self.sync_from_model(force=True)

    def sync_from_model(self, *, force: bool = False) -> None:
        if self._editing and not force:
            return
        area = self.query_one("#yaml-area", TextArea)
        area.load_text(self._model.to_yaml())

    def apply_yaml(self) -> None:
        area = self.query_one("#yaml-area", TextArea)
        try:
            self._model.replace_from_yaml(area.text)
        except Exception as e:
            self.query_one("#yaml-status", Static).update(str(e))
            return
        self.query_one("#yaml-status", Static).update("")
        self.post_message(self.Changed())
