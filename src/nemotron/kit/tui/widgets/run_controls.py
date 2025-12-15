from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Select, Static

from nemotron.kit.run import list_run_profiles


class RunControls(Container):
    DEFAULT_CSS = """\
    RunControls Select { width: 30; }
    """

    class RunRequested(Message):
        def __init__(self, detached: bool) -> None:
            super().__init__()
            self.detached = detached

    def __init__(self, stage_name: str | None = None) -> None:
        super().__init__()
        self._stage_name = stage_name
        self._profiles: list[str] = []
        self._selected_profile: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static("", id="run-status")

            yield Select(
                options=[],
                prompt="Profile",
                id="run-profile",
                allow_blank=True,
            )
            yield Button("Run", id="btn-run")
            yield Button("Launch", id="btn-launch")

    def on_mount(self) -> None:
        self._profiles = list_run_profiles()

        select = self.query_one("#run-profile", Select)
        select.set_options([(p, p) for p in self._profiles])

        default = None
        if self._stage_name and self._stage_name in self._profiles:
            default = self._stage_name
        elif "local" in self._profiles:
            default = "local"
        elif self._profiles:
            default = self._profiles[0]

        if default:
            select.value = default
            self._selected_profile = default

    @property
    def selected_profile(self) -> str | None:
        return self._selected_profile

    def set_status(self, text: str) -> None:
        self.query_one("#run-status", Static).update(text)

    def on_select_changed(self, event: Select.Changed) -> None:  # type: ignore[name-defined]
        if event.select.id == "run-profile":
            self._selected_profile = None if event.value is Select.BLANK else str(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self.post_message(self.RunRequested(detached=False))
        elif event.button.id == "btn-launch":
            self.post_message(self.RunRequested(detached=True))
