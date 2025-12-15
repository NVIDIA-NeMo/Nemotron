from __future__ import annotations

from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.theme import BUILTIN_THEMES
from textual.widgets import Footer

from nemotron.kit.tui import StageCommandMeta
from nemotron.kit.tui.screens.stage_config import StageConfigScreen
from nemotron.kit.tui.widgets.banner import NeMoTronBanner


class NemotronTUIApp(App[None]):
    CSS_PATH = "app.tcss"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+r", "run_attached", "Run", show=False),
        Binding("ctrl+l", "launch_detached", "Launch", show=False),
        Binding("ctrl+t", "cycle_theme", "Theme", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("escape", "quit", "Quit", show=False),
    ]

    def __init__(self, stage: StageCommandMeta, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stage = stage
        self._screen: StageConfigScreen | None = None
        self._themes = sorted(k for k in BUILTIN_THEMES if k != "textual-ansi")
        self._theme_name = "textual-dark"

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            yield NeMoTronBanner(stage_name=self.stage.name)
            self._screen = StageConfigScreen(self.stage)
            yield self._screen
            yield Footer()

    def on_mount(self) -> None:
        if self._theme_name in self._themes:
            self.theme = self._theme_name

    def action_cycle_theme(self) -> None:
        if not self._themes:
            return
        try:
            idx = self._themes.index(self.theme)
        except ValueError:
            idx = 0
        self.theme = self._themes[(idx + 1) % len(self._themes)]

    async def action_run_attached(self) -> None:
        if self._screen is not None:
            await self._screen.run_stage(detached=False)

    async def action_launch_detached(self) -> None:
        if self._screen is not None:
            await self._screen.run_stage(detached=True)
