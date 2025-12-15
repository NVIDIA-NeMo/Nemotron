from __future__ import annotations

from textual.widgets import Static


class NeMoTronBanner(Static):
    def __init__(self, stage_name: str) -> None:
        super().__init__(self._render_text(stage_name))

    @staticmethod
    def _render_text(stage_name: str) -> str:
        stage = stage_name.upper()
        return (
            f"[bold]NeMoTron[/] · Nano3 · {stage}\n"
            "[dim]Ctrl+R Run  Ctrl+L Launch  Ctrl+T Theme  Q/Esc Quit[/]"
        )
