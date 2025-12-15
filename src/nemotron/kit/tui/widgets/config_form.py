from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Input, Static

from nemotron.kit.tui.form_utils import coerce_value, flatten_dataclass, set_dotted_attr


class ConfigForm(Container):
    """Auto-generated form for nested dataclass configs."""

    DEFAULT_CSS = """\
    ConfigForm { padding: 1 2; }
    .section { border: round $border; padding: 1; margin-bottom: 1; }
    .section-title { text-style: bold; }
    .field-label { color: $text-muted; }
    .field-error { border: round $error; }
    """

    def __init__(self, config_type: type) -> None:
        super().__init__()
        self._config_type = config_type
        self._base = config_type()
        if not is_dataclass(self._base):
            raise TypeError("ConfigForm requires a dataclass config type")

        flat = flatten_dataclass(self._base)
        # path -> (annotation, widget_id)
        self._fields: dict[str, Any] = {path: ann for path, (_val, ann) in flat.items()}

        self._filter: str = ""

    def compose(self) -> ComposeResult:
        flat = flatten_dataclass(self._base)

        # Group by first segment for readability.
        grouped: dict[str, list[tuple[str, Any, Any]]] = {}
        for path, (val, ann) in flat.items():
            section = path.split(".", 1)[0] if "." in path else "config"
            grouped.setdefault(section, []).append((path, val, ann))

        with VerticalScroll():
            yield Input(
                placeholder="Filter (e.g. lr, seq_length, max_steps)",
                id="cfg-filter",
            )

            for section, items in grouped.items():
                filtered = [
                    (path, val, ann)
                    for (path, val, ann) in items
                    if not self._filter or self._filter in path.lower()
                ]
                if not filtered:
                    continue

                with Container(classes="section"):
                    yield Static(section, classes="section-title")
                    with Vertical():
                        for path, val, _ann in sorted(filtered, key=lambda x: x[0]):
                            yield Static(path, classes="field-label")
                            yield Input(
                                value="" if val is None else str(val),
                                id=self._id_for(path),
                            )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "cfg-filter":
            return
        self._filter = event.value.strip().lower()
        self.refresh(layout=True)

    def build_config_instance(self) -> Any:
        cfg = self._config_type()
        flat = flatten_dataclass(cfg)

        errors: list[str] = []
        for path, (_cur, ann) in flat.items():
            wid = self._id_for(path)
            inp = self.query_one(f"#{wid}", Input)
            try:
                value = coerce_value(inp.value, ann)
                inp.remove_class("field-error")
                set_dotted_attr(cfg, path, value)
            except Exception as e:
                inp.add_class("field-error")
                errors.append(f"{path}: {e}")

        if errors:
            raise ValueError("Invalid config values:\n" + "\n".join(errors))

        return cfg

    @staticmethod
    def _id_for(path: str) -> str:
        return "cfg-" + path.replace(".", "-")
