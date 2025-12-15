from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Select, Static

from nemotron.kit.artifact import ArtifactInput
from nemotron.kit.run import load_wandb_config


@dataclass
class ArtifactSlotState:
    slot: str
    source: Literal["wandb", "hf", "manual"] = "wandb"
    value: str = ""
    wb_selected: str = "latest"
    wb_latest_version: str | None = None
    wb_type: str | None = None


class ArtifactPicker(VerticalScroll):
    """Collects artifact references for App-style artifact slots.

    This produces a mapping suitable for nemotron.kit.app._apply_artifact_refs_to_config.
    """

    DEFAULT_CSS = """\
    ArtifactPicker { padding: 1 2; }
    .wbctx {
        border: round #76b900;
        background: $background;
        padding: 0;
        margin-bottom: 1;
    }
    .wbctx CollapsibleTitle { padding: 0 1; }
    .wbctx Contents { padding: 0 1; }
    .wbctx.-collapsed { padding-bottom: 0; padding-left: 0; }
    .slot { border: round #76b900; padding: 1; margin-bottom: 1; height: 1fr; }
    .slot-title { color: $text; text-style: bold; }
    .slot-help { color: $text-muted; }
    .slot-status { color: $text-muted; }
    .src-select { width: 18; height: 3; }
    .wbver-select { width: 1fr; }
    .ref-input { width: 1fr; height: 3; }

    .wbctx-entity { width: 22; height: 3; }
    .wbctx-project { width: 28; height: 3; }

    .refresh-btn {
        width: 10;
        height: 3;
        padding: 0 1;
        background: $surface;
        color: $primary;
        border: round $primary;
    }
    """

    def __init__(self, artifacts: dict[str, ArtifactInput]) -> None:
        super().__init__()
        self._artifacts = artifacts
        self._state: dict[str, ArtifactSlotState] = {
            slot: ArtifactSlotState(slot=slot, value="") for slot in artifacts
        }
        self._wb_entity, self._wb_project = _load_default_wandb_context()

    def compose(self) -> ComposeResult:
        if not self._artifacts:
            yield Static("No artifact inputs for this stage.", classes="placeholder")
            return

        with Collapsible(
            title=self._wbctx_title(),
            collapsed=True,
            classes="wbctx",
            id="wbctx",
        ):
            with Horizontal():
                yield Input(
                    value=self._wb_entity or "",
                    placeholder="entity",
                    id="wb-entity",
                    classes="wbctx-entity",
                )
                yield Input(
                    value=self._wb_project or "",
                    placeholder="project",
                    id="wb-project",
                    classes="wbctx-project",
                )
                yield Button("Refresh", id="wbrefresh-all", classes="refresh-btn")

        for slot, artifact in self._artifacts.items():
            with VerticalScroll(classes="slot", id=f"slot-{slot}"):
                yield Static(f"{slot}", classes="slot-title")
                yield Static(
                    (
                        f"Default: {artifact.default_name} · "
                        "Provide version (v10/latest) or full ref."
                    ),
                    classes="slot-help",
                )
                with Horizontal():
                    yield Select(
                        options=[("W&B", "wandb"), ("HF Hub", "hf"), ("Manual", "manual")],
                        value="wandb",
                        id=f"src-{slot}",
                        allow_blank=False,
                        classes="src-select",
                    )
                    yield Select(
                        options=[("latest", "latest")],
                        value="latest",
                        id=f"wbver-{slot}",
                        allow_blank=False,
                        prompt="Version",
                        classes="wbver-select",
                    )
                    yield Button(
                        "Refresh",
                        id=f"wbrefresh-{slot}",
                        classes="refresh-btn",
                    )
                    yield Input(
                        placeholder="v10 | latest | entity/project/name:v10 | art://...",
                        id=f"val-{slot}",
                        classes="ref-input",
                    )
                yield Static("", id=f"wbstatus-{slot}", classes="slot-status")

    def on_mount(self) -> None:
        self._update_wbctx_title()
        for slot in self._artifacts.keys():
            self._apply_slot_visibility(slot)
            # Kick off best-effort background refresh so users immediately see versions.
            self._refresh_wandb_versions(slot)

    def get_artifact_refs(self) -> dict[str, str]:
        refs: dict[str, str] = {}
        for slot in self._artifacts:
            state = self._state[slot]
            if state.source == "wandb":
                val = state.wb_selected.strip()
            else:
                val = state.value.strip()
            if val:
                refs[slot] = val
        return refs

    def on_select_changed(self, event: Select.Changed) -> None:  # type: ignore[name-defined]
        wid = event.select.id or ""
        if not wid.startswith("src-"):
            if wid.startswith("wbver-"):
                slot = wid.removeprefix("wbver-")
                if slot in self._state and event.value is not Select.BLANK:
                    self._state[slot].wb_selected = str(event.value)
                    self._update_slot_links(slot)
            return

        slot = wid.removeprefix("src-")
        if slot in self._state and event.value is not Select.BLANK:
            self._state[slot].source = str(event.value)  # type: ignore[assignment]
            self._set_wandb_controls_enabled(slot, enabled=self._state[slot].source == "wandb")
            self._apply_slot_visibility(slot)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        wid = event.button.id or ""
        if wid == "wbrefresh-all":
            for slot in self._artifacts.keys():
                self._refresh_wandb_versions(slot)
            return
        if not wid.startswith("wbrefresh-"):
            return
        slot = wid.removeprefix("wbrefresh-")
        if slot in self._state:
            self._refresh_wandb_versions(slot)

    def _on_wandb_context_changed(self) -> None:
        # Update header quickly; full refresh happens on explicit refresh.
        self._update_wbctx_title()

    def _set_wandb_controls_enabled(self, slot: str, enabled: bool) -> None:
        self.query_one(f"#wbrefresh-{slot}", Button).disabled = not enabled
        self.query_one(f"#wbver-{slot}", Select).disabled = not enabled

    def _apply_slot_visibility(self, slot: str) -> None:
        state = self._state[slot]
        is_wandb = state.source == "wandb"

        self.query_one(f"#wbrefresh-{slot}", Button).display = is_wandb
        self.query_one(f"#wbver-{slot}", Select).display = is_wandb

        inp = self.query_one(f"#val-{slot}", Input)
        inp.display = not is_wandb
        if state.source == "hf":
            inp.placeholder = "org/name[@rev]"
        elif state.source == "manual":
            inp.placeholder = "art://... | /path | entity/project/name:v10"

    def _set_slot_status(self, slot: str, text: str) -> None:
        self.query_one(f"#wbstatus-{slot}", Static).update(text)

    def _update_slot_links(self, slot: str) -> None:
        state = self._state.get(slot)
        artifact = self._artifacts.get(slot)
        if state is None or artifact is None:
            return

        if state.source != "wandb":
            self._set_slot_status(slot, "")
            return

        project_path = self._get_wandb_project_path()
        if not project_path:
            return

        # Prefer resolved type/version from API.
        type_name = state.wb_type
        if not type_name:
            return

        version = state.wb_selected
        if version == "latest" and state.wb_latest_version:
            version = state.wb_latest_version

        meta_url, lineage_url = _wandb_artifact_urls(
            project_path=project_path,
            type_name=type_name,
            artifact_name=artifact.default_name,
            version=version,
        )

        self._set_slot_status(
            slot,
            (
                f"[dim]W&B:[/dim] [link=\"{meta_url}\"]metadata[/link] · "
                f"[link=\"{lineage_url}\"]lineage[/link]"
            ),
        )

    @work(thread=True)
    def _refresh_wandb_versions(self, slot: str) -> None:
        if self.app is None:
            return

        # Only meaningful for wandb source; still safe to refresh for all slots.
        project_path = self._get_wandb_project_path()
        if project_path is None:
            self.app.call_from_thread(
                self._set_slot_status,
                slot,
                "W&B project/entity not configured (run.toml [wandb]).",
            )
            self.app.call_from_thread(self._update_wbctx_title)
            return

        artifact = self._artifacts.get(slot)
        if artifact is None:
            return

        state = self._state.get(slot)
        if state is None:
            return

        self.app.call_from_thread(self._set_slot_status, slot, "Loading W&B versions...")

        try:
            import wandb

            api = wandb.Api()
            latest = api.artifact(f"{project_path}/{artifact.default_name}:latest")
            type_name = latest.type
            latest_version = latest.version

            # Fetch newest versions (latest page first).
            versions = list(
                api.artifacts(
                    type_name=type_name,
                    name=f"{project_path}/{artifact.default_name}",
                    per_page=25,
                )
            )
            options = _build_wandb_version_options(latest=latest, versions=versions)
        except Exception as e:
            self.app.call_from_thread(self._set_slot_status, slot, f"W&B error: {e}")
            return

        def apply() -> None:
            state.wb_type = type_name
            state.wb_latest_version = latest_version
            sel = self.query_one(f"#wbver-{slot}", Select)
            sel.set_options(options)
            self._update_wbctx_title()
            self._update_slot_links(slot)

        self.app.call_from_thread(apply)

    def _wbctx_title(self) -> str:
        project_path = self._get_wandb_project_path()
        return "W&B context" if not project_path else f"W&B context: {project_path}"

    def _update_wbctx_title(self) -> None:
        try:
            wbctx = self.query_one("#wbctx", Collapsible)
        except Exception:
            return
        wbctx.title = self._wbctx_title()

    def _get_wandb_project_path(self) -> str | None:
        entity = (self._wb_entity or "").strip()
        project = (self._wb_project or "").strip()
        if entity and project:
            return f"{entity}/{project}"

        if project and not entity:
            # Try to infer entity from wandb client config.
            try:
                import wandb

                default_entity = getattr(wandb.Api(), "default_entity", None)
                if default_entity:
                    return f"{default_entity}/{project}"
            except Exception:
                pass

        return _resolve_wandb_project_path()

    def on_input_changed(self, event: Input.Changed) -> None:
        # W&B context inputs
        if event.input.id == "wb-entity":
            self._wb_entity = event.value.strip() or None
            self._on_wandb_context_changed()
            return
        if event.input.id == "wb-project":
            self._wb_project = event.value.strip() or None
            self._on_wandb_context_changed()
            return

        # Slot ref input
        wid = event.input.id or ""
        if not wid.startswith("val-"):
            return
        slot = wid.removeprefix("val-")
        if slot in self._state:
            self._state[slot].value = event.value


def _resolve_wandb_project_path() -> str | None:
    """Return '<entity>/<project>' used for W&B artifact browsing."""
    try:
        cfg = load_wandb_config()
    except Exception:
        cfg = None

    if cfg is not None and cfg.project:
        if cfg.entity:
            return f"{cfg.entity}/{cfg.project}"
        # If entity is not set in run.toml, fall back to env.

    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    if entity and project:
        return f"{entity}/{project}"

    # Last resort: if wandb has a default entity configured, use it with run.toml project.
    if cfg is not None and cfg.project:
        try:
            import wandb

            default_entity = getattr(wandb.Api(), "default_entity", None)
            if default_entity:
                return f"{default_entity}/{cfg.project}"
        except Exception:
            return None

    return None


def _load_default_wandb_context() -> tuple[str | None, str | None]:
    """Best-effort initial values for W&B entity/project inputs."""
    try:
        cfg = load_wandb_config()
    except Exception:
        cfg = None

    entity = None
    project = None

    if cfg is not None:
        entity = cfg.entity or None
        project = cfg.project or None

    if not entity:
        entity = os.environ.get("WANDB_ENTITY")
    if not project:
        project = os.environ.get("WANDB_PROJECT")

    return entity, project


def _format_wandb_dt(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return raw


def _build_wandb_version_options(
    *,
    latest: object,
    versions: list[object],
) -> list[tuple[str, str]]:
    def _get(obj: object, name: str, default: str = "") -> str:
        return str(getattr(obj, name, default) or default)

    options: list[tuple[str, str]] = []

    latest_version = _get(latest, "version")
    latest_created = _format_wandb_dt(_get(latest, "created_at"))
    options.append((f"latest ({latest_version} · {latest_created})", "latest"))

    for a in versions:
        version = _get(a, "version")
        if not version:
            continue
        created = _format_wandb_dt(_get(a, "created_at"))
        options.append((f"{version} · {created}", version))

    # Deduplicate (can happen if list includes the 'latest' artifact).
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for label, value in options:
        if value in seen:
            continue
        seen.add(value)
        deduped.append((label, value))

    return deduped


def _wandb_artifact_urls(
    *,
    project_path: str,
    type_name: str,
    artifact_name: str,
    version: str,
) -> tuple[str, str]:
    base = f"https://wandb.ai/{project_path}/artifacts/{type_name}/{artifact_name}/{version}"
    return f"{base}/metadata", f"{base}/lineage"
