from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Select, Static, TabbedContent

from nemotron.kit.tui.recipe_tui import RecipeTuiMeta, RecipeTuiResult
from nemotron.kit.tui.widgets.artifact_picker import ArtifactPicker
from nemotron.kit.tui.widgets.config_editor import ConfigEditor
from nemotron.kit.tui.widgets.run_controls import RunControls
from nemotron.kit.tui.widgets.yaml_editor import YamlEditor


class RecipeConfigScreen(Vertical):
    def __init__(self, meta: RecipeTuiMeta) -> None:
        super().__init__()
        self.meta = meta
        self._artifact_picker: ArtifactPicker | None = None
        self._config_editor: ConfigEditor | None = None
        self._yaml_editor: YamlEditor | None = None
        self._config_paths: dict[str, Path] = {}
        self._selected_config: Path | None = None
        self._run_controls: RunControls | None = None
        self._tabs: TabbedContent | None = None

    def compose(self) -> ComposeResult:
        with TabbedContent(
            "Input Artifact",
            "Job Config",
            "YAML",
            "Eval",
            initial="tab-1",
        ) as tabs:
            self._tabs = tabs
            self._artifact_picker = ArtifactPicker(self.meta.artifacts)
            yield self._artifact_picker

            config_path, options = _default_config_and_options(
                self.meta.config_dir,
                default_config=self.meta.default_config,
            )
            self._selected_config = config_path
            self._config_paths = {str(p): p for _label, p in options}

            with Vertical(id="jobcfg"):
                with Horizontal(id="jobcfg-header"):
                    yield Static("Config", classes="field-label")
                    yield Select(
                        options=[(label, str(p)) for (label, p) in options],
                        value=str(config_path),
                        allow_blank=False,
                        id="jobcfg-config",
                    )

                self._config_editor = ConfigEditor.from_yaml(config_path)
                yield self._config_editor

            self._yaml_editor = YamlEditor(self._config_editor.model, show_line_numbers=True)
            yield self._yaml_editor

            yield Static("Eval configuration coming soon.", classes="placeholder")

        self._run_controls = RunControls(stage_name=None)
        yield self._run_controls

    def on_select_changed(self, event: Select.Changed) -> None:  # type: ignore[name-defined]
        if event.select.id != "jobcfg-config" or event.value is Select.BLANK:
            return

        path = self._config_paths.get(str(event.value))
        if path is None or self._config_editor is None:
            return
        self._selected_config = path
        self._config_editor.load_yaml(path)
        if self._yaml_editor is not None:
            self._yaml_editor.sync_from_model(force=True)

    def on_mount(self) -> None:
        if self._tabs is None:
            return

        def _post_mount() -> None:
            if not self._tabs.active:
                self._tabs.active = "tab-1"
            self._tabs.disable_tab("tab-4")

        self.call_after_refresh(_post_mount)

    async def request_run(self, detached: bool) -> None:
        if self._config_editor is None:
            return

        train_cfg = self._config_editor.build_config_dictconfig()

        # Apply artifact refs based on the recipe's declared ArtifactInputs.
        if self._artifact_picker is not None and self.meta.artifacts:
            refs = self._artifact_picker.get_artifact_refs()
            if refs:
                try:
                    _apply_artifacts_to_train_config(train_cfg, refs, self.meta.artifacts)
                except Exception as e:
                    if self._run_controls is not None:
                        self._run_controls.set_status(str(e))
                    return

        profile = self._run_controls.selected_profile if self._run_controls else None
        if not profile or profile == "local":
            profile = None

        self.app.exit(
            result=RecipeTuiResult(
                train_config=train_cfg,
                config_path=str(self._selected_config) if self._selected_config else "",
                profile=profile,
                detached=detached,
            ),
        )

    async def on_run_controls_run_requested(self, message: RunControls.RunRequested) -> None:
        await self.request_run(detached=message.detached)

    def on_config_editor_changed(self, _message: ConfigEditor.Changed) -> None:
        if self._yaml_editor is not None:
            self._yaml_editor.sync_from_model()

    def on_yaml_editor_changed(self, _message: YamlEditor.Changed) -> None:
        if self._config_editor is not None:
            self._config_editor.refresh_from_model()


def _default_config_and_options(
    config_dir: str,
    *,
    default_config: str,
) -> tuple[Path, list[tuple[str, Path]]]:
    cfg_dir = Path(config_dir)
    files = sorted(cfg_dir.glob("*.y*ml"))
    if not files:
        raise FileNotFoundError(f"No config files found in {config_dir}")

    preferred = _resolve_default_config(cfg_dir, default_config)
    if preferred is None:
        preferred = cfg_dir / "default.yaml" if (cfg_dir / "default.yaml").exists() else files[0]

    # Put preferred first in the list for convenience.
    files = [preferred] + [p for p in files if p != preferred]

    options: list[tuple[str, Path]] = []
    for p in files:
        label = p.stem
        if p == preferred:
            label = f"{label} (default)"
        options.append((label, p))

    return preferred, options


def _resolve_default_config(cfg_dir: Path, default_config: str) -> Path | None:
    # Path-like
    if "/" in default_config or default_config.endswith((".yaml", ".yml")):
        p = Path(default_config)
        if p.exists():
            return p
        # allow relative-to-config-dir
        rel = cfg_dir / default_config
        return rel if rel.exists() else None

    # Name-like (stem)
    for ext in (".yaml", ".yml"):
        p = cfg_dir / f"{default_config}{ext}"
        if p.exists():
            return p
    return None


def _apply_artifacts_to_train_config(
    train_cfg: Any,
    refs: dict[str, str],
    artifacts: dict[str, Any],
) -> None:
    from nemotron.kit.app import _build_art_uri, _load_artifact_metadata_from_path
    from nemotron.kit.config import resolve_artifact_uri

    for slot, ref in refs.items():
        if slot not in artifacts:
            continue

        art_input = artifacts[slot]
        art_uri = _build_art_uri(ref, art_input.default_name)
        base_artifact_path = resolve_artifact_uri(art_uri)

        # Record for provenance; removed from train.yaml by ConfigBuilder.
        OmegaConf.update(train_cfg, f"run.{slot}", ref, merge=False)

        artifact_metadata = None
        if any(k.startswith("metadata.") for k in art_input.mappings.keys()):
            artifact_metadata = _load_artifact_metadata_from_path(base_artifact_path)

        for artifact_field, config_field in art_input.mappings.items():
            if config_field.startswith("fn."):
                continue

            if artifact_field.startswith("metadata."):
                metadata_key = artifact_field[9:]
                if artifact_metadata and metadata_key in artifact_metadata:
                    OmegaConf.update(
                        train_cfg,
                        config_field,
                        artifact_metadata[metadata_key],
                        merge=False,
                    )
                continue

            if artifact_field:
                full_uri = f"{art_uri}/{artifact_field}"
                local_path = resolve_artifact_uri(full_uri)
                OmegaConf.update(train_cfg, config_field, local_path, merge=False)
            else:
                OmegaConf.update(train_cfg, config_field, base_artifact_path, merge=False)
