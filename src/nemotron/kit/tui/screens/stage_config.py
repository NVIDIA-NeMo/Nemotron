from __future__ import annotations

import sys
import tempfile
from dataclasses import is_dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, TabbedContent

from nemotron.kit.app import _apply_artifact_refs_to_config
from nemotron.kit.run import load_run_profile, resolve_partition, run_with_nemo_run
from nemotron.kit.tui import StageCommandMeta
from nemotron.kit.tui.form_utils import dataclass_to_primitive_dict
from nemotron.kit.tui.widgets.artifact_picker import ArtifactPicker
from nemotron.kit.tui.widgets.config_editor import ConfigEditor
from nemotron.kit.tui.widgets.run_controls import RunControls
from nemotron.kit.tui.widgets.yaml_editor import YamlEditor


class StageConfigScreen(Vertical):
    def __init__(self, stage: StageCommandMeta) -> None:
        super().__init__()
        self.stage = stage
        self._artifact_picker: ArtifactPicker | None = None
        self._config_editor: ConfigEditor | None = None
        self._yaml_editor: YamlEditor | None = None
        self._run_controls: RunControls | None = None
        self._tabs: TabbedContent | None = None

    def compose(self) -> ComposeResult:
        # NOTE: Textual doesn't always auto-select the first tab when `initial` is empty
        # (varies by version / internal state). Force the initial pane.
        with TabbedContent(
            "Input Artifact",
            "Job Config",
            "YAML",
            "Eval",
            initial="tab-1",
        ) as tabs:
            self._tabs = tabs
            self._artifact_picker = ArtifactPicker(self.stage.artifacts)
            yield self._artifact_picker

            self._config_editor = ConfigEditor.from_dataclass(self.stage.config_type)
            yield self._config_editor

            self._yaml_editor = YamlEditor(self._config_editor.model, show_line_numbers=True)
            yield self._yaml_editor

            yield Static("Eval configuration coming soon.", classes="placeholder")

        self._run_controls = RunControls(stage_name=self.stage.name)
        yield self._run_controls

    def on_mount(self) -> None:
        # Tab widgets aren't fully composed until after `compose` yields; disable after mount.
        if self._tabs is None:
            return

        def _post_mount() -> None:
            # Ensure the first pane is active and disable the placeholder Eval tab.
            if not self._tabs.active:
                self._tabs.active = "tab-1"
            self._tabs.disable_tab("tab-4")

        self.call_after_refresh(_post_mount)

    async def run_stage(self, detached: bool) -> None:
        if self._config_editor is None or self._run_controls is None:
            return

        try:
            cfg = self._config_editor.build_config_instance()
        except Exception as e:
            self._run_controls.set_status(str(e))
            return

        # Apply artifact selections (App-style artifact slots)
        if self._artifact_picker is not None and self.stage.artifacts:
            art_refs = self._artifact_picker.get_artifact_refs()
            cfg = _apply_artifact_refs_to_config(
                cfg,
                art_refs=art_refs,
                stdin_artifacts=None,
                artifacts=self.stage.artifacts,
            )

        profile = self._run_controls.selected_profile
        if profile is None:
            # No run.toml profile found; run in-process via handler.
            self._run_controls.set_status("Running locally...")
            self.stage.handler(cfg)
            return

        if self.stage.entry_module is None:
            self._run_controls.set_status("Missing stage entry module")
            return

        if not is_dataclass(cfg):
            self._run_controls.set_status("Config is not a dataclass; cannot serialize")
            return

        # Write config as YAML for kit.cli() consumption.
        payload = {"config": dataclass_to_primitive_dict(cfg)}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml  # type: ignore[import-untyped]

            yaml.safe_dump(payload, f, sort_keys=False)
            config_path = f.name

        run_config = load_run_profile(profile)
        if detached:
            run_config.detach = True
            run_config.ray_mode = "job"
        else:
            run_config.detach = False

        # Resolve partition based on execution mode (detached maps to launch mode)
        run_config.partition = resolve_partition(run_config, is_launch=detached)

        self._run_controls.set_status(
            f"Launching via profile '{profile}' ({'detached' if detached else 'attached'})..."
        )

        exit_code = run_with_nemo_run(
            script_path=sys.executable,
            script_args=[
                "-m",
                self.stage.entry_module,
                "--config-file",
                config_path,
            ],
            run_config=run_config,
            ray=self.stage.ray,
            pre_ray_start_commands=None,
        )

        self._run_controls.set_status(f"Exited with code {exit_code}")
        if detached:
            self.app.exit(message=None)
        else:
            self.app.exit(result=None)

    async def on_run_controls_run_requested(self, message: RunControls.RunRequested) -> None:
        await self.run_stage(detached=message.detached)

    def on_config_editor_changed(self, _message: ConfigEditor.Changed) -> None:
        if self._yaml_editor is not None:
            self._yaml_editor.sync_from_model()

    def on_yaml_editor_changed(self, _message: YamlEditor.Changed) -> None:
        if self._config_editor is not None:
            self._config_editor.refresh_from_model()
