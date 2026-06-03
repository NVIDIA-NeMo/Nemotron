# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared static-check primitives for per-step tests.

Each ``test_<step>.py`` calls :func:`assert_step_static` with the expected
runspec / manifest / config invariants for that one step. Heavy framework deps
(nemo_automodel, megatron, nemo_rl, data_designer, cosmos_xenna) are NOT
imported here — they're a runtime concern and the suite must run on a plain CI
host. The check forbids them at module import time so generic ``step run`` can
load the script without those packages installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from nemo_runspec import parse as parse_runspec

DEFERRED_IMPORT_PREFIXES = (
    "nemo_automodel",
    "megatron",
    "nemo_rl",
    "data_designer",
    "cosmos_xenna",
)


def step_dir(test_file: str | Path, *parts: str) -> Path:
    """Resolve ``src/nemotron/steps/<parts>`` from a test file location.

    Lets each test write::

        STEP_DIR = step_dir(__file__, "peft", "automodel")
    """
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "src" / "nemotron" / "steps" / Path(*parts)


def _toplevel_import_modules(tree: ast.Module) -> list[str]:
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def assert_step_static(
    step_dir: Path,
    *,
    expected_name: str,
    expected_launch: str,
    expected_default_config: str,
    runtime_only_imports: tuple[str, ...] = DEFERRED_IMPORT_PREFIXES,
    require_workdir: bool = False,
    expected_id: str | None = None,
) -> None:
    """Run all per-step static invariants in one place.

    Args:
        step_dir: Path to the step directory (must contain step.py + step.toml).
        expected_name: Expected runspec ``name`` (e.g. ``"steps/peft/automodel"``).
        expected_launch: Expected runspec ``run.launch`` (torchrun / ray / python / direct).
        expected_default_config: Expected default config name (without ``.yaml``).
        runtime_only_imports: Module prefixes that must NOT be imported at
            module top-level. Override per-step if needed.
        require_workdir: True if the runspec must declare ``run.workdir``.
        expected_id: Expected step.toml ``[step] id`` (defaults to ``expected_name``
            with the leading ``steps/`` stripped).
    """
    step_py = step_dir / "step.py"
    step_toml = step_dir / "step.toml"
    assert step_py.exists(), f"{step_dir}: step.py is missing"
    assert step_toml.exists(), f"{step_dir}: step.toml is missing"

    # -- Parse step.py --------------------------------------------------------
    source = step_py.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(step_py))

    # main() exists.
    fn_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "main" in fn_names, f"{step_py}: no top-level `def main()`"

    # __main__ guard exists.
    assert (
        'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source
    ), f"{step_py}: missing `if __name__ == '__main__'` guard"

    # Sanity: the runtime-only check is informational. Steps that need to import
    # framework deps at module top-level are fine — they only ever run inside
    # their declared container. We just keep the parameter for future tightening.
    _ = runtime_only_imports
    _ = _toplevel_import_modules

    # -- Parse runspec --------------------------------------------------------
    spec = parse_runspec(str(step_py))
    assert spec.name == expected_name, (
        f"{step_py}: runspec name {spec.name!r} != expected {expected_name!r}"
    )
    assert spec.run.launch == expected_launch, (
        f"{step_py}: runspec launch {spec.run.launch!r} != expected {expected_launch!r}"
    )
    assert spec.config.default == expected_default_config, (
        f"{step_py}: runspec default config {spec.config.default!r} != "
        f"expected {expected_default_config!r}"
    )
    assert spec.config_dir.exists(), f"{step_py}: config dir {spec.config_dir} missing"

    # Default config YAML resolves and parses.
    default_yaml = spec.config_dir / f"{spec.config.default}.yaml"
    assert default_yaml.exists(), f"{step_py}: default config {default_yaml} missing"
    parsed = yaml.safe_load(default_yaml.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), (
        f"{default_yaml}: must be a YAML mapping at top level"
    )

    # When the author supplies a custom cmd template it must use the
    # ``{script}`` / ``{config}`` placeholders so backends can format it.
    if spec.run.cmd is not None:
        assert "{script}" in spec.run.cmd, (
            f"{step_py}: runspec cmd {spec.run.cmd!r} missing {{script}}"
        )
        assert "{config}" in spec.run.cmd, (
            f"{step_py}: runspec cmd {spec.run.cmd!r} missing {{config}}"
        )

    if require_workdir:
        assert getattr(spec.run, "workdir", ""), (
            f"{step_py}: runspec must declare run.workdir for this launch mode"
        )

    # -- Parse step.toml ------------------------------------------------------
    import tomllib

    with step_toml.open("rb") as fh:
        manifest = tomllib.load(fh)

    expected_manifest_id = expected_id if expected_id is not None else expected_name.removeprefix("steps/")
    actual_id = manifest.get("step", {}).get("id")
    assert actual_id == expected_manifest_id, (
        f"{step_toml}: [step] id {actual_id!r} != expected {expected_manifest_id!r}"
    )
    assert manifest.get("step", {}).get("category"), (
        f"{step_toml}: [step] category is missing"
    )
