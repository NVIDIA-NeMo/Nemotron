# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""PEP 723 inline script metadata parser for [tool.runspec] blocks.

Uses only stdlib: pathlib, re, tomllib (3.11+) / tomli (3.10).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from nemo_runspec._models import Runspec, RunspecConfig, RunspecResources, RunspecRun

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


def extract_pep723_toml(script_path: str | Path) -> str | None:
    """Extract PEP 723 ``# /// script`` block from a Python file.

    Strips the ``# `` prefix from each line inside the block and returns
    the raw TOML string.  Returns ``None`` if no block is found.
    """
    path = Path(script_path)
    text = path.read_text(encoding="utf-8")

    # Match the PEP 723 block: starts with ``# /// script``, ends with ``# ///``
    pattern = re.compile(
        r"^# /// script\s*\n((?:#[^\n]*\n)*?)# ///$",
        re.MULTILINE,
    )
    m = pattern.search(text)
    if m is None:
        return None

    # Strip leading "# " (or bare "#") from each line
    lines = m.group(1).splitlines()
    stripped = []
    for line in lines:
        if line.startswith("# "):
            stripped.append(line[2:])
        elif line == "#":
            stripped.append("")
        else:
            stripped.append(line.lstrip("# "))
    return "\n".join(stripped) + "\n"


def parse_runspec(toml_dict: dict) -> Runspec:
    """Build a :class:`Runspec` from the ``[tool.runspec]`` section of a TOML dict."""
    rs = toml_dict.get("tool", {}).get("runspec", {})
    if not rs:
        raise ValueError("No [tool.runspec] section found in TOML data")

    run_data = rs.get("run", {})
    config_data = rs.get("config", {})
    resources_data = rs.get("resources", {})

    return Runspec(
        schema=str(rs.get("schema", "1")),
        docs=rs.get("docs", ""),
        name=rs.get("name", ""),
        image=rs.get("image"),
        setup=rs.get("setup", "").strip(),
        run=RunspecRun(
            launch=run_data.get("launch", "torchrun"),
            cmd=run_data.get("cmd", "python {script} --config {config}"),
            workdir=run_data.get("workdir"),
        ),
        config=RunspecConfig(
            dir=config_data.get("dir", "./config"),
            default=config_data.get("default", "default"),
            format=config_data.get("format", "omegaconf"),
        ),
        resources=RunspecResources(
            nodes=resources_data.get("nodes", 1),
            gpus_per_node=resources_data.get("gpus_per_node", 8),
        ),
        env=rs.get("env", {}),
    )


def parse(script_path: str | Path) -> Runspec:
    """Parse ``[tool.runspec]`` from a Python script's PEP 723 metadata.

    This is the main public entry point.

    Args:
        script_path: Path to the recipe script (absolute or relative to cwd).

    Returns:
        A populated :class:`Runspec` instance with ``script_path`` set.

    Raises:
        FileNotFoundError: If the script does not exist.
        ValueError: If no PEP 723 block or [tool.runspec] section is found.
    """
    path = Path(script_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    toml_str = extract_pep723_toml(path)
    if toml_str is None:
        raise ValueError(f"No PEP 723 metadata block found in {path}")

    toml_dict = tomllib.loads(toml_str)
    spec = parse_runspec(toml_dict)

    # Replace with a copy that has script_path set
    return Runspec(
        schema=spec.schema,
        docs=spec.docs,
        name=spec.name,
        image=spec.image,
        setup=spec.setup,
        run=spec.run,
        config=spec.config,
        resources=spec.resources,
        env=spec.env,
        script_path=path,
    )
