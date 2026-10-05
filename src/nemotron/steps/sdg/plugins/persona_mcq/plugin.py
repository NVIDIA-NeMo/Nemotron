# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point descriptor for the Persona MCQ Data Designer plugin."""

from data_designer.plugins.plugin import Plugin, PluginType
from data_designer.plugins.registry import PluginRegistry

persona_mcq = Plugin(
    impl_qualified_name="nemotron.steps.sdg.plugins.persona_mcq.generator.PersonaMCQGenerator",
    config_qualified_name="nemotron.steps.sdg.plugins.persona_mcq.config.PersonaMCQConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)


def ensure_registered() -> None:
    """Register the bundled plugin when package entry-point metadata is unavailable.

    Remote ``nemotron steps`` jobs stage the repository's source tree on
    ``PYTHONPATH`` without installing the project wheel. Data Designer 0.5.x
    has no public programmatic registration method, so add the same descriptor
    exposed by the package entry point before its column-type module is loaded.
    """
    registry = PluginRegistry()
    if not registry.plugin_exists(persona_mcq.name):
        registry._plugins[persona_mcq.name] = persona_mcq  # noqa: SLF001
