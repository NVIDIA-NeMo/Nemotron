# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A visual investigation agent built on Nemotron 3.5 Super VL. This module is what the notebook imports.

    run = load_recording("model-switch-visual")   # replay a saved run
    run = run_live("model-switch-visual")         # or run live against the endpoint set in .env
    show_agent_view(run)

The implementation separates an orchestration loop (harness.py), bounded tools (tools.py), an investigation guide
(playbook/helpdesk.md), and enforced policy checks (policy/helpdesk.json and policy.py).
"""

from .harness import investigate
from .present import (Card, show_agent_view, show_build_guide, show_live_step, show_missing_evidence, show_reasoned_decision,
                      show_request, show_setup, show_visual_triage)
from .runs import Run, load_recording, load_run
from .version import AGENT_VERSION, check_config

__all__ = ["AGENT_VERSION", "Run", "check_config", "investigate", "load_recording", "load_run", "run_live",
           "show_agent_view", "show_build_guide", "show_missing_evidence", "show_reasoned_decision", "show_request",
           "show_setup", "show_visual_triage"]


def run_live(scenario):
    """Validate the selected endpoint, then run the agent with one progress line per model reply."""
    from IPython.display import display
    from . import endpoint

    from . import sandbox

    if sandbox.available() is None:
        raise RuntimeError("No sandbox for the agent's Python on this machine, so its Python calls would be refused. "
                           "macOS includes one; on Linux, install bubblewrap and allow unprivileged user namespaces "
                           "(see the README's Troubleshooting).")
    checks = endpoint.check()
    failed = [(name, detail) for name, passed, detail in checks if not passed]
    if failed:
        details = "; ".join(f"{name}: {detail}" for name, detail in failed)
        raise RuntimeError(f"Live endpoint validation failed. {details}")

    live = display(Card("<p><b>Live endpoint validated.</b> Text, thinking, image, and tool-call checks passed. "
                        "Investigation starting…</p>", "Live investigation"), display_id=True)

    def update(partial):
        live.update(show_live_step(partial))

    folder = investigate(scenario, on_step=update)
    live.update(Card(f"<p><b>Live investigation complete.</b> Saved to <code>{folder.name}/</code>; results render below.</p>",
                     "Live investigation"))
    return load_run(folder)
