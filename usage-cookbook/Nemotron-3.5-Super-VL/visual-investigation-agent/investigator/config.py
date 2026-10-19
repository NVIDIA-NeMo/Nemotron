# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's configuration: every prompt and design choice an experiment can vary, in one dict.

AGENT is the agent the notebook runs. Any other configuration is AGENT with some keys changed; make() merges the changes
and checks every key is known, so a typo cannot silently fall back to a default.
"""

import copy
import hashlib
import json

AGENT = {
    "name": "agent",
    "contract": "v1",          # v1: the system contract; v2: also asks it to test the explanation the request gives
    "playbook": "generic",     # generic: step 2 names the kinds of driver (volume, price, usage, mix); none: no playbook
    "schema": "neutral",       # neutral: the report's drivers field doesn't list cost metrics; v1: it does
    "attribution": True,       # True: the report names the recorded changes or documents that explain the driver (or none);
                               # "per_driver": the same for each driver that moved, with the day it started to move
    "request": "neutral",      # neutral: states only the size of the change; scenario: the world's own request;
                               # hint: a true clue; flip: a wrong explanation
    "objective": "explicit",   # explicit: the policy states "cheapest model within the bar"; implicit: only the bar
    "first_look": True,        # contract asks for a note on the request and images before the first tool call
    "why": True,               # action tools require a `why`
    "labels": False,           # True: evidence must be labelled Observed / Calculated / Hypothesis
    "tools": {"notes": False, "zoom": False, "check_report": False, "savings_inputs": False},
    "plan_first": False,       # contract asks for a written plan (in notes) before investigating
    "review": False,           # the first report is a draft: one fixed review message, then the final submit_report
    "blind": False,            # every image the model would receive is replaced by a one-line placeholder (a baseline)
    "image_budget": 4,         # images per request to the model, including the user's
    "max_nudges": 2,
    "sampling": {"temperature": 0.6, "top_p": 0.95, "thinking": True, "max_tokens": 16384, "seed": None},
}


def _merge(base, changes, path=""):
    out = copy.deepcopy(base)
    for key, value in changes.items():
        if key not in base:
            raise KeyError(f"unknown config key {path + key!r}")
        out[key] = _merge(base[key], value, f"{path}{key}.") if isinstance(base[key], dict) and isinstance(value, dict) else value
    return out


def make(changes=None, base=AGENT, name=None):
    config = _merge(base, changes or {})
    if name:
        config["name"] = name
    return config


LATER_KEYS = {"review": False, "blind": False}  # hashed only when set, so turning them off leaves the hash unchanged


def config_hash(config):
    body = {k: v for k, v in config.items() if k != "name" and not (k in LATER_KEYS and v == LATER_KEYS[k])}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


DEFAULT = AGENT  # the configuration the notebook and the suite runner use by default
VERSIONS = {AGENT["name"]: AGENT}  # the frozen configuration, with its manifest <name>.json


def for_version(name):
    """The configuration a frozen version (or a recording made with it) used."""
    return VERSIONS[name]


def from_ablate(ablate):
    """The default configuration, or one of the two ablations the suite runner offers."""
    if ablate == "playbook":
        return make({"playbook": "none"}, base=DEFAULT, name=f"{DEFAULT['name']}-no-playbook")
    return DEFAULT if not ablate else make(base=DEFAULT, name=f"{DEFAULT['name']}-{ablate}")
