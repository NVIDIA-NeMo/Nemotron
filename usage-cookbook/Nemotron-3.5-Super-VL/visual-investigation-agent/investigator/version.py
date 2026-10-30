# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which agent this is. Anything the model sees has a hash; each frozen version's manifest records its tested hashes.

Change the contract, a tool, the playbook, the policy, or a request, and the hashes stop matching: that is a
new agent version, and it needs its own test runs.
"""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import world

from .config import DEFAULT, VERSIONS, config_hash
from .contract import contract_text, playbook, request_for, system_prompt
from .policy import DEFAULT_WORLD, POLICY, policy_text
from .tools import tools_for

AGENT_VERSION = DEFAULT["name"]
ROOT = Path(__file__).resolve().parents[1]


def manifest_file(version=AGENT_VERSION):
    return ROOT / f"{version}.json"


MANIFEST_FILE = manifest_file()


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def model_facing(config=None):
    """Hashes of everything the model is given under a configuration (the default one if none is given)."""
    config = config or DEFAULT
    return {
        "system_contract": sha(contract_text(DEFAULT_WORLD, config)),
        "policy_text": sha(policy_text()),
        "system_prompt": {s: sha(system_prompt(s, config=config)) for s in world.SCENARIOS},
        "tool_schemas": sha(json.dumps(tools_for(DEFAULT_WORLD, config), sort_keys=True)),
        "request": {s: sha(request_for(s, config["request"])) for s in world.SCENARIOS},
        "playbook": sha(playbook(DEFAULT_WORLD, config["playbook"])),
        "policy": sha(json.dumps(POLICY, sort_keys=True)),
    }


def check_config(version=AGENT_VERSION):
    """Compare what the model sees now with a version's frozen manifest: {item: True if unchanged}."""
    frozen = json.loads(manifest_file(version).read_text())["model_facing_sha256"]
    current = model_facing(VERSIONS[version])
    return {item: current[item] == frozen[item] for item in frozen}


def run_manifest(scenario, ablate, stamp, config=None):
    from .harness import MAX_TURNS

    config = config or DEFAULT
    version = next((name for name, c in VERSIONS.items() if config_hash(c) == config_hash(config)), None)
    unchanged = bool(version) and not ablate and all(check_config(version).values())
    world.ensure(scenario)
    tools = tools_for(scenario, config)
    sampling = config["sampling"]
    return {
        "agent_version": version if unchanged else (f"{version}+modified" if version else config["name"]),
        "matches_frozen": unchanged,
        "config": config["name"], "config_hash": config_hash(config),
        "scenario": scenario, "ablate": ablate, "utc": stamp,
        "recorded": dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d"),
        "model": os.environ.get("MODEL_NAME", ""),
        "sampling": {"temperature": sampling["temperature"], "top_p": sampling["top_p"], "enable_thinking": sampling["thinking"],
                     "max_tokens": sampling["max_tokens"], **({"seed": sampling["seed"]} if sampling["seed"] is not None else {})},
        "limits": {"max_turns": MAX_TURNS, "max_nudges": config["max_nudges"], "max_images_in_context": config["image_budget"]},
        "sha256": {"system_prompt": sha(system_prompt(scenario, ablate, config)),
                   "request": sha(request_for(scenario, config["request"])),
                   "tools": sha(json.dumps(tools, sort_keys=True)), "playbook": sha(playbook(DEFAULT_WORLD, config["playbook"])),
                   "policy": sha(json.dumps(POLICY, sort_keys=True)), "world_content": world.content_sha256(scenario)},
    }
