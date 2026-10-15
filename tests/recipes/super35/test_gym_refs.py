"""Check super35 RL configs against the pinned NeMo-Gym tree.

The RL stages only pin NeMo-RL (whose Gym submodule is the commit below), so a
typo or rename in a Gym reference would only surface at job start on a GPU
cluster. These tests catch that on CPU by resolving every reference against a
Gym checkout at the pinned commit.

Set ``NEMO_GYM_DIR`` to a NeMo-Gym checkout at ``GYM_COMMIT``; the tests are
skipped otherwise:

    git init gym && cd gym
    git fetch --depth 1 https://github.com/NVIDIA-NeMo/Gym.git e24db4e6f470333e10180148a72cc8f25f25a912
    git checkout FETCH_HEAD
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

GYM_COMMIT = "e24db4e6f470333e10180148a72cc8f25f25a912"
SERVER_TYPES = ("resources_servers", "responses_api_models", "responses_api_agents")

REPO_ROOT = Path(__file__).resolve().parents[3]
RL_ROOT = REPO_ROOT / "src/nemotron/recipes/super35/stage2_rl"
RL_CONFIGS = sorted(RL_ROOT.glob("**/config/*.yaml"))

# References that are known not to exist in public Gym at GYM_COMMIT. Each
# entry must also be called out with a TODO in the config and stage README.
KNOWN_MISSING = {
    # Agentic teacher is blocked on the legacy Terminal-Bench agent
    # (swe_agents_orig + swe_agents/configs/terminalbench.yaml).
    ("stage2_teachers/agentic/config/default.yaml", "config_paths"),
    ("stage2_teachers/agentic/config/default.yaml", "entrypoint"),
    # Vision-reasoning stage 2 uses an unpublished structured-answer verifier.
    ("stage2_teachers/vision_reasoning/config/stage2.yaml", "entrypoint"),
}

GYM_DIR = Path(os.environ["NEMO_GYM_DIR"]) if os.environ.get("NEMO_GYM_DIR") else None
pytestmark = pytest.mark.skipif(GYM_DIR is None, reason="NEMO_GYM_DIR not set")


def _ids(paths):
    return [str(p.relative_to(RL_ROOT)) for p in paths]


def _gym_cfg(config: Path) -> dict:
    return yaml.safe_load(config.read_text())["env"]["nemo_gym"]


def _instances(block: dict) -> dict[str, tuple[str, str, dict]]:
    """Map instance name -> (server type, impl dir, settings) for a Gym config dict."""
    found = {}
    for name, value in block.items():
        if not isinstance(value, dict):
            continue
        for server_type in SERVER_TYPES:
            impls = value.get(server_type)
            if isinstance(impls, dict):
                for impl, settings in impls.items():
                    found[name] = (server_type, impl, settings if isinstance(settings, dict) else {})
    return found


def _merged_instances(config: Path) -> dict[str, tuple[str, str, dict]]:
    gym = _gym_cfg(config)
    merged = {}
    for rel in gym.get("config_paths", []):
        path = GYM_DIR / rel
        if path.exists():
            merged.update(_instances(yaml.safe_load(path.read_text()) or {}))
    for name, (server_type, impl, settings) in _instances(gym).items():
        base = merged.get(name, (server_type, impl, {}))[2]
        merged[name] = (server_type, impl, {**base, **settings})
    return merged


def _server_refs(node, path=""):
    """Yield (path, name, type) for every ``{name: ..., type: <server type>}`` reference."""
    if isinstance(node, dict):
        if isinstance(node.get("name"), str) and node.get("type") in SERVER_TYPES:
            yield path, node["name"], node["type"]
        for key, value in node.items():
            yield from _server_refs(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _server_refs(value, f"{path}[{i}]")


def _known_missing(config: Path, kind: str) -> bool:
    return (str(config.relative_to(RL_ROOT)), kind) in KNOWN_MISSING


def test_gym_checkout_is_pinned_commit():
    head = subprocess.run(["git", "-C", str(GYM_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == GYM_COMMIT


@pytest.mark.parametrize("config", RL_CONFIGS, ids=_ids(RL_CONFIGS))
def test_config_paths_exist(config):
    missing = [rel for rel in _gym_cfg(config).get("config_paths", []) if not (GYM_DIR / rel).exists()]
    if missing and _known_missing(config, "config_paths"):
        pytest.xfail(f"known missing: {missing}")
    assert not missing


@pytest.mark.parametrize("config", RL_CONFIGS, ids=_ids(RL_CONFIGS))
def test_server_impls_and_entrypoints_exist(config):
    problems = []
    for name, (server_type, impl, settings) in _instances(_gym_cfg(config)).items():
        impl_dir = GYM_DIR / server_type / impl
        if not impl_dir.is_dir():
            problems.append(f"{name}: {server_type}/{impl} missing")
        elif settings.get("entrypoint") and not (impl_dir / settings["entrypoint"]).exists():
            problems.append(f"{name}: {server_type}/{impl}/{settings['entrypoint']} missing")
    if problems and _known_missing(config, "entrypoint"):
        pytest.xfail(f"known missing: {problems}")
    assert not problems


@pytest.mark.parametrize("config", RL_CONFIGS, ids=_ids(RL_CONFIGS))
def test_server_references_resolve(config):
    instances = _merged_instances(config)
    dangling = [
        f"{path}: {ref_type}/{name}"
        for path, name, ref_type in _server_refs(_gym_cfg(config))
        if name not in instances and name != "policy_model"
    ]
    if dangling and (_known_missing(config, "config_paths") or _known_missing(config, "entrypoint")):
        pytest.xfail(f"depends on known-missing Gym code: {dangling}")
    assert not dangling


MOPD_CONFIGS = sorted((RL_ROOT / "stage3_mopd/config").glob("*.yaml"))


@pytest.mark.parametrize("config", MOPD_CONFIGS, ids=_ids(MOPD_CONFIGS))
def test_mopd_routes_name_real_agents(config):
    raw = yaml.safe_load(config.read_text())
    routes = raw["on_policy_distillation"]["teacher_model_by_agent_name"]
    agents = {n for n, (t, _, _) in _merged_instances(config).items() if t == "responses_api_agents"}
    canonical = {k for k, v in routes.items() if not str(v).startswith("${on_policy_distillation.")}
    unknown = sorted(set(routes) - canonical - agents)
    assert not unknown, f"routed agent names not defined in the Gym config: {unknown}"
