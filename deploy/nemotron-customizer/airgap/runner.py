#!/usr/bin/env python3
"""Lightweight airgap image runner for Nemotron Customizer.

This file intentionally lives under deploy/nemotron-customizer/airgap instead
of adding a new step. It is a connected-machine helper that validates requested
steps, discovers small execution-image Python gaps, builds launcher/execution images, and
saves image tarballs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomllib
import yaml

AIRGAP_DIR = Path(__file__).resolve().parent
REPO_ROOT = AIRGAP_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
STEP_ROOT = SRC_ROOT / "nemotron" / "steps"
DEFAULT_OUTPUT_DIR = AIRGAP_DIR / "out"
UV_VERSION = "0.11.1"
PROGRESS_STATE = "airgap-build-state.yaml"
COMPLETE_STATE = "airgap-build-complete.yaml"
LOCAL_PREFIXES = ("nemotron", "nemo_runspec")
CORE_IMPORTS = {
    "datasets",
    "megatron",
    "nemo",
    "numpy",
    "ray",
    "torch",
    "transformers",
    "triton",
    "vllm",
}
IMPORT_ALIASES = {
    "yaml": "pyyaml",
    "pydantic_settings": "pydantic-settings",
    "huggingface_hub": "huggingface-hub",
    "cosmos_xenna": "cosmos-xenna",
    "data_designer": "data-designer",
    "nemo_curator": "nemo-curator",
}


@dataclass(frozen=True)
class Target:
    step_id: str
    config: str | None = None

    @property
    def spec(self) -> str:
        return f"{self.step_id}:{self.config}" if self.config else self.step_id


@dataclass
class StepInfo:
    target: Target
    step_dir: Path
    step_py: Path
    step_toml: Path
    config_path: Path | None
    module: str
    mounts: list[Any] = field(default_factory=list)
    repo_overlays: list[RepoOverlay] = field(default_factory=list)


@dataclass(frozen=True)
class RepoOverlay:
    repo: str
    url: str
    ref: str
    target: str


@dataclass
class ExecutionGroup:
    name: str
    base_image: str
    tag: str
    tar: Path
    steps: list[str]
    platform: str | None = None
    required_imports: set[str] = field(default_factory=set)
    repo_overlays: list[RepoOverlay] = field(default_factory=list)
    pip_no_deps: bool = True
    candidate_imports: set[str] = field(default_factory=set)
    missing_imports: list[str] = field(default_factory=list)
    missing_core_imports: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    requirements_path: Path | None = None
    repo_overlays_path: Path | None = None
    selected_image: str | None = None
    image_names: set[str] = field(default_factory=set)


@dataclass
class RunState:
    path: Path
    done_path: Path
    data: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Nemotron Customizer airgap images from one YAML file.")
    parser.add_argument("--config", default=str(AIRGAP_DIR / "airgap.yaml"), help="Airgap runner YAML.")
    parser.add_argument("--execute", action="store_true", help="Run docker/git commands. Default prints the plan.")
    parser.add_argument("--stage", action="append", help="Stage to run. Repeatable. Defaults to config stages.")
    parser.add_argument(
        "--target",
        action="append",
        help="Nemotron step target step-id[:config]. Repeatable. Overrides workflow.stages.",
    )
    args = parser.parse_args(argv)

    config_path = resolve_input_path(Path(args.config))
    cfg = load_yaml(config_path)
    if args.target:
        cfg = with_workflow_targets(cfg, normalize_target_specs(args.target))
    stages = normalize_stages(args.stage or cfg.get("build_stages") or cfg.get("stages") or [])
    output_dir = resolve_repo_path(Path(cfg.get("paths", {}).get("output_dir", DEFAULT_OUTPUT_DIR)))
    if "build-execution-images" in stages:
        validate_docker_context_path(output_dir, field="paths.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_state = load_or_start_run_state(
        output_dir,
        config_path=config_path,
        cfg=cfg,
        stages=stages,
        execute=args.execute,
    )
    saved_images: list[dict[str, Any]] = []
    workflow = cfg.get("workflow") if isinstance(cfg.get("workflow"), Mapping) else {}

    print(f"[airgap] config={config_path}")
    print(f"[airgap] mode={'execute' if args.execute else 'plan'}")
    print(f"[airgap] stages={', '.join(stages)}")

    expanded_targets: list[Target] = []
    step_infos: dict[str, StepInfo] = {}
    groups: list[ExecutionGroup] = []
    workflow_manifest: dict[str, Any] = {
        "stages": list(workflow.get("stages") or []),
    }
    if workflow.get("name"):
        workflow_manifest["name"] = workflow.get("name")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "workflow": workflow_manifest,
        "output_dir": str(output_dir),
        "build_stages": stages,
    }

    if "validate" in stages or any(stage_needs_targets(stage) for stage in stages):
        begin_action(run_state, "validate")
        expanded_targets = expand_targets(cfg)
        step_infos = validate_targets(expanded_targets)
        manifest["targets"] = [step_to_manifest(info) for info in step_infos.values()]
        print(f"[validate] {len(step_infos)} target(s) ok")
        complete_action(run_state, "validate", {"targets": [target.spec for target in expanded_targets]})

    if any(stage in stages for stage in ("discover-execution-deps", "build-execution-images", "save-images")):
        groups = execution_groups(cfg, output_dir=output_dir, step_infos=step_infos)
        manifest["execution_groups"] = [execution_group_manifest(group) for group in groups]

    if "discover-execution-deps" in stages:
        if action_completed(run_state, "discover-execution-deps") and hydrate_discovered_groups(run_state, groups):
            print("[resume] skipping discover-execution-deps; using saved probe results")
        else:
            begin_action(run_state, "discover-execution-deps")
            locked_versions = locked_package_versions(REPO_ROOT / "uv.lock")
            for group in groups:
                discover_execution_deps(
                    group,
                    step_infos=step_infos,
                    locked_versions=locked_versions,
                    execute=args.execute,
                )
            remember_discovered_groups(run_state, groups)
            complete_action(run_state, "discover-execution-deps", {"groups": [group.name for group in groups]})
        manifest["execution_groups"] = [execution_group_manifest(group) for group in groups]

    if "build-launcher-image" in stages:
        launcher_image = cfg.get("launcher_image", {})
        launcher_image_tag = str(launcher_image.get("tag") or "nemotron-customizer-launcher-airgap:latest")
        platform = launcher_image_platform(launcher_image)
        action = "build-launcher-image"
        if action_completed(run_state, action) and docker_image_exists(launcher_image_tag, platform=platform):
            print(f"[resume] skipping {action}; image exists: {launcher_image_tag}")
        else:
            begin_action(run_state, action)
            status = build_launcher_image(launcher_image, execute=args.execute)
            if status:
                return status
            complete_action(run_state, action, {"image": launcher_image_tag})
        manifest["launcher_image"] = launcher_image_manifest(launcher_image)

    if "build-execution-images" in stages:
        clean_stale_group_dirs(output_dir, groups, execute=args.execute)
        for group in groups:
            action = f"build-execution-image:{group.name}"
            if action_completed(run_state, action) and docker_image_exists(group.tag, platform=group.platform):
                print(f"[resume] skipping {action}; image exists: {group.tag}")
            else:
                begin_action(run_state, action)
                status = build_execution_image(group, output_dir=output_dir, execute=args.execute)
                if status:
                    return status
                complete_action(run_state, action, {"image": group.tag})
        manifest["execution_groups"] = [execution_group_manifest(group) for group in groups]

    if "save-images" in stages:
        launcher_image = cfg.get("launcher_image", {})
        if launcher_image:
            output = output_dir / str(launcher_image.get("tar", "launcher-image.tar"))
            launcher_image_tag = str(launcher_image.get("tag") or "nemotron-customizer-launcher-airgap:latest")
            action = f"save-image:{launcher_image_tag}"
            if action_completed(run_state, action) and output.exists():
                print(f"[resume] skipping {action}; tar exists: {output}")
            else:
                begin_action(run_state, action)
                status = save_image(launcher_image_tag, output, args.execute)
                if status:
                    return status
                complete_action(run_state, action, {"tar": str(output)})
            saved_images.append(
                saved_image_manifest(
                    launcher_image_tag,
                    output,
                    execute=args.execute,
                    role="launcher",
                    name="launcher",
                )
            )
        for group in groups:
            action = f"save-image:{group.tag}"
            if action_completed(run_state, action) and group.tar.exists():
                print(f"[resume] skipping {action}; tar exists: {group.tar}")
            else:
                begin_action(run_state, action)
                status = save_image(group.tag, group.tar, args.execute)
                if status:
                    return status
                complete_action(run_state, action, {"tar": str(group.tar)})
            saved_images.append(
                saved_image_manifest(group.tag, group.tar, execute=args.execute, role="execution", name=group.name)
            )

    manifest["persistent_assets"] = {
        "policy": "models, datasets, checkpoints, and customer data stay on executor-visible persistent storage",
        "mounts_from_configs": collect_mounts(step_infos.values()),
        "baked_repo_overlays": [repo_overlay_manifest(item) for item in collect_repo_overlays(step_infos.values())],
    }
    manifest["step_execution_images"] = step_execution_image_manifest(groups)
    manifest["saved_images"] = saved_images
    manifest_path = output_dir / "airgap-manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    complete_run_state(run_state, manifest_path=manifest_path)
    print(f"[airgap] wrote {manifest_path}")
    if groups:
        print("[airgap] selected execution images:")
        for group in groups:
            image = group.selected_image or group.tag
            for step_id in group.steps:
                print(f"  - {step_id}: {image}")
    return 0


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: top-level YAML must be a mapping")
    return data


def normalize_target_specs(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for item in str(raw).split(","):
            target = item.strip()
            if target:
                out.append(target)
    return out


def with_workflow_targets(cfg: Mapping[str, Any], targets: list[str]) -> dict[str, Any]:
    out = dict(cfg)
    existing = out.get("workflow")
    workflow = dict(existing) if isinstance(existing, Mapping) else {}
    workflow["stages"] = targets
    out["workflow"] = workflow
    return out


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else path


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def docker_context_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise SystemExit(f"{path} must live under the repo root because docker build context is {REPO_ROOT}") from exc


def validate_docker_context_path(path: Path, *, field: str) -> None:
    try:
        docker_context_path(path)
    except SystemExit as exc:
        message = f"{field}={path} must live under the repo root because Docker builds use {REPO_ROOT}"
        raise SystemExit(message) from exc


def load_or_start_run_state(
    output_dir: Path,
    *,
    config_path: Path,
    cfg: Mapping[str, Any],
    stages: list[str],
    execute: bool,
) -> RunState | None:
    if not execute:
        return None
    path = output_dir / PROGRESS_STATE
    done_path = output_dir / COMPLETE_STATE
    signature = run_signature(config_path=config_path, cfg=cfg, stages=stages)
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise SystemExit(f"{path} must contain YAML mapping state")
        if data.get("signature") != signature:
            raise SystemExit(
                f"{path} is an incomplete airgap run for a different plan. "
                f"Finish it, move it aside, or remove it before starting a new plan."
            )
        print(f"[resume] found incomplete run state: {path}")
        return RunState(path=path, done_path=done_path, data=data)

    workflow = cfg.get("workflow") if isinstance(cfg.get("workflow"), Mapping) else {}
    data = {
        "schema_version": 1,
        "signature": signature,
        "config": str(config_path.resolve()),
        "workflow_stages": list(workflow.get("stages") or []),
        "build_stages": stages,
        "started_at": timestamp(),
        "completed_actions": {},
        "discovered_groups": {},
    }
    if done_path.exists():
        data["previous_complete"] = str(done_path)
    state = RunState(path=path, done_path=done_path, data=data)
    write_run_state(state)
    print(f"[airgap] progress state={path}")
    return state


def run_signature(*, config_path: Path, cfg: Mapping[str, Any], stages: list[str]) -> str:
    payload = {
        "config": str(config_path.resolve()),
        "stages": stages,
        "workflow": cfg.get("workflow"),
        "dependencies": cfg.get("dependencies"),
        "step_execution_images": cfg.get("step_execution_images"),
        "execution_images": cfg.get("execution_images"),
        "launcher_image": cfg.get("launcher_image"),
    }
    text = yaml.safe_dump(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_run_state(state: RunState | None) -> None:
    if state is None:
        return
    state.data["updated_at"] = timestamp()
    state.path.write_text(yaml.safe_dump(state.data, sort_keys=False), encoding="utf-8")


def begin_action(state: RunState | None, action: str) -> None:
    if state is None:
        return
    state.data["current_action"] = {"name": action, "started_at": timestamp()}
    write_run_state(state)


def complete_action(state: RunState | None, action: str, details: Mapping[str, Any] | None = None) -> None:
    if state is None:
        return
    completed = state.data.setdefault("completed_actions", {})
    completed[action] = {"completed_at": timestamp(), **dict(details or {})}
    if (state.data.get("current_action") or {}).get("name") == action:
        state.data.pop("current_action", None)
    write_run_state(state)


def action_completed(state: RunState | None, action: str) -> bool:
    if state is None:
        return False
    return action in (state.data.get("completed_actions") or {})


def remember_discovered_groups(state: RunState | None, groups: Iterable[ExecutionGroup]) -> None:
    if state is None:
        return
    state.data["discovered_groups"] = {
        group.name: {
            "candidate_imports": sorted(group.candidate_imports),
            "missing_imports": group.missing_imports,
            "missing_core_imports": group.missing_core_imports,
            "requirements": group.requirements,
        }
        for group in groups
    }
    write_run_state(state)


def hydrate_discovered_groups(state: RunState | None, groups: Iterable[ExecutionGroup]) -> bool:
    if state is None:
        return False
    saved = state.data.get("discovered_groups") or {}
    groups = list(groups)
    if not all(group.name in saved for group in groups):
        return False
    for group in groups:
        item = saved[group.name]
        group.candidate_imports = set(item.get("candidate_imports") or [])
        group.missing_imports = list(item.get("missing_imports") or [])
        group.missing_core_imports = list(item.get("missing_core_imports") or [])
        group.requirements = list(item.get("requirements") or [])
    return True


def complete_run_state(state: RunState | None, *, manifest_path: Path) -> None:
    if state is None:
        return
    state.data.pop("current_action", None)
    state.data["manifest"] = str(manifest_path)
    state.data["completed_at"] = timestamp()
    state.done_path.write_text(yaml.safe_dump(state.data, sort_keys=False), encoding="utf-8")
    state.path.unlink(missing_ok=True)
    print(f"[airgap] complete state={state.done_path}")


def normalize_stages(stages: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in stages:
        for item in str(raw).split(","):
            stage = item.strip()
            if stage and stage not in out:
                out.append(stage)
    out = out or [
        "validate",
        "discover-execution-deps",
        "build-launcher-image",
        "build-execution-images",
        "save-images",
    ]

    def ensure_before(required: str, requested: str) -> None:
        if requested not in out or required in out:
            return
        index = out.index(requested)
        out.insert(index, required)
        print(f"[airgap] auto-adding stage {required!r} because {requested!r} was requested")

    # Apply prerequisite edges from later stages toward earlier stages. Each
    # insertion is idempotent, so a user can ask for any suffix of the pipeline.
    ensure_before("build-execution-images", "save-images")
    ensure_before("build-launcher-image", "save-images")
    ensure_before("discover-execution-deps", "build-execution-images")
    ensure_before("validate", "discover-execution-deps")
    ensure_before("validate", "build-execution-images")
    ensure_before("validate", "save-images")
    order = {
        "validate": 0,
        "discover-execution-deps": 1,
        "build-launcher-image": 2,
        "build-execution-images": 3,
        "save-images": 4,
    }
    out.sort(key=lambda stage: order.get(stage, len(order)))
    return out


def stage_needs_targets(stage: str) -> bool:
    return stage in {"discover-execution-deps", "build-execution-images", "save-images"}


def expand_targets(cfg: Mapping[str, Any]) -> list[Target]:
    workflow = cfg.get("workflow") or {}
    raw_targets = [parse_target(item) for item in workflow.get("stages") or []]
    deps = cfg.get("dependencies") or workflow.get("dependencies") or {}
    out: list[Target] = []
    seen: set[str] = set()
    visiting: set[str] = set()
    stack: list[str] = []

    def add(target: Target) -> None:
        if target.spec in visiting:
            start = stack.index(target.spec) if target.spec in stack else 0
            cycle = " -> ".join([*stack[start:], target.spec])
            raise SystemExit(f"cyclic airgap dependency detected: {cycle}")
        if target.spec in seen:
            return
        visiting.add(target.spec)
        stack.append(target.spec)
        for dep in deps.get(target.step_id, []) or []:
            add(parse_target(dep))
        stack.pop()
        visiting.remove(target.spec)
        seen.add(target.spec)
        out.append(target)

    for target in raw_targets:
        add(target)
    if not out:
        raise SystemExit("workflow.stages must list at least one step")
    return out


def parse_target(value: str) -> Target:
    step_id, sep, config = str(value).partition(":")
    step_id = step_id.strip()
    config = config.strip() if sep else ""
    if not step_id:
        raise SystemExit(f"invalid target {value!r}; expected step-id[:config]")
    return Target(step_id=step_id, config=config or None)


def validate_targets(targets: Iterable[Target]) -> dict[str, StepInfo]:
    out: dict[str, StepInfo] = {}
    for target in targets:
        step_dir = STEP_ROOT / target.step_id
        step_py = step_dir / "step.py"
        step_toml = step_dir / "step.toml"
        config_path = step_dir / "config" / f"{target.config}.yaml" if target.config else None
        missing = [
            path for path in (step_dir, step_py, step_toml, config_path) if path is not None and not path.exists()
        ]
        if missing:
            raise SystemExit(f"{target.spec}: missing required path(s): {', '.join(str(path) for path in missing)}")
        module = "nemotron.steps." + target.step_id.replace("/", ".") + ".step"
        out[target.step_id] = StepInfo(
            target=target,
            step_dir=step_dir,
            step_py=step_py,
            step_toml=step_toml,
            config_path=config_path,
            module=module,
            mounts=read_config_mounts(config_path),
            repo_overlays=read_config_repo_overlays(config_path),
        )
    return out


def read_config_mounts(config_path: Path | None) -> list[Any]:
    if config_path is None or not config_path.exists():
        return []
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    if not isinstance(data, Mapping):
        return []
    run = data.get("run") if isinstance(data.get("run"), Mapping) else {}
    env = run.get("env") if isinstance(run.get("env"), Mapping) else {}
    mounts = env.get("mounts") if isinstance(env, Mapping) else []
    return mounts if isinstance(mounts, list) else []


def execution_groups(
    cfg: Mapping[str, Any],
    *,
    output_dir: Path,
    step_infos: Mapping[str, StepInfo] | None = None,
) -> list[ExecutionGroup]:
    if not step_infos:
        raise SystemExit("validate must run before execution images can be planned")
    if not cfg.get("step_execution_images"):
        raise SystemExit("airgap.yaml must define step_execution_images for the selected workflow stages")
    return execution_groups_from_step_execution_images(cfg, output_dir=output_dir, step_infos=step_infos)


def execution_groups_from_step_execution_images(
    cfg: Mapping[str, Any],
    *,
    output_dir: Path,
    step_infos: Mapping[str, StepInfo],
) -> list[ExecutionGroup]:
    step_execution_images = normalize_step_execution_images(cfg.get("step_execution_images") or {})
    image_defs = normalize_execution_images(cfg.get("execution_images") or {})
    merged: dict[str, ExecutionGroup] = {}

    for step_id in step_infos:
        image_name = step_execution_images.get(step_id)
        if not image_name:
            raise SystemExit(f"{step_id}: missing step_execution_images entry in airgap.yaml")
        image_def = image_defs.get(image_name)
        if image_def is None:
            raise SystemExit(f"{step_id}: step_execution_images points to unknown execution image {image_name!r}")
        base = str(image_def.get("base_image") or "").strip()
        if not base:
            raise SystemExit(f"execution_images.{image_name}.base_image is required")
        repo_overlays = getattr(step_infos[step_id], "repo_overlays", [])
        group_key = execution_group_key(base, repo_overlays)
        group = merged.get(group_key)
        if group is None:
            suffix = short_hash(
                {
                    "base_image": base,
                    "repo_overlays": [repo_overlay_manifest(item) for item in repo_overlays],
                }
            )
            group = ExecutionGroup(
                name=f"{image_name}-{suffix}",
                base_image=base,
                tag="",
                tar=output_dir / "execution-image.tar",
                steps=[],
                platform=str(image_def["platform"]) if image_def.get("platform") else None,
                pip_no_deps=bool(image_def.get("pip_no_deps", True)),
                repo_overlays=list(repo_overlays),
            )
            merged[group_key] = group
        group.image_names.add(image_name)
        group.steps.append(step_id)
        group.required_imports.update(str(name) for name in image_def.get("required_imports") or [])
        group.repo_overlays = merge_repo_overlays(
            group.repo_overlays,
            repo_overlays,
        )
    for group in merged.values():
        finalize_execution_group_name(group, image_defs=image_defs, output_dir=output_dir)
    return list(merged.values())


def finalize_execution_group_name(
    group: ExecutionGroup,
    *,
    image_defs: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> None:
    names = sorted(group.image_names)
    suffix = short_hash(
        {
            "base_image": group.base_image,
            "repo_overlays": [repo_overlay_manifest(item) for item in group.repo_overlays],
        }
    )
    if len(names) == 1:
        image_name = names[0]
        image_def = image_defs[image_name]
        tag = str(image_def.get("tag") or f"nemotron-execution-{sanitize(image_name)}:airgap")
        tar = output_dir / str(image_def.get("tar") or f"execution-{sanitize(image_name)}.tar")
        group.name = f"{image_name}-{suffix}"
    else:
        merged_name = "-".join(sanitize(name) for name in names)
        tag = f"nemotron-customizer-{merged_name}-airgap:latest"
        tar = output_dir / f"execution-{merged_name}-image.tar"
        group.name = f"{merged_name}-{suffix}"
    group.tag = tag_with_suffix(tag, suffix)
    group.tar = tar_with_suffix(tar, suffix)
    group.selected_image = group.tag


def execution_group_key(base_image: str, repo_overlays: Iterable[RepoOverlay]) -> str:
    overlays = sorted(
        (repo_overlay_manifest(item) for item in repo_overlays),
        key=lambda item: (item["target"], item["url"], item["ref"], item["repo"]),
    )
    payload = {
        "base_image": base_image,
        "repo_overlays": overlays,
    }
    return json.dumps(payload, sort_keys=True)


def short_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def tag_with_suffix(tag: str, suffix: str) -> str:
    image, separator, digest = tag.partition("@")
    last = image.rsplit("/", 1)[-1]
    if ":" in last:
        name, version = image.rsplit(":", 1)
        image = f"{name}-{suffix}:{version}"
    else:
        image = f"{image}-{suffix}"
    return f"{image}{separator}{digest}" if separator else image


def tar_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}-{suffix}{path.suffix}")


def normalize_step_execution_images(raw: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for step_id, value in raw.items():
        if isinstance(value, str):
            out[str(step_id)] = value
        elif isinstance(value, Mapping) and value.get("execution_image"):
            out[str(step_id)] = str(value["execution_image"])
    return out


def normalize_execution_images(raw: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        return {str(name): spec for name, spec in raw.items() if isinstance(spec, Mapping)}
    return {}


def read_config_repo_overlays(config_path: Path | None) -> list[RepoOverlay]:
    if config_path is None or not config_path.exists():
        return []
    text = config_path.read_text(encoding="utf-8")
    overlays: list[RepoOverlay] = []
    pattern = re.compile(r"\$\{auto_mount:(git\+[^,}]+),([^}]+)\}")
    for spec, target in pattern.findall(text):
        overlays.append(parse_git_overlay(spec, target))
    return merge_repo_overlays([], overlays)


def parse_git_overlay(spec: str, target: str) -> RepoOverlay:
    if not spec.startswith("git+"):
        raise SystemExit(f"invalid auto_mount git spec: {spec!r}")
    url_and_ref = spec[4:]
    if "@" not in url_and_ref:
        raise SystemExit(f"invalid auto_mount git spec missing @ref: {spec!r}")
    url, ref = url_and_ref.rsplit("@", 1)
    repo = url.rstrip("/").split("/")[-1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return RepoOverlay(repo=repo, url=url, ref=ref, target=target.strip())


def merge_repo_overlays(existing: list[RepoOverlay], incoming: Iterable[RepoOverlay]) -> list[RepoOverlay]:
    out = list(existing)
    seen = {(item.repo, item.url, item.ref, item.target) for item in out}
    for item in incoming:
        key = (item.repo, item.url, item.ref, item.target)
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def discover_execution_deps(
    group: ExecutionGroup,
    *,
    step_infos: Mapping[str, StepInfo],
    locked_versions: Mapping[str, str],
    execute: bool,
) -> None:
    imports: set[str] = set(group.required_imports)
    for step_id in group.steps:
        imports.update(discover_external_imports(step_infos[step_id].step_py))
    group.candidate_imports = imports
    if execute:
        missing = probe_step_modules(
            group.base_image,
            [step_infos[step_id].module for step_id in group.steps],
            required_imports=imports,
            locked_versions=locked_versions,
            pip_no_deps=group.pip_no_deps,
            platform=group.platform,
        )
    else:
        missing = probe_missing_imports(group.base_image, sorted(imports), execute=False, platform=group.platform)
    group.missing_imports = sorted(set(missing))
    group.missing_core_imports = sorted(name for name in missing if name.split(".", 1)[0] in CORE_IMPORTS)
    installable = sorted(name for name in group.missing_imports if name not in group.missing_core_imports)
    group.requirements = sorted(requirement_for_import(name, locked_versions) for name in installable)


def discover_external_imports(start: Path) -> set[str]:
    external: set[str] = set()
    try:
        tree = ast.parse(start.read_text(encoding="utf-8"))
    except SyntaxError:
        return external
    for node in ast.walk(tree):
        imported: list[str] = []
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imported = [node.module]
        for name in imported:
            root = name.split(".", 1)[0]
            if root in LOCAL_PREFIXES or is_stdlib(root):
                continue
            external.add(root)
    return external


def is_stdlib(root: str) -> bool:
    if root in sys.builtin_module_names:
        return True
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    if root in stdlib_names:
        return True
    return False


def probe_missing_imports(image: str, imports: list[str], *, execute: bool, platform: str | None = None) -> list[str]:
    if not imports:
        return []
    code = (
        "import importlib.util,json;"
        f"mods={imports!r};"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "print(json.dumps(missing))"
    )
    cmd = ["docker", "run", "--rm", "--pull", "never"]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.extend([image, "python", "-c", code])
    if not execute:
        print_cmd(cmd)
        return []
    ensure_image(image, platform=platform)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        raise SystemExit(result.returncode)
    return [str(item) for item in json.loads(result.stdout.strip() or "[]")]


def probe_step_modules(
    image: str,
    modules: list[str],
    *,
    required_imports: Iterable[str],
    locked_versions: Mapping[str, str],
    pip_no_deps: bool,
    platform: str | None = None,
) -> list[str]:
    """Import selected step modules in the execution image and discover missing imports.

    The loop installs only the packages it has already identified, in an
    ephemeral container, so the final requirements file stays based on actual
    import failures rather than broad static guesses.
    """

    ensure_image(image, platform=platform)
    missing: list[str] = []
    requirements: list[str] = []
    imports = sorted(set(required_imports))
    import_code = "import importlib;"
    import_code += "".join(f"importlib.import_module({module!r});" for module in imports)
    import_code += "".join(f"importlib.import_module({module!r});" for module in modules)
    for _ in range(20):
        install = ""
        if requirements:
            no_deps = "--no-deps " if pip_no_deps else ""
            install = "python -m pip install " + no_deps
            install += " ".join(shlex_quote(req) for req in requirements)
            install += (
                " >/tmp/nemotron-airgap-pip.log 2>&1 "
                "|| { echo '[airgap-pip] failed:'; cat /tmp/nemotron-airgap-pip.log; exit 1; } && "
            )
        cmd = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--mount",
            f"type=volume,source={pip_cache_volume(platform)},target=/root/.cache/pip",
            "-v",
            f"{REPO_ROOT}:/workspace/Nemotron:ro",
            "-w",
            "/workspace/Nemotron",
            "-e",
            "PYTHONPATH=/workspace/Nemotron/src",
        ]
        if platform:
            cmd.extend(["--platform", platform])
        cmd.extend([image, "bash", "-lc", install + "python -c " + shlex_quote(import_code)])
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode == 0:
            return missing
        text = result.stderr + "\n" + result.stdout
        match = re.search(r"(?:ModuleNotFoundError|ImportError):\s+No module named ['\"]([^'\"]+)['\"]", text)
        if not match:
            print(text, file=sys.stderr)
            raise SystemExit(result.returncode)
        import_name = match.group(1).split(".", 1)[0]
        if import_name not in missing:
            missing.append(import_name)
        if import_name in CORE_IMPORTS:
            print(f"[probe] base image is missing core import {import_name!r}; choose a compatible execution image")
            return missing
        requirement = requirement_for_import(import_name, locked_versions)
        if requirement in requirements:
            return missing
        requirements.append(requirement)
    raise SystemExit(f"import probe did not converge for {image}")


def requirement_for_import(import_name: str, locked_versions: Mapping[str, str]) -> str:
    package = package_for_import(import_name)
    version = locked_versions.get(normalize_package(package))
    return f"{package}=={version}" if version else package


def package_for_import(import_name: str) -> str:
    if import_name in IMPORT_ALIASES:
        return IMPORT_ALIASES[import_name]
    packages = metadata.packages_distributions().get(import_name)
    if packages:
        return normalize_package(packages[0])
    return import_name.replace("_", "-")


def locked_package_versions(lock_path: Path) -> dict[str, str]:
    if not lock_path.exists():
        return {}
    data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    versions: dict[str, str] = {}
    for package in data.get("package", []) or []:
        name = package.get("name")
        version = package.get("version")
        if isinstance(name, str) and isinstance(version, str):
            versions[normalize_package(name)] = version
    return versions


def normalize_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def build_launcher_image(launcher_image: Mapping[str, Any], *, execute: bool) -> int:
    image = str(launcher_image.get("tag") or "nemotron-customizer-launcher-airgap:latest")
    base = str(launcher_image.get("base_image") or "python:3.12-slim")
    platform = launcher_image_platform(launcher_image)
    cmd = [
        "docker",
        "build",
        "-f",
        str(AIRGAP_DIR / "Dockerfile.launcher"),
        "--build-arg",
        f"BASE_IMAGE={base}",
        "--build-arg",
        f"UV_VERSION={UV_VERSION}",
        "-t",
        image,
        ".",
    ]
    if platform:
        cmd[2:2] = ["--platform", platform]
    if execute:
        ensure_image(base, platform=platform)
    return run_or_print(cmd, execute)


def launcher_image_platform(launcher_image: Mapping[str, Any]) -> str | None:
    return str(launcher_image["platform"]) if launcher_image.get("platform") else None


def build_execution_image(group: ExecutionGroup, *, output_dir: Path, execute: bool) -> int:
    group_dir = output_dir / "execution-context" / group.name
    group_dir.mkdir(parents=True, exist_ok=True)
    group.requirements_path = group_dir / f"requirements-{group.name}.txt"
    group.requirements_path.write_text(
        "\n".join(group.requirements) + ("\n" if group.requirements else ""),
        encoding="utf-8",
    )
    repos_root = output_dir / "repo-overlays" / group.name
    prepare_repo_overlays(group, repos_root=repos_root, execute=execute)
    group.repo_overlays_path = group_dir / f"repo-overlays-{group.name}.json"
    group.repo_overlays_path.write_text(
        json.dumps([repo_overlay_build_manifest(item) for item in group.repo_overlays], indent=2) + "\n",
        encoding="utf-8",
    )
    cmd = [
        "docker",
        "build",
        "-f",
        str(AIRGAP_DIR / "Dockerfile.execution"),
        "--build-arg",
        f"BASE_IMAGE={group.base_image}",
        "--build-arg",
        f"EXECUTION_REQUIREMENTS={docker_context_path(group.requirements_path)}",
        "--build-arg",
        f"REPO_OVERLAYS={docker_context_path(group.repo_overlays_path)}",
        "--build-arg",
        f"REPO_OVERLAYS_DIR={docker_context_path(repos_root)}",
        "--build-arg",
        f"PIP_NO_DEPS={'true' if group.pip_no_deps else 'false'}",
        "-t",
        group.tag,
        ".",
    ]
    if group.platform:
        cmd[2:2] = ["--platform", group.platform]
    if execute:
        ensure_image(group.base_image, platform=group.platform)
    return run_or_print(cmd, execute)


def prepare_repo_overlays(group: ExecutionGroup, *, repos_root: Path, execute: bool) -> None:
    repos_root.mkdir(parents=True, exist_ok=True)
    (repos_root / ".keep").touch()
    for overlay in group.repo_overlays:
        dest = repos_root / repo_overlay_dir_name(overlay)
        if dest.exists():
            run_or_print(["git", "-C", str(dest), "fetch", "--all", "--tags", "--force", "--prune"], execute)
        else:
            run_or_print(["git", "clone", overlay.url, str(dest)], execute)
        run_or_print(["git", "-C", str(dest), "checkout", overlay.ref], execute)


def save_image(image: str, output: Path, execute: bool) -> int:
    return run_or_print(["docker", "save", "-o", str(output), image], execute, mkdir=output.parent)


def ensure_image(image: str, *, platform: str | None = None) -> None:
    if docker_image_exists(image, platform=platform):
        return
    suffix = f" for {platform}" if platform else ""
    print(f"[docker] pulling missing base image{suffix}: {image}")
    cmd = ["docker", "pull"]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.append(image)
    result = subprocess.run(cmd, check=False, cwd=REPO_ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)


def docker_image_exists(image: str, *, platform: str | None = None) -> bool:
    cached = docker_image_platform(image)
    return cached is not None and platform_matches(cached, platform)


def docker_image_platform(image: str) -> str | None:
    inspect = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Os}}/{{.Architecture}}{{if .Variant}}/{{.Variant}}{{end}}",
            image,
        ],
        stdout=subprocess.PIPE,
        text=True,
        stderr=subprocess.DEVNULL,
        cwd=REPO_ROOT,
    )
    if inspect.returncode != 0:
        return None
    return (inspect.stdout.strip().splitlines() or [None])[0]


def platform_matches(cached: str | None, requested: str | None) -> bool:
    if cached is None:
        return False
    if not requested:
        return True
    return cached == requested or cached.startswith(f"{requested}/")


def pip_cache_volume(platform: str | None = None) -> str:
    suffix = sanitize(platform or "default")
    return f"nemotron-airgap-pip-cache-{suffix}"


def run_or_print(cmd: list[str], execute: bool, *, mkdir: Path | None = None) -> int:
    print_cmd(cmd)
    if not execute:
        return 0
    if mkdir is not None:
        mkdir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, check=False, cwd=REPO_ROOT).returncode


def clean_stale_group_dirs(output_dir: Path, groups: Iterable[ExecutionGroup], *, execute: bool) -> None:
    keep = {group.name for group in groups}
    for relative in ("execution-context", "repo-overlays"):
        parent = output_dir / relative
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if not child.is_dir() or child.name in keep:
                continue
            if execute:
                shutil.rmtree(child)
                print(f"[clean] removed stale {child}")
            else:
                print_cmd(["rm", "-rf", str(child)])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_image_manifest(
    image: str,
    output: Path,
    *,
    execute: bool,
    role: str,
    name: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "name": name,
        "image": image,
        "tar": str(output),
        "sha256": sha256_file(output) if execute and output.exists() else None,
    }


def print_cmd(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex_quote(part) for part in cmd))


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def collect_mounts(infos: Iterable[StepInfo]) -> list[Any]:
    out: list[Any] = []
    for info in infos:
        out.extend(info.mounts)
    return out


def collect_repo_overlays(infos: Iterable[StepInfo]) -> list[RepoOverlay]:
    out: list[RepoOverlay] = []
    for info in infos:
        out = merge_repo_overlays(out, info.repo_overlays)
    return out


def repo_overlay_manifest(item: RepoOverlay) -> dict[str, str]:
    return {
        "repo": item.repo,
        "url": item.url,
        "ref": item.ref,
        "target": item.target,
    }


def repo_overlay_build_manifest(item: RepoOverlay) -> dict[str, str]:
    data = repo_overlay_manifest(item)
    data["source"] = repo_overlay_dir_name(item)
    return data


def repo_overlay_dir_name(item: RepoOverlay) -> str:
    return f"{sanitize(item.repo)}-{short_hash(repo_overlay_manifest(item))}"


def step_to_manifest(info: StepInfo) -> dict[str, Any]:
    return {
        "target": info.target.spec,
        "step_py": str(info.step_py.relative_to(REPO_ROOT)),
        "step_toml": str(info.step_toml.relative_to(REPO_ROOT)),
        "config": str(info.config_path.relative_to(REPO_ROOT)) if info.config_path else None,
        "module": info.module,
    }


def execution_group_manifest(group: ExecutionGroup) -> dict[str, Any]:
    return {
        "name": group.name,
        "image_names": sorted(group.image_names),
        "base_image": group.base_image,
        "platform": group.platform,
        "tag": group.tag,
        "selected_image": group.selected_image or group.tag,
        "tar": str(group.tar),
        "steps": group.steps,
        "pip_no_deps": group.pip_no_deps,
        "candidate_imports": sorted(group.candidate_imports),
        "missing_imports": group.missing_imports,
        "missing_core_imports": group.missing_core_imports,
        "requirements": group.requirements,
        "requirements_path": str(group.requirements_path) if group.requirements_path else None,
        "repo_overlays": [repo_overlay_manifest(item) for item in group.repo_overlays],
        "repo_overlays_path": str(group.repo_overlays_path) if group.repo_overlays_path else None,
    }


def step_execution_image_manifest(groups: Iterable[ExecutionGroup]) -> dict[str, str]:
    out: dict[str, str] = {}
    for group in groups:
        image = group.selected_image or group.tag
        for step_id in group.steps:
            out[step_id] = image
    return out


def launcher_image_manifest(launcher_image: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_image": launcher_image.get("base_image") or "python:3.12-slim",
        "platform": launcher_image.get("platform"),
        "tag": launcher_image.get("tag") or "nemotron-customizer-launcher-airgap:latest",
        "tar": launcher_image.get("tar") or "launcher-image.tar",
    }


def sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower() or "image"


if __name__ == "__main__":
    raise SystemExit(main())
