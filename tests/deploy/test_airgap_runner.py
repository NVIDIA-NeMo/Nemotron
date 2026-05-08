from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_runspec.config.loader import load_config


def _runner_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "deploy/nemotron-customizer/airgap/runner.py"
    spec = importlib.util.spec_from_file_location("airgap_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_airgap_runner_expands_and_validates_sft_dependency():
    runner = _runner_module()
    cfg = {
        "workflow": {"stages": ["sft/megatron_bridge:tiny"]},
        "dependencies": {"sft/megatron_bridge": ["prep/sft_packing:tiny"]},
    }

    targets = runner.expand_targets(cfg)
    infos = runner.validate_targets(targets)

    assert [target.spec for target in targets] == ["prep/sft_packing:tiny", "sft/megatron_bridge:tiny"]
    assert infos["sft/megatron_bridge"].module == "nemotron.steps.sft.megatron_bridge.step"
    assert infos["prep/sft_packing"].config_path.name == "tiny.yaml"
    assert [item.target for item in infos["sft/megatron_bridge"].repo_overlays] == [
        "/opt/megatron-lm",
        "/opt/Megatron-Bridge",
    ]


def test_airgap_runner_groups_task_images_by_base_image_and_repo_overlays(tmp_path):
    runner = _runner_module()
    overlay = runner.RepoOverlay(
        repo="Megatron-Bridge",
        url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
        ref="main",
        target="/opt/Megatron-Bridge",
    )
    cfg = {
        "step_images": {
            "prep/sft_packing": "a",
            "sft/megatron_bridge": "b",
        },
        "task_images": {
            "a": {
                "base_image": "image:base",
                "tag": "image:a",
                "tar": "a.tar",
                "required_imports": ["omegaconf"],
            },
            "b": {
                "base_image": "image:base",
                "tag": "image:b",
                "tar": "b.tar",
                "required_imports": ["yaml"],
            },
        },
    }

    groups = runner.task_groups(
        cfg,
        output_dir=tmp_path,
        step_infos={
            "prep/sft_packing": runner.StepInfo(
                target=runner.Target("prep/sft_packing"),
                step_dir=tmp_path,
                step_py=tmp_path / "step.py",
                step_toml=tmp_path / "step.toml",
                config_path=None,
                module="x",
            ),
            "sft/megatron_bridge": runner.StepInfo(
                target=runner.Target("sft/megatron_bridge"),
                step_dir=tmp_path,
                step_py=tmp_path / "step.py",
                step_toml=tmp_path / "step.toml",
                config_path=None,
                module="y",
                repo_overlays=[overlay],
            ),
        },
    )

    assert len(groups) == 2
    by_step = {group.steps[0]: group for group in groups}
    assert by_step["prep/sft_packing"].base_image == "image:base"
    assert by_step["prep/sft_packing"].required_imports == {"omegaconf"}
    assert by_step["prep/sft_packing"].repo_overlays == []
    assert by_step["sft/megatron_bridge"].base_image == "image:base"
    assert by_step["sft/megatron_bridge"].required_imports == {"yaml"}
    assert by_step["sft/megatron_bridge"].repo_overlays == [overlay]
    assert len({group.tag for group in groups}) == 2


def test_airgap_runner_only_builds_images_for_selected_steps(tmp_path):
    runner = _runner_module()
    cfg = {
        "step_images": {
            "prep/sft_packing": "nemo-megatron",
            "sft/automodel": "nemo-automodel",
        },
        "task_images": {
            "nemo-megatron": {"base_image": "nemo:base"},
            "nemo-automodel": {"base_image": "automodel:base"},
        },
    }

    groups = runner.task_groups(cfg, output_dir=tmp_path, step_infos={"prep/sft_packing": object()})

    assert len(groups) == 1
    assert groups[0].name.startswith("nemo-megatron-")
    assert groups[0].steps == ["prep/sft_packing"]


def test_airgap_runner_maps_sdg_to_light_sdk_image(tmp_path):
    runner = _runner_module()
    cfg = runner.load_yaml(runner.AIRGAP_DIR / "airgap.yaml")
    cfg["workflow"]["stages"] = ["sdg/data_designer:tiny"]

    targets = runner.expand_targets(cfg)
    infos = runner.validate_targets(targets)
    groups = runner.task_groups(cfg, output_dir=tmp_path, step_infos=infos)

    assert [target.spec for target in targets] == ["sdg/data_designer:tiny"]
    assert len(groups) == 1
    assert groups[0].name.startswith("nemo-data-designer-")
    assert groups[0].base_image == "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
    assert "data_designer" in groups[0].required_imports


def test_airgap_runner_maps_byob_to_data_designer_image(tmp_path):
    runner = _runner_module()
    cfg = runner.load_yaml(runner.AIRGAP_DIR / "airgap.yaml")
    cfg["workflow"]["stages"] = ["byob:tiny"]

    targets = runner.expand_targets(cfg)
    infos = runner.validate_targets(targets)
    groups = runner.task_groups(cfg, output_dir=tmp_path, step_infos=infos)

    assert [target.spec for target in targets] == ["byob:tiny"]
    assert len(groups) == 1
    assert groups[0].name.startswith("nemo-data-designer-")
    assert "data_designer" in groups[0].required_imports


def test_airgap_runner_target_override_selects_sdg_and_sft():
    runner = _runner_module()
    cfg = runner.load_yaml(runner.AIRGAP_DIR / "airgap.yaml")
    cfg = runner.with_workflow_targets(
        cfg,
        runner.normalize_target_specs(["sdg/data_designer:tiny", "sft/megatron_bridge:tiny"]),
    )

    targets = runner.expand_targets(cfg)
    infos = runner.validate_targets(targets)
    groups = runner.task_groups(cfg, output_dir=runner.AIRGAP_DIR / "out", step_infos=infos)

    assert [target.spec for target in targets] == [
        "sdg/data_designer:tiny",
        "prep/sft_packing:tiny",
        "sft/megatron_bridge:tiny",
    ]
    by_steps = {tuple(group.steps): group for group in groups}
    merged = by_steps[("sdg/data_designer", "prep/sft_packing")]
    assert merged.image_names == {"nemo-data-designer", "nemo-megatron"}
    assert merged.tag.startswith("nemotron-customizer-nemo-data-designer-nemo-megatron-airgap-")


def test_airgap_runner_splits_same_base_image_when_repo_overlays_differ(tmp_path):
    runner = _runner_module()
    overlay = runner.RepoOverlay(
        repo="Megatron-Bridge",
        url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
        ref="feature",
        target="/opt/Megatron-Bridge",
    )
    cfg = {
        "step_images": {
            "prep/sft_packing": "nemo-megatron",
            "sft/megatron_bridge": "nemo-megatron",
        },
        "task_images": {
            "nemo-megatron": {
                "base_image": "nemo:base",
                "tag": "nemo-airgap:latest",
                "tar": "nemo-airgap.tar",
            },
        },
    }
    groups = runner.task_groups(
        cfg,
        output_dir=tmp_path,
        step_infos={
            "prep/sft_packing": runner.StepInfo(
                target=runner.Target("prep/sft_packing"),
                step_dir=tmp_path,
                step_py=tmp_path / "step.py",
                step_toml=tmp_path / "step.toml",
                config_path=None,
                module="x",
            ),
            "sft/megatron_bridge": runner.StepInfo(
                target=runner.Target("sft/megatron_bridge"),
                step_dir=tmp_path,
                step_py=tmp_path / "step.py",
                step_toml=tmp_path / "step.toml",
                config_path=None,
                module="y",
                repo_overlays=[overlay],
            ),
        },
    )

    assert len(groups) == 2
    assert sorted([group.steps for group in groups]) == [["prep/sft_packing"], ["sft/megatron_bridge"]]
    assert len({group.tag for group in groups}) == 2


def test_airgap_runner_uses_collision_safe_repo_overlay_dirs():
    runner = _runner_module()
    first = runner.RepoOverlay(
        repo="Megatron-Bridge",
        url="https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
        ref="main",
        target="/opt/Megatron-Bridge",
    )
    second = runner.RepoOverlay(
        repo="Megatron-Bridge",
        url="https://github.com/example/Megatron-Bridge.git",
        ref="main",
        target="/opt/Other-Bridge",
    )

    assert runner.repo_overlay_dir_name(first) != runner.repo_overlay_dir_name(second)
    assert runner.repo_overlay_build_manifest(first)["source"] == runner.repo_overlay_dir_name(first)


def test_airgap_runner_auto_adds_stage_prerequisites():
    runner = _runner_module()

    assert runner.normalize_stages(["build-task-images"]) == [
        "validate",
        "discover-task-deps",
        "build-task-images",
    ]
    assert runner.normalize_stages(["save-images"]) == [
        "validate",
        "discover-task-deps",
        "build-submitter",
        "build-task-images",
        "save-images",
    ]


def test_airgap_runner_rejects_build_output_outside_docker_context(tmp_path):
    runner = _runner_module()

    with pytest.raises(SystemExit, match="paths.output_dir=.*must live under the repo root"):
        runner.validate_docker_context_path(tmp_path, field="paths.output_dir")


def test_airgap_runner_reports_dependency_cycles():
    runner = _runner_module()
    cfg = {
        "workflow": {"stages": ["a/b"]},
        "dependencies": {
            "a/b": ["c/d"],
            "c/d": ["a/b"],
        },
    }

    with pytest.raises(SystemExit, match=r"cyclic airgap dependency detected: a/b -> c/d -> a/b"):
        runner.expand_targets(cfg)


def test_airgap_runner_tag_suffix_handles_ports_and_digests():
    runner = _runner_module()

    assert runner.tag_with_suffix("registry:5000/team/image:latest", "abc123") == (
        "registry:5000/team/image-abc123:latest"
    )
    assert runner.tag_with_suffix("repo/image:latest@sha256:deadbeef", "abc123") == (
        "repo/image-abc123:latest@sha256:deadbeef"
    )
    assert runner.tag_with_suffix("repo/image@sha256:deadbeef", "abc123") == "repo/image-abc123@sha256:deadbeef"


def test_airgap_runner_saved_image_manifest_has_checksum(tmp_path):
    runner = _runner_module()
    image_tar = tmp_path / "image.tar"
    image_tar.write_text("image bytes", encoding="utf-8")

    saved = runner.saved_image_manifest("image:tag", image_tar, execute=True, role="task", name="group")

    assert saved["role"] == "task"
    assert saved["name"] == "group"
    assert saved["image"] == "image:tag"
    assert saved["tar"] == str(image_tar)
    assert saved["sha256"] == runner.sha256_file(image_tar)


def test_airgap_runner_platform_matching_accepts_variant_only_when_compatible():
    runner = _runner_module()

    assert runner.platform_matches("linux/amd64", "linux/amd64")
    assert runner.platform_matches("linux/arm64/v8", "linux/arm64")
    assert not runner.platform_matches("linux/amd64", "linux/arm64")
    assert runner.pip_cache_volume("linux/amd64") == "nemotron-airgap-pip-cache-linux-amd64"


def test_airgap_runner_progress_state_resumes_and_completes(tmp_path):
    runner = _runner_module()
    cfg = {"workflow": {"stages": ["byob:tiny"]}}
    config_path = tmp_path / "airgap.yaml"
    stages = ["validate"]

    state = runner.load_or_start_run_state(
        tmp_path,
        config_path=config_path,
        cfg=cfg,
        stages=stages,
        execute=True,
    )
    assert state is not None
    runner.begin_action(state, "validate")
    assert state.path.exists()
    assert not state.done_path.exists()

    runner.complete_action(state, "validate", {"targets": ["byob:tiny"]})
    resumed = runner.load_or_start_run_state(
        tmp_path,
        config_path=config_path,
        cfg=cfg,
        stages=stages,
        execute=True,
    )
    assert runner.action_completed(resumed, "validate")

    manifest = tmp_path / "airgap-manifest.yaml"
    manifest.write_text("schema_version: 1\n", encoding="utf-8")
    runner.complete_run_state(resumed, manifest_path=manifest)

    assert not state.path.exists()
    assert state.done_path.exists()


def test_airgap_runner_static_import_scan_stays_direct():
    runner = _runner_module()
    step_py = runner.STEP_ROOT / "prep/sft_packing/step.py"

    imports = runner.discover_external_imports(step_py)

    assert "omegaconf" in imports
    assert "cosmos_xenna" not in imports


def test_sft_airgap_overlay_clears_auto_mounts_but_inherits_config():
    runner = _runner_module()
    config = load_config(runner.AIRGAP_DIR / "configs/sft_megatron_bridge_tiny.yaml")
    plain = OmegaConf.to_container(config, resolve=False)

    assert plain["run"]["env"]["mounts"] == []
    assert plain["hf_model_path"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    assert plain["dataset"]["packed_sequence_specs"]["packed_sequence_size"] == 4096
