"""Contract tests for the super35 RL stage configs and shared runner.

Each RL config is a port of one stage of the Nemotron 3.5 Super reference
run. These tests pin the invariants the port must keep: artifact wiring,
the shared MTP setup, MOPD teacher routing and a public-repo hygiene check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_runspec import parse as parse_runspec
from nemotron.recipes.super35.stage2_rl._nemo_rl import is_megatron_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
RL_ROOT = REPO_ROOT / "src/nemotron/recipes/super35/stage2_rl"
RL_SCRIPTS = sorted(RL_ROOT.glob("**/train.py"))
RL_CONFIGS = sorted(p for p in RL_ROOT.glob("**/config/*.yaml"))
MOPD_CONFIGS = sorted((RL_ROOT / "stage3_mopd/config").glob("*.yaml"))

# Internal references that must never land in the public repo.
_INTERNAL_PATTERNS = re.compile(r"/lustre|gitlab-master|wandb\.ai/|/home/(?!\$\{oc\.env:USER\})|nvidian/")


def _load(path: Path):
    return OmegaConf.load(path)


def _ids(paths):
    return [str(p.relative_to(RL_ROOT)) for p in paths]


def test_every_stage_has_a_script_and_config():
    stages = {p.parent.relative_to(RL_ROOT).as_posix() for p in RL_SCRIPTS}
    assert stages == {
        "stage1_rlvr",
        "stage2_teachers/reasoning",
        "stage2_teachers/vision",
        "stage2_teachers/video",
        "stage2_teachers/vision_reasoning",
        "stage2_teachers/rlhf",
        "stage2_teachers/agentic",
        "stage3_mopd",
    }


@pytest.mark.parametrize("script", RL_SCRIPTS, ids=_ids(RL_SCRIPTS))
def test_runspec_default_config_exists(script):
    spec = parse_runspec(str(script.relative_to(REPO_ROOT)))
    assert spec.run.launch == "ray"
    assert spec.name.startswith("super35/rl/")
    assert (script.parent / "config" / f"{spec.config.default}.yaml").exists()


@pytest.mark.parametrize("config", RL_CONFIGS, ids=_ids(RL_CONFIGS))
class TestStageConfig:
    def test_artifact_wiring(self, config):
        cfg = _load(config)
        assert OmegaConf.select(cfg, "run.model", default="").endswith(":latest")
        assert OmegaConf.select(cfg, "run.output_artifact", default="").startswith("super35-rl-")
        raw = OmegaConf.to_container(cfg, resolve=False)
        # Warm starts (vision-reasoning stage 2) take architecture/tokenizer from a
        # separate base artifact and load weights via pretrained_checkpoint.
        assert raw["policy"]["model_name"] in ("${art:model,path}", "${art:base_model,path}")

    def test_shared_mtp_setup(self, config):
        megatron = _load(config).policy.megatron_cfg
        assert megatron.mtp_num_layers == 5
        assert megatron.mtp_use_repeated_layer is True
        assert megatron.mtp_detach_heads is True
        assert megatron.mtp_loss_scaling_factor == pytest.approx(0.3)

    def test_no_internal_references(self, config):
        for lineno, line in enumerate(config.read_text().splitlines(), 1):
            assert not _INTERNAL_PATTERNS.search(line), f"{config.name}:{lineno}: {line.strip()}"


@pytest.mark.parametrize("for_remote", [True, False], ids=["remote", "local"])
@pytest.mark.parametrize("config", RL_CONFIGS, ids=_ids(RL_CONFIGS))
def test_script_config_keeps_runner_metadata(config, for_remote):
    """The CLI hands the runner an extracted config; it must still carry the
    ``run`` keys ``_nemo_rl.run`` reads (output artifact, HF architecture)."""
    from nemo_runspec.config.loader import extract_train_config

    run = extract_train_config(_load(config), for_remote=for_remote).run
    assert run.output_artifact == _load(config).run.output_artifact
    assert run.hf_architecture == _load(config).run.hf_architecture


@pytest.mark.parametrize("config", RL_CONFIGS, ids=_ids(RL_CONFIGS))
def test_slurm_gpus_match_cluster(config):
    """The Slurm allocation must request the GPUs/node NeMo-RL expects; a
    missing ``run.env.gpus_per_node`` silently inherits the env profile's."""
    cfg = _load(config)
    assert cfg.run.env.gpus_per_node == cfg.cluster.gpus_per_node


_ALL_SUPER35_FILES = sorted(
    p
    for p in (REPO_ROOT / "src/nemotron/recipes/super35").rglob("*")
    if p.suffix in {".yaml", ".py", ".md", ".sh"} or p.name == "Dockerfile"
)


@pytest.mark.parametrize("path", _ALL_SUPER35_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_internal_references_anywhere(path):
    """Every super35 recipe file (not only the RL configs) stays public-safe."""
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        assert not _INTERNAL_PATTERNS.search(line), f"{path.name}:{lineno}: {line.strip()}"


def test_output_artifacts_are_unique():
    names = [_load(p).run.output_artifact for p in RL_CONFIGS]
    assert len(names) == len(set(names)), names


@pytest.mark.parametrize("config", MOPD_CONFIGS, ids=_ids(MOPD_CONFIGS))
def test_mopd_teachers_resolve_to_run_artifacts(config):
    raw = OmegaConf.to_container(_load(config), resolve=False)
    run_keys = set(raw["run"])
    routes = raw["on_policy_distillation"]["teacher_model_by_agent_name"]
    assert "general" in routes
    for agent, target in routes.items():
        art = re.fullmatch(r"\$\{art:(\w+),path\}", str(target))
        alias = re.fullmatch(r"\$\{on_policy_distillation\.teacher_model_by_agent_name\.(\w+)\}", str(target))
        assert art or alias, f"{agent} -> {target}"
        if art:
            assert art.group(1) in run_keys, f"{agent} uses undeclared artifact {art.group(1)}"
        else:
            assert alias.group(1) in routes, f"{agent} aliases unknown teacher {alias.group(1)}"


class TestIsMegatronCheckpoint:
    def test_hf_checkpoint(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert not is_megatron_checkpoint(tmp_path)

    def test_megatron_iter_dirs(self, tmp_path):
        (tmp_path / "iter_0000100").mkdir()
        assert is_megatron_checkpoint(tmp_path)

    def test_megatron_tracker_file(self, tmp_path):
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("100")
        assert is_megatron_checkpoint(tmp_path)

    def test_missing_path(self, tmp_path):
        assert not is_megatron_checkpoint(tmp_path / "missing")


class TestArtifactOnlyResolution:
    """Only ${art:...} is resolved; secrets in ${oc.env:...} stay interpolations."""

    def test_keeps_env_interpolations(self, monkeypatch):
        from nemotron.recipes.super35.stage2_rl._nemo_rl import _resolve_artifact_refs

        monkeypatch.setenv("NVIDIA_API_KEY", "secret")
        if not OmegaConf.has_resolver("art"):
            OmegaConf.register_new_resolver("art", lambda name, field: f"/ckpt/{name}/{field}")
        cfg = OmegaConf.create(
            {
                "policy": {"model_name": "${art:model,path}", "tokenizer": {"name": "${policy.model_name}"}},
                "judge": {"api_key": "${oc.env:NVIDIA_API_KEY}"},
                "paths": ["${art:data,train}", "plain"],
            }
        )
        out = _resolve_artifact_refs(OmegaConf.to_container(cfg, resolve=False), cfg)
        assert out["policy"]["model_name"].endswith("model/path")
        assert out["policy"]["tokenizer"]["name"] == "${policy.model_name}"
        assert out["judge"]["api_key"] == "${oc.env:NVIDIA_API_KEY}"
        assert out["paths"][0].endswith("data/train") and out["paths"][1] == "plain"


class TestPrepareCheckpoints:
    def _nemo_rl_step(self, root: Path, converted: bool) -> Path:
        step = root / "step_30"
        (step / "policy" / "weights" / "iter_0000000").mkdir(parents=True)
        if converted:
            (step / "hf").mkdir()
            (step / "hf" / "config.json").write_text("{}")
        return step

    def test_policy_teachers_and_warm_start(self, tmp_path):
        from nemotron.recipes.super35.stage2_rl._nemo_rl import prepare_checkpoints

        student = self._nemo_rl_step(tmp_path / "student", converted=True)
        teacher = self._nemo_rl_step(tmp_path / "teacher", converted=True)
        hf_teacher = tmp_path / "hf_teacher"
        hf_teacher.mkdir()
        (hf_teacher / "config.json").write_text("{}")
        config = {
            "policy": {"model_name": str(student), "pretrained_checkpoint": {"path": str(student)}},
            "on_policy_distillation": {
                "teacher_model_by_agent_name": {
                    "general": str(student),
                    "Reasoning": str(teacher),
                    "Vision": str(hf_teacher),
                    "math_agent": "${on_policy_distillation.teacher_model_by_agent_name.Reasoning}",
                }
            },
        }
        prepare_checkpoints(config, {"hf_architecture": "nvidia/arch"})

        routes = config["on_policy_distillation"]["teacher_model_by_agent_name"]
        assert config["policy"]["model_name"] == str(student / "hf")
        assert config["policy"]["pretrained_checkpoint"]["path"] == str(student / "policy" / "weights")
        assert routes["general"] == str(student / "hf")
        assert routes["Reasoning"] == str(teacher / "hf")
        assert routes["Vision"] == str(hf_teacher)
        assert routes["math_agent"].startswith("${")

    def test_hub_id_passes_through(self):
        from nemotron.recipes.super35.stage2_rl._nemo_rl import to_hf_checkpoint

        assert to_hf_checkpoint("nvidia/some-model", "nvidia/arch") == "nvidia/some-model"
