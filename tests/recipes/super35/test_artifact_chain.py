"""Pin the super35 artifact lineage to the Nemotron 3.5 Super reference run.

Each stage reads its inputs from W&B artifacts named in its config's ``run:``
section. A wrong name silently trains from the wrong parent (or fails only at
job start), so these tests check that every model input is produced by some
stage or import command and that the edges match the reference lineage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPES = REPO_ROOT / "src/nemotron/recipes/super35"
RL_ROOT = RECIPES / "stage2_rl"

# `nemotron super35 model import {pretrain,sft,rl}` default artifact names.
IMPORTED_MODELS = {"super35-pretrain-model", "super35-sft-model", "super35-rl-rlvr-model"}

# Reference lineage (stage config -> {run key: parent artifact}); see
# stage2_rl/README.md. MOPD's self-anchor teacher is its student (run.model).
EXPECTED_LINEAGE = {
    "stage1_rlvr/config/default.yaml": {"model": "super35-sft-model"},
    "stage2_teachers/reasoning/config/default.yaml": {"model": "super35-rl-rlvr-model"},
    "stage2_teachers/vision/config/default.yaml": {"model": "super35-rl-rlvr-model"},
    "stage2_teachers/video/config/default.yaml": {"model": "super35-rl-rlvr-model"},
    "stage2_teachers/agentic/config/default.yaml": {"model": "super35-rl-rlvr-model"},
    "stage2_teachers/vision_reasoning/config/stage1.yaml": {"model": "super35-rl-rlvr-model"},
    "stage2_teachers/vision_reasoning/config/stage2.yaml": {
        "model": "super35-rl-teacher-vision-reasoning-stage1-model",
        "base_model": "super35-rl-rlvr-model",
    },
    # RLHF teacher 1 branches from SFT; its stage 2 resumes stage 1's checkpoint_dir.
    "stage2_teachers/rlhf/config/teacher1_stage1.yaml": {"model": "super35-sft-model"},
    "stage2_teachers/rlhf/config/teacher1_stage2.yaml": {"model": "super35-sft-model"},
    "stage2_teachers/rlhf/config/teacher2_stage1.yaml": {"model": "super35-rl-rlvr-model"},
    "stage2_teachers/rlhf/config/teacher2_stage2.yaml": {"model": "super35-rl-teacher-rlhf2-stage1-model"},
    "stage3_mopd/config/phase1.yaml": {
        "model": "super35-rl-rlvr-model",
        "teacher_agentic": "super35-rl-teacher-agentic-model",
        "teacher_reasoning": "super35-rl-teacher-reasoning-model",
        "teacher_rlhf": "super35-rl-teacher-rlhf1-model",
        "teacher_vision": "super35-rl-teacher-vision-model",
    },
    # NeMo-RL guide variant: phase 1 without the agentic teacher.
    "stage3_mopd/config/single_stage.yaml": {
        "model": "super35-rl-rlvr-model",
        "teacher_reasoning": "super35-rl-teacher-reasoning-model",
        "teacher_rlhf": "super35-rl-teacher-rlhf1-model",
        "teacher_vision": "super35-rl-teacher-vision-model",
    },
    "stage3_mopd/config/phase2.yaml": {
        "model": "super35-rl-mopd-phase1-model",
        "teacher_rlhf": "super35-rl-teacher-rlhf2-model",
        "teacher_vision": "super35-rl-teacher-vision-model",
        "teacher_vision_reasoning": "super35-rl-teacher-vision-reasoning-model",
        "teacher_video": "super35-rl-teacher-video-model",
    },
}


def _run(path: Path) -> dict:
    return yaml.safe_load(path.read_text()).get("run", {})


def _name(ref: str) -> str:
    return ref.rsplit(":", 1)[0]


def _mb_configs() -> list[Path]:
    return sorted((RECIPES / "stage0_pretrain/config").glob("*.yaml")) + sorted(
        (RECIPES / "stage1_sft/config").glob("*.yaml")
    )


def _produced_models() -> set[str]:
    produced = set(IMPORTED_MODELS)
    for path in sorted(RL_ROOT.glob("**/config/*.yaml")):
        produced.add(_run(path)["output_artifact"])
    for path in _mb_configs():
        # Megatron-Bridge names the checkpoint artifact after the save dir.
        save = yaml.safe_load(path.read_text())["checkpoint"]["save"]
        produced.add(Path(save).name)
    return produced


def _model_inputs(path: Path) -> dict[str, str]:
    run = _run(path)
    return {k: _name(v) for k, v in run.items() if k in ("model", "base_model") or k.startswith("teacher_")}


def test_rl_lineage_matches_reference():
    actual = {str(p.relative_to(RL_ROOT)): _model_inputs(p) for p in sorted(RL_ROOT.glob("**/config/*.yaml"))}
    assert actual == EXPECTED_LINEAGE


@pytest.mark.parametrize(
    "path",
    sorted(RL_ROOT.glob("**/config/*.yaml")) + _mb_configs() + [RECIPES / "stage3_eval/config/default.yaml"],
    ids=lambda p: str(p.relative_to(RECIPES)),
)
def test_every_model_input_is_produced(path):
    missing = {k: v for k, v in _model_inputs(path).items() if v not in _produced_models()}
    assert not missing, f"inputs no stage or import command produces: {missing}"


def test_final_checkpoint_feeds_eval():
    assert _name(_run(RECIPES / "stage3_eval/config/default.yaml")["model"]) == "super35-rl-mopd-phase2-model"


def test_sft_stages_chain():
    stage1 = yaml.safe_load((RECIPES / "stage1_sft/config/default.yaml").read_text())
    stage2 = yaml.safe_load((RECIPES / "stage1_sft/config/stage2.yaml").read_text())
    assert Path(stage1["checkpoint"]["save"]).name == _name(stage2["run"]["model"])
    assert Path(stage2["checkpoint"]["save"]).name == "super35-sft-model"


def test_every_data_artifact_is_documented():
    """Data is user-imported, so each name a stage reads must be in its README."""
    readmes = "\n".join(p.read_text() for p in RECIPES.glob("**/README.md"))
    data = {_name(_run(p)["data"]) for p in sorted(RL_ROOT.glob("**/config/*.yaml"))}
    data |= {_name(_run(p)["data"]) for p in _mb_configs() if "data" in _run(p)}
    undocumented = sorted(d for d in data if d not in readmes)
    assert not undocumented
