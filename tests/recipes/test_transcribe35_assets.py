# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static checks for the link-first Nemotron 3.5 Transcribe integration."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROFILES = {
    "bf16": {
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
        "revision": "525be0eeaa30b1b0782fe1c96d172e0c012a2ec6",
    },
    "fp8": {
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
        "revision": "94edf6deb1b42bdae3529b010e283de855dc473e",
    },
    "nvfp4": {
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
        "revision": "499c4e5aa456ef25a36def3033f206e8c07e62eb",
    },
}
NEMO_SPEECH_REVISION = "cbc617bb94a31451fe0a3503741e2a8e7568ff22"


def test_transcribe35_discovery_targets_exist() -> None:
    expected = [
        "docs/nemotron/transcribe35/README.md",
        "skills/nemotron-transcribe/SKILL.md",
        "skills/nemotron-transcribe/INDEX.md",
        "skills/nemotron-transcribe/context/index.toml",
        "skills/nemotron-transcribe/context/quick-reference.md",
        "skills/nemotron-transcribe/model-card.md",
        "skills/nemotron-transcribe/recipes/overview.md",
        "usage-cookbook/Nemotron-3.5-Transcribe/README.md",
        "usage-cookbook/Nemotron-3.5-Transcribe/vllm_cookbook.ipynb",
    ]

    missing = [path for path in expected if not (REPO_ROOT / path).is_file()]
    assert not missing, f"Missing Transcribe integration files: {missing}"

    context_index = (REPO_ROOT / "skills/nemotron-transcribe/context/index.toml").read_text(encoding="utf-8")
    assert 'model = "nemotron-transcribe"' in context_index
    assert 'default_entry = "quick-reference"' in context_index


def test_transcribe35_skill_routes_to_canonical_speech_skills() -> None:
    skill = (REPO_ROOT / "skills/nemotron-transcribe/SKILL.md").read_text(encoding="utf-8")

    assert skill.startswith("---\nname: nemotron-transcribe\n")
    for upstream_skill in ("train-speech-llm", "finetune-speech-llm", "train-dflash", "quantize-speech-llm"):
        assert f"NVIDIA-NeMo/Speech/blob/main/.claude/skills/{upstream_skill}/SKILL.md" in skill
    assert "NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/conf/salm_nemotron.yaml" in skill

    assert "nvidia.slack.com" not in skill
    assert "/work/" not in skill


def test_transcribe35_vllm_notebook_is_valid_and_complete() -> None:
    path = REPO_ROOT / "usage-cookbook/Nemotron-3.5-Transcribe/vllm_cookbook.ipynb"
    notebook_text = path.read_text(encoding="utf-8")
    notebook = json.loads(notebook_text)

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(code)
    profiles_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PROFILES" for target in node.targets)
    )
    profiles = ast.literal_eval(profiles_assignment.value)
    for variant, expected in EXPECTED_PROFILES.items():
        assert profiles[variant]["model_id"] == expected["model_id"]
        assert profiles[variant]["revision"] == expected["revision"]
        assert notebook_text.count(expected["revision"]) == 1
    for required in (
        NEMO_SPEECH_REVISION,
        "vLLM 0.32.0",
        "VLLM_PLUGINS=nemo_speechlm",
        "VLLM_MAX_AUDIO_DECODE_DURATION_S=3600",
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180",
        "VLLM_USE_V2_MODEL_RUNNER=0",
        "av==19.0.2",
        "--host 127.0.0.1",
        "--use-replayssm",
        "--mamba-backend flashinfer",
        "--mamba-cache-mode none",
        "--trust-request-mm-kwargs",
        "Nemotron3_5TranscribeForConditionalGeneration",
        "expected_quant_algo",
        "Set MODEL_VARIANT explicitly when MODEL_DIR is set.",
        "capture_ctc_timestamps",
        "A dense one-hour conversation can need about 20,000 output tokens.",
        "Put speaker tags <spk:0> <spk:1> <spk:2>",
        "enable_thinking",
        "ctc_timestamps",
        "--revision {profile['revision']}",
    ):
        assert required in notebook_text

    assert "--mamba-cache-mode all" not in notebook_text
    assert "vLLM 0.28.0" not in notebook_text
    assert "uv pip install openai" not in notebook_text


def test_transcribe35_local_checkpoint_requires_renamed_architecture(tmp_path, monkeypatch) -> None:
    notebook = json.loads(
        (REPO_ROOT / "usage-cookbook/Nemotron-3.5-Transcribe/vllm_cookbook.ipynb").read_text(encoding="utf-8")
    )
    profile_code = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "PROFILES =" in "".join(cell["source"])
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    monkeypatch.setenv("MODEL_VARIANT", "bf16")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    config_path.write_text(
        json.dumps(
            {
                "model_type": "nemo_speechlm",
                "architectures": ["NeMoSpeechLMForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="instead of mixing pre-rename and post-rename components"):
        exec(profile_code, {})

    config_path.write_text(
        json.dumps(
            {
                "model_type": "nemo_speechlm",
                "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    namespace: dict[str, object] = {}
    exec(profile_code, namespace)
    assert namespace["MODEL_SOURCE"] == str(model_dir)


def test_transcribe35_public_docs_do_not_contain_internal_launch_material() -> None:
    public_files = [
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs/deployment-guides.md",
        REPO_ROOT / "docs/index.md",
        REPO_ROOT / "docs/nemotron/transcribe35/README.md",
        REPO_ROOT / "skills/INDEX.md",
        REPO_ROOT / "skills/nemotron-transcribe/SKILL.md",
        REPO_ROOT / "skills/nemotron-transcribe/INDEX.md",
        REPO_ROOT / "skills/nemotron-transcribe/context/index.toml",
        REPO_ROOT / "skills/nemotron-transcribe/context/quick-reference.md",
        REPO_ROOT / "skills/nemotron-transcribe/model-card.md",
        REPO_ROOT / "skills/nemotron-transcribe/recipes/overview.md",
        REPO_ROOT / "usage-cookbook/README.md",
        REPO_ROOT / "usage-cookbook/Nemotron-3.5-Transcribe/README.md",
        REPO_ROOT / "usage-cookbook/Nemotron-3.5-Transcribe/vllm_cookbook.ipynb",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_files)

    assert "nvidia.slack.com" not in combined
    assert "/work/" not in combined
    assert "hf_" not in combined
    assert "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2" in combined
    assert "nvidia/Nemotron-3.5-Transcribe-30B-A4B-DFlash2" not in combined
