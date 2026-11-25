"""Audit trainable parameters, then execute the standard AutoModel VLM recipe."""

import json
import os
from pathlib import Path

import nemo_automodel.recipes.vlm.finetune as vlm
from nemo_automodel.cli.app import main

original_build = vlm.build_model


def audited_build(*args, **kwargs):
    model = original_build(*args, **kwargs)
    params = {name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad}

    def count(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    mode = os.environ["TUTORIAL_MODE"]
    if mode == "lora":
        assert len(params) > 20 and count(model) < 10**9, "LoRA did not attach correctly"
        for name in params:
            assert "lora_" in name and name.startswith("language_model."), name
            assert not any(part in name for part in ("out_proj", "lm_head", "vision", "sound", "audio")), name
        assert count(model.vision_model) == count(model.vision_projector) == 0
    else:
        assert not any("lora_" in name for name in params)
        assert count(model) > 10**9
        assert all(count(module) > 0 for module in (model.language_model, model.vision_model, model.vision_projector))
    for path in ("language_model.model.embed_tokens", "language_model.lm_head"):
        assert count(model.get_submodule(path)) == 0, path
    audio = getattr(model, "sound_encoder", None)
    assert audio is None or count(audio) == 0
    report = dict(status="passed", mode=mode, trainable_count=count(model), parameters=params)
    directory = Path(os.environ["RUN_DIR"]) / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"rank{os.environ['RANK']}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Trainability audit passed: {mode}, {count(model):,} parameters", flush=True)
    return model


vlm.build_model = audited_build
if __name__ == "__main__":
    main()
