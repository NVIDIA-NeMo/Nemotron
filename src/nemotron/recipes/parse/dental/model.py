"""Prepare a local model view without changing checkpoint tensors."""

import json
import hashlib
from pathlib import Path
import shutil

BASE_MODEL = "nvidia/Nemotron-3.5-Parse"
BASE_REVISION = "62cb1e4ba153d113ff64aaf2be6e068c377cf53c"


def sft_checkpoint(output):
    """Find the latest completed SFT checkpoint for the next notebook."""
    checkpoints = list((Path(output) / "checkpoints").glob("epoch_*_step_*/model/consolidated"))
    if not checkpoints:
        raise FileNotFoundError("Complete the SFT notebook before starting RLVR")
    return max(checkpoints, key=lambda path: int(path.parents[1].name.rsplit("_", 1)[1]))


def prepare_model(checkpoint, output, scope="full", processor_source=None):
    if checkpoint == BASE_MODEL:
        from huggingface_hub import snapshot_download

        checkpoint = snapshot_download(BASE_MODEL, revision=BASE_REVISION)
    checkpoint, output = Path(checkpoint).absolute(), Path(output).absolute()
    if scope not in ("full", "decoder"):
        raise ValueError("scope must be full or decoder")
    config = json.loads((checkpoint / "config.json").read_text())
    if config["is_encoder_decoder"] or not config["decoder"]["causal_vision_prefix"]:
        raise ValueError("This lesson requires causal visual-prefix Nemotron Parse 3.5")
    wrapper = Path(__file__).with_name("hf_dental_parse.py")
    identity = dict(
        checkpoint=str(checkpoint.resolve()),
        processor=str(Path(processor_source or checkpoint).resolve()),
        scope=scope,
        config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        weights={
            p.name: [p.stat().st_size, p.stat().st_mtime_ns]
            for p in checkpoint.glob("*.safetensors")
        },
    )
    record = output / "model-source.json"
    if output.exists():
        if record.is_file() and json.loads(record.read_text()) == identity:
            return output
        raise FileExistsError(
            f"Model view already exists with different or unknown inputs: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    for path in checkpoint.iterdir():
        if path.is_file() and path.name not in ("config.json", "hf_dental_parse.py"):
            (output / path.name).symlink_to(path)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = Path(processor_source or checkpoint) / name
        if path.is_file() and not (output / name).exists():
            (output / name).symlink_to(path.absolute())
    config["architectures"] = ["RadioMBartPrefixForCausalLM"]
    for key in ("AutoModel", "AutoModelForCausalLM", "AutoModelForImageTextToText"):
        config.setdefault("auto_map", {})[key] = (
            "hf_dental_parse.DentalParseForConditionalGeneration"
        )
    config["dental_train_scope"] = scope
    (output / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copyfile(wrapper, output / "hf_dental_parse.py")
    record.write_text(json.dumps(identity, indent=2))
    return output
