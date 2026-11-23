"""Prepare a local model view without changing checkpoint tensors."""

import hashlib
import json
import os
import shutil
from pathlib import Path

BASE_MODEL = "nvidia/Nemotron-3.5-Parse"
BASE_REVISION = "62cb1e4ba153d113ff64aaf2be6e068c377cf53c"


def sft_checkpoint(output):
    """Find the latest completed SFT checkpoint for downstream stages."""
    checkpoints = list((Path(output) / "checkpoints").glob("epoch_*_step_*/model/consolidated"))
    if not checkpoints:
        raise FileNotFoundError("Run the SFT stage before starting RLVR")
    return max(checkpoints, key=lambda path: int(path.parents[1].name.rsplit("_", 1)[1]))


def resolve_model(checkpoint, revision=None):
    """Resolve the configured local checkpoint or Hub model before training."""
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        if revision is not None:
            raise ValueError("model_revision applies to Hub models, not local checkpoints")
        return local.resolve(), None
    if local.is_absolute() or str(checkpoint).startswith((".", "~")):
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
    if revision is None and checkpoint == BASE_MODEL:
        revision = BASE_REVISION
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(str(checkpoint), revision=revision)).resolve()
    resolved_revision = snapshot.name if snapshot.parent.name == "snapshots" else revision
    return snapshot, resolved_revision


def materialize_base_model(
    checkpoint: str | Path, output: str | Path, revision: str | None = None
) -> tuple[Path, str | None]:
    """Keep unchanged base bytes beside SFT outputs, independent of the Hub cache.

    Dereference cache symlinks. Hard links retain large immutable weight files
    without another same-filesystem copy; different filesystems use a real copy.
    The adapted training view is prepared separately from this base snapshot.
    """
    checkpoint, resolved_revision = resolve_model(checkpoint, revision)
    output = Path(output).absolute()
    if output.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("SFT output must be outside the source model directory to snapshot its base")

    def link_or_copy(source: str, destination: str) -> str:
        source_path = Path(source).resolve()
        try:
            os.link(source_path, destination)
        except OSError:
            shutil.copy2(source_path, destination)
        return destination

    shutil.copytree(checkpoint, output, symlinks=False, copy_function=link_or_copy)
    return output, resolved_revision


def prepare_model(checkpoint, output, scope="full", processor_source=None, revision=None):
    source = str(checkpoint)
    checkpoint, resolved_revision = resolve_model(checkpoint, revision)
    checkpoint, output = Path(checkpoint).absolute(), Path(output).absolute()
    if scope not in ("full", "decoder"):
        raise ValueError("scope must be full or decoder")
    config = json.loads((checkpoint / "config.json").read_text())
    if config["is_encoder_decoder"] or not config["decoder"]["causal_vision_prefix"]:
        raise ValueError("This recipe requires causal visual-prefix Nemotron Parse 3.5")
    wrapper = Path(__file__).with_name("hf_parse.py")
    identity = dict(
        source=source,
        revision=resolved_revision,
        checkpoint=str(checkpoint.resolve()),
        processor=str(Path(processor_source or checkpoint).resolve()),
        scope=scope,
        config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        weights={p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in checkpoint.glob("*.safetensors")},
    )
    record = output / "model-source.json"
    if output.exists():
        if record.is_file() and json.loads(record.read_text()) == identity:
            return output
        raise FileExistsError(f"Model view already exists with different or unknown inputs: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for path in checkpoint.iterdir():
        if path.is_file() and path.name not in ("config.json", "hf_parse.py", "hf_dental_parse.py"):
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
        config.setdefault("auto_map", {})[key] = "hf_parse.ParseForConditionalGeneration"
    config.pop("dental_train_scope", None)
    config["parse_train_scope"] = scope
    (output / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copyfile(wrapper, output / "hf_parse.py")
    record.write_text(json.dumps(identity, indent=2))
    return output
