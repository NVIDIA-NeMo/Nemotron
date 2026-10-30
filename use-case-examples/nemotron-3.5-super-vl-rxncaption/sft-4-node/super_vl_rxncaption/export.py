"""Export this run alone: merge LoRA into its downloaded base, or consolidate SFT."""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from assets import write_json

# The training recipe disables MTP and uses the native legacy vision projector.
# These base-only buffers are not part of that model's exported state dictionary.
OMITTED_KEYS = {
    "vision_projector.vision_final_layernorm.weight",
    "vision_projector.vision_final_layernorm.bias",
    "vision_model.summary_idxs",
    "vision_model.radio_model.input_conditioner.norm_mean",
    "vision_model.radio_model.input_conditioner.norm_std",
}


def exported_key(key):
    return not key.startswith("language_model.mtp.") and key not in OMITTED_KEYS


def base_key(name):
    name = name.removeprefix("base_model.model.")
    name = ".".join(part for part in name.split(".") if part not in ("_checkpoint_wrapped_module", "_orig_mod"))
    name = name.replace("language_model.model.layers.", "language_model.backbone.layers.", 1)
    if not name.startswith("language_model.backbone.layers.") or any(
        part in name for part in ("out_proj", "lm_head", "vision", "sound", "audio")
    ):
        raise ValueError(f"Unexpected adapter key: {name}")
    return name


def verify_export(model, expected_keys=None):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weights = index["weight_map"]
    if expected_keys is not None and set(weights) != set(expected_keys):
        raise ValueError("Export does not contain the complete expected model state")
    if not weights or not (model / "config.json").is_file():
        raise ValueError("Export lacks weights/configuration")
    size = 0
    for name in sorted(set(weights.values())):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Invalid shard filename: {name}")
        path = model / name
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < length < 32 * 1024**2:
                raise ValueError("Invalid safetensors header")
            header = json.loads(stream.read(length))
        entries = {k: v for k, v in header.items() if k != "__metadata__"}
        if set(entries) != {k for k, v in weights.items() if v == name}:
            raise ValueError(f"Index/header mismatch in {name}")
        end = max(v["data_offsets"][1] for v in entries.values())
        if path.stat().st_size != 8 + length + end:
            raise ValueError(f"Incomplete shard: {name}")
        size += end
    if size != index["metadata"]["total_size"]:
        raise ValueError("Export tensor byte count differs from index")
    return dict(model_path=str(model), tensor_keys=len(weights), shards=len(set(weights.values())), tensor_bytes=size)


def copy_metadata(base, output):
    for path in base.iterdir():
        if path.is_file() and path.suffix in (".json", ".py", ".jinja", ".txt") and not path.name.startswith("."):
            if path.name != "model.safetensors.index.json":
                shutil.copyfile(path, output / path.name)
    config = json.loads((output / "config.json").read_text())
    config["rxn_vision_final_layernorm"] = False
    write_json(output / "config.json", config)


def merge_lora(base, adapter, output, expected_adapter_keys, rank=64, scale=2):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    weights = load_file(adapter / "adapter_model.safetensors", device="cpu")
    if {base_key(k) for k in weights} != {base_key(k) for k in expected_adapter_keys}:
        raise ValueError("Saved adapter parameters differ from the trainability audit")
    pairs = {}
    for name, tensor in weights.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])\.weight", name)
        if not match or tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid adapter tensor: {name}")
        pair = pairs.setdefault(base_key(match[1] + ".weight"), {})
        if match[2] in pair:
            raise ValueError(f"Duplicate adapter tensor: {name}")
        pair[match[2]] = tensor
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Incomplete LoRA A/B pairs")
    if not any(torch.count_nonzero(pair["B"]).item() for pair in pairs.values()):
        raise ValueError("No adapter updates were saved")
    original = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {key: shard for key, shard in original.items() if exported_key(key)}
    if not set(pairs) <= set(selected):
        raise ValueError("Adapter cannot be mapped to this base checkpoint")
    output.mkdir(exist_ok=False)
    (output / ".export_incomplete").touch()
    copy_metadata(base, output)
    mapping, total, used = {}, 0, set()
    shards = sorted(set(selected.values()))
    with torch.no_grad():
        for number, source_name in enumerate(shards, 1):
            result = {}
            with safe_open(base / source_name, framework="pt", device="cpu") as source:
                for key in sorted(k for k, shard in selected.items() if shard == source_name):
                    tensor = source.get_tensor(key)
                    if key in pairs:
                        a, b = pairs[key]["A"], pairs[key]["B"]
                        if a.shape[0] != rank or b.shape[1] != rank or (b.shape[0], a.shape[1]) != tensor.shape:
                            raise ValueError(f"Adapter/base shape mismatch: {key}")
                        if tensor.dtype != torch.bfloat16:
                            raise ValueError("Expected a BF16 base for this recipe")
                        # Match native LinearLoRA.materialize_effective_weight BF16 arithmetic.
                        tensor = tensor + scale * (b @ a)
                        if not torch.isfinite(tensor).all():
                            raise ValueError(f"Nonfinite merged tensor: {key}")
                        used.add(key)
                    result[key] = tensor.contiguous()
                    total += tensor.numel() * tensor.element_size()
            name = f"model-{number:05d}-of-{len(shards):05d}.safetensors"
            save_file(result, output / name, metadata={"format": "pt"})
            mapping.update({key: name for key in result})
            del result
            print(f"Exported shard {number}/{len(shards)}", flush=True)
    if used != set(pairs):
        raise ValueError("Not all adapters were merged")
    write_json(output / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": mapping})
    report = verify_export(output)
    (output / ".export_incomplete").unlink()
    return dict(**report, merged_layers=len(pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--mode", choices=["lora", "sft"], required=True)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()
    run = args.run_dir
    ready = json.loads((run / "ready.json").read_text())
    complete = json.loads((run / "training_complete.json").read_text())
    if complete["fingerprint"] != ready["fingerprint"]:
        raise ValueError("No completed training for this configuration")
    steps = ready["steps"]
    checkpoint = run / "checkpoints" / f"epoch_0_step_{steps - 1}" / "model"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Final checkpoint is missing: {checkpoint}")
    audits = [json.loads((run / "audit" / f"rank{rank}.json").read_text()) for rank in range(args.world_size)]
    if not all(a["status"] == "passed" and a["mode"] == args.mode for a in audits):
        raise ValueError("Training audit failed")
    if args.mode == "lora":
        if any(set(a["parameters"]) != set(audits[0]["parameters"]) for a in audits):
            raise ValueError("Ranks disagree about trainable adapters")
        import torch

        torch.set_num_threads(8)
        report = merge_lora(args.base_model, checkpoint, run / "export", audits[0]["parameters"])
    else:
        output = checkpoint / "consolidated"
        base_index = json.loads((args.base_model / "model.safetensors.index.json").read_text())
        expected_keys = {key for key in base_index["weight_map"] if exported_key(key)}
        try:
            report = verify_export(output, expected_keys)
        except (ValueError, OSError, KeyError, struct.error):
            if len(list(checkpoint.glob("shard-*.safetensors"))) != args.world_size:
                raise ValueError("Incomplete rank checkpoint; cannot consolidate")
            output = run / "export"
            output.mkdir(exist_ok=False)
            (output / ".export_incomplete").touch()
            tool = Path(os.environ["AUTOMODEL_SOURCE"]) / "tools/offline_hf_consolidation.py"
            subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "--model-name",
                    str(args.base_model),
                    "--input-dir",
                    str(checkpoint),
                    "--output-dir",
                    str(output),
                    "--backend",
                    "gloo",
                    "--num-threads",
                    "8",
                ],
                check=True,
            )
            report = verify_export(output, expected_keys)
            (output / ".export_incomplete").unlink()
        # Supply missing processor/tokenizer files without replacing trained export config.
        for path in args.base_model.iterdir():
            if path.is_file() and path.suffix in (".json", ".py", ".jinja", ".txt") and not path.name.startswith("."):
                if not (output / path.name).exists():
                    shutil.copyfile(path, output / path.name)
    write_json(
        run / "export.json",
        dict(**report, status="verified", training_steps=steps, smoke_test=ready["smoke_steps"] is not None),
    )


if __name__ == "__main__":
    main()
