#!/usr/bin/env python3
"""Create an Ultra checkpoint compatibility overlay.

Some Ultra checkpoints store architecture layout as explicit
``layers_block_type`` / ``mtp_layers_block_type`` lists.  Some NeMo RL runtime
stacks expect the older compact fields:
``hybrid_override_pattern``, ``mtp_hybrid_override_pattern``, ``num_hidden_layers``,
and local ``auto_map`` implementation files.

This script builds a lightweight overlay directory: it writes a converted
``config.json`` and creates relative symlinks to the other files from the source
checkpoint.  If runtime Python implementation files are available, the overlay
copies those small files and adds a local ``auto_map``.  Modern runtimes that
already support ``model_type=nemotron_h`` do not need those local files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any


BLOCK_TO_PATTERN = {
    "mamba": "M",
    "moe": "E",
    "attention": "*",
}

DEFAULT_AUTO_MAP = {
    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compatibility overlay for an Ultra HF checkpoint."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="New Ultra checkpoint directory.",
    )
    parser.add_argument(
        "--reference-old",
        type=Path,
        help=(
            "Optional old compatible checkpoint directory. Used as a fallback "
            "for compact hybrid fields and Ultra Python implementation files."
        ),
    )
    parser.add_argument(
        "--runtime-source",
        type=Path,
        help=(
            "Optional directory containing configuration_nemotron_h.py and "
            "modeling_nemotron_h.py for older runtimes that require local "
            "trust_remote_code files."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output overlay directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy sidecar files instead of symlinking them. Model weights are always symlinked unless --copy-weights is also set.",
    )
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        help="Copy model weight shards too. This is usually not what you want on Lustre.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return data


def finite_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return sys.float_info.max
    if isinstance(value, list):
        return [finite_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: finite_json_value(v) for k, v in value.items()}
    return value


def pattern_from_blocks(blocks: list[Any], field_name: str) -> str:
    chars: list[str] = []
    for idx, block in enumerate(blocks):
        key = str(block).lower()
        try:
            chars.append(BLOCK_TO_PATTERN[key])
        except KeyError as exc:
            expected = ", ".join(sorted(BLOCK_TO_PATTERN))
            raise ValueError(
                f"Unsupported {field_name}[{idx}]={block!r}; expected one of {expected}"
            ) from exc
    return "".join(chars)


def converted_config(
    source_cfg: dict[str, Any], old_cfg: dict[str, Any] | None
) -> dict[str, Any]:
    cfg = dict(source_cfg)

    layers = cfg.pop("layers_block_type", None)
    if layers is not None:
        if not isinstance(layers, list):
            raise TypeError("layers_block_type must be a list when present")
        cfg["hybrid_override_pattern"] = pattern_from_blocks(layers, "layers_block_type")
        cfg["num_hidden_layers"] = len(layers)
    elif "hybrid_override_pattern" not in cfg and old_cfg is not None:
        cfg["hybrid_override_pattern"] = old_cfg["hybrid_override_pattern"]
        cfg["num_hidden_layers"] = old_cfg["num_hidden_layers"]
    elif "hybrid_override_pattern" not in cfg:
        raise ValueError(
            "config.json has neither layers_block_type nor hybrid_override_pattern"
        )

    mtp_layers = cfg.pop("mtp_layers_block_type", None)
    if mtp_layers is not None:
        if not isinstance(mtp_layers, list):
            raise TypeError("mtp_layers_block_type must be a list when present")
        cfg["mtp_hybrid_override_pattern"] = pattern_from_blocks(
            mtp_layers, "mtp_layers_block_type"
        )
    elif (
        "mtp_hybrid_override_pattern" not in cfg
        and old_cfg is not None
        and "mtp_hybrid_override_pattern" in old_cfg
    ):
        cfg["mtp_hybrid_override_pattern"] = old_cfg["mtp_hybrid_override_pattern"]

    if old_cfg is not None and old_cfg.get("auto_map"):
        cfg["auto_map"] = old_cfg["auto_map"]

    if old_cfg is not None and "time_step_limit" not in cfg and "time_step_limit" in old_cfg:
        cfg["time_step_limit"] = finite_json_value(old_cfg["time_step_limit"])

    return finite_json_value(cfg)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.relpath(src, start=dst.parent), dst)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    reference_old = args.reference_old.resolve() if args.reference_old else None
    runtime_source = args.runtime_source.resolve() if args.runtime_source else None
    output = args.output.resolve()

    if not source.is_dir():
        raise NotADirectoryError(source)
    if reference_old is not None and not reference_old.is_dir():
        raise NotADirectoryError(reference_old)
    if runtime_source is not None and not runtime_source.is_dir():
        raise NotADirectoryError(runtime_source)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} already exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_cfg = load_json(source / "config.json")
    old_cfg = load_json(reference_old / "config.json") if reference_old else None
    cfg = converted_config(source_cfg, old_cfg)

    with (output / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")

    for src in sorted(source.iterdir()):
        if src.name == "config.json":
            continue
        dst = output / src.name
        if src.is_dir():
            continue
        is_weight = src.suffix == ".safetensors"
        link_or_copy(src, dst, copy=args.copy and (args.copy_weights or not is_weight))

    runtime_files = ("configuration_nemotron_h.py", "modeling_nemotron_h.py")
    if runtime_source is None:
        if all((source / name).exists() for name in runtime_files):
            runtime_source = source
        elif reference_old is not None and all((reference_old / name).exists() for name in runtime_files):
            runtime_source = reference_old

    if runtime_source is not None:
        for name in runtime_files:
            src = runtime_source / name
            if not src.exists():
                raise FileNotFoundError(
                    f"{src} does not exist. Pass --runtime-source pointing to a "
                    "compatible Ultra checkpoint with local implementation files."
                )
            shutil.copy2(src, output / name)

        cfg["auto_map"] = old_cfg.get("auto_map", DEFAULT_AUTO_MAP) if old_cfg else DEFAULT_AUTO_MAP
        with (output / "config.json").open("w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"Wrote overlay: {output}")
    print(f"hybrid_override_pattern length: {len(cfg['hybrid_override_pattern'])}")
    print(f"num_hidden_layers: {cfg['num_hidden_layers']}")
    print(f"mtp_hybrid_override_pattern: {cfg.get('mtp_hybrid_override_pattern')}")
    print(f"runtime files included: {runtime_source is not None}")
    print(f"auto_map: {cfg.get('auto_map')}")


if __name__ == "__main__":
    main()
