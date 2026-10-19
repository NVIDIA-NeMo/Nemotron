"""Pinned BF16 snapshot validation and runtime precision policy.

Standard library only: usable on the host before installing AutoModel.
No weights are converted, downloaded, or modified by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import struct


MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "environment.json"
ENVIRONMENT = json.loads(MANIFEST.read_text())
MODEL = ENVIRONMENT["model"]
MODEL_ID = MODEL["repo_id"]
MODEL_REVISION = MODEL["revision"]
MODEL_LOCAL_DIR = MODEL["local_dir"]


def _read_json(path: Path):
    return json.loads(path.read_text())


def _check_revision(snapshot: Path, name: str) -> None:
    # Both `hf download --local-dir` and a complete Hub cache snapshot work.
    metadata = snapshot / ".cache" / "huggingface" / "download" / (name + ".metadata")
    if metadata.is_file():
        if metadata.read_text().splitlines()[0] != MODEL_REVISION:
            raise ValueError(f"Wrong Hugging Face revision for {name}; use {MODEL_REVISION}")
    elif not (snapshot.name == MODEL_REVISION and snapshot.parent.name == "snapshots"):
        raise ValueError(
            f"Missing revision evidence for {name}. Preserve the .cache/huggingface/download "
            "metadata when copying an hf --local-dir download, or use a complete Hub snapshot."
        )


def inspect_snapshot(snapshot: Path, *, check_shards: bool = True, verify_hashes: bool = False) -> dict:
    """Validate exact revision, unquantized metadata and stored tensor headers.

    Header checks detect missing/truncated shards and wrong storage dtypes, not
    arbitrary payload corruption. They are not cryptographic weight checksums.
    """
    snapshot = Path(snapshot).expanduser().absolute()
    _check_revision(snapshot, "config.json")
    config = _read_json(snapshot / "config.json")
    llm = config.get("llm_config", {})
    if config.get("quantization_config") or llm.get("quantization_config") or (snapshot / "hf_quant_config.json").exists():
        raise ValueError("BF16-only: quantized checkpoints are not accepted for this training run")
    if config.get("dtype", config.get("torch_dtype")) != "bfloat16":
        raise ValueError("Expected BF16 checkpoint metadata")
    if config.get("architectures") != ["NemotronH_Omni_Reasoning_V3"]:
        raise ValueError("Checkpoint architecture differs from the reviewed Nemotron VLM")
    mtp = llm.get("num_nextn_predict_layers")
    if mtp != MODEL["source_mtp_layers"]:
        raise ValueError("Pinned checkpoint must preserve its one MTP prediction head")
    result = {
        "repo_id": MODEL_ID, "revision": MODEL_REVISION,
        "checkpoint_format": "BF16", "quant_method": None,
        "source_mtp_layers": mtp,
        "mtp_training_enabled": False, "snapshot": str(snapshot),
        "training_backend_status": "configured_not_yet_validated", "serving_backend_status": "unvalidated",
    }
    if not check_shards:
        return result
    for name in ("model.safetensors.index.json", "tokenizer_config.json", "tokenizer.json",
                 "processor_config.json", "preprocessor_config.json"):
        _check_revision(snapshot, name)
        if not (snapshot / name).is_file():
            raise FileNotFoundError(snapshot / name)
    index = _read_json(snapshot / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    if not shards:
        raise ValueError("Snapshot index contains no weight shards")
    dtypes = Counter()
    payload_size = 0
    seen = set()
    dtype_bytes = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                   "BF16": 2, "F16": 2, "I16": 2, "I32": 4, "F32": 4,
                   "I64": 8, "F64": 8, "BOOL": 1}
    for name in shards:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Unsafe or unsupported shard filename: {name}")
        _check_revision(snapshot, name)
        path = snapshot / name
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors header: {name}")
            size = struct.unpack("<Q", prefix)[0]
            if size > 100_000_000:
                raise ValueError(f"Unreasonable safetensors header length: {name}")
            header = json.loads(handle.read(size))
        end = 0
        intervals = []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in seen or weight_map.get(key) != name:
                raise ValueError(f"Tensor/index mismatch: {key}")
            seen.add(key)
            begin, finish = value["data_offsets"]
            if begin < 0 or finish < begin:
                raise ValueError(f"Invalid tensor offsets: {key}")
            expected_bytes = math.prod(value["shape"]) * dtype_bytes[value["dtype"]]
            if finish - begin != expected_bytes:
                raise ValueError(f"Tensor shape/dtype/byte-length mismatch: {key}")
            intervals.append((begin, finish))
            end = max(end, finish)
            dtypes[value["dtype"]] += 1
        position = 0
        for begin, finish in sorted(intervals):
            if begin != position:
                raise ValueError(f"Overlapping or non-contiguous tensor data: {name}")
            position = finish
        if path.stat().st_size != 8 + size + end:
            raise ValueError(f"Incomplete or malformed weight shard: {name}")
        payload_size += end
    if seen != set(weight_map):
        raise ValueError("Indexed tensors are missing from the stored shards")
    if not dtypes.get("BF16") or set(dtypes) - {"BF16", "F32", "I64", "I32", "BOOL"}:
        raise ValueError("BF16 weight storage required; packed/FP8/F16 tensors are not accepted")
    # Tensor headers and byte bounds are authoritative; do not edit source metadata.
    declared_size = index.get("metadata", {}).get("total_size")
    result.update(shards=len(shards), tensor_payload_bytes=payload_size, tensor_dtypes=dict(dtypes),
                  index_declared_payload_bytes=declared_size,
                  index_payload_size_matches=(payload_size == declared_size),
                  warnings=[])
    if payload_size != declared_size:
        result["warnings"].append(
            "Index aggregate total_size differs from validated tensor headers; "
            "verify shard SHA-256 hashes against Hub metadata. Source index left unchanged."
        )
    if verify_hashes:
        with ThreadPoolExecutor(max_workers=8) as executor:
            verified = list(executor.map(lambda name: verify_shard_hash(snapshot, name), shards))
        result["sha256_verified_shards"] = len(verified)
    return result


def verify_shard_hash(snapshot: Path, name: str) -> str:
    metadata = snapshot / ".cache/huggingface/download" / (name + ".metadata")
    if metadata.is_file():
        expected = metadata.read_text().splitlines()[1].strip('"')
    else:
        # Hub cache snapshots link to blobs named with the LFS SHA-256.
        expected = (snapshot / name).resolve().name
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Missing LFS SHA-256 evidence for {name}")
    digest = hashlib.sha256()
    with (snapshot / name).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {name}")
    return name


def validate_download_destination(snapshot: Path) -> None:
    """Allow a same-revision partial download, but never overwrite another base."""
    snapshot = Path(snapshot).absolute()
    for path in snapshot.iterdir() if snapshot.exists() else ():
        if path.is_file() and path.name != ".gitattributes":
            _check_revision(snapshot, path.name)
    config = snapshot / "config.json"
    if config.is_file():
        inspect_snapshot(snapshot, check_shards=False)


def require_runtime_support(stage: str) -> None:
    """Enforce BF16 policy, not a claim of validated training or serving success."""
    if stage not in {"training", "serving"}:
        raise ValueError(f"Unknown runtime stage: {stage}")
    if MODEL["precision"] != "BF16" or MODEL.get("quant_method") is not None:
        raise RuntimeError(f"BF16 {stage} requires the original, unquantized pinned checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--check-download-destination", action="store_true")
    parser.add_argument("--require-runtime", choices=("training", "serving"))
    args = parser.parse_args()
    if args.model_path:
        if args.check_download_destination:
            validate_download_destination(args.model_path)
            print("BF16 download destination is new or contains same-revision assets")
            return
        if args.metadata_only and args.verify_hashes:
            parser.error("--verify-hashes requires complete shards, not --metadata-only")
        print(json.dumps(inspect_snapshot(args.model_path, check_shards=not args.metadata_only,
                                         verify_hashes=args.verify_hashes), indent=2))
    if args.require_runtime:
        require_runtime_support(args.require_runtime)
    if not args.model_path and not args.require_runtime:
        parser.error("Provide --model-path or --require-runtime")


if __name__ == "__main__":
    main()
