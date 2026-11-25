"""Support code for the chemical-reaction LoRA tutorial.

Copy this file beside chemical_reaction_parsing_lora.ipynb. This is local tutorial
code, not a separately installed package. It uses the notebook's explicit inputs
and has no dependency on the surrounding repository.

The notebook keeps configuration, the training recipe, and launch commands visible.
This module handles downloads, dataset validation, container entry scripts, and
RxnCaption prediction conversion/scoring. Importing it does not download or launch
anything. Training and merge scripts are written into the mounted run directory.
"""

import base64
import hashlib
import json
import math
import re
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# Download and validate versioned inputs.


def sha256(path):
    """Hash a file in chunks so large archives do not fill RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Replace a JSON report only after its temporary file is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require_sources(config):
    """Check public repositories, immutable revisions, images, and release size."""
    missing = []
    for section, keys in {
        "model": ["repo_id", "revision"],
        "data": ["repo_id", "revision", "train_images", "test_images"],
        "automodel": ["url", "revision"],
    }.items():
        for key in keys:
            value = config[section].get(key, "")
            if value in ("", None) or (isinstance(value, str) and "${" in value):
                missing.append(f"{section}.{key}")
    for key in ("train_image", "eval_image"):
        value = config.get(key, "")
        if not isinstance(value, str) or not value or "${" in value:
            missing.append(key)
    if missing:
        raise ValueError(
            "Set download sources before running: " + ", ".join(missing) + ". Fill in the CONFIG cell above."
        )
    for name in ("model", "data", "automodel"):
        if not re.fullmatch(r"[a-fA-F0-9]{40}", config[name]["revision"]):
            raise ValueError(f"{name}.revision must be an immutable 40-character commit")
    if config["data"]["train_images"] != 15_000 or config["data"]["test_images"] != 400:
        raise ValueError("This recipe expects the pinned 15,000-train/400-test public data release")


def extract_zip(archive, destination, required):
    """Accept an archive root or one wrapping folder, rejecting traversal/links."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            # Reject entries that could escape the extraction directory.
            for item in source.infolist():
                path = Path(item.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16)
                ):
                    raise ValueError(f"Unsafe ZIP member: {item.filename}")
            source.extractall(root)
        # Accept files at the ZIP root or inside one enclosing folder.
        candidates = [
            p for p in [root, *root.iterdir()] if p.is_dir() and all((p / name).is_file() for name in required)
        ]
        if len(candidates) != 1:
            raise ValueError(f"Archive must contain {required}, at root or in one folder")
        shutil.move(str(candidates[0]), destination)


def public_data_identity(spec):
    """Return the complete identity of generated public data."""
    return {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "train_images": spec["train_images"],
        "test_images": spec["test_images"],
        "converter_sha256": sha256(Path(__file__).with_name("prepare_rxncaption_data.py")),
    }


def prepare_public_data(spec, root, download_file=None):
    """Download the pinned public release and build both tutorial inputs."""
    from prepare_rxncaption_data import prepare_dataset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    identity = public_data_identity(spec)
    marker = root / ".public_data.json"
    required = (
        root / "data/train/dataset.jsonl",
        root / "data/test/dataset.jsonl",
        root / "benchmark/ground_truth/ground_truth.json",
        root / "benchmark/mapdicts/mapdict_gt.json",
    )
    if marker.is_file():
        if json.loads(marker.read_text()) != identity or not all(path.is_file() for path in required):
            raise ValueError(
                "Public data identity changed or its completed extraction is incomplete; use a new workspace"
            )
        return
    if (root / "data").exists() or (root / "benchmark").exists():
        raise FileExistsError("Unverified data/benchmark directory exists; choose a new workspace")

    if download_file is None:
        from huggingface_hub import hf_hub_download

        download_file = hf_hub_download
    downloads = root / "U-RxnDiagram-15k-download"
    source = root / "U-RxnDiagram-15k-source"
    source.mkdir(exist_ok=True)
    for split in ("train", "test"):
        filename = f"{split}_set.zip"
        archive = Path(
            download_file(
                repo_id=spec["repo_id"],
                filename=filename,
                repo_type="dataset",
                revision=spec["revision"],
                local_dir=downloads,
            )
        )
        destination = source / f"{split}_archive"
        expected = f"{split}_set/ground_truth.json"
        if destination.exists():
            if not (destination / expected).is_file():
                raise ValueError(f"Incomplete extracted public archive: {destination}")
        else:
            extract_zip(archive, destination, [expected])

    converted = root / "U-RxnDiagram-15k-converted"
    if not converted.exists():
        prepare_dataset(
            source,
            converted,
            expected_train=spec["train_images"],
            expected_test=spec["test_images"],
            dataset_url=f"https://huggingface.co/datasets/{spec['repo_id']}",
            source_revision=spec["revision"],
        )
    manifest = json.loads((converted / "manifest.json").read_text())
    if manifest["source_revision"] != spec["revision"]:
        raise ValueError("Converted public data revision differs from CONFIG")
    shutil.move(str(converted / "finetuning-data"), root / "data")
    shutil.move(str(converted / "benchmark"), root / "benchmark")
    write_json(marker, identity)


def download(config, workspace):
    """Fetch the checkpoint, datasets, AutoModel source, and container images."""
    require_sources(config)  # Fail before downloading hundreds of GB if an input is missing.
    for command in ("git", "docker"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required host command is unavailable: {command}")
    docker_info = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if docker_info.returncode:
        raise RuntimeError(f"Docker daemon is unavailable: {docker_info.stderr.strip()}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(workspace.parent).free < 600 * 2**30:
        raise RuntimeError("Keep at least 600 GiB free for model download, training, merge, and evaluation")
    root = workspace / "assets"
    root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    sources = {
        "model": config["model"],
        "automodel_revision": config["automodel"]["revision"],
        "data": public_data_identity(config["data"]),
        "train_image": config["train_image"],
        "eval_image": config["eval_image"],
    }
    sources_path = root / "sources.json"
    if sources_path.is_file() and json.loads(sources_path.read_text()) != sources:
        raise ValueError("Configured sources differ from this workspace; choose a new workspace")
    if not sources_path.exists():
        write_json(sources_path, sources)
    # 1. Download and convert the pinned public U-RxnDiagram release.
    prepare_public_data(config["data"], root)
    # 2. Download the pinned HF checkpoint into this workspace.
    model = config["model"]
    model_dir = root / "model"
    model_marker = model_dir / ".download.json"
    if model_marker.exists() and json.loads(model_marker.read_text()) != model:
        raise ValueError("Model source changed; choose a new workspace")
    snapshot_download(repo_id=model["repo_id"], revision=model["revision"], local_dir=model_dir)
    write_json(model_marker, model)
    # 3. Check out the source revision that training will import in the container.
    source = root / "Automodel"
    spec = config["automodel"]
    if not source.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", spec["url"], str(source)], check=True)
    actual = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, capture_output=True)
    if actual.returncode or actual.stdout.strip() != spec["revision"] or not (source / "nemo_automodel").is_dir():
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
        # A fresh --no-checkout clone reports deleted files until the first checkout.
        if (source / "nemo_automodel").exists() and dirty.strip():
            raise ValueError("AutoModel checkout has local edits")
        subprocess.run(["git", "-C", str(source), "fetch", "origin", spec["revision"]], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", spec["revision"]], check=True)
    # 4. Pull the AutoModel NGC image for training/merge and vLLM for evaluation.
    for name in ("train_image", "eval_image"):
        subprocess.run(["docker", "pull", config[name]], check=True)
    # sources.json already records immutable public identities, not authentication tokens.


# Clean labels and prepare disjoint training/validation/benchmark splits.

IMAGE_TAG_RE = re.compile(r"(<image\s*>)", re.IGNORECASE)

CONTAMINATION_RE = re.compile(
    r"("
    r"<ctrl\d+>thought"
    r"|the image (?:shows|contains)"
    r"|determine content type"
    r"|according to (?:the )?(?:ocr )?(?:transcription )?rule"
    r"|however,?\s*rule\s*\d+"
    r")",
    re.IGNORECASE,
)

REACTION_FIELDS = ("reactants", "conditions", "products")


def _clean_target(
    target: str,
    *,
    remove_empty_text: bool,
) -> tuple[str, bool, int]:
    """Validate and clean one assistant target.

    Returns ``(serialized_target, contaminated, removed_empty_items)``.
    """
    try:
        reactions = json.loads(target)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant target is not valid JSON: {exc}") from exc
    if not isinstance(reactions, list):
        raise ValueError(f"assistant target must be a JSON list, got {type(reactions).__name__}")

    contaminated = False
    removed_empty = 0
    cleaned_reactions = []

    for reaction_index, reaction in enumerate(reactions):
        if not isinstance(reaction, dict) or set(reaction) != set(REACTION_FIELDS):
            raise ValueError(f"reaction {reaction_index} must contain exactly {REACTION_FIELDS}; got {reaction!r}")

        cleaned_reaction = {}
        for field in REACTION_FIELDS:
            items = reaction[field]
            if not isinstance(items, list):
                raise ValueError(f"reaction {reaction_index}.{field} must be a list")

            cleaned_items = []
            for item_index, item in enumerate(items):
                if not isinstance(item, dict):
                    raise ValueError(f"reaction {reaction_index}.{field}[{item_index}] must be an object")
                item_type = item.get("type")
                if item_type == "mol":
                    if set(item) != {"type", "index"} or not isinstance(item.get("index"), int):
                        raise ValueError(f"reaction {reaction_index}.{field}[{item_index}] has invalid mol schema")
                elif item_type in {"txt", "idt"}:
                    if set(item) != {"type", "content"} or not isinstance(item.get("content"), str):
                        raise ValueError(f"reaction {reaction_index}.{field}[{item_index}] has invalid text schema")
                    content = item["content"]
                    contaminated = contaminated or bool(CONTAMINATION_RE.search(content))
                    if remove_empty_text and not content.strip():
                        removed_empty += 1
                        continue
                else:
                    raise ValueError(f"reaction {reaction_index}.{field}[{item_index}] has unknown type {item_type!r}")
                cleaned_items.append(dict(item))
            cleaned_reaction[field] = cleaned_items
        cleaned_reactions.append(cleaned_reaction)

    serialized = json.dumps(cleaned_reactions, ensure_ascii=False, separators=(",", ":"))
    return serialized, contaminated, removed_empty


def prepare_records(data, benchmark, output, validation_size=400):
    records, seen, filtered, empty_items = [], set(), 0, 0
    for line in (data / "train/dataset.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if len(row["images"]) != 1:
            raise ValueError("Expected exactly one image per row")
        relative = Path(row["images"][0])
        image = data / "train" / relative
        if relative.is_absolute() or not image.resolve().is_relative_to((data / "train").resolve()):
            raise ValueError("Image must be relative to its split directory")
        if not image.is_file() or image.stem in seen:
            raise ValueError("Missing image or duplicate image ID")
        seen.add(image.stem)
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        answers = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
        if len(users) != 1 or len(answers) != 1 or len(IMAGE_TAG_RE.findall(users[0])) != 1:
            raise ValueError("Expected one user/assistant pair and one <image> marker")
        target, contaminated, removed = _clean_target(answers[0], remove_empty_text=True)
        if contaminated:
            filtered += 1
            continue
        empty_items += removed
        records.append(dict(id=image.stem, image=str(Path("train") / relative), prompt=users[0], target=target))
    if not 0 < validation_size < len(records):
        raise ValueError("Validation size must be smaller than the clean training dataset")
    ranking = sorted(records, key=lambda row: hashlib.sha256(f"42:{row['id']}".encode()).digest())
    validation_ids = {row["id"] for row in ranking[:validation_size]}
    validation = [row for row in records if row["id"] in validation_ids]
    unique_train = [row for row in records if row["id"] not in validation_ids]
    train = []
    for row in unique_train:
        count = len(json.loads(row["target"]))
        train.extend([row] * (3 if count >= 9 else 2 if count >= 5 else 1))
    gold = json.loads((benchmark / "ground_truth/ground_truth.json").read_text())["images"]
    names = [row["file_name"] for row in gold]
    if len(names) != 400 or len(set(names)) != 400 or any(Path(n).name != n for n in names):
        raise ValueError("Expected 400 distinct benchmark image basenames")
    if {p.name for p in (benchmark / "images/gt_bivp").iterdir() if p.is_file()} != set(names):
        raise ValueError("Benchmark images differ from ground truth")
    mappings = json.loads((benchmark / "mapdicts/mapdict_gt.json").read_text())
    mapped_names = [Path(row["file_name"]).name for row in mappings]
    if len(mapped_names) != 400 or set(mapped_names) != set(names):
        raise ValueError("Benchmark mappings do not cover all 400 images")
    test = [json.loads(line) for line in (data / "test/dataset.jsonl").read_text().splitlines() if line.strip()]
    test_ids = [Path(row["images"][0]).stem for row in test]
    benchmark_ids = {Path(name).stem for name in names}
    if len(test_ids) != 400 or set(test_ids) != benchmark_ids or seen & benchmark_ids:
        raise ValueError("Test IDs differ from the benchmark or overlap training/validation")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("validation", validation)):
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = dict(
        unique_train_rows=len(unique_train),
        weighted_train_rows=len(train),
        validation_rows=len(validation),
        benchmark_images=400,
        filtered_rows=filtered,
        removed_empty_text_items=empty_items,
        full_steps=math.ceil(len(train) / 16),
        benchmark_sha256={name: sha256(benchmark / "images/gt_bivp" / name) for name in sorted(names)},
        benchmark_labels_sha256={
            name: sha256(benchmark / name) for name in ("ground_truth/ground_truth.json", "mapdicts/mapdict_gt.json")
        },
    )
    write_json(
        output / "split_ids.json",
        dict(
            train=sorted(row["id"] for row in unique_train),
            validation=sorted(validation_ids),
            benchmark=sorted(benchmark_ids),
        ),
    )
    write_json(output / "report.json", report)
    return report


# Entry scripts run inside the training container, which supplies PyTorch.

TRAINING_SOURCE = r"""
import json
import os
from pathlib import Path
import re
from PIL import Image


class ReactionDataset:
    def __init__(self, path_or_dataset, split, image_root):
        lines = (Path(path_or_dataset) / f"{split}.jsonl").read_text().splitlines()
        self.rows = [json.loads(line) for line in lines if line.strip()]
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_root / row["image"]) as opened:
            image = opened.convert("RGB")
        content = []
        for part in re.split(r"(<image\s*>)", row["prompt"], flags=re.IGNORECASE):
            if re.fullmatch(r"<image\s*>", part, flags=re.IGNORECASE):
                content.append({"type": "image", "image": image})
            elif part:
                content.append({"type": "text", "text": part})
        return {"id": row["id"], "conversation": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]}]}


def make_dataset(path_or_dataset, split, image_root):
    return ReactionDataset(path_or_dataset, split, image_root)


if __name__ == "__main__":
    import nemo_automodel.recipes.vlm.finetune as vlm
    from nemo_automodel.cli.app import main
    original_build = vlm.build_model

    def audited_build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        parameters = {name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad}
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert len(parameters) > 20 and count < 10**9, "Adapters did not attach correctly"
        for name in parameters:
            assert "lora_" in name and name.startswith("language_model."), name
            assert not any(part in name for part in ("out_proj", "lm_head", "vision", "sound", "audio")), name
        assert all(not p.requires_grad for p in model.vision_model.parameters())
        assert all(not p.requires_grad for p in model.vision_projector.parameters())
        audit = Path(os.environ["RUN_DIR"]) / "audit"
        audit.mkdir(parents=True, exist_ok=True)
        (audit / f"rank{os.environ['RANK']}.json").write_text(json.dumps(
            dict(status="passed", trainable_count=count, parameters=parameters), indent=2))
        print(f"LoRA audit passed: {count:,} trainable parameters", flush=True)
        return model

    vlm.build_model = audited_build
    main()
"""

MERGE_SOURCE = r"""
import json
import os
from pathlib import Path
import re
import shutil
import struct

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
OMITTED_KEYS = {'vision_projector.vision_final_layernorm.weight',
                'vision_projector.vision_final_layernorm.bias', 'vision_model.summary_idxs',
                'vision_model.radio_model.input_conditioner.norm_mean',
                'vision_model.radio_model.input_conditioner.norm_std'}

def exported_key(key):
    return not key.startswith('language_model.mtp.') and key not in OMITTED_KEYS

def base_key(name):
    name = name.removeprefix('base_model.model.')
    name = '.'.join(part for part in name.split('.') if part not in ('_checkpoint_wrapped_module', '_orig_mod'))
    name = name.replace('language_model.model.layers.', 'language_model.backbone.layers.', 1)
    if not name.startswith('language_model.backbone.layers.') or any(
            part in name for part in ('out_proj', 'lm_head', 'vision', 'sound', 'audio')):
        raise ValueError(f'Unexpected adapter key: {name}')
    return name

def verify_export(model, expected_keys=None):
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    weights = index['weight_map']
    if expected_keys is not None and set(weights) != set(expected_keys):
        raise ValueError('Export does not contain the complete expected model state')
    if not weights or not (model / 'config.json').is_file():
        raise ValueError('Export lacks weights/configuration')
    size = 0
    for name in sorted(set(weights.values())):
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError(f'Invalid shard filename: {name}')
        path = model / name
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < length < 32 * 1024**2:
                raise ValueError('Invalid safetensors header')
            header = json.loads(stream.read(length))
        entries = {k: v for k, v in header.items() if k != '__metadata__'}
        if set(entries) != {k for k, v in weights.items() if v == name}:
            raise ValueError(f'Index/header mismatch in {name}')
        end = max(v['data_offsets'][1] for v in entries.values())
        if path.stat().st_size != 8 + length + end:
            raise ValueError(f'Incomplete shard: {name}')
        size += end
    if size != index['metadata']['total_size']:
        raise ValueError('Export tensor byte count differs from index')
    return dict(model_path=str(model), tensor_keys=len(weights),
                shards=len(set(weights.values())), tensor_bytes=size)

def copy_metadata(base, output):
    for path in base.iterdir():
        if path.is_file() and path.suffix in ('.json', '.py', '.jinja', '.txt') and not path.name.startswith('.'):
            if path.name != 'model.safetensors.index.json':
                shutil.copyfile(path, output / path.name)
    config = json.loads((output / 'config.json').read_text())
    config['rxn_vision_final_layernorm'] = False
    write_json(output / 'config.json', config)

def merge_lora(base, adapter, output, expected_adapter_keys, rank=64, scale=2):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    weights = load_file(adapter / 'adapter_model.safetensors', device='cpu')
    if {base_key(k) for k in weights} != {base_key(k) for k in expected_adapter_keys}:
        raise ValueError('Saved adapter parameters differ from the trainability audit')
    pairs = {}
    for name, tensor in weights.items():
        match = re.fullmatch(r'(.+)\.lora_([AB])\.weight', name)
        if not match or tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid adapter tensor: {name}')
        pair = pairs.setdefault(base_key(match[1] + '.weight'), {})
        if match[2] in pair:
            raise ValueError(f'Duplicate adapter tensor: {name}')
        pair[match[2]] = tensor
    if not pairs or any(set(pair) != {'A', 'B'} for pair in pairs.values()):
        raise ValueError('Incomplete LoRA A/B pairs')
    if not any(torch.count_nonzero(pair['B']).item() for pair in pairs.values()):
        raise ValueError('No adapter updates were saved')
    original = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    selected = {key: shard for key, shard in original.items() if exported_key(key)}
    if not set(pairs) <= set(selected):
        raise ValueError('Adapter cannot be mapped to this base checkpoint')
    output.mkdir(exist_ok=False)
    (output / '.export_incomplete').touch()
    copy_metadata(base, output)
    mapping, total, used = {}, 0, set()
    shards = sorted(set(selected.values()))
    with torch.no_grad():
        for number, source_name in enumerate(shards, 1):
            result = {}
            with safe_open(base / source_name, framework='pt', device='cpu') as source:
                for key in sorted(k for k, shard in selected.items() if shard == source_name):
                    tensor = source.get_tensor(key)
                    if key in pairs:
                        a, b = pairs[key]['A'], pairs[key]['B']
                        if a.shape[0] != rank or b.shape[1] != rank or (b.shape[0], a.shape[1]) != tensor.shape:
                            raise ValueError(f'Adapter/base shape mismatch: {key}')
                        if tensor.dtype != torch.bfloat16:
                            raise ValueError('Expected a BF16 base for this recipe')
                        # Match native LinearLoRA.materialize_effective_weight BF16 arithmetic.
                        tensor = tensor + scale * (b @ a)
                        if not torch.isfinite(tensor).all():
                            raise ValueError(f'Nonfinite merged tensor: {key}')
                        used.add(key)
                    result[key] = tensor.contiguous()
                    total += tensor.numel() * tensor.element_size()
            name = f'model-{number:05d}-of-{len(shards):05d}.safetensors'
            save_file(result, output / name, metadata={'format': 'pt'})
            mapping.update({key: name for key in result})
            del result
            print(f'Exported shard {number}/{len(shards)}', flush=True)
    if used != set(pairs):
        raise ValueError('Not all adapters were merged')
    write_json(output / 'model.safetensors.index.json', {'metadata': {'total_size': total}, 'weight_map': mapping})
    report = verify_export(output)
    (output / '.export_incomplete').unlink()
    return dict(**report, merged_layers=len(pairs))

if __name__ == "__main__":
    import torch
    torch.set_num_threads(8)
    run = Path(os.environ["RUN_DIR"])
    complete = json.loads((run / "training_complete.json").read_text())
    checkpoint = run / "checkpoints" / f"epoch_0_step_{complete['steps'] - 1}" / "model"
    audits = [json.loads((run / "audit" / f"rank{rank}.json").read_text()) for rank in range(4)]
    assert all(a["status"] == "passed" for a in audits)
    assert all(set(a["parameters"]) == set(audits[0]["parameters"]) for a in audits)
    report = merge_lora(Path(os.environ["MODEL_PATH"]), checkpoint, run / "export", audits[0]["parameters"])
    write_json(run / "export.json", dict(**report, status="verified", **complete))
"""


def write_runtime_scripts(code_dir):
    """Write the audited training entry point and CPU adapter merge script."""
    code_dir = Path(code_dir)
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "train_entry.py").write_text(TRAINING_SOURCE)
    (code_dir / "merge_adapter.py").write_text(MERGE_SOURCE)


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Log:", log_path, flush=True)
    with log_path.open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


# Paper-defined RxnCaption SoftMatch and BIVP HybridMatch.


def _edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def get_bboxes_match(gt_boxes, pred_boxes, iou_threshold=0.5):
    n, m = len(gt_boxes), len(pred_boxes)
    if n == 0 or m == 0:
        return [-1] * n, m, np.zeros((n, m))
    iou_mat = np.array([[_iou(g, p) for p in pred_boxes] for g in gt_boxes])
    match = [-1] * n
    used_pred = set()
    for gi in range(n):
        best_iou, best_pi = 0.0, -1
        for pi in range(m):
            if pi in used_pred:
                continue
            if iou_mat[gi, pi] > best_iou:
                best_iou, best_pi = iou_mat[gi, pi], pi
        if best_iou >= iou_threshold:
            match[gi] = best_pi
            used_pred.add(best_pi)
    return match, m - len(used_pred), iou_mat


class BBox:
    def __init__(self, record):
        self.id = record["id"]
        self.category_id = record.get("category_id", 1)
        self.bbox = record.get("bbox", [0, 0, 0, 0])
        self.text = record.get("text", "")

    @property
    def is_molecule(self):
        return self.category_id == 1

    @property
    def is_text(self):
        return self.category_id in (2, 3)


class Reaction:
    def __init__(self, rxn_dict, bbox_lookup):
        self.reactants = [bbox_lookup[i] for i in rxn_dict.get("reactants", []) if i in bbox_lookup]
        self.conditions = [bbox_lookup[i] for i in rxn_dict.get("conditions", []) if i in bbox_lookup]
        self.products = [bbox_lookup[i] for i in rxn_dict.get("products", []) if i in bbox_lookup]

    def is_empty(self, mol_only=False):
        members = self.reactants + self.conditions + self.products
        if mol_only:
            members = [b for b in members if b.is_molecule]
        return len(members) == 0


def _soft_match(gr, pr):
    """RxnCaption SoftMatch: ignore text and merge conditions into reactants."""

    def split(rxn):
        bboxes, groups = [], []
        for b in rxn.reactants + rxn.conditions:
            if b.is_molecule:
                bboxes.append(b.bbox)
                groups.append("rc")
        for b in rxn.products:
            if b.is_molecule:
                bboxes.append(b.bbox)
                groups.append("p")
        return bboxes, groups

    gb, gg = split(gr)
    pb, pg = split(pr)
    if len(gb) != len(pb):
        return False
    if not gb:
        return True
    m, _, _ = get_bboxes_match(gb, pb)
    if any(v == -1 for v in m):
        return False
    return all(gg[gi] == pg[m[gi]] for gi in range(len(gb)))


def _hybrid_match(gr, pr, ratio=0.2):
    """RxnCaption HybridMatch for BIVP.

    - Molecules: 3-way role-aware IoU matching (RxnScribe Hard-Match style).
    - Reactants text: multiset must match exactly (string equality).
    - Products text: multiset must match exactly (string equality).
    - Conditions text: count must match; each within normalized edit distance <= ratio.
    """
    for role in ("reactants", "conditions", "products"):
        gb = [b.bbox for b in getattr(gr, role) if b.is_molecule]
        pb = [b.bbox for b in getattr(pr, role) if b.is_molecule]
        if len(gb) != len(pb):
            return False
        if gb:
            m, _, _ = get_bboxes_match(gb, pb)
            if any(v == -1 for v in m):
                return False

    for role in ("reactants", "products"):
        gt_t = sorted(b.text for b in getattr(gr, role) if b.is_text)
        pr_t = sorted(b.text for b in getattr(pr, role) if b.is_text)
        if gt_t != pr_t:
            return False

    gt_t = sorted(b.text for b in gr.conditions if b.is_text)
    pr_t = sorted(b.text for b in pr.conditions if b.is_text)
    if len(gt_t) != len(pr_t):
        return False
    used = set()
    for g in gt_t:
        ok = False
        for j, p in enumerate(pr_t):
            if j in used:
                continue
            d = _edit_distance(g, p)
            ml = max(len(g), len(p), 1)
            if d / ml <= ratio:
                used.add(j)
                ok = True
                break
        if not ok:
            return False
    return True


def evaluate_image(gold_img, pred_img, match_fn, mol_only=False):
    """Evaluate one image, return TP/FP/FN."""
    gl = {b["id"]: BBox(b) for b in gold_img.get("bboxes", [])}
    pl = {b["id"]: BBox(b) for b in pred_img.get("bboxes", [])}

    gt_rxns = [Reaction(r, gl) for r in gold_img.get("reactions", [])]
    pr_rxns = [Reaction(r, pl) for r in pred_img.get("reactions", [])]

    gt = [r for r in gt_rxns if not r.is_empty(mol_only)]
    pr = [r for r in pr_rxns if not r.is_empty(mol_only)]

    if not gt and not pr:
        return 0, 0, 0, True

    matched_pr = set()
    tp = 0
    for gi, gr in enumerate(gt):
        for pi, prr in enumerate(pr):
            if pi in matched_pr:
                continue
            if match_fn(gr, prr):
                matched_pr.add(pi)
                tp += 1
                break

    return tp, len(pr) - tp, len(gt) - tp, False


def convert(pred_path, mapdict_path, gt_path, output_path):
    """Convert validated predictions from numbered boxes to the scoring format."""
    predictions = json.loads(Path(pred_path).read_text())
    mappings = json.loads(Path(mapdict_path).read_text())
    ground_truth = json.loads(Path(gt_path).read_text())
    lookup = {entry["file_name"].split("/")[-1]: entry.get("map_dict", {}) for entry in mappings}
    output = {**ground_truth, "images": []}

    for name, entry in predictions.items():
        mapping = lookup.get(name, {})
        image = {"file_name": name, "bboxes": [], "reactions": []}
        for index, info in mapping.items():
            image["bboxes"].append({"id": int(index) - 1, "category_id": 1, "bbox": info["yolo_bbox"]})
        next_id = max((int(index) for index in mapping), default=0)

        for reaction in json.loads(entry["response"]):
            converted = {}
            for role in REACTION_FIELDS:
                converted[role] = []
                for item in reaction[role]:
                    if item["type"] == "mol":
                        converted[role].append(item["index"] - 1)
                    else:
                        converted[role].append(next_id)
                        image["bboxes"].append(
                            {"id": next_id, "category_id": 2, "bbox": [0, 0, 0, 0], "text": item["content"]}
                        )
                        next_id += 1
            image["reactions"].append(converted)

        valid_ids = {box["id"] for box in image["bboxes"]}
        for reaction in image["reactions"]:
            for role in REACTION_FIELDS:
                reaction[role] = [index for index in reaction[role] if index in valid_ids]
        output["images"].append(image)

    write_json(output_path, output)


def normalize_response(response):
    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        value, _, _ = _clean_target(cleaned, remove_empty_text=False)
        return value, False
    except (ValueError, KeyError, TypeError):
        return "[]", True


def score(ground_truth, converted):
    gold = {row["file_name"]: row for row in json.loads(ground_truth.read_text())["images"]}
    predictions = {row["file_name"]: row for row in json.loads(converted.read_text())["images"]}
    if set(gold) != set(predictions):
        raise ValueError("Every benchmark image must be included in scoring")
    result = {}
    for name, function, mol_only in (
        ("SoftMatch", _soft_match, True),
        ("HybridMatch", _hybrid_match, False),
    ):
        tp = fp = fn = 0
        for filename in sorted(gold):
            a, b, c, _ = evaluate_image(gold[filename], predictions[filename], function, mol_only)
            tp, fp, fn = tp + a, fp + b, fn + c
        result[name] = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=100 * tp / (tp + fp) if tp + fp else 0,
            recall=100 * tp / (tp + fn) if tp + fn else 0,
            f1=200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        )
    return result


# Use the same reaction prompt and decoding settings as the reference runs.

PROMPT_BEFORE = 'You are an expert in chemical image structure analysis. You will be given an image in which all "molecular structures" have been boxed and numbered. Your task is to identify and reconstruct the chemical reaction(s) in the image based on these boxed structures. Please follow the rules below:\n1. Each reaction must contain three fields: reactants, conditions, products (each is a List[Dict])\n2. Each Dict must include:\n   - type: one of "mol", "txt", or "idt"\n   - if type is "mol", second field is index (box ID);\n   - if type is "txt" or "idt", second field is content (raw text)\n3. Boxed molecular structures must be type "mol". Other elements must use "txt" or "idt"\n4. Ignore decorations, arrows, and illustrations; only extract real elements\n5. Clean up identifiers and retain only the core ID\n6. If no reaction exists in the image, return an empty list []\n7. Use arrow direction: tail = reactants, head = products\nExample format:\n[\n  {\n    "reactants": [\n      {"type": "mol", "index": 1},\n      {"type": "txt", "content": "NaCl"}\n    ],\n    "conditions": [\n      {"type": "mol", "index": 3},\n      {"type": "txt", "content": "H2O, 25°C"}\n    ],\n    "products": [\n      {"type": "mol", "index": 2},\n      {"type": "idt", "content": "1a"}\n    ]\n  }\n]\nDo not include any explanations, comments, or non-structured data.'  # noqa: E501

PROMPT_AFTER = "Please extract the list of reactions from the image. Return only the JSON reaction list without adding any other content."  # noqa: E501


def predict(image, endpoint):
    encoded = base64.b64encode(image.read_bytes()).decode()
    payload = dict(
        model="model",
        temperature=0,
        max_tokens=2048,
        chat_template_kwargs={"enable_thinking": False},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_BEFORE},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    {"type": "text", "text": PROMPT_AFTER},
                ],
            }
        ],
    )
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as reply:
        content = json.load(reply)["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("Server returned no answer content")
    return content


def evaluate_checkpoint(run_dir, benchmark, *, port, docker_command, container_path):
    """Serve the export, score all images, and stop only this run's container.

    The notebook supplies its Docker command builder and path mapping so the
    visible launch settings are also used for evaluation. No notebook globals
    are read by this module.
    """
    exported = json.loads((run_dir / "export.json").read_text())
    model = run_dir / "export"
    assert exported["status"] == "verified" and not (model / ".export_incomplete").exists()
    report = json.loads((run_dir / "prepared/report.json").read_text())
    for name, digest in report["benchmark_labels_sha256"].items():
        assert sha256(benchmark / name) == digest, "Benchmark labels changed"
    for name, digest in report["benchmark_sha256"].items():
        assert sha256(benchmark / "images/gt_bivp" / name) == digest, "Benchmark image changed"
    output = run_dir / "evaluation"
    output.mkdir(exist_ok=False)
    view = output / "serving_model"
    view.mkdir()
    for path in model.iterdir():
        if path.is_file() and path.suffix in (".json", ".py", ".txt", ".jinja"):
            shutil.copyfile(path, view / path.name)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        assert Path(name).name == name
        (view / name).symlink_to(container_path(model / name))
    config = json.loads((view / "config.json").read_text())
    config["rxn_vision_final_layernorm"] = False
    write_json(view / "config.json", config)
    tokenizer = json.loads((view / "tokenizer_config.json").read_text())
    tokenizer["tokenizer_class"] = "PreTrainedTokenizerFast"
    write_json(view / "tokenizer_config.json", tokenizer)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    cidfile = output / "container.id"
    command = docker_command(
        "eval",
        [
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            container_path(view),
            "--served-model-name",
            "model",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "2",
            "--max-model-len",
            "65536",
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.85",
            "--no-enable-prefix-caching",
            "--reasoning-parser",
            "nemotron_v3",
        ],
        gpu=True,
        extra_options=["--cidfile", str(cidfile)],
    )
    endpoint = f"http://127.0.0.1:{port}"
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError("vLLM exited; inspect evaluation/server.log")
                try:
                    with urllib.request.urlopen(endpoint + "/v1/models", timeout=2) as response:
                        available = json.load(response)["data"]
                    if any(row["id"] == "model" for row in available):
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Model startup exceeded 20 minutes")
                time.sleep(5)
            raw, normalized, invalid, errors = {}, {}, [], []
            for number, name in enumerate(sorted(report["benchmark_sha256"]), 1):
                try:
                    response = predict(benchmark / "images/gt_bivp" / name, endpoint)
                except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as error:
                    errors.append(name)
                    response = f"ERROR: {type(error).__name__}"
                value, bad = normalize_response(response)
                if bad:
                    invalid.append(name)
                raw[name], normalized[name] = {"response": response}, {"response": value}
                write_json(output / "raw.json", raw)
                if number % 10 == 0:
                    print(f"Evaluated {number}/400", flush=True)
            write_json(output / "normalized.json", normalized)
            gold = benchmark / "ground_truth/ground_truth.json"
            convert(
                output / "normalized.json", benchmark / "mapdicts/mapdict_gt.json", gold, output / "converted.json"
            )
            results = dict(
                images=len(raw),
                inference_errors=errors,
                invalid_json_images=invalid,
                smoke_test=exported["smoke_test"],
                metrics=score(gold, output / "converted.json"),
            )
            write_json(output / "metrics.json", results)
            if errors:
                raise RuntimeError(f"{len(errors)} inference requests failed; inspect the saved report")
            return results
        finally:
            # Only stop the container created by this invocation, identified by Docker.
            if cidfile.exists():
                cid = cidfile.read_text().strip()
                if re.fullmatch(r"[a-f0-9]{64}", cid):
                    subprocess.run(
                        ["docker", "stop", "--time", "10", cid],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)
