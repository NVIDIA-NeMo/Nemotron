"""Download versioned tutorial inputs; no machine-specific input paths."""

import hashlib
import json
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


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
    """Check required settings, HTTPS URLs, checksums, and pinned revisions."""
    missing = []
    for section, keys in {
        "model": ["repo_id", "revision"],
        "data": ["url", "sha256"],
        "benchmark": ["url", "sha256"],
        "automodel": ["url", "revision"],
    }.items():
        for key in keys:
            value = config[section].get(key, "")
            if not isinstance(value, str) or not value or "${" in value:
                missing.append(f"{section}.{key}")
    for key in ("train_image", "eval_image"):
        value = config.get(key, "")
        if not isinstance(value, str) or not value or "${" in value:
            missing.append(key)
    if missing:
        raise ValueError(
            "Set download sources before running: "
            + ", ".join(missing)
            + ". Fill in settings.yaml using settings.example.yaml."
        )
    for name in ("data", "benchmark"):
        if not config[name]["url"].startswith("https://"):
            raise ValueError(f"{name}.url must use HTTPS")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", config[name]["sha256"]):
            raise ValueError(f"{name}.sha256 must be a SHA-256 digest")
    for name in ("model", "automodel"):
        if not re.fullmatch(r"[a-fA-F0-9]{40}", config[name]["revision"]):
            raise ValueError(f"{name}.revision must be an immutable 40-character commit")


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


def download_zip(spec, destination, required):
    """Reuse a verified dataset, or download, check, and extract its ZIP."""
    destination = Path(destination)
    marker = destination / ".download.json"
    identity = {"sha256": spec["sha256"].lower()}
    # The marker records the archive checksum of a completed extraction.
    if marker.is_file():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"Different download already present at {destination}; use another workspace")
        if not all((destination / name).is_file() for name in required):
            raise ValueError(f"Incomplete download: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Unverified directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download URLs may contain signed query strings; log only artifact identities.
    archive = destination.with_suffix(".download.zip")
    with urllib.request.urlopen(spec["url"], timeout=120) as response, archive.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    # Verify the downloaded bytes before extracting them.
    if sha256(archive) != identity["sha256"]:
        raise ValueError(f"Download checksum mismatch for {destination.name}")
    extract_zip(archive, destination, required)
    # Record success only after the expected files have been extracted.
    write_json(marker, identity)
    archive.unlink()


def download(config, workspace):
    """Fetch the checkpoint, datasets, AutoModel source, and container images."""
    require_sources(config)  # Fail before downloading hundreds of GB if a URL is missing.
    root = workspace / "assets"
    root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    # 1. Download the pinned HF checkpoint into this workspace.
    model = config["model"]
    model_dir = root / "model"
    model_marker = model_dir / ".download.json"
    if model_marker.exists() and json.loads(model_marker.read_text()) != model:
        raise ValueError("Model source changed; choose a new workspace")
    snapshot_download(repo_id=model["repo_id"], revision=model["revision"], local_dir=model_dir)
    write_json(model_marker, model)
    # 2. Fetch and verify the training data and benchmark separately.
    download_zip(config["data"], root / "data", ["train/dataset.jsonl", "test/dataset.jsonl"])
    download_zip(
        config["benchmark"], root / "benchmark", ["ground_truth/ground_truth.json", "mapdicts/mapdict_gt.json"]
    )
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
    # Store identities, not signed URLs or authentication tokens.
    write_json(
        root / "sources.json",
        {
            "model": model,
            "automodel_revision": spec["revision"],
            "data_sha256": config["data"]["sha256"],
            "benchmark_sha256": config["benchmark"]["sha256"],
            "train_image": config["train_image"],
            "eval_image": config["eval_image"],
        },
    )
