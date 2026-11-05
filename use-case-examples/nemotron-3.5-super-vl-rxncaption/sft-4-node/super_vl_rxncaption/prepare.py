"""Validate downloaded BIVP inputs and reserve a deterministic validation split."""

import json
import math
from pathlib import Path

from assets import sha256, write_json
from rxncaption_tuning_adapter import make_tuning_dataset


def prepare(workspace, run, validation_size=400):
    data = workspace / "assets/data"
    benchmark = workspace / "assets/benchmark"
    for split in ("train", "test"):
        seen = set()
        for line in (data / split / "dataset.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("images"), list) or len(row["images"]) != 1:
                raise ValueError("Each data row must have exactly one image")
            relative = Path(row["images"][0])
            if relative.is_absolute() or not (data / split / relative).resolve().is_relative_to(
                (data / split).resolve()
            ):
                raise ValueError("Training image paths must stay inside their split directory")
            if relative.stem in seen:
                raise ValueError(f"Duplicate image ID in {split}: {relative.stem}")
            seen.add(relative.stem)
    common = dict(
        path_or_dataset=str(data),
        validation_size=validation_size,
        validation_seed=42,
        filter_contaminated=True,
        remove_empty_text=True,
        verify_images=True,
        include_system_prompt=False,
    )
    train = make_tuning_dataset(**common, split="train", medium_repeats=2, dense_repeats=3)
    valid = make_tuning_dataset(**common, split="validation")
    gold = json.loads((benchmark / "ground_truth/ground_truth.json").read_text())["images"]
    names = [row["file_name"] for row in gold]
    if len(names) != len(set(names)) or len(names) != 400:
        raise ValueError("Expected exactly 400 distinct benchmark images")
    if any(Path(name).name != name for name in names):
        raise ValueError("Benchmark filenames must be plain basenames")
    image_dir = benchmark / "images/gt_bivp"
    if {p.name for p in image_dir.iterdir() if p.is_file()} != set(names):
        raise ValueError("Benchmark image coverage differs from its ground truth")
    maps = json.loads((benchmark / "mapdicts/mapdict_gt.json").read_text())
    map_names = [Path(row["file_name"]).name for row in maps]
    if len(map_names) != len(set(map_names)) or set(map_names) != set(names):
        raise ValueError("Benchmark mapping must cover every image exactly once")
    test = [json.loads(line) for line in (data / "test/dataset.jsonl").read_text().splitlines() if line.strip()]
    test_ids = [Path(row["images"][0]).stem for row in test]
    train_ids = {row["id"] for row in train.records}
    valid_ids = {row["id"] for row in valid.records}
    benchmark_ids = {Path(name).stem for name in names}
    if len(test_ids) != 400 or set(test_ids) != benchmark_ids:
        raise ValueError("Packaged test IDs must match the separate benchmark")
    if train_ids & valid_ids or (train_ids | valid_ids) & benchmark_ids:
        raise ValueError("Training, validation, and benchmark splits overlap")
    run.mkdir(parents=True, exist_ok=True)
    split_ids = dict(train=sorted(train_ids), validation=sorted(valid_ids), benchmark=sorted(benchmark_ids))
    write_json(run / "split_ids.json", split_ids)
    # Decode a real image and materialize the conversation to catch schema errors early.
    sample = train[0]
    image = next(item["image"] for item in sample["conversation"][0]["content"] if item["type"] == "image")
    image.save(run / "example.png")
    report = dict(
        unique_train_rows=len(train_ids),
        weighted_train_rows=len(train),
        validation_rows=len(valid),
        benchmark_images=len(names),
        full_steps=math.ceil(len(train) / 16),
        train_cleanup=train.stats,
        input_sha256={split: sha256(data / split / "dataset.jsonl") for split in ("train", "test")},
        benchmark_sha256={name: sha256(image_dir / name) for name in sorted(names)},
    )
    report["benchmark_labels_sha256"] = {
        name: sha256(benchmark / name) for name in ("ground_truth/ground_truth.json", "mapdicts/mapdict_gt.json")
    }
    write_json(run / "prepared.json", report)
    return report
