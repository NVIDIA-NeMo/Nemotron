#!/usr/bin/env python3
"""Convert U-RxnDiagram-15k into the BIVP data used by these tutorials.

The source dataset is CC BY-NC 4.0. This script does not download or redistribute
it; follow the adjacent README to obtain and extract the public archives.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

DATASET_URL = "https://huggingface.co/datasets/opendatalab/U-RxnDiagram-15k"
DATASET_REVISION = "274be957df104c4fde2104c2e02578b643aad3d1"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
REACTION_FIELDS = ("reactants", "conditions", "products")

PROMPT_BEFORE = (
    'You are an expert in chemical image structure analysis. You will be given an image in which all "molecular '
    'structures" have been boxed and numbered. Your task is to identify and reconstruct the chemical reaction(s) '
    "in the image based on these boxed structures. Please follow the rules below:\n"
    "1. Each reaction must contain three fields: reactants, conditions, products (each is a List[Dict])\n"
    '2. Each Dict must include:\n   - type: one of "mol", "txt", or "idt"\n'
    '   - if type is "mol", second field is index (box ID);\n'
    '   - if type is "txt" or "idt", second field is content (raw text)\n'
    '3. Boxed molecular structures must be type "mol". Other elements must use "txt" or "idt"\n'
    "4. Ignore decorations, arrows, and illustrations; only extract real elements\n"
    "5. Clean up identifiers and retain only the core ID\n"
    "6. If no reaction exists in the image, return an empty list []\n"
    "7. Use arrow direction: tail = reactants, head = products\n"
    "Example format:\n"
    "[\n"
    "  {\n"
    '    "reactants": [\n'
    '      {"type": "mol", "index": 1},\n'
    '      {"type": "txt", "content": "NaCl"}\n'
    "    ],\n"
    '    "conditions": [\n'
    '      {"type": "mol", "index": 3},\n'
    '      {"type": "txt", "content": "H2O, 25°C"}\n'
    "    ],\n"
    '    "products": [\n'
    '      {"type": "mol", "index": 2},\n'
    '      {"type": "idt", "content": "1a"}\n'
    "    ]\n"
    "  }\n"
    "]\n"
    "Do not include any explanations, comments, or non-structured data."
)
PROMPT_AFTER = (
    "Please extract the list of reactions from the image. "
    "Return only the JSON reaction list without adding any other content."
)
USER_PROMPT = f"{PROMPT_BEFORE}\n<image>\n{PROMPT_AFTER}"


def data_license(dataset_url: str = DATASET_URL, revision: str = DATASET_REVISION) -> str:
    """Return the attribution notice embedded in every converted data bundle."""
    return f"""# U-RxnDiagram-15k data license

Source: {dataset_url}
Revision: {revision}

The source images and annotations are licensed under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0):
{LICENSE_URL}

- Attribution is required.
- Commercial use is prohibited unless you obtain separate permission from the
  dataset rights holder.
- The numbered-box images and JSONL files produced by this script are adaptations
  of the source dataset. When sharing them, retain this notice, link the source
  and license, and state that bounding boxes/numeric labels and training-format
  records were generated for the Nemotron Super VL RxnCaption tutorial.

Citation:

Song, Jiahe, et al. "RxnCaption: Reformulating Reaction Diagram Parsing as
Visual Prompt Guided Captioning." arXiv:2511.02384.

```bibtex
@article{{song2025rxncaption,
  title={{RxnCaption: Reformulating Reaction Diagram Parsing as Visual Prompt Guided Captioning}},
  author={{Song, Jiahe and Wang, Chuang and Jiang, Bowen and Wang, Yinfan and Zheng, Hao and others}},
  journal={{arXiv preprint arXiv:2511.02384}},
  year={{2025}}
}}
```
"""


DATA_LICENSE = data_license()


@dataclass(frozen=True)
class ConvertedImage:
    """One training row and its numbered-box mapping."""

    file_name: str
    row: dict[str, Any]
    mapping: dict[str, Any]


def _find_split(source: Path, split_name: str) -> Path:
    """Find one extracted ``train_set`` or ``test_set`` directory."""
    direct = source / split_name
    if (direct / "ground_truth.json").is_file() and (direct / "images").is_dir():
        return direct
    candidates = [
        path.parent
        for path in source.rglob("ground_truth.json")
        if path.parent.name == split_name and (path.parent / "images").is_dir()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one extracted {split_name}/ground_truth.json and {split_name}/images under {source}; "
            f"found {len(candidates)}. Extract {split_name}.zip first."
        )
    return candidates[0]


def _reading_order(records: list[tuple[list[float], int]]) -> list[tuple[list[float], int]]:
    """Match RxnCaption's top-to-bottom, then left-to-right BIVP ordering."""
    if not records:
        return []
    sorted_records = sorted(records, key=lambda record: (record[0][1], record[0][0]))
    rows: list[list[tuple[list[float], int]]] = []
    current = [sorted_records[0]]
    for record in sorted_records[1:]:
        first = current[0][0]
        height = first[3] - first[1]
        if first[1] - height / 2 <= record[0][1] <= first[1] + height / 2:
            current.append(record)
        else:
            rows.append(current)
            current = [record]
    rows.append(current)
    return [record for row in rows for record in sorted(row, key=lambda value: value[0][0])]


def _draw_index_label(
    draw: ImageDraw.ImageDraw,
    number: int,
    x_min: float,
    y_min: float,
    font: ImageFont.ImageFont,
    padding: int = 2,
) -> None:
    box = draw.textbbox((0, 0), str(number), font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.rectangle((x_min, y_min, x_min + width + 2 * padding, y_min + 1.8 * height), fill="black")
    draw.text((x_min + padding, y_min + padding), str(number), fill="white", font=font)


def _annotate(source: Path, destination: Path, ordered: list[tuple[list[float], int]]) -> None:
    """Draw the same blue boxes and one-based numeric labels used for training."""
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    if ordered:
        minimum = min(min(box[2] - box[0], box[3] - box[1]) for box, _ in ordered)
        if minimum < 50:
            font_size = max(int(minimum * 0.5), 24)
        elif minimum < 100:
            font_size = max(int(minimum * 0.4), 36)
        else:
            font_size = max(int(minimum * 0.3), 48)
        font = ImageFont.load_default(size=min(font_size, 48))
        for number, (box, _) in enumerate(ordered, 1):
            x_min, y_min, x_max, y_max = box
            draw.rectangle((x_min, y_min, x_max, y_max), outline="blue", width=4)
            _draw_index_label(draw, number, x_min, y_min, font)
    image.save(destination)


def _target(image: dict[str, Any], visual_indices: dict[int, int]) -> list[dict[str, list[dict[str, Any]]]]:
    """Convert source bbox IDs into numbered molecules and inline text items."""
    boxes = {box["id"]: box for box in image.get("bboxes", [])}
    if len(boxes) != len(image.get("bboxes", [])):
        raise ValueError(f"{image['file_name']}: duplicate bbox IDs")
    converted_reactions = []
    for reaction in image.get("reactions", []):
        converted: dict[str, list[dict[str, Any]]] = {}
        for role in REACTION_FIELDS:
            converted[role] = []
            for bbox_id in reaction.get(role, []):
                if bbox_id not in boxes:
                    raise ValueError(f"{image['file_name']}: reaction references missing bbox {bbox_id}")
                box = boxes[bbox_id]
                category = box.get("category_id")
                if category == 1:
                    converted[role].append({"type": "mol", "index": visual_indices[bbox_id]})
                elif category in (2, 3):
                    text = box.get("text", "")
                    if not isinstance(text, str):
                        raise ValueError(f"{image['file_name']}: bbox {bbox_id} has non-string text")
                    converted[role].append({"type": "txt" if category == 2 else "idt", "content": text})
                elif category != 4:
                    raise ValueError(f"{image['file_name']}: bbox {bbox_id} has unknown category {category}")
        converted_reactions.append(converted)
    return converted_reactions


def _convert_image(image: dict[str, Any], source_images: Path, output_images: Path) -> ConvertedImage:
    file_name = image.get("file_name")
    if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
        raise ValueError(f"Image names must be nonempty basenames; got {file_name!r}")
    source = source_images / file_name
    if not source.is_file():
        raise FileNotFoundError(source)

    structures: list[tuple[list[float], int]] = []
    for box in image.get("bboxes", []):
        if box.get("category_id") != 1:
            continue
        coordinates = box.get("bbox")
        if not isinstance(coordinates, list) or len(coordinates) != 4:
            raise ValueError(f"{file_name}: bbox {box.get('id')} must be [x, y, width, height]")
        x, y, width, height = coordinates
        if not all(isinstance(value, (int, float)) for value in coordinates) or width <= 0 or height <= 0:
            raise ValueError(f"{file_name}: bbox {box.get('id')} has invalid coordinates")
        structures.append(([x, y, x + width, y + height], box["id"]))
    ordered = _reading_order(structures)
    visual_indices = {bbox_id: number for number, (_, bbox_id) in enumerate(ordered, 1)}
    _annotate(source, output_images / file_name, ordered)

    target = _target(image, visual_indices)
    row = {
        "messages": [
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ],
        "images": [f"images/{file_name}"],
    }
    source_boxes = {box["id"]: box for box in image.get("bboxes", [])}
    mapping = {
        str(number): {
            "gt_id": bbox_id,
            "yolo_bbox": source_boxes[bbox_id]["bbox"],
            "gt_bbox": source_boxes[bbox_id]["bbox"],
        }
        for number, (_, bbox_id) in enumerate(ordered, 1)
    }
    return ConvertedImage(file_name, row, mapping)


def _load_ground_truth(split: Path, expected: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truth = json.loads((split / "ground_truth.json").read_text())
    images = ground_truth.get("images")
    if not isinstance(images, list) or len(images) != expected:
        raise ValueError(
            f"{split}: expected {expected} annotated images, found {len(images) if isinstance(images, list) else 0}"
        )
    names = [image.get("file_name") for image in images]
    if len(set(names)) != len(names):
        raise ValueError(f"{split}: duplicate image names")
    return ground_truth, images


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _convert_split(
    source: Path,
    output: Path,
    split: str,
    expected: int,
    workers: int,
) -> tuple[dict[str, Any], list[ConvertedImage]]:
    raw_name = f"{split}_set"
    source_split = _find_split(source, raw_name)
    ground_truth, images = _load_ground_truth(source_split, expected)
    destination = output / "finetuning-data" / split
    image_output = destination / "images"
    image_output.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        converted = list(
            pool.map(
                lambda image: _convert_image(image, source_split / "images", image_output),
                images,
            )
        )
    (destination / "dataset.jsonl").write_text(
        "".join(json.dumps(record.row, ensure_ascii=False, separators=(",", ":")) + "\n" for record in converted)
    )
    return ground_truth, converted


def prepare_dataset(
    source: Path,
    output: Path,
    *,
    expected_train: int = 15_000,
    expected_test: int = 400,
    workers: int = 8,
    dataset_url: str = DATASET_URL,
    source_revision: str = DATASET_REVISION,
) -> dict[str, Any]:
    """Build tutorial data and benchmark directories from an extracted public release."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if workers < 1:
        raise ValueError("workers must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Output or partial output already exists: {output}")
    partial.mkdir()
    try:
        _, train = _convert_split(source, partial, "train", expected_train, workers)
        test_ground_truth, test = _convert_split(source, partial, "test", expected_test, workers)
        train_ids = {Path(record.file_name).stem for record in train}
        test_ids = {Path(record.file_name).stem for record in test}
        if train_ids & test_ids:
            raise ValueError("Train and test image IDs overlap")

        benchmark = partial / "benchmark"
        benchmark_images = benchmark / "images" / "gt_bivp"
        benchmark_images.mkdir(parents=True)
        test_images = partial / "finetuning-data" / "test" / "images"
        for record in test:
            source_image = test_images / record.file_name
            destination = benchmark_images / record.file_name
            try:
                os.link(source_image, destination)
            except OSError:
                shutil.copy2(source_image, destination)
        _write_json(benchmark / "ground_truth" / "ground_truth.json", test_ground_truth)
        _write_json(
            benchmark / "mapdicts" / "mapdict_gt.json",
            [{"file_name": record.file_name, "map_dict": record.mapping} for record in test],
        )

        for directory in (partial, partial / "finetuning-data", benchmark):
            (directory / "DATA_LICENSE.md").write_text(data_license(dataset_url, source_revision))
        report = {
            "source": dataset_url,
            "source_revision": source_revision,
            "license": "CC BY-NC 4.0",
            "train_images": len(train),
            "test_images": len(test),
            "benchmark_images": len(test),
            "format": "BIVP numbered molecule boxes with typed JSON reaction targets",
        }
        _write_json(partial / "manifest.json", report)
        partial.replace(output)
        return report
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing extracted train_set/ and test_set/ from U-RxnDiagram-15k",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for converted data")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.source_dir, args.output_dir, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
