"""CPU tests for the public U-RxnDiagram-15k conversion."""

import json
import os
import re
import runpy
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "lora-1-node"))

from prepare_rxncaption_data import (  # noqa: E402
    DATASET_REVISION,
    DATASET_URL,
    PROMPT_AFTER,
    PROMPT_BEFORE,
    prepare_dataset,
)
from rxncaption_support import prepare_public_data, require_sources  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class DatasetConversionTests(unittest.TestCase):
    def test_notebook_defaults_are_complete_and_smoke_first(self):
        notebook_path = ROOT / "lora-1-node/chemical_reaction_parsing_lora.ipynb"
        notebook = json.loads(notebook_path.read_text())
        namespace = {"Path": Path, "os": os, "re": re}
        with patch.dict(os.environ, {}, clear=True):
            exec("".join(notebook["cells"][7]["source"]), namespace)
        require_sources(namespace["CONFIG"])
        self.assertEqual(namespace["SMOKE_STEPS"], 5)
        self.assertNotIn("benchmark", namespace["CONFIG"])
        self.assertIn("@sha256:", namespace["CONFIG"]["train_image"])
        self.assertIn("@sha256:", namespace["CONFIG"]["eval_image"])

    def test_prompt_matches_training_and_evaluation(self):
        shared = runpy.run_path(Path(__file__).parent / "sft-4-node/super_vl_rxncaption/prompt.py")
        self.assertEqual(PROMPT_BEFORE, shared["PROMPT_BEFORE"])
        self.assertEqual(PROMPT_AFTER, shared["PROMPT_AFTER"])

    def make_split(self, root, split, name):
        directory = root / f"{split}_set"
        images = directory / "images"
        images.mkdir(parents=True)
        Image.new("RGB", (100, 100), "white").save(images / name)
        image = {
            "id": 1,
            "file_name": name,
            "bboxes": [
                {"id": 10, "category_id": 1, "bbox": [60, 40, 20, 20]},
                {"id": 11, "category_id": 1, "bbox": [5, 5, 20, 20]},
                {"id": 12, "category_id": 2, "bbox": [30, 30, 5, 5], "text": "H2O"},
                {"id": 13, "category_id": 3, "bbox": [80, 80, 5, 5], "text": "1a"},
                {"id": 14, "category_id": 4, "bbox": [0, 90, 5, 5]},
            ],
            "reactions": [
                {
                    "reactants": [11, 12],
                    "conditions": [14],
                    "products": [10, 13],
                }
            ],
        }
        write_json(directory / "ground_truth.json", {"images": [image]})

    def test_builds_training_and_benchmark_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "raw", root / "converted"
            self.make_split(source, "train", "train.png")
            self.make_split(source, "test", "test.png")

            report = prepare_dataset(source, output, expected_train=1, expected_test=1, workers=2)
            self.assertEqual(report["source"], DATASET_URL)
            self.assertEqual((report["train_images"], report["test_images"]), (1, 1))

            row = json.loads((output / "finetuning-data/train/dataset.jsonl").read_text())
            self.assertEqual(row["images"], ["images/train.png"])
            self.assertEqual([message["role"] for message in row["messages"]], ["user", "assistant"])
            self.assertEqual(row["messages"][0]["content"].count("<image>"), 1)
            target = json.loads(row["messages"][1]["content"])
            self.assertEqual(
                target,
                [
                    {
                        "reactants": [
                            {"type": "mol", "index": 1},
                            {"type": "txt", "content": "H2O"},
                        ],
                        "conditions": [],
                        "products": [
                            {"type": "mol", "index": 2},
                            {"type": "idt", "content": "1a"},
                        ],
                    }
                ],
            )

            mapping = json.loads((output / "benchmark/mapdicts/mapdict_gt.json").read_text())
            self.assertEqual(mapping[0]["map_dict"]["1"]["gt_id"], 11)
            self.assertEqual(mapping[0]["map_dict"]["2"]["gt_id"], 10)
            self.assertEqual(
                json.loads((output / "benchmark/ground_truth/ground_truth.json").read_text())["images"][0][
                    "file_name"
                ],
                "test.png",
            )
            with Image.open(output / "finetuning-data/train/images/train.png") as image:
                self.assertNotEqual(image.getpixel((10, 10)), (255, 255, 255))
            self.assertTrue((output / "benchmark/images/gt_bivp/test.png").is_file())
            self.assertIn("NonCommercial", (output / "finetuning-data/DATA_LICENSE.md").read_text())

    def test_rejects_wrong_release_size_without_leaving_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "raw", root / "converted"
            self.make_split(source, "train", "train.png")
            self.make_split(source, "test", "test.png")
            with self.assertRaisesRegex(ValueError, "expected 15000"):
                prepare_dataset(source, output, workers=1)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(".converted.partial").exists())

    def test_public_download_builds_and_reuses_both_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, archives, assets = root / "raw", root / "archives", root / "assets"
            archives.mkdir()
            self.make_split(raw, "train", "train.png")
            self.make_split(raw, "test", "test.png")
            for split in ("train", "test"):
                archive = archives / f"{split}_set.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    for path in (raw / f"{split}_set").rglob("*"):
                        if path.is_file():
                            stream.write(path, path.relative_to(raw))

            def download_file(**kwargs):
                return archives / kwargs["filename"]

            spec = {
                "repo_id": "opendatalab/U-RxnDiagram-15k",
                "revision": DATASET_REVISION,
                "train_images": 1,
                "test_images": 1,
            }
            prepare_public_data(spec, assets, download_file=download_file)
            self.assertTrue((assets / "data/train/dataset.jsonl").is_file())
            self.assertTrue((assets / "benchmark/images/gt_bivp/test.png").is_file())

            prepare_public_data(
                spec,
                assets,
                download_file=lambda **kwargs: self.fail("completed data must be reused"),
            )


if __name__ == "__main__":
    unittest.main()
