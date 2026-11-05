"""CPU regression checks for the direct workflow; no model downloads or GPU jobs."""

import ast
import copy
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml
from assets import download_zip, extract_zip, require_sources, sha256, write_json
from convert_predictions import convert
from evaluate import normalize_response, score
from export import merge_lora, verify_export
from metrics import run_evaluation
from pipeline import Pipeline
from prepare import prepare

HERE = Path(__file__).resolve().parent


def config(workspace):
    return dict(
        workspace=str(workspace),
        run_id="test_run",
        model=dict(repo_id="test/model", revision="a" * 40),
        data=dict(url="https://example.org/data.zip", sha256="b" * 64),
        benchmark=dict(url="https://example.org/benchmark.zip", sha256="c" * 64),
        automodel=dict(url="https://github.com/NVIDIA-NeMo/Automodel.git", revision="d" * 40),
        train_image="test/train",
        eval_image="test/eval",
    )


class AssetTests(unittest.TestCase):
    def test_required_sources_fail_before_download(self):
        value = config(Path("/tmp/example"))
        value["model"]["repo_id"] = "${SUPER_VL_MODEL_ID}"
        with self.assertRaisesRegex(ValueError, "model.repo_id"):
            require_sources(value)

    def test_download_extract_cache_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("package/train/dataset.jsonl", "{}\n")
                stream.writestr("package/test/dataset.jsonl", "{}\n")
            spec = dict(url=archive.as_uri(), sha256=sha256(archive))
            target = root / "dataset"
            required = ["train/dataset.jsonl", "test/dataset.jsonl"]
            download_zip(spec, target, required)
            self.assertEqual((target / required[0]).read_text(), "{}\n")
            with patch("urllib.request.urlopen", side_effect=AssertionError("Must reuse verified archive")):
                download_zip(spec, target, required)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                download_zip(dict(spec, sha256="0" * 64), root / "bad", required)

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../escape", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
                extract_zip(archive, root / "out", ["dataset.jsonl"])
            self.assertFalse((root.parent / "escape").exists())


class LaunchTests(unittest.TestCase):
    def make_flow(self, root, mode="sft", smoke=5):
        settings = root / "settings.yaml"
        settings.write_text(yaml.safe_dump(config(root / "workspace")))
        return Pipeline(settings, mode, smoke)

    def test_topology_and_no_write_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow = self.make_flow(root)
            with self.assertRaisesRegex(ValueError, "master-addr"):
                flow.training_command()
            command = flow.training_command(3, "10.0.0.10")
            self.assertIn("--nnodes=4", command)
            self.assertIn("--nproc-per-node=4", command)
            self.assertIn("--node-rank=3", command)
            self.assertEqual(command[0:2], ["docker", "run"])
            self.assertFalse(flow.workspace.exists())

    def test_workspace_is_relative_to_settings_and_interpreters_are_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_dir = root / "settings"
            settings_dir.mkdir()
            values = config(Path("my-workspace"))
            values.update(train_python="python-train", eval_python="python-serve")
            settings = settings_dir / "settings.yaml"
            settings.write_text(yaml.safe_dump(values))
            flow = Pipeline(settings, "sft", 5)
            self.assertEqual(flow.workspace, settings_dir / "my-workspace")
            train = flow.training_command(master_addr="10.0.0.10")
            self.assertEqual(train[train.index("--entrypoint") + 1], "python-train")
            evaluate = flow.docker("eval", ["evaluate.py"])
            self.assertEqual(evaluate[evaluate.index("--entrypoint") + 1], "python-serve")
            self.assertIn("AUTOMODEL_SOURCE=/workspace/assets/Automodel", train)
            self.assertFalse(flow.workspace.exists())

    def test_sft_example_settings_use_documented_environment_variables(self):
        environment = dict(
            SUPER_VL_MODEL_ID="test/super-vl",
            SUPER_VL_MODEL_REVISION="a" * 40,
            RXN_DATA_URL="https://example.org/train.zip",
            RXN_DATA_SHA256="b" * 64,
            RXN_BENCHMARK_URL="https://example.org/benchmark.zip",
            RXN_BENCHMARK_SHA256="c" * 64,
            AUTOMODEL_REVISION="d" * 40,
            AUTOMODEL_IMAGE="test/train:v1",
            VLLM_IMAGE="test/serve:v1",
        )
        for mode in ("sft",):
            path = HERE.parent / f"super_vl_rxncaption_{mode}" / "settings.example.yaml"
            with patch.dict(os.environ, environment):
                flow = Pipeline(path, mode)
            self.assertEqual(flow.config["model"]["repo_id"], environment["SUPER_VL_MODEL_ID"])
            self.assertEqual(flow.config["model"]["revision"], environment["SUPER_VL_MODEL_REVISION"])
            self.assertEqual(flow.config["automodel"]["revision"], environment["AUTOMODEL_REVISION"])
            self.assertEqual(flow.config["train_image"], environment["AUTOMODEL_IMAGE"])
            require_sources(flow.config)

    def test_smoke_recipe_and_no_training_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            flow = self.make_flow(Path(tmp))
            flow.initialize()
            recipe = flow.recipe(5)
            self.assertEqual(recipe["step_scheduler"]["max_steps"], 5)
            self.assertEqual(recipe["step_scheduler"]["val_every_steps"], 5)
            self.assertNotIn("peft", recipe)
            self.assertFalse(recipe["freeze_config"]["freeze_vision_tower"])
            self.assertTrue(recipe["optimizer"]["master_weights"])
            (flow.run / "train.yaml").write_text(yaml.safe_dump(recipe))
            write_json(
                flow.run / "ready.json",
                dict(**flow.identity(), steps=5, recipe_sha256=sha256(flow.run / "train.yaml")),
            )
            for rank in range(4):
                write_json(
                    flow.run / f"workspace_visible_{rank}.json",
                    dict(fingerprint=flow.fingerprint, node_rank=rank),
                )
            for rank in range(1, 4):
                write_json(flow.run / f"train_complete_{rank}.json", flow.identity())
            with patch.object(flow, "logged") as launch:
                flow.train(master_addr="10.0.0.10")
                launch.assert_called_once()
                with self.assertRaises(FileExistsError):
                    flow.train(master_addr="10.0.0.10")
            self.assertTrue((flow.run / "training_complete.json").exists())
            changed = config(flow.workspace)
            changed["eval_port"] = 12345
            flow.settings.write_text(yaml.safe_dump(changed))
            with self.assertRaisesRegex(ValueError, "fresh run_id"):
                Pipeline(flow.settings, "sft", 5).initialize()

    def test_failure_stops_export_and_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            flow = self.make_flow(Path(tmp))
            with (
                patch.object(flow, "download"),
                patch.object(flow, "prepare"),
                patch.object(flow, "train", side_effect=RuntimeError("training failure")),
                patch.object(flow, "export") as export,
                patch.object(flow, "evaluate") as evaluate,
            ):
                with self.assertRaisesRegex(RuntimeError, "training failure"):
                    flow.all(master_addr="10.0.0.10")
                export.assert_not_called()
                evaluate.assert_not_called()
            self.assertTrue((flow.run / "failed_node0.json").exists())

    def test_full_stage_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            flow = self.make_flow(Path(tmp))
            calls = []
            with (
                patch.object(flow, "download", side_effect=lambda: calls.append("download")),
                patch.object(flow, "prepare", side_effect=lambda: calls.append("prepare")),
                patch.object(flow, "train", side_effect=lambda *args: calls.append("train")),
                patch.object(flow, "export", side_effect=lambda: calls.append("export")),
                patch.object(flow, "evaluate", side_effect=lambda: calls.append("eval")),
            ):
                flow.all(master_addr="10.0.0.10")
            self.assertEqual(calls, ["download", "prepare", "train", "export", "eval"])


class DataAndEvaluationTests(unittest.TestCase):
    def test_standalone_metrics_reject_missing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ground = {
                "images": [
                    {"file_name": "a.png", "bboxes": [], "reactions": []},
                    {"file_name": "b.png", "bboxes": [], "reactions": []},
                ]
            }
            predictions = {
                "images": [
                    {"file_name": "a.png", "bboxes": [], "reactions": []},
                ]
            }
            write_json(root / "ground.json", ground)
            write_json(root / "predictions.json", predictions)
            with self.assertRaisesRegex(ValueError, "exactly match ground truth"):
                run_evaluation(root / "ground.json", root / "predictions.json")

    def test_real_preparation_and_disjoint_splits(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "assets/data"
            benchmark = root / "assets/benchmark"
            (benchmark / "images/gt_bivp").mkdir(parents=True)
            gold, maps = [], []
            for split, count in [("train", 12), ("test", 400)]:
                directory = data / split
                (directory / "images").mkdir(parents=True)
                rows = []
                for i in range(count):
                    name = f"{split}_{i}.png"
                    image = Image.new("RGB", (4, 4), "white")
                    image.save(directory / "images" / name)
                    reaction = dict(reactants=[dict(type="mol", index=1)], conditions=[], products=[])
                    rows.append(
                        dict(
                            images=["images/" + name],
                            messages=[
                                dict(role="user", content="<image> Parse reactions."),
                                dict(role="assistant", content=json.dumps([reaction] * 5)),
                            ],
                        )
                    )
                    if split == "test":
                        image.save(benchmark / "images/gt_bivp" / name)
                        gold.append(dict(file_name=name, bboxes=[], reactions=[]))
                        maps.append(dict(file_name=name, map_dict={}))
                (directory / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_json(benchmark / "ground_truth/ground_truth.json", dict(images=gold))
            write_json(benchmark / "mapdicts/mapdict_gt.json", maps)
            result = prepare(root, root / "run", validation_size=2)
            self.assertEqual(result["unique_train_rows"], 10)
            self.assertEqual(result["weighted_train_rows"], 20)
            self.assertEqual(result["validation_rows"], 2)
            self.assertEqual(result["benchmark_images"], 400)
            self.assertEqual(result["full_steps"], 2)
            self.assertTrue((root / "run/example.png").is_file())
            rows_path = data / "train/dataset.jsonl"
            rows_path.write_text(rows_path.read_text() + rows_path.read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate image ID"):
                prepare(root, root / "run2", validation_size=2)

    def test_invalid_predictions_count_as_misses_and_role_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boxes = [dict(id=i, category_id=1, bbox=[i * 20, 0, 10, 10]) for i in range(2)]
            ground = dict(
                images=[
                    dict(file_name=name, bboxes=boxes, reactions=[dict(reactants=[0], conditions=[], products=[1])])
                    for name in ("a.png", "b.png")
                ]
            )
            write_json(root / "gt.json", ground)
            write_json(
                root / "map.json",
                [
                    dict(file_name=name, map_dict={str(i + 1): dict(yolo_bbox=b["bbox"]) for i, b in enumerate(boxes)})
                    for name in ("a.png", "b.png")
                ],
            )
            correct = json.dumps(
                [dict(reactants=[dict(type="mol", index=1)], conditions=[], products=[dict(type="mol", index=2)])]
            )
            invalid, bad = normalize_response("not JSON")
            self.assertTrue(bad)
            self.assertEqual(normalize_response('[{"reactants": null}]'), ("[]", True))
            write_json(root / "raw.json", {"a.png": dict(response=correct), "b.png": dict(response=invalid)})
            convert(root / "raw.json", root / "map.json", root / "gt.json", root / "pred.json")
            values = score(root / "gt.json", root / "pred.json")
            self.assertEqual(set(values), {"SoftMatch", "HybridMatch"})
            self.assertEqual(values["SoftMatch"]["fn"], 1)
            self.assertAlmostEqual(values["SoftMatch"]["f1"], 200 / 3)
            flipped = copy.deepcopy(ground)
            for row in flipped["images"]:
                row["reactions"][0] = dict(reactants=[1], conditions=[], products=[0])
            write_json(root / "flipped.json", flipped)
            values = score(root / "gt.json", root / "flipped.json")
            self.assertEqual(values["SoftMatch"]["f1"], 0)

    def test_paper_hybrid_text_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boxes = [
                dict(id=0, category_id=1, bbox=[0, 0, 10, 10]),
                dict(id=1, category_id=1, bbox=[20, 0, 10, 10]),
                dict(id=2, category_id=2, bbox=[0, 0, 0, 0], text="heat"),
                dict(id=3, category_id=2, bbox=[0, 0, 0, 0], text="NaOH"),
            ]
            reaction = dict(reactants=[0, 3], conditions=[2], products=[1])
            ground = dict(images=[dict(file_name="a.png", bboxes=boxes, reactions=[reaction])])
            prediction = copy.deepcopy(ground)
            prediction["images"][0]["bboxes"][2]["text"] = "heatt"
            write_json(root / "gt.json", ground)
            write_json(root / "pred.json", prediction)
            values = score(root / "gt.json", root / "pred.json")
            self.assertEqual(values["SoftMatch"]["f1"], 100)
            self.assertEqual(values["HybridMatch"]["f1"], 100)

            prediction["images"][0]["bboxes"][3]["text"] = "Na0H"
            write_json(root / "pred.json", prediction)
            self.assertEqual(score(root / "gt.json", root / "pred.json")["HybridMatch"]["f1"], 0)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
    "Run in the training container to exercise real BF16 tensor export",
)
class TensorExportTests(unittest.TestCase):
    def test_real_bf16_merge_uses_current_base_and_validates_shards(self):
        import torch
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter = root / "base", root / "adapter"
            base.mkdir()
            adapter.mkdir()
            key = "language_model.backbone.layers.0.mixer.q_proj.weight"
            w = torch.ones(4, 3, dtype=torch.bfloat16)
            a = torch.full((2, 3), 0.25, dtype=torch.bfloat16)
            b = torch.full((4, 2), 0.5, dtype=torch.bfloat16)
            frozen = torch.full((3,), 7, dtype=torch.bfloat16)
            base_weights = {
                key: w,
                "vision_model.weight": frozen,
                "language_model.mtp.layers.0.weight": torch.ones(2, dtype=torch.bfloat16),
            }
            save_file(base_weights, base / "model.safetensors")
            write_json(
                base / "model.safetensors.index.json",
                dict(metadata=dict(total_size=34), weight_map={k: "model.safetensors" for k in base_weights}),
            )
            write_json(base / "config.json", dict(model_type="nemotron_h_omni"))
            prefix = "base_model.model.language_model.model.layers.0.mixer.q_proj"
            adapter_weights = {prefix + ".lora_A.weight": a, prefix + ".lora_B.weight": b}
            save_file(adapter_weights, adapter / "adapter_model.safetensors")
            report = merge_lora(base, adapter, root / "export", adapter_weights, rank=2)
            self.assertEqual(report["merged_layers"], 1)
            merged = load_file(root / "export/model-00001-of-00001.safetensors")
            torch.testing.assert_close(merged[key], w + 2 * (b @ a), rtol=0, atol=0)
            torch.testing.assert_close(merged["vision_model.weight"], frozen, rtol=0, atol=0)
            torch.testing.assert_close(load_file(base / "model.safetensors")[key], w, rtol=0, atol=0)
            self.assertNotIn("language_model.mtp.layers.0.weight", merged)
            shard = root / "export/model-00001-of-00001.safetensors"
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete shard"):
                verify_export(root / "export")


class NotebookTests(unittest.TestCase):
    def test_outputs_cleared_and_code_parses(self):
        directory = HERE.parent / "super_vl_rxncaption_lora"
        for path in directory.glob("*.ipynb"):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    source = "".join(cell["source"])
                    if not source.startswith("%pip"):
                        ast.parse(source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
