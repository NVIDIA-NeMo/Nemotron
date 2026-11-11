"""Regression contracts for the dental objective and stage handoffs.

The fixed synthetic fixture is independent of the implementation under test.
"""

import copy
import json
import re
from pathlib import Path

import pytest
import yaml

from nemotron.recipes.parse.dental.config import recipe_root, sft_config
from nemotron.recipes.parse.dental.curriculum import select_training_pages
from nemotron.recipes.parse.dental.evaluate import pass_at_k
from nemotron.recipes.parse.dental.metrics import components, exact_page_reward, successful_answer
from nemotron.recipes.parse.dental.model import prepare_model, sft_checkpoint


@pytest.fixture
def page():
    return json.loads((Path(__file__).parent / "fixtures/example.json").read_text())[0]


def test_content_reward_and_reported_success_have_distinct_eos_contracts(page):
    target = page["target"]
    assert exact_page_reward(target, page) == 1
    assert successful_answer({"text": target, "ended": True}, page)
    assert not successful_answer({"text": target, "ended": False}, page)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("| 14 | filling | M |", "| 15 | filling | M |"),
        ("| 17 | filling | MO |", "| 17 | filling | M |"),
        ("Sora Tavin", "Sora Tavim"),
        ("700.00", "701.00"),
        ("| 14 | filling | M |", "| 14 | filling | M |\n| 14 | filling | M |"),
    ],
)
def test_wrong_content_or_duplicate_chart_row_fails_exact_reward(page, before, after):
    assert before in page["target"]
    assert exact_page_reward(page["target"].replace(before, after), page) == 0


def test_layout_changes_do_not_earn_or_remove_dental_content_credit(page):
    prediction = re.sub(r"<[xy]_[^>]+>", "<x_0>", page["target"])
    prediction = prediction.replace("<class_Chart>", "<class_Text>")
    assert exact_page_reward(prediction, page) == 1


def test_surface_alias_is_semantically_valid_but_changes_visible_transcription(page):
    prediction = page["target"].replace("| 14 | filling | M |", "| 14 | filling | mesial |")
    scores = components(prediction, page)
    assert scores["chart_exact"] == 1
    assert scores["text_cer"] > 0
    assert exact_page_reward(prediction, page) == 0


def test_pass_at_k_uses_all_samples_and_rejects_insufficient_sample_budget():
    answers = [{"answers": [{"correct": True}, {"correct": False}, {"correct": False}, {"correct": False}]}]
    assert pass_at_k(answers, 1) == 0.25
    assert pass_at_k(answers, 2) == 0.5
    assert pass_at_k(answers, 4) == 1
    with pytest.raises(ValueError, match="k cannot exceed"):
        pass_at_k(answers, 5)


def test_curriculum_rejects_held_out_contexts_and_mismatched_answer_ids(page):
    candidate = dict(copy.deepcopy(page), id="candidate")
    answers = [{"id": "candidate", "answers": [{"correct": True}, {"correct": False}]}]
    with pytest.raises(ValueError, match="outside other splits"):
        select_training_pages([candidate], answers, pages=1, excluded=[page])
    answers[0]["id"] = "another-page"
    with pytest.raises(ValueError, match="IDs differ"):
        select_training_pages([candidate], answers, pages=1)


def test_packaged_sft_config_resolves_explicit_stage_paths_from_any_cwd(tmp_path, monkeypatch):
    root = recipe_root()
    monkeypatch.chdir(tmp_path)
    assert recipe_root() == root
    cfg = sft_config(root / "stage1_sft/config/automodel.yaml", "model", "data", "sft")
    assert cfg["dataset"]["path"] == str(tmp_path / "data/train.jsonl")
    assert cfg["validation_dataset"]["path"] == str(tmp_path / "data/monitor.jsonl")
    assert cfg["checkpoint"]["checkpoint_dir"] == str(tmp_path / "sft/checkpoints")
    assert cfg["dataset"]["_target_"] == "nemotron.recipes.parse.dental.sft_data.make_dataset"
    assert cfg["step_scheduler"]["max_steps"] == 2852
    assert cfg["step_scheduler"]["global_batch_size"] == 1
    assert cfg["optimizer"]["lr"] == 3e-6


def test_packaged_rl_config_preserves_checkpoint_loading_and_scientific_budget():
    cfg = yaml.safe_load((recipe_root() / "stage2_rlvr/config/grpo.yaml").read_text())
    assert cfg["policy"]["generation"]["vllm_cfg"]["load_format"] == "auto"
    assert cfg["data"]["train"]["dataset_name"] == "nemotron.recipes.parse.dental.rl_data.DentalPages"
    assert cfg["grpo"]["max_num_steps"] == 128
    assert cfg["grpo"]["num_prompts_per_step"] * cfg["grpo"]["num_generations_per_prompt"] == 64
    assert cfg["grpo"]["overlong_filtering"] is True


def test_checkpoint_handoff_selects_latest_numeric_step(tmp_path):
    for suffix in ("epoch_0_step_9", "epoch_1_step_10"):
        (tmp_path / "checkpoints" / suffix / "model/consolidated").mkdir(parents=True)
    assert sft_checkpoint(tmp_path) == tmp_path / "checkpoints/epoch_1_step_10/model/consolidated"


def test_model_view_preserves_weights_and_rejects_changed_identity(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    original = {"is_encoder_decoder": False, "decoder": {"causal_vision_prefix": True}}
    (checkpoint / "config.json").write_text(json.dumps(original))
    (checkpoint / "model.safetensors").write_bytes(b"synthetic unit-test placeholder")
    (checkpoint / "tokenizer.json").write_text("{}")
    output = tmp_path / "view"
    assert prepare_model(checkpoint, output) == output
    assert (output / "model.safetensors").resolve() == checkpoint / "model.safetensors"
    assert json.loads((checkpoint / "config.json").read_text()) == original
    assert json.loads((output / "config.json").read_text())["dental_train_scope"] == "full"
    assert (output / "hf_dental_parse.py").is_file()
    assert prepare_model(checkpoint, output) == output
    with pytest.raises(FileExistsError, match="different or unknown inputs"):
        prepare_model(checkpoint, output, scope="decoder")
