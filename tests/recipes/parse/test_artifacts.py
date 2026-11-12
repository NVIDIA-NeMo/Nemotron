"""Publishing stage bookkeeping must not mutate an evaluated model payload."""

import json

from nemotron.recipes.parse import RECIPE_VERSION
from nemotron.recipes.parse.common import artifact, model_from_artifact
from nemotron.recipes.parse.dental.evaluate import evaluation_key


def test_model_artifact_preserves_checkpoint_inventory_and_evaluation_identity(tmp_path):
    output = tmp_path / "rlvr"
    checkpoint = output / "checkpoints" / "step_128"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"fixture": true}\n')
    (checkpoint / "tokenizer_config.json").write_text('{}\n')
    (checkpoint / "model.safetensors").write_bytes(b"unit fixture, not model weights")
    image = tmp_path / "held-out.png"
    image.write_bytes(b"unit fixture, not an image")
    rows = [{"id": "held-out", "image": str(image)}]

    def inventory():
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in checkpoint.iterdir()}

    before = inventory()
    key = evaluation_key(checkpoint, rows, samples=8)
    result = artifact(output, "parse/model", checkpoint, processor_source=str(checkpoint))

    assert inventory() == before
    assert evaluation_key(checkpoint, rows, samples=8) == key
    assert not (checkpoint / "metadata.json").exists()
    metadata = json.loads((output / "metadata.json").read_text())
    handoff = json.loads(result.read_text())
    assert metadata["path"] == str(output)
    assert metadata["metadata"]["payload_path"] == str(checkpoint)
    assert handoff["path"] == str(checkpoint)
    assert handoff["schema_version"] == 1
    assert handoff["recipe_version"] == metadata["metadata"]["recipe_version"] == RECIPE_VERSION
    assert "source_revision" not in handoff and "source_revision" not in metadata["metadata"]
    assert model_from_artifact(result) == checkpoint

    # Existing v1 handoffs may have additional source metadata instead of a recipe version.
    handoff.pop("recipe_version")
    handoff["source_revision"] = "prior-release"
    result.write_text(json.dumps(handoff))
    assert model_from_artifact(result) == checkpoint


def test_file_payload_is_recorded_without_treating_it_as_metadata_directory(tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    payload = output / "evaluation.json"
    payload.write_text('{"pass_at_1": 0.5}\n')
    original = payload.read_bytes()

    result = artifact(output, "parse/evaluation", payload)

    assert payload.read_bytes() == original
    assert json.loads((output / "metadata.json").read_text())["metadata"]["payload_path"] == str(payload)
    assert json.loads(result.read_text())["path"] == str(payload)
