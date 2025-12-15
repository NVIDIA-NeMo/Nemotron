import importlib
import sys
import types


def test_patch_wandb_init_for_lineage_registers_artifacts_and_tags(monkeypatch):
    import nemotron.kit.wandb as wb

    wb = importlib.reload(wb)

    used: list[str] = []

    class FakeRun:
        def __init__(self):
            self.tags = []

        def use_artifact(self, qname: str):
            used.append(qname)

    fake_run = FakeRun()

    def fake_init(*args, **kwargs):
        fake_wandb.run = fake_run
        return fake_run

    fake_wandb = types.SimpleNamespace(run=None, init=fake_init)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    wb.patch_wandb_init_for_lineage(
        artifact_qualified_names=["ent/proj/DataBlendsArtifact-pretrain:v5"],
        tags=["pretrain"],
    )

    fake_wandb.init()

    assert used == ["ent/proj/DataBlendsArtifact-pretrain:v5"]
    assert "pretrain" in fake_run.tags
