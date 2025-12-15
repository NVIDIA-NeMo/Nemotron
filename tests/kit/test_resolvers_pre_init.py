import sys
import types

from omegaconf import OmegaConf

from nemotron.kit import resolvers


def test_register_resolvers_from_config_pre_init(monkeypatch, tmp_path):
    resolvers.clear_artifact_cache()

    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    downloaded_dir = tmp_path / "artifact"
    downloaded_dir.mkdir()

    class FakeArtifact:
        def __init__(self, ref: str):
            self.qualified_name = ref
            self.version = "v5"
            self.name = "DataBlendsArtifact-pretrain"
            self.type = "dataset"

        def download(self, skip_cache: bool = True):
            return str(downloaded_dir)

    class FakeApi:
        def __init__(self):
            self.last_ref = None

        def artifact(self, ref: str):
            self.last_ref = ref
            return FakeArtifact(ref)

    fake_api = FakeApi()

    class FakeWandb(types.SimpleNamespace):
        def Api(self):  # noqa: N802
            return fake_api

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())

    cfg = OmegaConf.create(
        {
            "run": {"data": "DataBlendsArtifact-pretrain:v5"},
            "recipe": {"per_split_data_args_path": "${art:data,path}"},
        }
    )

    qualified_names = resolvers.register_resolvers_from_config(cfg, mode="pre_init")
    assert qualified_names == ["ent/proj/DataBlendsArtifact-pretrain:v5"]
    assert fake_api.last_ref == "ent/proj/DataBlendsArtifact-pretrain:v5"

    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["recipe"]["per_split_data_args_path"] == str(downloaded_dir)
