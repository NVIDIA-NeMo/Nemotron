# Artifact Types & Lineage

Nemotron uses typed Pydantic artifacts to version data and models across training stages. Each artifact stores its output path plus typed metadata fields that are accessible via `${art:NAME,FIELD}` config resolvers.

> For the tracking infrastructure (manifest + wandb backends, `setup_artifact_tracking`, `log_artifact`, env.toml configuration), see [nemo_runspec artifacts](../nemo_runspec/artifacts.md).

## End-to-End Lineage

The training pipeline produces six artifact types across three stages:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333', 'clusterBkg': '#ffffff', 'clusterBorder': '#333333'}}}%%
flowchart TB
    subgraph stage0["Stage 0: Pretraining"]
        raw0["Raw Text Data"] --> dp0["data_prep.py"]
        dp0 --> data0["PretrainBlendsArtifact<br/>(bin/idx)"]
        data0 --> train0["train.py"]
        train0 --> model0["ModelArtifact-pretrain"]
    end

    subgraph stage1["Stage 1: SFT"]
        raw1["Instruction Data"] --> dp1["data_prep.py"]
        dp1 --> data1["SFTDataArtifact<br/>(Packed Parquet)"]
        model0 --> train1["train.py"]
        data1 --> train1
        train1 --> model1["ModelArtifact-sft"]
    end

    subgraph stage2["Stage 2: RL"]
        raw2["RL Prompts"] --> dp2["data_prep.py"]
        dp2 --> data2["SplitJsonlDataArtifact<br/>(JSONL)"]
        model1 --> train2["train.py"]
        data2 --> train2
        train2 --> model2["ModelArtifact-rl<br/>(Final Model)"]
    end

    style stage0 fill:#e1f5fe,stroke:#2196f3
    style stage1 fill:#f3e5f5,stroke:#9c27b0
    style stage2 fill:#e8f5e9,stroke:#4caf50
```

## Artifact Types

| Artifact | Stage | Format | Description |
|----------|-------|--------|-------------|
| `PretrainBlendsArtifact` | [0](./nano3/pretrain.md) | bin/idx | Tokenized pretraining data in Megatron format |
| `ModelArtifact` | 0, 1, 2 | checkpoint | Model checkpoints (pretrain, sft, rl) |
| `SFTDataArtifact` | [1](./nano3/sft.md) | Packed Parquet | Packed SFT sequences with loss masks |
| `SplitJsonlDataArtifact` | [2](./nano3/rl.md) | JSONL | RL prompts for [NeMo-RL](./nvidia-stack.md#nemo-rl) |

## Artifact Naming

| Concept | Example | Where Used |
|---------|---------|------------|
| **Python class** | `PretrainBlendsArtifact` | Code imports (`from nemotron.kit import ...`) |
| **Registered name** | `super3-pretrain-data-tiny` | W&B artifact names, manifest directories |
| **Config reference** | `super3-pretrain-data-tiny:latest` | YAML configs (`run.data`), CLI overrides |

## Using Artifacts in Configs

### Semantic URIs

```
art://super3-pretrain-data-tiny:latest    # Latest version
art://ModelArtifact-sft:v3                # Specific version
```

### Config Resolvers

```yaml
run:
  data: super3-pretrain-data-tiny:latest

recipe:
  per_split_data_args_path: ${art:data,path}/blend.json
  seq_length: ${art:data,pack_size}
```

The `${art:data,FIELD}` resolver reads typed fields from the artifact's `metadata.json`.

### CLI Overrides

```bash
uv run nemotron nano3 pretrain run.data=PretrainBlendsArtifact-tiny:v2
uv run nemotron nano3 sft run.model=my-custom-pretrain:latest
```

## Creating Custom Artifacts

Subclass `Artifact` to create typed artifacts. Fields are automatically synced to `metadata.json` and available via resolvers.

```python
from pathlib import Path
from typing import Annotated
from pydantic import Field
from nemotron.kit.artifacts.base import Artifact


class MyDataArtifact(Artifact):
    """Custom data artifact with typed metadata."""

    num_samples: Annotated[int, Field(ge=0, description="Number of samples")]
    source_url: Annotated[str | None, Field(default=None)]
```

### Saving

```python
from nemo_runspec.artifacts import setup_artifact_tracking, log_artifact

tracking = setup_artifact_tracking(config)

artifact = MyDataArtifact(path=Path("/output/data"), num_samples=10000)
log_artifact(artifact, tracking, name="my-data")
```

### Loading

```python
artifact = MyDataArtifact.from_uri("art://my-data:latest")
print(artifact.path)         # /output/data
print(artifact.num_samples)  # 10000 — IDE autocomplete works
```

### Input Lineage

Track dependencies by overriding `get_input_uris()`:

```python
class ProcessedDataArtifact(Artifact):
    source_artifact: str

    def get_input_uris(self) -> list[str]:
        return [self.source_artifact]
```

## Importing External Assets

### Model Import

```bash
uv run nemotron nano3 model import pretrain /path/to/checkpoint --step 50000
uv run nemotron nano3 model import sft /path/to/sft_model --step 10000
```

### Data Import

```bash
uv run nemotron nano3 data import pretrain /path/to/blend.json
uv run nemotron nano3 data import sft /path/to/sft_data/
```

See [Importing Models & Data](./nano3/import.md) for detailed directory structures.

## Further Reading

- [Artifact Tracking](../nemo_runspec/artifacts.md) – tracking infrastructure, manifest backend, env.toml config
- [Nemotron Kit](./kit.md) – kit internals
- [OmegaConf Configuration](../nemo_runspec/omegaconf.md) – `${art:...}` resolver details
- [W&B Integration](./wandb.md) – credentials and configuration
