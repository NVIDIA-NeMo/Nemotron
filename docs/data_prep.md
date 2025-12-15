# Data Preparation Module

The `nemotron.data_prep` module processes raw text data from HuggingFace, S3, or local sources into various training formats compatible with Megatron-Bridge and Megatron-Core.

## Supported Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `binidx` | Tokenized `.bin/.idx` indexed datasets | Pretraining (default) |
| `jsonl` | JSONL files with optional transforms | SFT/RL training |
| `packed` | Packed sequences in `.npy` format | Efficient SFT training |

## Quick Start

### High-Level API (DataPrepConfig)

For simple tokenization to binidx format:

```python
from nemotron.data_prep import DataPrepConfig, run_data_prep
from pathlib import Path

config = DataPrepConfig(
    blend_path=Path("data_blend.json"),
    output_dir=Path("./output"),
    tokenizer_model="meta-llama/Llama-3.2-1B",
)

artifact = run_data_prep(config)
print(f"Blend path: {artifact.path}")
```

### Low-Level API (last_mile_process)

For more control over output format:

```python
from nemotron.data_prep import last_mile_process, DataBlend, PipelineConfig
from nemotron.data_prep.config import OutputConfig, JsonlOutputConfig
from nemotron.data_prep.formats.transforms import sft

blend = DataBlend.load("data_blend.json")

config = PipelineConfig(
    output=OutputConfig(
        dir=Path("./sft_data"),
        format=JsonlOutputConfig(
            transform=sft(input="instruction", output="response"),
        ),
    ),
)

result = last_mile_process(blend, config)
```

## Output Formats

### BinIdx (Default)

Tokenized binary format for Megatron pretraining:

```python
from nemotron.data_prep.config import BinIdxOutputConfig

config = PipelineConfig(
    tokenizer=TokenizerConfig(model="meta-llama/Llama-3.2-1B"),
    output=OutputConfig(
        dir=Path("./tokenized"),
        format=BinIdxOutputConfig(
            shard_size="256MB",  # Or num_shards=128
            dtype="int32",
        ),
    ),
)
```

### JSONL

Structured JSONL for SFT/RL training (no tokenization):

```python
from nemotron.data_prep.config import JsonlOutputConfig
from nemotron.data_prep.formats.transforms import sft, openai_chat

# SFT format: {"input": "...", "output": "..."}
config = PipelineConfig(
    output=OutputConfig(
        dir=Path("./sft_data"),
        format=JsonlOutputConfig(
            transform=sft(input="instruction", output="response"),
            compression="zstd",  # Optional compression
        ),
    ),
)

# OpenAI chat format: {"messages": [...]}
config = PipelineConfig(
    output=OutputConfig(
        dir=Path("./rl_data"),
        format=JsonlOutputConfig(
            transform=openai_chat(),
        ),
    ),
)
```

### Packed (Coming Soon)

Packed sequences for efficient SFT training:

```python
from nemotron.data_prep.config import PackedOutputConfig

config = PipelineConfig(
    tokenizer=TokenizerConfig(model="meta-llama/Llama-3.2-1B"),
    output=OutputConfig(
        dir=Path("./packed_data"),
        format=PackedOutputConfig(
            pack_size=4096,
            algorithm="first_fit_shuffle",
        ),
    ),
)
```

## Transforms

Transforms convert input records to the desired output format. They are callables that take a dict and return a dict (or `None` to skip the record).

### Built-in Transform Factories

```python
from nemotron.data_prep.formats.transforms import (
    sft,           # SFT format: {input, output}
    openai_chat,   # OpenAI format: {messages: [...]}
    sharegpt,      # ShareGPT format: {conversations: [...]}
    passthrough,   # Pass records unchanged
    select,        # Select specific fields
    rename,        # Rename fields
)
```

### sft()

Creates SFT format output:

```python
transform = sft(
    input="instruction",   # Source field for input
    output="response",     # Source field for output
    system="system_prompt" # Optional system prompt field
)

# Input:  {"instruction": "Hello", "response": "Hi!", "system_prompt": "Be helpful"}
# Output: {"input": "Hello", "output": "Hi!", "system": "Be helpful"}
```

### openai_chat()

Creates OpenAI chat format:

```python
transform = openai_chat(messages="conversation")

# Input:  {"conversation": [{"role": "user", "content": "Hi"}]}
# Output: {"messages": [{"role": "user", "content": "Hi"}]}
```

### sharegpt()

Creates ShareGPT format:

```python
transform = sharegpt(conversations="turns")

# Input:  {"turns": [{"from": "human", "value": "Hi"}]}
# Output: {"conversations": [{"from": "human", "value": "Hi"}]}
```

### passthrough()

Passes records unchanged:

```python
transform = passthrough()

# Input:  {"any": "data"}
# Output: {"any": "data"}
```

### select()

Selects specific fields:

```python
transform = select("id", "text")

# Input:  {"id": 1, "text": "hello", "extra": "ignored"}
# Output: {"id": 1, "text": "hello"}
```

### rename()

Renames fields:

```python
transform = rename(input="question", output="answer")

# Input:  {"question": "What?", "answer": "This."}
# Output: {"input": "What?", "output": "This."}
```

### Custom Transforms

You can use any callable:

```python
# Lambda
transform = lambda r: {"input": r["q"], "output": r["a"]} if r.get("valid") else None

# Function
def my_transform(record: dict) -> dict | None:
    if len(record.get("text", "")) < 10:
        return None  # Skip short records
    return {"input": record["question"], "output": record["answer"]}
```

## Sharding Configuration

Both `shard_size` and `num_shards` are supported (mutually exclusive):

```python
# Target shard size (default)
format=JsonlOutputConfig(shard_size="256MB")

# Explicit shard count
format=JsonlOutputConfig(num_shards=64)
```

Supported size formats: `"256MB"`, `"1G"`, `"500MiB"`, etc.

## Type Definitions

TypedDicts are provided for type safety:

```python
from nemotron.data_prep.formats.transforms import (
    SftRecord,         # {"input": str, "output": str}
    OpenAIChatRecord,  # {"messages": list[Message]}
    ShareGPTRecord,    # {"conversations": list[Conversation]}
    Message,           # {"role": str, "content": str}
)
```

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `run_data_prep(config)` | High-level entry point for tokenization |
| `last_mile_process(blend, config)` | Low-level entry point with format dispatch |
| `tokenize(blend, config)` | Deprecated alias for `last_mile_process` |

### Configuration Classes

| Class | Description |
|-------|-------------|
| `DataPrepConfig` | High-level configuration for `run_data_prep` |
| `PipelineConfig` | Low-level pipeline configuration |
| `TokenizerConfig` | Tokenizer settings (model, add_bos, add_eos) |
| `OutputConfig` | Output directory and format |
| `BinIdxOutputConfig` | Tokenized binary format options |
| `JsonlOutputConfig` | JSONL format options |
| `PackedOutputConfig` | Packed sequence format options |

### Result Classes

| Class | Description |
|-------|-------------|
| `PipelineResult` | Complete pipeline result with all splits |
| `SplitResult` | Result for a single split (train/valid/test) |
| `DataBlendsArtifact` | Artifact with blend.json path and metrics |

## Compression

JSONL output supports optional zstd compression:

```python
format=JsonlOutputConfig(
    compression="zstd",  # Output .jsonl.zst files
)
```

Requires the `zstandard` package: `pip install zstandard`

## Dependencies

Core dependencies:
- `ray` - Parallel processing
- `pyarrow` - Parquet file reading
- `xxhash` - Fast checksums

Optional dependencies:
- `orjson` - Fast JSON serialization (falls back to stdlib json)
- `zstandard` - Zstd compression for JSONL output
