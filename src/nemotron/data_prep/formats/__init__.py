"""Output format implementations."""

from nemotron.data_prep.formats.indexed_dataset import IndexedDatasetBuilder
from nemotron.data_prep.formats.jsonl_dataset import JsonlDatasetBuilder
from nemotron.data_prep.formats.transforms import (
    Transform,
    SftRecord,
    SftRecordWithSystem,
    Message,
    OpenAIChatRecord,
    Conversation,
    ShareGPTRecord,
    sft,
    openai_chat,
    sharegpt,
    passthrough,
    select,
    rename,
)

__all__ = [
    # Writers
    "IndexedDatasetBuilder",
    "JsonlDatasetBuilder",
    # Transform types
    "Transform",
    "SftRecord",
    "SftRecordWithSystem",
    "Message",
    "OpenAIChatRecord",
    "Conversation",
    "ShareGPTRecord",
    # Transform factories
    "sft",
    "openai_chat",
    "sharegpt",
    "passthrough",
    "select",
    "rename",
]
