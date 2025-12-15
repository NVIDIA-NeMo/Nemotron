# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenizer provider factories."""

from typing import Protocol


class TokenizerFn(Protocol):
    """Protocol for tokenizer callable with vocab_size attribute."""

    vocab_size: int

    def __call__(self, texts: list[str]) -> list[list[int]]:
        """Tokenize a batch of texts."""
        ...


def create_tokenizer(resolved_config: dict) -> TokenizerFn:
    """
    Create tokenizer from resolved config.

    IMPORTANT: Uses resolved_revision, not user-provided revision.

    Supported types:
    - huggingface: HuggingFace AutoTokenizer
    - sentencepiece: SentencePiece model file
    - tiktoken: OpenAI tiktoken encodings (cl100k_base, o200k_base, etc.)
    """
    tokenizer_type = resolved_config["type"]

    if tokenizer_type == "huggingface":
        return _create_huggingface_tokenizer(resolved_config)
    elif tokenizer_type == "sentencepiece":
        return _create_sentencepiece_tokenizer(resolved_config)
    elif tokenizer_type == "tiktoken":
        return _create_tiktoken_tokenizer(resolved_config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def _create_huggingface_tokenizer(resolved_config: dict) -> TokenizerFn:
    """Create HuggingFace tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_config["model"],
        revision=resolved_config["resolved_revision"],  # Use resolved SHA
        trust_remote_code=resolved_config.get("trust_remote_code", False),
        use_fast=True,
    )

    add_bos = resolved_config.get("add_bos", False)
    add_eos = resolved_config.get("add_eos", True)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    def tokenize_batch(texts: list[str]) -> list[list[int]]:
        """Vectorized batch tokenization."""
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        results = []
        for ids in encoded["input_ids"]:
            ids = list(ids)
            if add_bos and bos_id is not None:
                ids = [bos_id] + ids
            if add_eos and eos_id is not None:
                ids = ids + [eos_id]
            results.append(ids)
        return results

    # Attach vocab_size as attribute
    tokenize_batch.vocab_size = vocab_size  # type: ignore
    return tokenize_batch  # type: ignore


def _create_sentencepiece_tokenizer(resolved_config: dict) -> TokenizerFn:
    """Create SentencePiece tokenizer."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=resolved_config["model"])
    add_bos = resolved_config.get("add_bos", False)
    add_eos = resolved_config.get("add_eos", True)
    vocab_size = sp.vocab_size()

    def tokenize_batch(texts: list[str]) -> list[list[int]]:
        results = []
        for text in texts:
            ids = sp.encode(text)
            if add_bos:
                ids = [sp.bos_id()] + ids
            if add_eos:
                ids = ids + [sp.eos_id()]
            results.append(ids)
        return results

    # Attach vocab_size as attribute
    tokenize_batch.vocab_size = vocab_size  # type: ignore
    return tokenize_batch  # type: ignore


def _create_tiktoken_tokenizer(resolved_config: dict) -> TokenizerFn:
    """Create tiktoken tokenizer.

    Supports standard encodings (cl100k_base, o200k_base, etc.) and custom patterns.
    Compatible with Megatron Bridge TikTokenizer configuration.
    """
    import tiktoken

    model = resolved_config["model"]
    add_bos = resolved_config.get("add_bos", False)
    add_eos = resolved_config.get("add_eos", True)

    # Try to get encoding by name (e.g., "cl100k_base", "o200k_base")
    try:
        enc = tiktoken.get_encoding(model)
    except ValueError:
        # Fall back to model-based encoding (e.g., "gpt-4", "gpt-3.5-turbo")
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError as e:
            raise ValueError(
                f"Unknown tiktoken encoding or model: {model}. "
                f"Valid encodings: cl100k_base, o200k_base, p50k_base, r50k_base. "
                f"Or use a model name like gpt-4, gpt-3.5-turbo."
            ) from e

    vocab_size = enc.n_vocab

    # Get special tokens for BOS/EOS if needed
    # tiktoken doesn't have standard BOS/EOS, so we use common conventions
    # For GPT models: <|endoftext|> is typically used as EOS
    try:
        eot_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        eos_id = eot_token[0] if eot_token else None
    except Exception:
        eos_id = None

    # BOS is less common in tiktoken; typically not used
    bos_id = None

    def tokenize_batch(texts: list[str]) -> list[list[int]]:
        results = []
        for text in texts:
            ids = enc.encode(text)
            if add_bos and bos_id is not None:
                ids = [bos_id] + ids
            if add_eos and eos_id is not None:
                ids = ids + [eos_id]
            results.append(ids)
        return results

    # Attach vocab_size as attribute
    tokenize_batch.vocab_size = vocab_size  # type: ignore
    return tokenize_batch  # type: ignore
