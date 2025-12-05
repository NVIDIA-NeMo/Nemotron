"""Tokenizer provider factories."""

from typing import Callable, Protocol


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
    """
    tokenizer_type = resolved_config["type"]

    if tokenizer_type == "huggingface":
        return _create_huggingface_tokenizer(resolved_config)
    elif tokenizer_type == "sentencepiece":
        return _create_sentencepiece_tokenizer(resolved_config)
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
