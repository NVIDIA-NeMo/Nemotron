# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural, language, exact, and MinHash deduplication for authored MCQs."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from nemotron.steps.sdg.persona_mcq.runtime.languages import compile_script_pattern, script_fraction

LATIN_GLOSS = re.compile(r"\s*\([A-Za-z0-9 ,./'\-]+\)")
PRIME = (1 << 61) - 1


def strip_latin_gloss(text: str, script_pattern: str, *, enabled: bool) -> str:
    pattern = compile_script_pattern(script_pattern)
    return LATIN_GLOSS.sub("", text).strip() if enabled and pattern.search(text) else text


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\wऀ-ॿ ]", "", text)


def shingles(text: str, size: int) -> set[str]:
    text = normalize(text)
    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _signature(values: set[str], permutations: int, seed: int) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    multipliers = rng.integers(1, PRIME, size=permutations, dtype=np.int64)
    offsets = rng.integers(0, PRIME, size=permutations, dtype=np.int64)
    signature = [PRIME] * permutations
    for value in values:
        token = int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big") % PRIME
        for index in range(permutations):
            hashed = (int(multipliers[index]) * token + int(offsets[index])) % PRIME
            if hashed < signature[index]:
                signature[index] = hashed
    return tuple(signature)


def _extract(record: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]] | None:
    conversation = record.get("conversation", record)
    if isinstance(conversation, str):
        import json

        try:
            conversation = json.loads(conversation)
        except json.JSONDecodeError:
            return None
    if not isinstance(conversation, dict):
        return None
    metadata = conversation.get("metadata") or {}
    parsed = metadata.get("parsed_question") or {}
    question = parsed.get("question")
    choices = parsed.get("choices")
    if not isinstance(question, str) or not question.strip() or not isinstance(choices, list):
        return None
    choices = [str(choice).strip() for choice in choices]
    if len(choices) != 4 or not all(choices) or len({normalize(choice) for choice in choices}) != 4:
        return None
    return question.strip(), choices, metadata


def deduplicate(
    records: list[dict[str, Any]],
    *,
    language: str,
    script_pattern: str,
    min_script_fraction: float,
    max_script_fraction: float,
    strip_latin_glosses: bool,
    source_model: str,
    threshold: float = 0.80,
    shingle_size: int = 4,
    permutations: int = 128,
    bands: int = 16,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if permutations % bands:
        raise ValueError("lexical permutations must be divisible by bands")
    stats: Counter[str] = Counter(loaded=len(records))
    exact: set[str] = set()
    accepted: list[dict[str, Any]] = []
    accepted_shingles: list[set[str]] = []
    buckets: defaultdict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    rows = permutations // bands

    for record in records:
        extracted = _extract(record)
        if extracted is None:
            stats["drop_structural"] += 1
            continue
        question, choices, metadata = extracted
        alphabetic = sum(character.isalpha() for character in question)
        fraction = script_fraction(question, script_pattern)
        if alphabetic and not min_script_fraction <= fraction <= max_script_fraction:
            stats["drop_wrong_language"] += 1
            continue
        question = strip_latin_gloss(question, script_pattern, enabled=strip_latin_glosses)
        choices = [strip_latin_gloss(choice, script_pattern, enabled=strip_latin_glosses) for choice in choices]
        exact_key = hashlib.sha256(
            (normalize(question) + "\0" + "\0".join(sorted(normalize(choice) for choice in choices))).encode()
        ).hexdigest()
        if exact_key in exact:
            stats["drop_exact"] += 1
            continue
        tokens = shingles(question, shingle_size)
        signature = _signature(tokens, permutations, seed)
        candidates: set[int] = set()
        for band in range(bands):
            key = (band, signature[band * rows : (band + 1) * rows])
            candidates.update(buckets[key])
        if any(_jaccard(tokens, accepted_shingles[index]) >= threshold for index in candidates):
            stats["drop_near"] += 1
            continue
        query_id = hashlib.sha256(
            (language.lower() + "\0" + normalize(question) + "\0" + "\0".join(normalize(c) for c in choices)).encode()
        ).hexdigest()
        output = {
            "query_id": query_id,
            "question": question,
            "choices": choices,
            "language": language.lower(),
            "metadata": {
                "source_model": source_model,
                "difficulty": metadata.get("difficulty"),
                "topic": metadata.get("topic"),
                "facet": metadata.get("facet"),
                "region": metadata.get("region"),
                "persona_uuid": (metadata.get("persona") or {}).get("uuid")
                if isinstance(metadata.get("persona"), dict)
                else None,
            },
        }
        index = len(accepted)
        accepted.append(output)
        accepted_shingles.append(tokens)
        exact.add(exact_key)
        for band in range(bands):
            buckets[(band, signature[band * rows : (band + 1) * rows])].append(index)

    stats["accepted"] = len(accepted)
    return accepted, dict(stats)
