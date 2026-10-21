# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer-seed preparation, voting, quality gates, and SFT export."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any

from nemotron.steps.sdg.persona_mcq.runtime.answers import answer_prompt, split_visible_reasoning
from nemotron.steps.sdg.persona_mcq.runtime.languages import script_fraction


def prepare_answer_seed(records: list[dict[str, Any]], _seed: int) -> list[dict[str, Any]]:
    """Render source-compatible ids and option shuffles for answer teachers."""
    prepared: list[dict[str, Any]] = []
    for record in records:
        choices = list(record["choices"])
        query_id = hashlib.blake2b(
            (record["language"] + "||" + record["question"] + "||" + "|".join(choices)).encode(),
            digest_size=12,
        ).hexdigest()
        rng = random.Random(int(query_id[:8], 16))
        permutation = list(range(len(choices)))
        rng.shuffle(permutation)
        metadata = record.get("metadata") or {}
        prepared.append(
            {
                "query_id": query_id,
                "language": record["language"],
                "question": record["question"],
                "choices": [choices[index] for index in permutation],
                "original_choices": choices,
                "shuffle_permutation": permutation,
                "gen_model": metadata.get("source_model"),
                "difficulty": metadata.get("difficulty"),
                "topic": metadata.get("topic"),
                "facet": metadata.get("facet"),
                "region": metadata.get("region"),
                "persona_uuid": metadata.get("persona_uuid"),
            }
        )
    return prepared


def vote(letters: list[str | None], agreement: str) -> str | None:
    valid = [letter for letter in letters if letter]
    if agreement == "unanimous":
        return valid[0] if len(valid) == len(letters) and len(set(valid)) == 1 else None
    if agreement == "majority":
        if not valid:
            return None
        letter, count = Counter(valid).most_common(1)[0]
        return letter if count > len(letters) / 2 else None
    raise ValueError(f"Unsupported agreement policy: {agreement!r}")


def build_sft_records(
    answers_by_model: dict[str, list[dict[str, Any]]],
    *,
    response_model: str,
    language: str,
    language_config: dict[str, Any],
    reasoning_config: dict[str, Any],
    agreement: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    indexed = {
        model: {str(record["query_id"]): record for record in records} for model, records in answers_by_model.items()
    }
    if response_model not in indexed:
        raise ValueError(f"Unknown response model {response_model!r}")
    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for query_id, teacher in indexed[response_model].items():
        letters = [indexed[model].get(query_id, {}).get("parsed_letter") for model in indexed]
        winner = vote(letters, agreement)
        if winner is None:
            skipped["no_agreement"] += 1
            continue
        if teacher.get("parsed_letter") != winner:
            skipped["teacher_dissents"] += 1
            continue
        if teacher.get("finish_reason") in {"length", "max_tokens"}:
            skipped["truncated"] += 1
            continue
        answer = str(teacher.get("answer") or "").strip()
        visible_reasoning, normalized_answer = split_visible_reasoning(
            answer,
            language_config["answer_label"],
            language_config.get("answer_label_aliases") or (),
        )
        if visible_reasoning:
            answer = normalized_answer
        if not answer:
            skipped["empty_answer"] += 1
            continue
        answer_range = language_config["answer_script_fraction"]
        answer_fraction = script_fraction(answer, language_config["script_pattern"])
        if not float(answer_range["min"]) <= answer_fraction <= float(answer_range["max"]):
            skipped["answer_language_impurity"] += 1
            continue
        reasoning_range = reasoning_config["script_fraction"]
        reasoning_candidates = [
            str(teacher.get("reasoning") or "").strip(),
            visible_reasoning,
        ]
        reasoning_candidates = [candidate for candidate in reasoning_candidates if candidate]
        if not reasoning_candidates:
            skipped["empty_reasoning"] += 1
            continue
        reasoning = next(
            (
                candidate
                for candidate in reasoning_candidates
                if float(reasoning_range["min"])
                <= script_fraction(candidate, reasoning_config["script_pattern"])
                <= float(reasoning_range["max"])
            ),
            None,
        )
        if reasoning is None:
            skipped["reasoning_language_impurity"] += 1
            continue
        prompt = answer_prompt(teacher, language_config, reasoning_config)
        output.append(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "reasoning_content": reasoning, "content": answer},
                ],
                "metadata": {
                    "query_id": query_id,
                    "language": language,
                    "answer_label": language_config["answer_label"],
                    "reasoning_language": reasoning_config["display_name"],
                    "response_model": response_model,
                    "agreement": agreement,
                    "voted_letter": winner,
                    "n_valid_votes": sum(letter is not None for letter in letters),
                    "difficulty": teacher.get("difficulty"),
                    "topic": teacher.get("topic"),
                    "facet": teacher.get("facet"),
                    "region": teacher.get("region"),
                    "gen_model": teacher.get("gen_model"),
                    "persona_uuid": teacher.get("persona_uuid"),
                },
            }
        )
    return output, {"kept": len(output), **dict(skipped)}


def _final_answer(answer_label: str, voted: str) -> str:
    return f"{answer_label}: {voted}"


def sample_aligned_datasets(
    datasets: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    sample_per_language: int,
    seed: int,
    reasoning_off_fraction: float,
    answer_variant: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not 0 <= reasoning_off_fraction <= 1:
        raise ValueError("reasoning_off_fraction must be between zero and one")
    indexed = {
        teacher: {
            language: {str(row["metadata"]["query_id"]): row for row in rows} for language, rows in by_language.items()
        }
        for teacher, by_language in datasets.items()
    }
    languages = sorted(set.intersection(*(set(by_language) for by_language in indexed.values())))
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for language in languages:
        common = sorted(set.intersection(*(set(by_language[language]) for by_language in indexed.values())))
        if len(common) < sample_per_language:
            raise ValueError(
                f"{language}: shared teacher intersection {len(common)} is smaller than requested "
                f"{sample_per_language}"
            )
        selected[language] = rng.sample(common, sample_per_language)

    reasoning_off_ids: dict[str, set[str]] = {}
    for language, query_ids in selected.items():
        count = int(len(query_ids) * reasoning_off_fraction)
        ranked = sorted(
            query_ids,
            key=lambda query_id: hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest(),
        )
        reasoning_off_ids[language] = set(ranked[:count])

    result: dict[str, list[dict[str, Any]]] = {}
    for teacher, by_language in indexed.items():
        rows: list[dict[str, Any]] = []
        for language, query_ids in selected.items():
            for query_id in query_ids:
                source = by_language[language][query_id]
                user = next(message["content"] for message in source["messages"] if message["role"] == "user")
                assistant_source = next(message for message in source["messages"] if message["role"] == "assistant")
                is_off = query_id in reasoning_off_ids[language]
                content = assistant_source["content"]
                if answer_variant == "stripped":
                    content = _final_answer(
                        source["metadata"]["answer_label"],
                        source["metadata"]["voted_letter"],
                    )
                elif answer_variant != "full":
                    raise ValueError(f"Unsupported answer variant: {answer_variant!r}")
                assistant = {"role": "assistant", "content": content}
                if not is_off:
                    assistant["reasoning_content"] = assistant_source["reasoning_content"]
                metadata = {**source["metadata"], "reasoning_mode": "off" if is_off else "on"}
                rows.append(
                    {
                        "messages": [{"role": "system", "content": ""}, {"role": "user", "content": user}, assistant],
                        "metadata": metadata,
                    }
                )
        random.Random(seed).shuffle(rows)
        result[teacher] = rows
    return result, {
        "selected_by_language": {language: len(ids) for language, ids in selected.items()},
        "reasoning_off_by_language": {language: len(query_ids) for language, query_ids in reasoning_off_ids.items()},
    }
