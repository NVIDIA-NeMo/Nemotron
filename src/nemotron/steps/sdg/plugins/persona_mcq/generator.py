# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Designer generator for persona-grounded MCQs."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)

from nemotron.steps.sdg.plugins.persona_mcq.config import PersonaMCQConfig
from nemotron.steps.sdg.plugins.persona_mcq.llm import completion_text
from nemotron.steps.sdg.plugins.persona_mcq.parsing import format_question, parse_question
from nemotron.steps.sdg.plugins.persona_mcq.prompts import (
    QUESTION_AUTHOR_SYSTEM_PROMPT_CONTEXTUAL,
    QUESTION_AUTHOR_SYSTEM_PROMPT_KNOWLEDGE_MCQ_FACET,
)
from nemotron.steps.sdg.plugins.persona_mcq.taxonomy import (
    CONTEXTUAL_FACETS,
    DIFFICULTY_WEIGHTS,
    FACET_WEIGHTS,
    GEOGRAPHY_FACET,
)

FACET_LABELS = {
    "arts_persona": "arts, music and literature",
    "culinary_persona": "food and cuisine",
    "religious_background": "religious background",
    "cultural_background": "cultural background",
    "professional_persona": "profession and field of work",
    "skills_and_expertise": "skills and expertise",
    "linguistic_background": "language",
    "sports_persona": "sports",
    "travel_persona": "travel",
    "hobbies_and_interests": "hobbies and interests",
    "healthcare_persona": "health and medicine",
    "finance_persona": "personal finance",
    GEOGRAPHY_FACET: "geography and regional knowledge",
}
GEO_FIELDS = (
    ("country", "Country"),
    ("state", "State/Region"),
    ("region", "State/Region"),
    ("district", "District"),
    ("city", "City"),
    ("zone", "Area type"),
    ("first_language", "Primary language"),
)
CONTEXT_FIELDS = (
    ("occupation", "Occupation"),
    ("age", "Age"),
    ("country", "Country"),
    ("state", "State/Region"),
    ("region", "State/Region"),
    ("city", "City"),
    ("zone", "Area type"),
)


def _field(persona: Any, key: str) -> str:
    if not isinstance(persona, dict):
        return ""
    value = persona.get(key)
    text = "" if value is None else str(value).strip()
    return "" if text.lower() == "none" else text


def _descriptor(persona: Any, fields: tuple[tuple[str, str], ...]) -> str:
    return "; ".join(f"{label}: {value}" for key, label in fields if (value := _field(persona, key)))


def _rng(persona: Any, seed: int) -> random.Random:
    stable = json.dumps(persona, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{seed}:{stable}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = [(key, max(0.0, float(weight))) for key, weight in weights.items() if float(weight) > 0]
    if not items:
        raise ValueError("sampling weights must include at least one positive value")
    return rng.choices([key for key, _ in items], weights=[weight for _, weight in items], k=1)[0]


class PersonaMCQGenerator(
    ColumnGeneratorCellByCell[PersonaMCQConfig],
    ColumnGeneratorWithModelRegistry[PersonaMCQConfig],
):
    """Author one MCQ from a deterministic persona facet and difficulty sample."""

    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        persona = data[cfg.persona_column]
        rng = _rng(persona, cfg.random_seed)
        try:
            prompt, metadata = self._build_prompt(persona, cfg, rng)
            raw = completion_text(
                self.get_model(cfg.model_alias),
                system_prompt=prompt,
                user_prompt="Please write one multiple-choice question now, following the required format exactly.",
            )
            parsed = parse_question(raw, num_options=cfg.num_options)
            metadata["parsed_question"] = parsed
            if not parsed["success"]:
                metadata.update({"error": "generation_failure", "raw_response": raw})
                messages: list[dict[str, str]] = []
            else:
                messages = [{"role": "user", "content": format_question(parsed)}]
            ok = bool(parsed["success"])
        except Exception as exc:  # noqa: BLE001 - plugin records row-level failures for later retry.
            metadata = {"track": "persona_grounded", "error": f"{type(exc).__name__}: {exc}"}
            messages = []
            ok = False
        data[cfg.name] = json.dumps({"messages": messages, "metadata": metadata}, ensure_ascii=False, default=str)
        data["conversation_status"] = ok
        return data

    @staticmethod
    def _build_prompt(
        persona: Any,
        cfg: PersonaMCQConfig,
        rng: random.Random,
    ) -> tuple[str, dict[str, Any]]:
        facet_weights = cfg.facet_weights or FACET_WEIGHTS
        eligible: dict[str, float] = {}
        facet_texts: dict[str, str] = {}
        for facet, weight in facet_weights.items():
            text = _descriptor(persona, GEO_FIELDS) if facet == GEOGRAPHY_FACET else _field(persona, facet)
            if text and weight > 0:
                eligible[facet] = weight
                facet_texts[facet] = text
        if not eligible:
            for fallback in ("persona", "detailed_persona"):
                if text := _field(persona, fallback):
                    eligible[fallback] = 1.0
                    facet_texts[fallback] = text
                    break
        if not eligible:
            raise ValueError("persona has no usable MCQ facet")

        facet = _weighted_choice(rng, eligible)
        difficulty = _weighted_choice(rng, cfg.difficulty_weights or DIFFICULTY_WEIGHTS)
        region = next(
            (value for key in ("region", "state", "country", "city", "district") if (value := _field(persona, key))),
            "",
        )
        region_hint = _descriptor(persona, GEO_FIELDS) or "the locale represented by the persona"
        metadata: dict[str, Any] = {
            "track": "persona_grounded",
            "persona": persona,
            "facet": facet,
            "difficulty": difficulty,
            "region": region or None,
        }
        if facet in CONTEXTUAL_FACETS:
            contextual = CONTEXTUAL_FACETS[facet]
            subtopic = rng.choice(contextual["subtopics"])
            subject = contextual["subject"]
            metadata.update({"topic": subject, "subtopic": subtopic})
            prompt = QUESTION_AUTHOR_SYSTEM_PROMPT_CONTEXTUAL.format(
                subject=subject,
                subtopic=subtopic,
                context_text=_descriptor(persona, CONTEXT_FIELDS) or region_hint,
                profile_text=facet_texts[facet],
                difficulty=difficulty,
                language=cfg.language,
                num_options=cfg.num_options,
            )
        else:
            topic = FACET_LABELS.get(facet, facet)
            metadata["topic"] = topic
            prompt = QUESTION_AUTHOR_SYSTEM_PROMPT_KNOWLEDGE_MCQ_FACET.format(
                facet_name=topic,
                facet_text=facet_texts[facet],
                region=region_hint,
                difficulty=difficulty,
                language=cfg.language,
                num_options=cfg.num_options,
            )
        return prompt, metadata
