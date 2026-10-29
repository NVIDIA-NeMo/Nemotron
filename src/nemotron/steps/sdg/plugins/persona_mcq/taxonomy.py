# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampling weights for persona-grounded MCQ generation."""

from __future__ import annotations

FACET_WEIGHTS = {
    "geography": 0.15,
    "arts_persona": 0.12,
    "cultural_background": 0.12,
    "culinary_persona": 0.10,
    "professional_persona": 0.10,
    "religious_background": 0.09,
    "skills_and_expertise": 0.08,
    "travel_persona": 0.05,
    "hobbies_and_interests": 0.04,
    "healthcare_persona": 0.06,
    "finance_persona": 0.04,
    "sports_persona": 0.03,
    "linguistic_background": 0.02,
}

GEOGRAPHY_FACET = "geography"

CONTEXTUAL_FACETS = {
    "finance_persona": {
        "subject": "personal finance and financial literacy",
        "subtopics": [
            "loans & credit (agricultural, small-business, home, and secured loans)",
            "insurance (life, health, crop, and public insurance programmes)",
            "taxation (income, consumption, withholding, and tax-saving rules)",
            "government savings and pension schemes",
            "banking and digital payments",
            "investment & markets (mutual funds, shares, gold, bonds)",
            "retirement & provident fund (EPF, gratuity, pension)",
            "micro-finance & self-help groups",
        ],
    },
    "healthcare_persona": {
        "subject": "health and medicine",
        "subtopics": [
            "nutrition & diet",
            "infectious diseases (malaria, tuberculosis, dengue, etc.)",
            "occupational health & work-related risks",
            "maternal & child health",
            "non-communicable diseases (diabetes, hypertension, heart disease)",
            "mental health",
            "public health programmes & immunisation",
            "first aid & disease prevention",
            "basics of pharmacology & treatment",
            "traditional and locally practiced medicine",
        ],
    },
}

DIFFICULTY_WEIGHTS = {
    "high-school level (easy)": 0.10,
    "undergraduate level (medium)": 0.30,
    "graduate level (hard)": 0.35,
    "postgraduate / expert level (very hard)": 0.25,
}
