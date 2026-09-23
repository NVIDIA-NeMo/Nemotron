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

"""Persona formatting (Nemotron-Personas style). Domain-agnostic.

The generator passes whatever persona record the seed provides; any subset of
these keys is rendered into a short persona blurb.
"""

from __future__ import annotations

import random
from datetime import date, datetime
from typing import Any, Dict, List


def format_persona_for_prompt(persona: Dict[str, Any]) -> str:
    def f(key: str) -> str:
        v = persona.get(key)
        return str(v) if v is not None else ""

    state = f("state") or f("region")
    age = f("age")
    if not age and f("birth_date"):
        try:
            bd = datetime.fromisoformat(str(persona["birth_date"])).date()
            age = str((date.today() - bd).days // 365)
        except Exception:
            age = ""

    name = " ".join(p for p in [f("first_name"), f("last_name")] if p) or "Unknown"
    demo: List[str] = []
    if age:
        demo.append(f"Age: {age}")
    for key, label in [("sex", "Sex"), ("marital_status", "Marital Status"),
                       ("occupation", "Occupation"), ("education_level", "Education Level")]:
        if f(key):
            demo.append(f"{label}: {f(key)}")
    loc = ", ".join(x for x in [f("city"), state, f("country")] if x)
    if loc:
        demo.append(f"Location: {loc}")

    sections = [f"Name: {name}"] if name != "Unknown" else []   # Nemotron-Personas has no name field
    if demo:
        sections.append("Demographics:\n" + "\n".join(demo))

    facets = [
        ("persona", "Overview"), ("professional_persona", "Professional"),
        ("finance_persona", "Finance"), ("healthcare_persona", "Healthcare"),
        ("skills_and_expertise", "Skills & Expertise"),
        ("hobbies_and_interests", "Hobbies & Interests"),
        ("career_goals_and_ambitions", "Career Goals"),
    ]
    random.shuffle(facets)
    for key, label in facets:
        if f(key):
            sections.append(f"{label}: {f(key).rstrip('.') + '.'}")
    return "\n\n".join(sections)
