"""Step 2b: join persona seeds with official procedure facts to make the Data Designer seed file.

Each output row = one persona + one procedure + the published facts for that procedure + an optional
region-specific note. The generator may only use those facts, which is what stops it inventing
documents, fees or deadlines.

    uv run python use-case-examples/french-public-service-assistant/scripts/build_seeds.py

Input : data/persona_seeds.jsonl (from sample_personas.py), sources/procedures.json
Output: data/seeds.jsonl

Persona names are removed here so they cannot leak into the training data. Overseas personas are
repeated (the Nemotron personas have few) so a batch contains enough local cases, e.g. the Guyane
passport stamp price.
"""

import argparse
import json
import random
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent

PASSPORT = "Passeport d'un majeur : première demande"
ID_CARD_FIRST = "Carte d'identité d'un majeur : première demande"
# Sampling weight per procedure, in the order they appear in procedures.json.
WEIGHTS = [3, 2, 2, 2, 2, 1]
OVERSEAS = ("Guyane", "Martinique", "Guadeloupe", "La Réunion", "Mayotte")


def strip_name(base: dict, full_name: str) -> None:
    """Replace the persona's name (and each part of it) with a neutral placeholder."""
    for token in sorted({full_name, *full_name.split()}, key=len, reverse=True):
        if len(token) > 2:
            for field in ("professional_persona", "persona"):
                base[field] = base[field].replace(token, "l'usager")


def local_note(procedure: str, departement: str) -> str:
    """Region-specific fact attached to the seed (empty when the procedure has none)."""
    if procedure != PASSPORT:
        return ""
    if departement == "Guyane":
        return "Le demandeur réside en Guyane : le timbre fiscal du passeport est de 43 € (et non 86 €)."
    return "Le demandeur ne réside pas en Guyane : le timbre fiscal du passeport est de 86 €."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--personas", type=Path, default=EXAMPLE_DIR / "data" / "persona_seeds.jsonl")
    parser.add_argument("--procedures", type=Path, default=EXAMPLE_DIR / "sources" / "procedures.json")
    parser.add_argument("--out", type=Path, default=EXAMPLE_DIR / "data" / "seeds.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    procs = json.load(open(args.procedures, encoding="utf-8"))
    by_name = {p["procedure"]: p for p in procs}

    n_rows = n_overseas = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for line in open(args.personas, encoding="utf-8"):
            p = json.loads(line)
            base = {
                k: p[k]
                for k in ("source_uuid", "professional_persona", "persona", "commune", "departement", "occupation")
            }
            strip_name(base, p["professional_persona"].split(",")[0].strip())

            overseas = p["departement"] in OVERSEAS
            repeats = (8 if p["departement"] == "Guyane" else 4) if overseas else 1
            for _ in range(repeats):
                if overseas:
                    passport_share = 0.8 if p["departement"] == "Guyane" else 0.6
                    chosen = by_name[PASSPORT] if random.random() < passport_share else by_name[ID_CARD_FIRST]
                else:
                    chosen = random.choices(procs, WEIGHTS)[0]
                row = dict(
                    base,
                    procedure=chosen["procedure"],
                    procedure_facts=chosen["facts"],
                    local_note=local_note(chosen["procedure"], p["departement"]),
                )
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                n_overseas += overseas

    print(f"wrote {n_rows} seed rows ({n_overseas} overseas) to {args.out}")


if __name__ == "__main__":
    main()
