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
import re
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent

PASSPORT = "Passeport d'un majeur : première demande"
ID_CARD_FIRST = "Carte d'identité d'un majeur : première demande"
# Sampling weight per procedure, in the order they appear in procedures.json.
WEIGHTS = [3, 2, 2, 2, 2, 1]
OVERSEAS = ("Guyane", "Martinique", "Guadeloupe", "La Réunion", "Mayotte")


# Words that may sit inside a name without a capital letter ("Jean de La Fontaine", "Amine ben Salah").
PARTICLES = {"de", "du", "des", "d'", "la", "le", "van", "von", "ben", "el", "di", "da"}
NAME_WORD = r"[A-ZÀ-ÖØ-Ý][\w'’\-]*"
LEADING_NAME = re.compile(rf"^({NAME_WORD}(?:\s+(?:(?:{'|'.join(sorted(PARTICLES))})\s+)?{NAME_WORD}){{0,3}})(?P<end>,?)")


MIDTEXT_NAME = re.compile(rf"[,.;]\s+({NAME_WORD}(?:\s+(?:(?:{'|'.join(sorted(PARTICLES))})\s+)?{NAME_WORD}){{1,3}})")


def extract_name(text: str) -> str | None:
    """Return the persona's name: the run of capitalised words that opens the text, or None.

    Nemotron-Personas-France profiles start with the name ("Yvonne Hostin, ancienne secrétaire ..." or
    "Marie Dupont qui gère ..."), so the name is the leading capitalised run, not everything before the
    first comma. A lone capitalised word is only accepted when a comma follows it, so a sentence opening
    like "Anciennement agent, ..." is not mistaken for a name.
    """
    match = LEADING_NAME.match(text.strip())
    if match and (len(match.group(1).split()) >= 2 or match.group("end")):
        return match.group(1)
    # Some profiles open with the job ("Ancien technicien ..., Rémi Perret combine ..."): take the first
    # two-or-more-word capitalised run after a punctuation mark.
    midtext = MIDTEXT_NAME.search(text)
    return midtext.group(1) if midtext else None


def strip_name(base: dict, full_name: str | None) -> bool:
    """Replace the persona's name (the full name and each capitalised part) with a neutral placeholder.

    Matches whole words only, so a name part never rewrites the inside of another word. Returns False
    when no name was found, so the caller can report it.
    """
    if not full_name:
        return False
    parts = [t for t in full_name.split() if t.lower() not in PARTICLES and len(t) > 2]
    for token in sorted({full_name, *parts}, key=len, reverse=True):
        pattern = re.compile(rf"(?<![\w\-]){re.escape(token)}(?![\w\-])")
        for field in ("professional_persona", "persona"):
            base[field] = pattern.sub("l'usager", base[field])
    return True


def local_note(procedure: str, departement: str) -> str:
    """Region-specific fact attached to the seed (empty when the procedure has none)."""
    if procedure != PASSPORT:
        return ""
    if departement == "Guyane":
        return "Le demandeur réside en Guyane : le timbre fiscal du passeport est de 43 € (et non 86 €)."
    return "Le demandeur ne réside pas en Guyane : le timbre fiscal du passeport est de 86 €."


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--personas", type=Path, default=EXAMPLE_DIR / "data" / "persona_seeds.jsonl")
    parser.add_argument("--procedures", type=Path, default=EXAMPLE_DIR / "sources" / "procedures.json")
    parser.add_argument("--out", type=Path, default=EXAMPLE_DIR / "data" / "seeds.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    procs = json.load(open(args.procedures, encoding="utf-8"))
    by_name = {p["procedure"]: p for p in procs}

    n_rows = n_overseas = n_unnamed = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for line in open(args.personas, encoding="utf-8"):
            p = json.loads(line)
            base = {
                k: p[k]
                for k in ("source_uuid", "professional_persona", "persona", "commune", "departement", "occupation")
            }
            name = extract_name(p["professional_persona"]) or extract_name(p["persona"])
            if not strip_name(base, name):
                n_unnamed += 1

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
    if n_unnamed:
        print(f"warning: no name found in {n_unnamed} personas; the audit step still drops self-introductions")


if __name__ == "__main__":
    main()
