"""Step 5: audit the generated records and split them 90/10 by source persona.

Drops a record when any check fails:
  text    - leftover reasoning, template placeholders, or an implausible length
  name    - the persona introduces themselves ("je m'appelle ...")
  route   - the answer offers a way to file (paper, post, kiosk/terminal) that the sources do not give;
            saying the route is NOT available is fine
  numbers - the answer contains a number that is not in the source facts for that procedure
            ("un an" and "1 an" count as the same number; "2" does not match "12")
  judge   - the judge column did not mark the answer as grounded

Split by source_uuid, not by row, so one persona never appears in both train and validation.

    uv run python use-case-examples/french-public-service-assistant/scripts/audit_and_split.py

Input : data/sdg/sft_raw.jsonl, sources/procedures.json
Output: data/train.jsonl, data/validation.jsonl

Automated checks are a filter, not a guarantee. Read a sample of the kept records yourself before
training: a judge can be wrong in both directions.
"""

import argparse
import json
import random
import re
from decimal import Decimal
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


INVENTED_ROUTE = re.compile(r"(?i)sur papier|par courrier|par voie postale|\bborne\b|terminal")
# A sentence that says the route is NOT available ("on ne peut pas déposer par courrier") is correct.
NEGATION = re.compile(r"(?i)\b(?:ne|n['’])\s*[^.;!?\n]*?\b(?:pas|jamais|plus|aucun\w*)\b|impossible|interdit")
NUMBER = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)?(?![\d])")
NUMBER_WORDS = {
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
    "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
}  # fmt: skip
SPELLED_AMOUNT = re.compile(rf"(?i)\b({'|'.join(NUMBER_WORDS)})\s+(ans?|années?|mois|jours?|semaines?|€|euros?)\b")


def route_ok(answer: str) -> bool:
    """False when the answer offers a filing route the sources do not give (paper, post, kiosk)."""
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", answer):
        if INVENTED_ROUTE.search(sentence) and not NEGATION.search(sentence):
            return False
    return True


def canonical_numbers(text: str) -> set[str]:
    """Numbers in the text as comparable strings: "un an" -> "1", "43,00" -> "43", "12" never matches "2"."""
    text = SPELLED_AMOUNT.sub(lambda m: f"{NUMBER_WORDS[m.group(1).lower()]} {m.group(2)}", text)
    return {str(Decimal(n.replace(",", ".")).normalize()) for n in NUMBER.findall(text)}


def numbers_ok(answer: str, source: str) -> bool:
    """Every number in the answer also appears in the source facts. A cheap filter, not a factuality check:
    it cannot tell that a number is attached to the wrong fact, which is what the judge column is for."""
    return canonical_numbers(answer) <= canonical_numbers(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=EXAMPLE_DIR / "data" / "sdg" / "sft_raw.jsonl")
    parser.add_argument("--procedures", type=Path, default=EXAMPLE_DIR / "sources" / "procedures.json")
    parser.add_argument("--out-dir", type=Path, default=EXAMPLE_DIR / "data")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    facts = {p["procedure"]: p["facts"] for p in json.load(open(args.procedures, encoding="utf-8"))}
    rows = [json.loads(line) for line in open(args.raw, encoding="utf-8")]

    def judge_ok(r: dict) -> bool:
        j = r.get("grounding")
        if isinstance(j, str):
            try:
                j = json.loads(j)
            except json.JSONDecodeError:
                return False
        return bool(j and j.get("grounded") is True)

    def route_ok_row(r: dict) -> bool:
        return route_ok(r["messages"][1]["content"])

    def name_ok(r: dict) -> bool:
        text = r["messages"][0]["content"] + " " + r["messages"][1]["content"]
        return not re.search(r"(?i)je m'appelle|mon nom est", text)

    def text_ok(r: dict) -> bool:
        answer = r["messages"][1]["content"]
        return not re.search(r"<think>|\[Nom|\[Votre|\[adresse", answer) and 40 < len(answer) < 1800

    checks = [
        ("text", text_ok),
        ("name", name_ok),
        ("route", route_ok_row),
        ("numbers", lambda r: numbers_ok(r["messages"][1]["content"], facts[r["procedure"]])),
        ("judge", judge_ok),
    ]
    kept, dropped = [], {name: 0 for name, _ in checks}
    for r in rows:
        for name, check in checks:
            if not check(r):
                dropped[name] += 1
                break
        else:
            kept.append(r)
    print(f"{len(rows)} generated -> {len(kept)} kept; dropped {dropped}")

    ids = sorted({r["source_uuid"] for r in kept})
    random.shuffle(ids)
    val_ids = set(ids[: max(1, int(len(ids) * args.val_fraction))])

    train_path, val_path = args.out_dir / "train.jsonl", args.out_dir / "validation.jsonl"
    with open(train_path, "w", encoding="utf-8") as train, open(val_path, "w", encoding="utf-8") as val:
        for r in kept:
            # Keep only what the trainer needs; drop judge output and other metadata.
            item = {"messages": r["messages"], "source_uuid": r["source_uuid"]}
            (val if r["source_uuid"] in val_ids else train).write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {train_path.name} and {val_path.name} to {args.out_dir} (split by source_uuid)")


if __name__ == "__main__":
    main()
