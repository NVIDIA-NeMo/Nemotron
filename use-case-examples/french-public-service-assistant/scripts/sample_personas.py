"""Step 2: sample a balanced slice of Nemotron-Personas-France into compact seed records.

Selects at most --per-group personas for each (departement, occupation) pair, keeps the source UUID
for lineage, and writes data/persona_seeds.jsonl. It then runs build_seeds.py, which joins each persona
with a procedure and its official facts into data/seeds.jsonl, the file the Data Designer config reads.
Pass --no-build-seeds to stop after the persona sample.

    uv run python use-case-examples/french-public-service-assistant/scripts/sample_personas.py

Inspect the coverage counts it prints before generating data. Matching population distributions is
not automatically the same as meeting product-quality or fairness goals.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

import build_seeds

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=2000, help="number of persona seeds to keep")
    parser.add_argument("--per-group", type=int, default=3, help="max personas per (departement, occupation)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-build-seeds", action="store_true", help="only write the persona sample")
    parser.add_argument("--out", type=Path, default=EXAMPLE_DIR / "data" / "persona_seeds.jsonl")
    args = parser.parse_args()

    ds = load_dataset("nvidia/Nemotron-Personas-France", split="train")
    random.seed(args.seed)
    quota: dict[tuple[str, str], int] = defaultdict(int)
    seeds = []

    for row in ds.shuffle(seed=args.seed):
        key = (row["departement"], row["occupation"])
        if quota[key] >= args.per_group:
            continue
        quota[key] += 1
        seeds.append(
            {
                "source_uuid": row["uuid"],
                "professional_persona": row["professional_persona"],
                "persona": row["persona"],
                "skills": row["skills_and_expertise_list"],
                "interests": row["hobbies_and_interests_list"],
                "commune": row["commune"],
                "departement": row["departement"],
                "occupation": row["occupation"],
            }
        )
        if len(seeds) == args.size:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in seeds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(seeds)} persona seeds to {args.out}")
    print("Départements covered:", len({s["departement"] for s in seeds}))
    print("Top occupations:", Counter(s["occupation"] for s in seeds).most_common(5))

    if not args.no_build_seeds:
        build_seeds.main(["--personas", str(args.out), "--seed", str(args.seed)])


if __name__ == "__main__":
    main()
