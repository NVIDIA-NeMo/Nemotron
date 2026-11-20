"""Select informative training pages; never select held-out evaluation pages."""

import random
from .data import context_id


def select_training_pages(candidates, answers, pages=256, seed=42, excluded=()):
    if len(candidates) != len(answers) or pages > len(candidates):
        raise ValueError("Need a sampled answer group for each candidate and enough candidates")
    if [r["id"] for r in candidates] != [r["id"] for r in answers]:
        raise ValueError("Candidate/answer IDs differ")
    forbidden = {context_id(r["record"]) for r in excluded}
    contexts = [context_id(r["record"]) for r in candidates]
    if len(set(contexts)) != len(contexts) or not set(contexts).isdisjoint(forbidden):
        raise ValueError("RL candidates must have independent contexts outside other splits")
    successes = [sum(a["correct"] for a in r["answers"]) for r in answers]
    sizes = [len(r["answers"]) for r in answers]
    if not all(sizes):
        raise ValueError("Each candidate needs sampled answers")
    rng = random.Random(seed)
    mixed = [i for i, c in enumerate(successes) if 0 < c < sizes[i]]
    rng.shuffle(mixed)
    mixed.sort(key=lambda i: abs(successes[i] / sizes[i] - 0.5))
    selected = mixed[: int(pages * 0.75)]
    remaining = [i for i in range(len(candidates)) if i not in set(selected)]
    rng.shuffle(remaining)
    selected += remaining[: pages - len(selected)]
    rng.shuffle(selected)
    return [candidates[i] for i in selected]
