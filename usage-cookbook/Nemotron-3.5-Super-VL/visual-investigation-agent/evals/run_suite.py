# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the agent several times on one world against the selected endpoint, then score every run.

    python -m evals.run_suite --scenario model-switch-visual --trials 3 [--ablate playbook|policy] [--config changes.json]
    python -m evals.run_suite --scenario g-price-2003 --trials 3

A world is a Fernledger month (model-switch, volume-growth, model-switch-visual, price-increase) or a generated world
(world/generator.py, for example g-price-2003). Runs go to runs/<scenario>-<stamp>/trial-N. Each is scored segment by segment
(evals/segments.py); Fernledger months also get the original exact checks (evals/checks.py).
"""

import argparse
import concurrent.futures
import datetime as dt
import json
from pathlib import Path

import world
from investigator.config import from_ablate, make
from investigator.harness import RUNS, investigate
from investigator.runs import load_run

from . import checks, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--ablate", choices=["playbook", "policy"])
    parser.add_argument("--config", help="JSON file of configuration changes (investigator/config.py)")
    args = parser.parse_args()
    world.spec(args.scenario)  # fails early on an unknown world
    world.ensure(args.scenario, images=True)  # render the documents once, before the trials start
    config = make(json.loads(Path(args.config).read_text()), name=Path(args.config).stem) if args.config else from_ablate(args.ablate)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = RUNS / f"{args.scenario}-{args.ablate or config['name']}-{stamp}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.trials) as pool:
        jobs = [pool.submit(investigate, args.scenario, folder / f"trial-{i + 1}", args.ablate, None, None, config)
                for i in range(args.trials)]
        trials = sorted(Path(j.result()) for j in jobs)
    for trial in trials:
        run = load_run(trial)
        s = segments.score(run)
        line = (f"{trial.name}: {run.result['seconds']:.0f} s, {run.result['turns']} steps, error={run.result['error']}; "
                f"end to end {'pass' if s['e2e'] else 'FAIL'}, interventions {s['interventions']}; segments failed: "
                f"{[f'{k} {segments.NAMES[k]}' for k in segments.SEGMENTS if s[k] is False] or 'none'}")
        if args.scenario in world.SCENARIOS:
            exact = checks.exact_checks(run)
            line += "; exact " + ", ".join(f"{checks.EXACT_LABELS[k]}={'pass' if v else 'FAIL'}" for k, v in exact.items())
        print(line)
    print(folder)


if __name__ == "__main__":
    main()
