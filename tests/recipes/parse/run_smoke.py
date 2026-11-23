"""Manual GPU integration check; not collected by pytest.

Run from an installed Nemotron checkout:
    python tests/recipes/parse/run_smoke.py --output /absolute/path/smoke
    python tests/recipes/parse/run_smoke.py --output /absolute/path/docker-smoke --run parse-docker

The output path must be mounted at the same absolute path for Docker execution.
Results check optimizer updates and checkpoint reload, not model quality.
RL uses an explicitly test-only dense serialization reward; all held-out scoring
retains the dental task definitions. Production recipe rewards are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def small_dataset(source: Path, output: Path) -> None:
    from nemotron.recipes.parse.common import artifact, dataset_identity

    output.mkdir()
    counts = {"train": 2, "monitor": 1, "test": 1}
    for split, count in counts.items():
        rows = [json.loads(line) for line in (source / f"{split}.jsonl").read_text().splitlines()][:count]
        if len(rows) != count:
            raise ValueError(f"Need {count} pages in {split}")
        for row in rows:
            image = Path(row["image"])
            if image.is_absolute() or ".." in image.parts:
                raise ValueError("The reference data must use images inside its bundle")
            target = output / image
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / image, target)
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "dataset.json").write_text(json.dumps({"smoke_only": True, "pages": counts}))
    artifact(output, "parse/dataset", output, identity=dataset_identity(output))


def check_training_steps(output: Path) -> dict:
    evidence = {}
    for stage in ("sft", "rlvr"):
        history = [json.loads(line) for line in (output / stage / "history.jsonl").read_text().splitlines()]
        training = [row for row in history if row["phase"] == "train"]
        if [row["step"] for row in training] != [1, 2]:
            raise AssertionError(f"Expected two completed {stage} updates")
        gradients = [row["metrics"].get("grad_norm", float("nan")) for row in training]
        if not all(math.isfinite(value) for value in gradients) or not any(value > 0 for value in gradients):
            raise AssertionError(f"{stage.upper()} did not report finite gradients with at least one nonzero update")
        evidence[stage] = {"completed_steps": [row["step"] for row in training], "grad_norms": gradients}
        if not any(
            math.isfinite(row["metrics"].get("grad_norm", float("nan")))
            and row["metrics"]["grad_norm"] > 0
            and math.isfinite(row["metrics"].get("lr", float("nan")))
            and row["metrics"].get("lr", 0) > 0
            for row in training
        ):
            raise AssertionError(
                f"{stage.upper()} did not report a finite, nonzero gradient with a positive learning rate"
            )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", help="NeMo-Run environment profile (omit for local uv)")
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Choose a new empty smoke directory: {output}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                str(ROOT / "src"),
                str(Path(__file__).resolve().parent),
                "/nemo_run/code/tests/recipes/parse",
                env.get("PYTHONPATH", ""),
            ],
        )
    )

    def run(stage: str, *overrides: str) -> None:
        cmd = [sys.executable, "-m", "nemotron", "parse", stage]
        if args.run:
            cmd += ["--run", args.run]
            if stage == "rlvr":
                # Profiles forward configured env vars, not the submitter's PYTHONPATH.
                cmd.append("run.env.env_vars.PYTHONPATH=/nemo_run/code/tests/recipes/parse:/nemo_run/code/src")
        subprocess.run([*cmd, *overrides], env=env, cwd=ROOT, check=True)

    run("data", f"data_dir={output / 'full-data'}")
    small_dataset(output / "full-data", output / "data")
    data = f"data_dir={output / 'data'}"
    run(
        "sft",
        "-c",
        "tiny",
        data,
        "task=null",
        f"output_dir={output / 'sft'}",
        # The reduced dataset needs multiple epochs to exercise both updates.
        "training.step_scheduler.num_epochs=null",
    )
    run(
        "rlvr",
        "-c",
        "tiny",
        data,
        f"sft_output={output / 'sft'}",
        f"output_dir={output / 'rlvr'}",
        "task=smoke_task.make_smoke_task",
        "candidate_contexts=2",
        "selection_pages=2",
        "mining_samples=2",
        "training.grpo.num_prompts_per_step=2",
        "training.grpo.num_generations_per_prompt=2",
        "training.policy.train_global_batch_size=4",
        "training.policy.train_micro_batch_size=2",
        "training.policy.logprob_batch_size=2",
        "training.policy.generation.vllm_cfg.enforce_eager=true",
    )
    run("eval", data, f"model_artifact={output / 'rlvr/artifact.json'}", f"output_dir={output / 'eval'}", "samples=1")
    optimizer = check_training_steps(output)
    comparison = json.loads((output / "rlvr/evaluation.json").read_text())
    report = json.loads((output / "eval/evaluation.json").read_text())
    if set(report["models"]) != {"base", "sft", "rlvr"}:
        raise AssertionError("Standalone evaluation did not compare base, SFT, and RLVR checkpoints")
    reloaded = report["models"]["rlvr"]
    if comparison["after"] != reloaded:
        raise AssertionError("Standalone checkpoint evaluation differs from the RLVR after-evaluation")
    trained_answers = json.loads((output / "rlvr/after/answers.json").read_text())
    reloaded_answers = json.loads((output / "eval/rlvr/predictions/answers.json").read_text())
    if trained_answers != reloaded_answers:
        raise AssertionError("Exported-checkpoint answer text or per-answer scores changed on reload")
    (output / "smoke-results.json").write_text(
        json.dumps(
            {
                "smoke_only": True,
                "profile": args.run,
                "rl_training_reward": "test-only serialization_similarity; not a quality metric",
                "optimizer_steps": optimizer,
                "reload_matches": True,
                "predictions_match": True,
                "evaluation": reloaded,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Smoke workflow and checkpoint reload passed: {output}")


if __name__ == "__main__":
    main()
