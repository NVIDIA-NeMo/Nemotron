"""Matched held-out comparison for the dental reference task."""

from nemotron.recipes.parse.common import write_json
from nemotron.recipes.parse.dental.evaluate import evaluate_isolated, evaluation_protocol, summarize
from nemotron.recipes.parse.training.lineage import verify_model


def uplift(before, after):
    """Positive pass@1 gain and positive CER reduction both mean improvement."""
    cer_before, cer_after = before["metrics"]["text_cer"], after["metrics"]["text_cer"]
    return dict(
        pass_at_1_gain_pp=100 * (after["pass_at_k"]["1"] - before["pass_at_k"]["1"]),
        text_cer_reduction_pp=100 * (cer_before - cer_after),
        text_cer_relative_reduction=(cer_before - cer_after) / cer_before if cer_before else None,
    )


def render_report(result):
    lines = ["# Parse dental evaluation", "",
             "| Checkpoint | Sampled pass@1 ↑ | Mean text CER ↓ |",
             "| --- | ---: | ---: |"]
    for name, summary in result["models"].items():
        label = "Base" if name == "base" else name.upper()
        lines.append(f"| {label} | {100 * summary['pass_at_k']['1']:.2f}% | "
                     f"{100 * summary['metrics']['text_cer']:.4f}% |")
    lines += ["", "| Comparison | Pass@1 gain (pp) ↑ | Text CER reduction (pp) ↑ |",
              "| --- | ---: | ---: |"]
    for name, change in result["uplift"].items():
        first, second = name.split("_to_")
        lines.append(f"| {first.upper()} → {second.upper()} | {change['pass_at_1_gain_pp']:+.2f} | "
                     f"{change['text_cer_reduction_pp']:+.4f} |")
    lines += ["", "All checkpoints use the same held-out pages and evaluation protocol.",
              "Scores and rewards are specific to the dental reference task.", ""]
    return "\n".join(lines)


def evaluate_comparison(lineage, rows, output, samples=8):
    """Use a fresh generation worker per checkpoint, with no training or mining."""
    summaries = {}
    expected_ids = [row["id"] for row in rows]
    for name in ("base", "sft", "rlvr"):
        reference = lineage["models"][name]
        directory = output / name
        directory.mkdir()
        answers = evaluate_isolated(reference["path"], rows, directory / "predictions",
                                    samples=samples, processor_source=reference["processor_source"])
        if [row["id"] for row in answers] != expected_ids:
            raise ValueError(f"{name} evaluation must return the same ordered held-out pages")
        summary = summarize(answers)
        if summary["samples"] != samples:
            raise ValueError(f"{name} evaluation returned a different sample count")
        summaries[name] = summary
        write_json(directory / "evaluation.json", summary)
    # A source modified during evaluation must not produce a completed comparison.
    for reference in lineage["models"].values():
        verify_model(reference)
    result = dict(
        schema_version=1,
        protocol=evaluation_protocol(samples),
        lineage=lineage,
        models=summaries,
        uplift={f"{first}_to_{second}": uplift(summaries[first], summaries[second])
                for first, second in (("base", "sft"), ("sft", "rlvr"), ("base", "rlvr"))},
    )
    write_json(output / "evaluation.json", result)
    (output / "comparison.md").write_text(render_report(result))
    return result
