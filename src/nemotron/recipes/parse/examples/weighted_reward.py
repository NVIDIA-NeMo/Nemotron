"""Run from the recipe directory: .venv/bin/python examples/weighted_reward.py"""

from pprint import pprint

from nemotron.recipes.parse.document_rewards import (
    box_quality, chart_quality, completion_quality, table_quality,
    text_quality, weighted_reward,
)


def main():
    # Toy annotations; use parsed predictions and reference fields for your task.
    table = "<table><tr><th>Item</th><th>Count</th></tr><tr><td>Box</td><td>2</td></tr></table>"
    chart = '''<chart type="bar">
title: Revenue
axis[x]: Region
axis[y]: Revenue (USD)
| Region | Revenue |
| --- | --- |
| West | 40 |
| East | 60 |
</chart>'''
    components = {
        "text": text_quality("Total: 42", "Total: 40"),
        "boxes": box_quality([[1, 0, 10, 10]], [[0, 0, 10, 10]]),
        "tables": table_quality(table, table),
        "charts": chart_quality(chart.replace("West | 40", "West | 30"), chart),
        # Real rollouts must supply their actual token IDs and tokenizer EOS ID.
        "completion": completion_quality("Total: 42", token_ids=[101, 102, 2],
                                          eos_token_id=2, finish_reason="stop"),
    }
    weights = {"text": 0.40, "boxes": 0.20, "tables": 0.20,
               "charts": 0.15, "completion": 0.05}
    combined = weighted_reward(components, weights)
    pprint(combined)

    # For a page without table/chart annotations, omit those terms explicitly.
    components.update(tables=None, charts=None)
    print("\nApplicable terms only:")
    pprint(weighted_reward(components, weights))

    # Optional: require natural, non-looping completion as a hard gate.
    gated_reward = combined["score"] * components["completion"]["score"]
    print(f"\nReward with completion gate: {gated_reward:.4f}")


if __name__ == "__main__":
    main()
