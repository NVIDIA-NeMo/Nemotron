# Document evaluation and reward building blocks

This package provides optional scorers for transcription, layout, tables,
charts, and completion, plus a weighted reward helper. They are building blocks
for domain-specific evaluation and rewards; the dental reference task uses its
own exact-page reward. Choose matching rules, thresholds, and composition for
your task, then connect the scorers to its evaluation and reward environment.

The [document scoring guide](../../../../../docs/nemotron/parse/rewards.md) in
`docs/` covers APIs, dependencies, examples, and limitations. The executable
[weighted reward example](../examples/weighted_reward.py) demonstrates the
component inputs and composition. See [third-party attribution](../../../../../docs/nemotron/parse/attribution.md)
for scorer and font licenses.
