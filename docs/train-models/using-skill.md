# Tips for Model Training with Agents

## Have a Goal in Mind

You might want to add instruction following, specialize a model on your domain or product voice, align behavior with preferences after supervised fine tuning, or land on a smaller or faster model through distillation, pruning, or quantization.
Those are concrete outcomes to name before you reach for an agent.

This page helps you get that plan and work efficiently with your agent.
You steer an agent through the *nemotron-customize* skill (`/nemotron-customize`).
The agent can use the skill to produce an explicit series of steps and commands before anything executes.
The full skill contract lives in `skills/nemotron-customize/SKILL.md` at the repository root.

Use the skill when you have a general direction, but you are unsure how to proceed inside this repository: which steps apply, how artifacts connect, or what YAML shape the runners expect.

## What to Tell the Agent

The agent maps your goal to a plan: an ordered set of steps, configs, and commands.
The more grounded the context you give it, the less it has to guess—and the less rework you do later.
The lists below use progressive disclosure: start with the short block, then add detail as the session develops.

### Have Ready Early

These items keep the first prompts grounded and cut down rework.
The nested lines are example sentences you can paste or adapt; they are not the only valid wording.

- Desired outcome in plain language, so the agent can narrow the stage chain and success checks.

  :::{div} sd-font-italic sd-font-weight-lighter
  - "We want the model to follow our runbook-style instructions on support tickets without drifting into free-form guesses when the ticket is incomplete."
  - "After training, reviewers should see fewer policy violations on a fixed red-team prompt set we already use in evaluation."
  :::

- Rough stage story when you already know it (for example SFT then alignment), or an explicit note that you want a recommendation, so the agent does not guess your pipeline shape.

  :::{div} sd-font-italic sd-font-weight-lighter
  - "We already ran SFT in another team’s repo; from here we only need alignment on preference data against that checkpoint."
  - "We do not know the right order; recommend a minimal path from the catalog given our data and hardware."
  :::

- Where inputs live, paths or Hugging Face Hub datasets, and what you call the starting weights, so manifests and `consumes` edges can line up without invented paths.

  :::{div} sd-font-italic sd-font-weight-lighter
  - "Training chat lives at `/data/support/train.jsonl` on the shared file system. The evaluation JSONL is next to it under `eval.jsonl`."
  - "Base weights are `organization/model-name` on the Hugging Face Hub. We have a local cache under `/scratch/hf-cache`."
  :::

- Hardware you schedule against at least at nodes, GPUs per node, and GPU family, so scale and parallelism assumptions stay realistic.

  :::{div} sd-font-italic sd-font-weight-lighter
  - "We can schedule up to four nodes with eight H100 GPUs per node in Lepton for training; prefer fewer nodes if the step supports it cleanly."
  - "Slurm jobs must write checkpoints under `/project/checkpoints/$USER/run-id` so retention policies pick them up."
  :::

You do not need step identifiers or perfect file formats before you start if you are willing to discover those with the agent in the next block.

### Fine to Clarify During the Session

The skill uses an _orient phase_ so that you and the agent can tighten these together.

- How your data layout lines up with what a chosen step expects, using [Data and Checkpoint Formats](how-to/data-and-checkpoint-formats.md) and the step’s `step.toml` as you go.
- Backend choice inside a family, using [Choose an SFT Backend](how-to/choose-sft-backend.md), [Choose a PEFT Backend](how-to/choose-peft-backend.md), or [Choose an RL Alignment Step](how-to/choose-rl-step.md) when you want a human-readable decision before configs land.
- Whether the first execution should validate end-to-end execution with a short sample run or target production-scale training, so defaults match intent.
- Tokenizer, template, and sequence-length consistency across prep and train when the pipeline touches more than one stage.

### Where to Look When You Are Unsure

Use the [How-To Guides](how-to/index.md) for task-level procedures.
Use [Reference](reference/index.md) for naming and configuration conventions.
Use [Explanation](explanation/index.md) for how artifacts and libraries fit together.
You can also ask the agent to walk a specific step with `nemotron steps show` once you know its name.

## Your Role in the Loop

The skill follows _orient_, _plan_, _act_, and then _verify_.
You answer missing constraints, approve the plan before files change, and check that stage outputs line up with the next stage’s inputs.
`SKILL.md` spells out each phase, file naming, and verification expectations for the whole catalog, not only training.
