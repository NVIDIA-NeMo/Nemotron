# New Benchmark Family Checklist

Use this checklist before changing BYOB for a new benchmark family such as GSM8K, coding tasks,
free-form QA, or tool-use benchmarks. The purpose is to keep BYOB agent-modifiable: the agent should
add a family-specific package instead of pushing non-MCQ behavior into the MCQ runtime.

## First Questions

Answer these before editing code. Ask the user only for items that cannot be inferred from the request,
the config, or existing files.

| Question | Why it matters |
| --- | --- |
| What is the family name? | This becomes `runtime/benchmark_families/<family>/` and the registry key. |
| What task format is expected? | MCQ, numeric answer, free-form answer, code, tool trace, or multi-turn dialogue require different schemas. |
| What final output schema must downstream evaluators read? | The final parquet contract controls export, translation, validation, and HITL review. |
| What source examples should guide style? | GSM8K may use grade-school math examples; coding may use problem/solution examples. |
| What user-provided corpus or topic source drives generation? | Some families generate from documents, while others generate from seed problems or structured specs. |
| What answer form must be validated? | MCQ validates answer letters; GSM8K needs numeric extraction; coding may need tests. |
| Which quality gates apply? | MCQ-only gates like distractor validity do not apply to GSM8K. |
| What stages are optional for smoke tests? | Embedding-heavy stages should be configurable so small runs can complete in constrained environments. |
| What config template should users start from? | Each family needs a minimal config with model aliases, input paths, and output paths. |
| What golden case proves the family works? | Add at least one small deterministic test or static asset check for the new family. |

## Implementation Rules

- Create `runtime/benchmark_families/<family>/`.
- Keep family prompts, response models, dataset parsing, postprocessing, and export code inside that package.
- Register the family in `runtime/benchmark_families/registry.py` with a `BenchmarkFamilySpec`.
- Keep `scripts/runtime.py` as a dispatcher. Do not add family-specific branches there.
- Add a family config template under `config/` only if the default MCQ template is not appropriate.
- Add or update references and patterns so another agent can discover when to use the new family.
- Preserve existing MCQ behavior and final MCQ schema.

## Stage Design

Define the family stages explicitly before coding.

| Stage decision | MCQ example | GSM8K-style example |
| --- | --- | --- |
| Seed preparation | Few-shot MCQ examples plus target documents | Math examples plus topic constraints or source documents |
| Generation | Question, choices, answer letter | Word problem, solution reasoning, final numeric answer |
| Judgement | Clarity and category | Solvability, grade level, arithmetic consistency |
| Deduplication | Semantic duplicate question detection | Semantic duplicate problem detection |
| Answer validation | Distractor validity and answer letter check | Recompute or parse final numeric answer |
| Difficulty filtering | Easiness/hallucination model answers | Independent solve pass or answer agreement |
| Final export | MMLU-Pro-style MCQ parquet | Family-specific parquet documented in references |

## GSM8K-Specific Notes

A GSM8K-style family should not inherit MCQ assumptions:

- Do not require `options`, `answer_index`, or answer letters unless the user explicitly requests MCQ math.
- Define a final schema such as `problem_id`, `question`, `answer`, `solution`, `category`, `src`, and optional
  `difficulty` or `reasoning_type`.
- Use a response model that separates the reasoning or solution from the final answer.
- Add a postprocessor that normalizes and validates the final answer, for example numeric extraction from a
  `#### 42`-style final line.
- Replace distractor stages with math-specific validation stages.
- Add a small golden example where the generated answer can be mechanically checked.

## Review Checklist

Before finishing a new family, verify:

- `python -m nemotron.steps.byob.scripts.run --list-families` lists the new family.
- The new family can run a tiny prepare/generate flow or has a static test when runtime dependencies are unavailable.
- The final schema is documented in `references/`.
- The new family has no imports that make skill discovery require heavy runtime dependencies.
- MCQ tests still pass.
- The family can be modified independently by editing files under its own package.
