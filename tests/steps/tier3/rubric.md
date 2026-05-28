# Tier 3 Judge Rubric

Rate the generated ML pipeline project on five criteria. Score each criterion from **1** to **5**.

## 1. Correctness
- **1** — Major errors in step chaining, artifact compatibility, or hardware/parallelism choices.
- **2** — Multiple correctness issues; the project would need significant fixes.
- **3** — Mostly correct, but has at least one meaningful pipeline or configuration issue.
- **4** — Correct overall with only minor issues or omissions.
- **5** — Fully correct: steps chain cleanly, artifact types are compatible, and parallelism fits the stated hardware.

## 2. Completeness
- **1** — Core stages or required configuration are missing.
- **2** — Several important stages, files, or config values are incomplete.
- **3** — Covers the main path, but some necessary details are missing or placeholder-like.
- **4** — Nearly complete; only small gaps remain.
- **5** — Complete end-to-end project for the stated request with no obvious missing stages or placeholders.

## 3. Readability
- **1** — Hard to understand without outside context.
- **2** — Some structure exists, but intent and flow are unclear.
- **3** — Understandable with effort; naming or documentation could be clearer.
- **4** — Clear and well organized for an engineer reader.
- **5** — Easy to understand and modify without external context.

## 4. Runability
- **1** — Would not run even with the right environment.
- **2** — Major missing dependencies, configs, or execution wiring.
- **3** — Plausibly runnable, but important details are shaky.
- **4** — Runnable with minor fixes or assumptions.
- **5** — Project appears ready to run with the right environment and inputs.

## 5. Forkability
- **1** — Too coupled or opaque to customize safely.
- **2** — Customization would require deep reverse engineering.
- **3** — Some customization points exist, but they are not obvious.
- **4** — Reasonably self-contained and customizable.
- **5** — Clearly structured, self-contained, and easy to adapt or swap components.

## Judge Instructions
- Evaluate the project only against the provided user request, plan, and project files.
- Prefer concrete evidence from the files over assumptions.
- Penalize placeholder text, TODOs, missing configs, incompatible stage wiring, and hardware mismatches.
- Return **JSON only**.

## Required JSON Response Shape

```json
{
  "scores": {
    "Correctness": 1,
    "Completeness": 1,
    "Readability": 1,
    "Runability": 1,
    "Forkability": 1
  },
  "reasoning": {
    "Correctness": "brief explanation",
    "Completeness": "brief explanation",
    "Readability": "brief explanation",
    "Runability": "brief explanation",
    "Forkability": "brief explanation"
  }
}
```

## Template Prompt

Use this template when sending content to the judge model:

```text
You are grading a generated ML pipeline project.

Apply the rubric below and return JSON only using the required schema.

=== RUBRIC ===
{{RUBRIC_TEXT}}

=== USER REQUEST ===
{{USER_REQUEST}}

=== GENERATED PLAN ===
{{GENERATED_PLAN}}

=== GENERATED PROJECT FILES ===
{{GENERATED_PROJECT_FILES}}
```
