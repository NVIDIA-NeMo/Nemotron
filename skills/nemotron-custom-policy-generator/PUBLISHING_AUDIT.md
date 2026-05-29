# Publishing Audit — nemotron-policy-generator

Audited against [NVIDIA Agent Skills — Publishing Onboarding Guide](https://docs.google.com/document/d/1SNFRQCv0_p3DC2a_IWIf0cB3c49tSE1d/edit) on 2026-05-27. Target: external publication on [github.com/nvidia/skills](https://github.com/nvidia/skills).

## Executive verdict

The skill is **structurally close to publish-ready**. Naming, length, frontmatter quality, and the "when NOT to trigger" boundary all already comply. The gaps are mechanical and now scaffolded in this folder: a license header inside `SKILL.md`, an `evals/` directory, a `BENCHMARK.md`, and a `SKILLCARD.yaml` template. The remaining work is process — OSRB sign-off, moving the skill into a real NVIDIA-owned GitHub repo at `skills/<name>/` at the repo root, and running `nv-base validate --external` plus a full `nv-aces` eval pass before the `components.d/<slug>.yml` PR.

## What already complies

| Guide requirement | Status | Evidence |
|---|---|---|
| Skill name is lowercase, hyphenated, CLI-safe, ≤64 chars, product-scoped | Pass | `nemotron-policy-generator` is product-prefixed and under length |
| Directory name matches `name:` field in frontmatter | Pass | Both equal `nemotron-policy-generator` |
| `SKILL.md` under 500 lines | Pass | ~262 lines after license header insertion |
| `description` frontmatter is specific, not categorical | Pass | Names exact models (NCS-Reasoning-4B, Nemotron-3), exact deployment patterns, exact output formats |
| `description` includes trigger keywords a user would actually say | Pass | "BYO-policy", "custom safety taxonomy", "eval rubric", "labeling rubric", "guardrail config", "moderation policy" |
| `description` states when NOT to trigger (implicitly via scope) | Pass | "Do not activate this skill when" block names the three failure modes (evaluate / test / legal-advice) |
| No `alwaysApply` or `globs` in frontmatter | Pass | Neither key present |
| References live under `references/` | Pass | `aegis_taxonomy.md`, `policy_patterns.md` |
| Assets live under `assets/` | Pass | Templates + HTML GUI present |
| License chosen from allowed set (Apache 2.0 / CC-BY 4.0 / dual) | Pass | Frontmatter declares `Apache-2.0 AND CC-BY-4.0` — appropriate dual license given the skill mixes prose (CC-BY) and the HTML GUI / templates (Apache) |
| Audience is "use the product" (not contributor tooling) | Pass | The skill helps customers use Nemotron content-safety models, not internal CI/code-style |

## What was missing (now fixed in this folder)

### 1. License header below frontmatter and H1 — FIXED

The publishing guide says explicitly: "Copyright / license headers go below the YAML frontmatter and the initial H1 heading, not above. Headers above the frontmatter interfere with agent parsing." The previous `SKILL.md` had no copyright block. An HTML-comment SPDX header has been inserted directly after the `# Nemotron Custom Policy Generator` H1, before `## When to Use This Skill`.

### 2. `evals/evals.json` — FIXED (scaffolded)

The guide requires every published skill to ship an `evals/` directory next to `SKILL.md` containing `evals.json`. The dataset shipped here has 8 cases (4 positive, 4 negative) covering:

- Rough-keywords-only input → clean V2 map
- Multimodal + multilingual BYO with custom categories → Nemotron-3 emit block
- Extending an existing policy → version bump + diff summary
- Labeling-rubric primary use case → binary severity branch
- Distractors for the three "Do not activate" failure modes plus one wholly unrelated LLM task

Each positive case has `expected_skill`, `expected_script`, `ground_truth`, and an ordered `expected_behavior` list that NV-ACES can score per the guide's evaluator spec. Negative cases set `expected_skill: null`.

### 3. `evals/EVAL.md` — FIXED (scaffolded)

Developer-facing how-to for running the eval set on both Claude Code and Codex harnesses, with the acceptance bar that gates publication (trigger precision = 1.0 on negative cases is the hard rule).

### 4. `BENCHMARK.md` — FIXED (scaffolded)

Per the guide: "Every published skill ships a BENCHMARK.md summarizing how it was evaluated… with BOTH with-skill and without-skill tables so the uplift is visible." The scaffolded report has the harness configuration block, both with-skill and without-skill result tables, a five-dimension NV-BASE rollup, and a known-limitations block. Cells are `_TBD_` and populate either via `nv-aces run … --output BENCHMARK.md` or manually after evaluation.

### 5. `SKILLCARD.yaml` — FIXED (scaffolded)

The NVCARPS pipeline auto-generates the bulk of this from the skill repo + pipeline outputs (identity, provenance, scan results, evaluation metrics). The template here pre-fills the team-owned fields (identity, ownership, license, compatibility, intended use, behavioral boundaries) and marks auto-populated fields with `_auto_` placeholders so the pipeline can fill them in. Per Michael Boone's NVCARPS pipeline note in the guide, this lands May 18 IST 2026 in NVCARPS CI.

## What still needs to happen (process work, not file changes)

These are upstream of the publication pipeline and the audit can't fix them — they require human / NVIDIA-process action:

### 1. Move to a real GitHub repo at `skills/<name>/`

Today the skill lives under `.claude/skills/nemotron-policy-generator/` in this Cowork session. The guide is explicit: the canonical release-facing path is `<repo>/skills/<skill-name>/SKILL.md` at the repo root. `nv-base validate --external` (the catalog publish gate) enforces this. Path migration is required before submitting `components.d/<slug>.yml`.

If you want to keep `.claude/skills/` for local Cowork use too, the guide permits adding it as a symlink alias to the canonical `skills/` path — but the canonical path is the source of truth.

### 2. OSRB IP-review clearance

Per Step 3 of the guide, the IP Review Process must complete with all six answers affirmative before the skill is eligible for publication. Since this introduces a new external skill repo, OSRB will need to clear: (a) the new repo, (b) the dual Apache-2.0 + CC-BY-4.0 license selection, and (c) any third-party dependencies (the HTML GUI in `assets/nemotron_policy_generator.html` — confirm it has no bundled third-party JS without OSRB clearance).

OSRB is the longest lead-time item in publication. Start now if you haven't.

### 3. NV-BASE local scan

Run `nv-base validate --external` locally before pushing. The publishing guide flags this twice: "Skills become public the moment you push to your product repo — before the pre-catalog scan fires at sync time. Run NV-BASE locally first to catch issues before they land in your public git history."

### 4. NVCARPS CI run + signing

Comment `/nvskills-ci` on the eventual PR to trigger the NVCARPS pipeline. The pipeline produces `skill.oms.sig`, fills in the `_auto_` fields of `SKILLCARD.yaml`, and produces the BENCHMARK.md from your evals dataset. Verify the signature commit `Attach NVSkills validation signatures` lands before merge.

### 5. `components.d/<slug>.yml` PR to NVIDIA/skills

One file, ~6 lines:

```yaml
- name: Nemotron Policy Generator
  repo: NVIDIA/<your-repo>
  description: >-
    Generates BYO content-safety policies for NVIDIA Nemotron content-safety
    guardrails (Reasoning-4B today; Nemotron-3 at Computex 2026).
  skills:
    - path: /skills/
      catalog_dir: nemotron-policy-generator
```

Use `git commit -s` (DCO sign-off is enforced).

## Quibbles and small wins

These aren't blockers, just polish:

### Frontmatter is richer than the spec minimum

The current frontmatter includes `title`, `license`, `compatibility`, and a full `metadata` block (`author`, `team`, `tags`, `languages`, `frameworks`, `domain`). The publishing guide's minimum required frontmatter is just `name` + `description`. The extras don't violate the agentskills.io spec, but if `nv-base validate --external` flags any of them as unrecognized, drop them into the `metadata:` block (which the spec treats as opaque) rather than top-level keys. The `metadata:` block here already nests author/team/tags/languages/frameworks/domain correctly — but `title`, `license`, and `compatibility` are at the top level and would be safer nested inside `metadata:`.

### "Aegis" naming carry-over

The reference file is `references/aegis_taxonomy.md` but the workflow has migrated to Nemotron Content Safety V2. The file's *content* is up-to-date, but the *filename* still references the older Aegis name. Consider renaming to `references/v2_taxonomy.md` and updating the two references in `SKILL.md` — pure hygiene, not a publishing blocker, but it removes a stale signal.

### `references/` and `assets/` token budget

Both folders are under the SKILL.md 500-line gate that the guide tracks at the SKILL.md level, but NVCARPS also scans for token budget overruns across the skill directory. The HTML GUI in `assets/` is 44 KB — fine — but worth confirming it doesn't contain inline bundled libraries (jQuery, Tailwind CDN copy) that might trip the dependency audit. Spot-check before pushing.

## File index (what's in this audit deliverable)

- `SKILL.md` — corrected, with SPDX license header inserted below the H1 (line 50 area)
- `PUBLISHING_AUDIT.md` — this document
- `BENCHMARK.md` — placeholder report with both result tables and the five-dimension rollup
- `SKILLCARD.yaml` — team-owned fields filled; auto fields marked `_auto_`
- `evals/evals.json` — 8-case dataset (4 positive, 4 negative)
- `evals/EVAL.md` — how to run + acceptance bar
- `assets/`, `references/` — unchanged from your working copy

## Sources

- [Skills_Publishing_Onboarding_Guide.docx](https://docs.google.com/document/d/1SNFRQCv0_p3DC2a_IWIf0cB3c49tSE1d/edit)
- [github.com/nvidia/skills](https://github.com/nvidia/skills) (catalog)
- [agentskills.io/specification](https://agentskills.io/specification) (spec)
