---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "mopd-teachers"
paper_sections: ["3.3.2"]
title: "Nemotron 3 Ultra MOPD: Specialized Teacher Models"
summary: |
  Details the more-than-ten domain-specialized teacher models distilled into the Nemotron 3 Ultra
  student via MOPD: Software Engineering, Office & Workplace Task (GDPval), Search, Terminal-use,
  Conversational Tool-use, Model Usability, Agentic Safety, Chat (Ultra-based GenRM),
  Instruction-following & Factuality, STEM / General Reasoning, and Competitive Coding. Captures each
  teacher's initialization, training recipe (SFT / PivotRL / RLVR / RLHF), data sources, and the
  General Reasoning teacher's benchmark table and SFT blend token budget.
key_facts:
  - "SWE teacher: three-stage pipeline = SFT on Ultra base with agentic data -> PivotRL on single-step agentic environments -> end-to-end SWE-RL (multi-turn repo interaction, hidden-test binary reward in GRPO)."
  - "SWE teacher mitigations: mask loss on unfinished trajectories (hit max turns or agent/eval timeouts); assign negative advantage to malformed reasoning/tool-call tokens; close gold-patch leak channels (rewrite repo to fresh clone at base commit with future commits physically deleted; runtime command filter blocking remote git ops and GitHub web/raw-content/Pages downloads)."
  - "Office & Workplace Task teacher: targets GDPval; initialized from a Nemotron 3 Ultra checkpoint that completed general SFT; trained on AfterQuery (AQ) tasks; light SFT (workflow priors) then pivot RL in the MOPD stage distilling the SFT teacher into the student using pivots from strong-model AQ rollouts."
  - "Search teacher: initialized from an Ultra checkpoint; SFT on trajectories augmented with context-management behavior (discard-all resets and summary-based compression), focusing primarily on discard-all; lets the model search for longer effective contexts than its official context length."
  - "Terminal-use teacher: starts from expert trajectories for long-timeout tasks (up to one hour); uses PivotRL with re-profiling steps whenever accuracy saturates."
  - "Conversational tool-use teacher: same data/recipe as Nemotron 3 Super, trained via PivotRL; data expanded to sequential and dependent multi-step actions to discourage premature termination."
  - "Model Usability teacher: extends Super's structured schema formatting to add document extraction, citation formatting, and freeform text formatting; structured schema covers five types (JSON, YAML, XML, TOML, CSV) across six task categories (direct extraction, translation, multistep-related, multistep-unrelated, schema-only, error correction); seed data via NeMo Data Designer with gpt-oss-120b, environments via Nemo-Gym."
  - "Agentic Safety teacher: targets indirect prompt-injection in tool-response data; four attack categories (unauthorized actions, data modification, denial of service, data exfiltration); automated red-teaming loop with Nemotron 3 Super as attacker and Nemotron 3 Nano as defender; deterministic verifier marks injection resisted only if the agent does not invoke the attacker's target tool with target arguments."
  - "Chat teacher: an Ultra-based Generative Reward Model (GenRM) scaled up in capacity and data; trained on top of the Ultra SFT model; evaluates a pair of candidate responses given conversational context, conditioned on user-defined principles when provided; predicts individual scores + ranking (triplet per principle, then overall); only overall scores used as RLHF reward; multiple RLHF iterations with targeted data per cycle."
  - "Instruction-following & Factuality teacher: domain-focused RLVR on top of the RL checkpoint from Section 3.2; combines instruction-following, abstention-focused, and RLHF environments; abstention reward dynamically calibrated; RLHF data added to avoid behavioral collapse/overfitting."
  - "STEM / General Reasoning teacher: starts from the student, adds SFT and RL on selected datasets; matches or outperforms DeepSeek V4 Pro (High) on GPQA, MMLU-Pro, LiveCodeBench v6, IMOAnswerBench, Apex Shortlist (Table 3)."
  - "General Reasoning teacher SFT blend = 40B generated tokens, token-level target proportions: Science reasoning 23.5B (58.75%); Math CoT/TIR + proof 9.45B (23.63%); Competitive coding 4.05B (10.13%); General-domain 3.0B (7.50%)."
  - "STEM teacher science data: traces from DeepSeek-V4-Pro (4 per problem, 16 for harder, 8 for long-form resample of median-correct-length >16k); graded by gpt-oss-120b; held-out RL eval set of 3,000 problems (pass rates 0.25-0.80, median correct-solution lengths <64k tokens)."
  - "STEM teacher coding data: ~14K problems from international competitions over past 10 years + 4K hard OpenCodeReasoning problems; 10 candidate solutions per problem with DeepSeek-V4, filtered on compilation success."
  - "STEM teacher math data: 95,164 unique math problems retained; DeepSeek-V4-Pro (high-inference) CoT+TIR, judged by gpt-oss-120b; validated pool = 285,516 CoT + 259,915 TIR = 545,431 total examples."
  - "STEM teacher proof data: 5,751 unique proof problems from AoPS section; DeepSeek-V4-Pro (max-inference); DeepSeekMath-V2 methodology (proof/verification/meta-verification); validated pool = 82,737 samples."
  - "STEM teacher RL stage: focuses on non-STEM domains (humanities, sociology); same setup as Section 3.2 except 128 prompts per batch and global batch size 2048; generalizes to gains across all domains."
  - "Competitive Coding teacher: Competitive Coding RL on Nemotron-Cascade coding data; filters out prompts Nemotron-Ultra SFT solves in all 8 of 8 rollouts, yielding a compact final set of 3.5K samples."
related_steps: []
currency: "frozen"
---

# Scope

- What specialized teacher models feed MOPD, and how is each trained?
- What data sources/sizes and initializations does each teacher use?
- How strong is the General Reasoning (STEM) teacher and what is its SFT blend?

# Teacher roster (3.3.2)

Claim: more than ten specialized teachers, each via its own domain-specific pipeline.

| Teacher | Init | Recipe | Notable data / settings |
| --- | --- | --- | --- |
| Software Engineering | Ultra base | SFT (agentic) -> PivotRL (single-step) -> end-to-end SWE-RL (GRPO, hidden-test binary reward) | Loss mask on unfinished trajectories; negative advantage on malformed reasoning/tool calls; gold-patch leak channels closed |
| Office & Workplace Task | Ultra checkpoint after general SFT | Light SFT (workflow priors) -> pivot RL in MOPD stage | AfterQuery (AQ) tasks; targets GDPval; pivots from strong-model AQ rollouts |
| Search | Ultra checkpoint | SFT on context-management-augmented trajectories | Discard-all resets (primary) + summary-based compression; longer effective context than official length |
| Terminal-use | Expert long-timeout trajectories (up to 1 hour) | PivotRL with re-profiling on saturation | — |
| Conversational Tool-use | (Super data/recipe) | PivotRL | Expanded to sequential/dependent multi-step actions |
| Model Usability | — | (formatting datasets) | 5 schema types (JSON/YAML/XML/TOML/CSV), 6 task categories; NeMo Data Designer + gpt-oss-120b; Nemo-Gym |
| Agentic Safety | — | Automated red-teaming loop | Attacker = Nemotron 3 Super, defender = Nemotron 3 Nano; 4 attack categories; deterministic verifier |
| Chat (GenRM) | Ultra SFT model | RLVR (NVIDIA 2026 method) + multiple RLHF iterations | Pairwise + principle-conditioned scoring; only overall score as reward |
| Instruction-following & Factuality | RL checkpoint from 3.2 | Domain-focused RLVR | Instruction-following + abstention + RLHF environments; dynamic abstention reward |
| STEM / General Reasoning | Student | SFT + RL on selected datasets | Matches/outperforms DeepSeek V4 Pro (High); 40B-token SFT blend (below) |
| Competitive Coding | (Nemotron-Cascade coding data) | Competitive Coding RL | Filter prompts solved 8/8 by Ultra SFT -> 3.5K samples |

# General Reasoning teacher benchmarks (Table 3)

Claim: highest scores bold in source; differences within one point considered noise.

| Benchmark | DeepSeek V4 Pro (High) | Student | General Reasoning Teacher |
| --- | --- | --- | --- |
| HLE (no tools) | 34.5 | 25.6 | 32.1 |
| GPQA | 89.1 | 85.0 | 88.5 |
| MMLU-Pro | 87.1 | 85.7 | 87.7 |
| LiveCodeBench v6 | 89.8 | 87.4 | 90.0 |
| IMOAnswerBench | 88.0 | 84.5 | 92.5 |
| Apex Shortlist | 85.5 | 68.9 | 85.4 |

# General Reasoning teacher SFT blend (40B tokens)

Token-level target proportions (not example counts):

| Component | Tokens | Share |
| --- | --- | --- |
| Science reasoning (STEM + non-STEM) | 23.5B | 58.75% |
| Math CoT/TIR + math proof | 9.45B | 23.63% |
| Competitive coding | 4.05B | 10.13% |
| General-domain | 3.0B | 7.50% |
| Total | 40B | 100% |

Data sub-pools for this teacher:

| Sub-pool | Figures |
| --- | --- |
| Science | DeepSeek-V4-Pro traces (4/problem, 16 hard, 8 long-form resample of median >16k); gpt-oss-120b judge; held-out RL eval = 3,000 problems (pass 0.25-0.80, median <64k tokens) |
| Coding | ~14K competition problems (10 yrs) + 4K hard OpenCodeReasoning; 10 candidates/problem (DeepSeek-V4); compilation-filtered |
| Math CoT/TIR | 95,164 unique problems; DeepSeek-V4-Pro high-inference; validated 285,516 CoT + 259,915 TIR = 545,431 total |
| Math proof | 5,751 unique proof problems (AoPS); DeepSeek-V4-Pro max-inference; DeepSeekMath-V2 method; validated 82,737 samples |

RL stage for this teacher: focuses on non-STEM (humanities, sociology); same as Section 3.2 except 128 prompts/batch and global batch size 2048; generalizes to gains across all domains.

# Caveats

- Some teachers (Model Usability, Agentic Safety, Conversational Tool-use, Competitive Coding) do not report an explicit initialization checkpoint here; do not assume one.
- Table 3 differences within one point are explicitly considered noise by the authors.
- The 40B-token blend and its sub-pool counts belong to the General Reasoning / STEM teacher only, not the main Ultra SFT stage (which trains on 204,800 samples — see sft.md).
- "more than ten" teachers; the table lists 11 named teachers but do not over-claim that this is the complete set.
- AQ = AfterQuery; "strong model" used to generate AQ rollouts is not named in the source.
