---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "mopd-warmup-results"
paper_sections: ["3.3.3", "3.3.4", "3.3.5"]
title: "Nemotron 3 Ultra MOPD: Warmup, Results & Discussion, Limitations"
summary: |
  Covers the MOPD warmup stage (a light SFT on teacher-distribution data to fix teacher/student
  distribution mismatch), the main MOPD results across domains (Tables 4 and 5) including recovery
  rates toward specialized teachers, the discussion of when MOPD works best vs falls short (HLE), and
  the limitations / open problems (logit matching, foundations for MOPD, long-horizon tasks).
key_facts:
  - "Key MOPD finding: teachers trained with substantially different pipelines cannot be effectively combined via a straightforward MOPD merge; distribution mismatch makes student trajectories out-of-distribution for the teacher, degrading supervision."
  - "MOPD warmup: a brief, very light SFT of the student on data drawn from the teacher's training distribution, to align the student's reasoning trajectories/output distribution with the teacher's; intentionally small scale, inducing minimal regression on unrelated domains (any residual loss recovered during MOPD)."
  - "Warmup ablation (Table 4): GDPVal Student 28.9 / Warmup 46.7 / No Warmup 35.3 / Teacher 49.5; BrowseComp 31.0 / 44.4 / 33.0 / 51.0; HLE (no tools) 25.6 / 26.7 / 26.3 / 32.1."
  - "Warmup substantially improves agentic-domain performance after MOPD but provides only negligible gains on general reasoning (HLE)."
  - "Recovery rate is defined as (MOPD2 - RLVR)/(Teacher - RLVR): the fraction of the teacher-student gap closed by MOPD."
  - "MOPD results (Table 5) columns: SFT, RLVR, MOPD1, MOPD2, Teacher, Recovery(%). Terminal Bench 2.0: 34.5/44.5/50.8/54.0/50.0/172.7%. GDPVal: 23.2/28.9/46.7/46.7/49.5/86.4%. SWE-Bench Verified: 63.5/65.8/70.1/71.7/72.5/88.1%. TauBench Telecom: 55.7/82.7/91.2/92.9/94.0/90.3%. BrowseComp: 14.3/31.0/41.0/44.4/51.0/67.0%. LiveCodeBench (v6): 85.5/87.4/90.0/89.0/92.4/32.0%. IMOAnswerBench (no tools): 85.1/84.5/88.1/88.6/92.5/51.3%. HLE (no tools): 19.7/25.6/25.9/26.7/32.1/16.9%. OmniScience Non-Hallucination: 4.8/46.3/77.9/78.7/87.0/79.6%. IFBench (prompt loose): 62.3/78.4/80.0/81.7/83.0/71.7%. Multi-Challenge: 53.3/60.3/62.8/63.8/63.3/116.7%."
  - "RLVR = initial student checkpoint; MOPD1/MOPD2 = checkpoints after the first/second MOPD iterations."
  - "On several benchmarks MOPD surpasses the corresponding specialized teacher (positive cross-domain generalization from merging supervision); e.g. MOPD2 outperforms the teacher on data-science tasks in Terminal Bench, suggesting transfer from office/productivity workflows."
  - "MOPD is most effective when the teacher's advantage is expressible as token-level preferences over trajectories the student already samples (tool-use decisions, environment interactions, abstention behavior, multi-step execution)."
  - "Gains are smallest on self-contained reasoning (especially HLE): the general reasoning teacher's advantage comes partly from extra off-policy SFT/RL on DeepSeek-V4-Pro data the student never saw, so missing-capability reasoning paths are rarely sampled and become out-of-distribution for the teacher, making token-level supervision less informative."
  - "Limitation - Logit matching: distribution-level (top-k / full-vocab) distillation did not improve MOPD and underperformed the sampled-token objective on some agentic benchmarks (e.g. Terminal Bench); full-distribution matching may over-constrain low-support prefixes and amplify noise from off-support states."
  - "Limitation - Foundations for MOPD: better teacher/student support overlap might come from a unified SFT stage before specialization, or from using specialized teachers to generate SFT data trained into the student before RL/MOPD; not systematically evaluated due to time/resource constraints."
  - "Limitation - Long-horizon tasks: mixing end-to-end agentic environments (many turns) with single-turn reasoning environments caused substantial training inefficiency from divergent rollout times; in practice single-turn rollouts (similar to PivotRL) were used for most agentic tasks."
related_steps: []
currency: "frozen"
---

# Scope

- What is MOPD warmup and why is it needed?
- How much does warmup help (Table 4)?
- What are the main MOPD results and recovery rates across domains (Table 5)?
- When does MOPD work best, and why does it underperform on HLE?
- What MOPD variants/limitations remain open?

# MOPD warmup (3.3.3)

Claim: teachers trained with substantially different pipelines cannot be combined by a straightforward MOPD merge — student trajectories become OOD for the teacher, degrading supervision. This arose because teachers and student were developed in parallel with their own specialized SFT pipelines.

Warmup = a brief, very light SFT of the student on data from the teacher's training distribution, aligning the student's reasoning trajectories/output distribution with the teacher's so student rollouts stay within teacher support. Intentionally small-scale -> minimal regression on unrelated domains; residual degradation recovered during MOPD.

## Warmup ablation (Table 4)

| Benchmark | Student | Warmup | No Warmup | Teacher |
| --- | --- | --- | --- | --- |
| GDPVal | 28.9 | 46.7 | 35.3 | 49.5 |
| BrowseComp | 31.0 | 44.4 | 33.0 | 51.0 |
| HLE (no tools) | 25.6 | 26.7 | 26.3 | 32.1 |

Warmup substantially helps agentic domains (GDPVal, BrowseComp) but gives negligible gains on general reasoning (HLE).

# MOPD results (3.3.4, Table 5)

RLVR = initial student checkpoint; MOPD1/MOPD2 = after first/second MOPD iterations. Recovery(%) = (MOPD2 - RLVR) / (Teacher - RLVR).

| Benchmark | SFT | RLVR | MOPD1 | MOPD2 | Teacher | Recovery(%) |
| --- | --- | --- | --- | --- | --- | --- |
| Terminal Bench 2.0 | 34.5 | 44.5 | 50.8 | 54.0 | 50.0 | 172.7% |
| GDPVal | 23.2 | 28.9 | 46.7 | 46.7 | 49.5 | 86.4% |
| SWE-Bench Verified | 63.5 | 65.8 | 70.1 | 71.7 | 72.5 | 88.1% |
| TauBench Telecom | 55.7 | 82.7 | 91.2 | 92.9 | 94.0 | 90.3% |
| BrowseComp | 14.3 | 31.0 | 41.0 | 44.4 | 51.0 | 67.0% |
| LiveCodeBench (v6) | 85.5 | 87.4 | 90.0 | 89.0 | 92.4 | 32.0% |
| IMOAnswerBench (no tools) | 85.1 | 84.5 | 88.1 | 88.6 | 92.5 | 51.3% |
| HLE (no tools) | 19.7 | 25.6 | 25.9 | 26.7 | 32.1 | 16.9% |
| OmniScience Non-Hallucination | 4.8 | 46.3 | 77.9 | 78.7 | 87.0 | 79.6% |
| IFBench (prompt loose) | 62.3 | 78.4 | 80.0 | 81.7 | 83.0 | 71.7% |
| Multi-Challenge | 53.3 | 60.3 | 62.8 | 63.8 | 63.3 | 116.7% |

Observations:
- MOPD improves over the RLVR student across the suite; strong recovery on agentic (Terminal Bench, GDPVal, SWE-Bench Verified, TauBench Telecom, BrowseComp) and instruction-following/factuality (OmniScience, IFBench, Multi-Challenge).
- On some benchmarks MOPD surpasses the teacher (e.g. Terminal Bench 2.0 MOPD2 54.0 > Teacher 50.0; Multi-Challenge 63.8 > 63.3), indicating positive cross-domain generalization (e.g. data-science transfer from office/productivity workflows).
- MOPD works best when the teacher's edge is token-level preferences over trajectories the student already samples (tool-use, env interaction, abstention, multi-step execution).
- Smallest gains on self-contained reasoning (HLE, recovery 16.9%): the general reasoning teacher's advantage partly comes from extra off-policy SFT/RL on DeepSeek-V4-Pro data the student never saw; those reasoning paths are rarely sampled, becoming OOD for the teacher and weakening token-level supervision. Consistent with the Table 4 warmup pattern.

# Limitations & open problems (3.3.5)

| Open problem | Finding |
| --- | --- |
| Logit matching | Top-k / full-vocab distribution-level distillation did not improve MOPD; underperformed the sampled-token objective on some agentic benchmarks (e.g. Terminal Bench). Hypothesis: over-strong local constraint on low-support prefixes amplifies noise from off-support states. |
| Foundations for MOPD | Better support overlap might come from a unified pre-specialization SFT stage, or from teacher-generated SFT data trained into the student before RL/MOPD. Not systematically evaluated (time/resource constraints). |
| Long-horizon tasks | Mixing many-turn agentic environments with single-turn reasoning caused training inefficiency from divergent rollout times. In practice single-turn rollouts (similar to PivotRL) used for most agentic tasks; end-to-end rollouts left as open exploration. |

# Caveats

- The authors explicitly state these variants "should not be interpreted as evidence that these approaches are fundamentally ineffective" — only that they did not help under the current setup.
- Recovery(%) can exceed 100% (Terminal Bench 2.0 172.7%, Multi-Challenge 116.7%) precisely because MOPD2 surpassed the teacher on those benchmarks.
- All Table 4/Table 5 numbers are point estimates as printed; no variance/CI reported.
- LiveCodeBench MOPD1 (90.0) is higher than MOPD2 (89.0); the trajectory is not monotonic for every benchmark — do not assume MOPD2 always beats MOPD1.
