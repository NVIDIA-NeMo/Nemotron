---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "sft"
paper_sections: ["3", "3.1", "3.1.1", "3.1.2"]
title: "Nemotron 3 Ultra Post-Training: Supervised Fine-Tuning Data Mixture and Data Packing"
summary: |
  Covers the opening of the redesigned Nemotron 3 Ultra post-training pipeline and the general SFT
  stage. The pipeline is SFT -> RLVR -> MOPD warmup -> MOPD (xN cycles) -> MTP Boosting (Figure 9).
  This chunk details the SFT hyperparameters, the per-domain SFT data sources and sizes (long-context,
  efficiency/control, safety, search, terminal-use, conversational tool-use, software issue resolution,
  math/proof, science, chat, code, CUDA, RTL, multilingual), and the length-aware best-fit data packing
  strategy (3.1.2).
key_facts:
  - "Post-training is substantially redesigned vs Nemotron 3 Super; the pipeline (Figure 9) is: Base (pretrained hybrid Mamba-Attention MoE) -> SFT -> RLVR -> MOPD Warmup -> MOPD (x N cycles) -> MTP Boosting -> Nemotron 3 Ultra."
  - "SFT trains on packed sequences of length 294,912 with a global batch size of 64."
  - "SFT trains on 204,800 samples using a cosine LR schedule: peak LR 1.5e-5, minimum LR 1e-6, and 9,600 warmup samples."
  - "The shared-weight MTP objective is retained during SFT, using two MTP layers with a per-token auxiliary-loss scaling factor of 0.1."
  - "Long-context SFT data is prepared at 512K following the synthetic data pipeline of NVIDIA (2026); targets long-context multi-document reasoning, sequential scanning, and SQL queries."
  - "Efficiency/control component 1: GPT-OSS-120B medium-effort-mode samples on math reasoning, STEM QA, and instruction following, to initiate Ultra's medium-effort mode (later optimized in RLVR)."
  - "Efficiency/control component 2: samples with reasoning traces truncated to random reasoning budgets (responses unchanged); design change vs Nano/Super: </think> tokens in truncated samples are masked from SFT training loss."
  - "Safety: retains the 45K safety data blend from Nemotron 3 Super, translated into six languages (German, Spanish, French, Japanese, Italian, Chinese) via NeMo Skills chunked translation with NVIDIA Riva Translate 4B v1.1 as backbone."
  - "Safety translation filtering: back-translated and compared to original; examples with semantic similarity below 0.8 filtered out, removing approximately 10-15% per translated language."
  - "Final safety blend: approximately 135K samples = approximately 45K English + approximately 15K from each of the six translated languages."
  - "Search: retains Nemotron 3 Super search trajectories (Wikidata seed, 4-8 hop random walks, solved via Tavily with MiniMax 2.1 as teacher)."
  - "Search: adds a commercially-cleared subset of OpenResearcher (originally 97K+ trajectories synthesized with gpt-oss-120b over 15M FineWeb docs, three browser tools search/open/find); commercial-OK curation yields approximately 21.7K SFT trajectories."
  - "Search: vendor-curated hard samples requiring 50-100 searches collected in the BrowseComp harness (Appendix B), using MiniMax 2.5 and GLM 5.1 as teacher models."
  - "Terminal-use dataset: approximately 370K multi-turn conversations (mix of reasoning and non-reasoning); DeepSeek-V3.2 acting agent within Terminus-2 / Harbor framework."
  - "Conversational tool-use data scaled via a fully synthetic six-stage generation pipeline (User and Environment simulation), similar to Nemotron 3 Super."
  - "Software issue resolution: synthetic GitHub-issue trajectories from Minimax-M2.5 (thinking) and Qwen3-Coder-480B-A35B-Instruct (non-thinking), captured with OpenHands, SWE-agent, Mini-SWE-agent, Opencode harnesses, filtered by a per-trajectory heuristic analyzer."
  - "Math/proof: Nemotron-Cascade-2 math data; non-proof = 1.8M tool-calling + 1.9M non-tool samples (responses from DeepSeek-V3.2 and DeepSeek-V3.2-Speciale); proof data from AOPS split of Nemotron-Math-Proofs-v1, responses from DeepSeek-V3.2-Speciale."
  - "Code: competitive programming (Codeforces, AtCoder, AIZU, CodeChef), GPT-OSS-120B teacher with rejection sampling; final dataset = 1.2M Python reasoning traces, 1.0M C++14 reasoning traces, 1.3M Python tool-calling reasoning traces."
  - "CUDA: approximately 100K samples for kernel generation/repair/optimization, built with DeepSeek-R1 and GPT-OSS-120B; CUDA-X libraries covered: Thrust, CUB, cuBLAS, cuDNN, cuSPARSE, cuRAND, cuSOLVER."
  - "RTL: ACE-RTL training data (spec-to-RTL, code editing, code debugging), generated with DeepSeek-R1 and GPT-OSS-120B; final dataset around 1.2M training samples."
  - "Multilingual: new end-to-end full-JSON-in/full-JSON-out translation pipeline using DeepSeek-V3-0324; newly synthesized for Hindi, Japanese, Korean, Brazilian Portuguese; remaining languages reuse Super-V3 synthetic data."
  - "Data packing: length-aware best-fit packing (Ding et al., 2024) into sequences up to max context length; round-robin source interleaving, fixed-size pool of open sequences, best-fit assignment to minimize padding, no truncation/splitting, in-pack deduplication, final shuffle over completed packs."
related_steps: []
currency: "frozen"
---

# Scope

- What is the full Nemotron 3 Ultra post-training pipeline and how does it differ from Super?
- What hyperparameters are used for the general SFT stage (sequence length, batch size, sample count, LR schedule, MTP)?
- What SFT data sources/domains are used and how big is each?
- How is the SFT data packed?

# Pipeline overview (3 opening, Figure 9)

Post-training was substantially redesigned vs Nemotron 3 Super (NVIDIA, 2026). Rather than relying solely on consecutive RL stages, the pipeline augments with Multi-teacher On-Policy Distillation (MOPD) for both broad capability acquisition and targeted specialization.

| Stage | Role |
| --- | --- |
| Base | Pretrained hybrid Mamba-Attention MoE checkpoint, extended for long context |
| SFT | Multi-domain SFT with shared-weight MTP objective |
| RLVR | Unified verifiable-reward training (agentic, reasoning, chat, safety, instruction following, long-context) |
| MOPD Warmup | One-time light SFT aligning student rollouts with teacher-supported distributions |
| MOPD (x N cycles) | Asynchronous on-policy distillation merging specialized teachers via dense token-level guidance |
| MTP Boosting | Head-only KL distillation aligning MTP drafts with backbone logits for faster speculative decoding |

Claim: two iterations of MOPD were performed for Nemotron 3 Ultra (see mopd/overview.md).

# SFT hyperparameters (3.1)

| Setting | Value |
| --- | --- |
| Packed sequence length | 294,912 |
| Global batch size | 64 |
| Number of samples | 204,800 |
| LR schedule | Cosine |
| Peak LR | 1.5e-5 |
| Minimum LR | 1e-6 |
| Warmup samples | 9,600 |
| MTP layers | 2 (shared-weight) |
| MTP per-token auxiliary-loss scaling | 0.1 |

# SFT data sources (3.1.1)

| Domain | Key sizes / teachers |
| --- | --- |
| Long Context | 512K long-context data; targets multi-doc reasoning, sequential scanning, SQL |
| Efficiency & Control | GPT-OSS-120B medium-effort samples (math/STEM/IF) + truncated-reasoning-budget samples with </think> masked from loss |
| Safety | ~135K total = ~45K English (from Super blend) + ~15K each in DE/ES/FR/JA/IT/ZH; semantic-similarity <0.8 filtered (~10-15% removed/language); Riva Translate 4B v1.1 backbone |
| Search | Super trajectories (Wikidata, 4-8 hop, Tavily, MiniMax 2.1 teacher) + ~21.7K OpenResearcher commercial-OK subset (orig 97K+, gpt-oss-120b, 15M FineWeb docs) + BrowseComp 50-100 search samples (MiniMax 2.5, GLM 5.1 teachers) |
| Terminal-use | ~370K multi-turn conversations; DeepSeek-V3.2 in Terminus-2/Harbor |
| Conversational Tool Use | Six-stage synthetic pipeline (user + env simulation), similar to Super |
| Software Issue Resolution | Synthetic GitHub-issue trajectories; Minimax-M2.5 (thinking) + Qwen3-Coder-480B-A35B-Instruct (non-thinking); heuristic-analyzer filtered |
| Math / Proof | Nemotron-Cascade-2: 1.8M tool-calling + 1.9M non-tool (DeepSeek-V3.2 / DeepSeek-V3.2-Speciale); proof from AOPS split, DeepSeek-V3.2-Speciale |
| Science | Nemotron Nano recipe (physics/chem/bio) + DeepSeek-V3.2 web-search and web-search-with-Python traces |
| Chat | Seeds from LMArena/WildChat; GLM-5 candidates; Nemotron-GenRM selection; multi-turn via simulated user; train only on final assistant turn |
| Code | Competitive programming; GPT-OSS-120B teacher + rejection sampling; 1.2M Python + 1.0M C++14 + 1.3M Python tool-calling traces |
| CUDA | ~100K samples (gen/repair/optimization); DeepSeek-R1 + GPT-OSS-120B; CUDA-X: Thrust, CUB, cuBLAS, cuDNN, cuSPARSE, cuRAND, cuSOLVER |
| RTL | ACE-RTL data; DeepSeek-R1 + GPT-OSS-120B; ~1.2M samples |
| Multilingual | End-to-end full-JSON translation via DeepSeek-V3-0324; new for Hindi/Japanese/Korean/Brazilian Portuguese; rest reuse Super-V3 |

# Data packing (3.1.2)

Claim: length-aware best-fit packing strategy (Ding et al., 2024).

- Reads/interleaves all source files round-robin; maintains a fixed-size pool of open sequences; retires a sequence when residual capacity falls below a small tolerance.
- Each conversation assigned to the partially filled sequence it most tightly fits (best-fit) to minimize padding.
- No truncation or splitting of conversations (preserves context, reduces hallucinations).
- In-pack deduplication prevents identical prompts co-occurring in one sequence.
- Final shuffle over all completed packs; ensures each packed sequence draws from a broad, well-mixed cross-section to avoid distributional locality.

# Caveats

- The 512K long-context figure is a data preparation length, not the same as the SFT packed sequence length of 294,912.
- "approximately"/"around"/"~" figures (135K, 45K, 15K, 21.7K, 97K, 370K, 100K, 1.2M RTL) are stated as approximate in the source; do not present as exact.
- The six-stage conversational tool-use pipeline is described only as "similar to Nemotron-3 Super"; no per-stage detail or size is reported here.
- Many domains (long-context, efficiency/control, science, chat, conversational tool use, software issue resolution) do not report a total sample/token count; do not invent one.
- This chunk does NOT cover the General Reasoning / STEM teacher's separate SFT blend token counts (40B etc.) - those belong to MOPD teachers (see mopd/teachers.md).
