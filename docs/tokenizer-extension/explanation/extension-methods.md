---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Add, Replace, and Expand: how each tokenizer_extension/extend method changes the vocabulary and merge table, and when to use each."
topics: ["Tokenizer Extension", "BPE"]
tags: ["Explanation", "Tokenizer Extension"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Add, Replace, and Expand

The `extend` step supports three construction methods, selected by `method`.
A job builds exactly one of them.
The methods differ in what happens to the base vocabulary and in whether the new tokens receive merge rules.

## The Three Methods

| `method` | Vocabulary change | Merge rules | Use when |
|----------|-------------------|-------------|----------|
| `add` | Keeps the base's residual target-script tokens and appends `N` new ones: `vocab = V + N` | New tokens are spliced into the merge table | Default choice; robust across a script family such as Hindi and Marathi |
| `replace` | Prunes the `R` residual tokens of the target script, then splices `N` fresh corpus-optimal tokens: `vocab = V − R + N` | New tokens are spliced into the merge table | One target language and the smallest possible vocabulary |
| `expand` | Registers decoded surfaces through `add_tokens()` | None; the tokens are atomic | A comparison baseline, or a target script that the base vocabulary barely covers |

*Residual tokens* are the tokens for the target script that the base tokenizer already contains.
For a language with its own Unicode block they are numerous; for a Latin-script language such as Vietnamese only tokens that carry a diacritic qualify, so Replace frees far fewer rows (761 for Vietnamese against approximately 1,569 for Hindi, according to `LANGUAGES.md`).

## Why the Splice Is Constructive

Appending trained merges below the base merges can leave tokens in the vocabulary that greedy BPE can never emit, because a higher-priority base rule shadows them.
Such *rank-dead* tokens inflate fertility silently while graph-reachability checks still pass.

The shared core in `continued_bpe.py` instead builds each new token from the pieces that the current merge table yields, so the appended rule is the one that fires.
After the splice, `find_rank_dead_tokens` asserts that zero tokens are unemittable, and the build fails if any remain.
The `step.toml` lists this condition as the `rank_dead_tokens` error.

## Expand Is Not Uniformly Worse

Because `expand` adds atomic tokens without merge rules, the new tokens do not compose, and the guidebook reports lower fertility gains than `add` for Hindi (−6.8%) and Vietnamese (−1.6%) at a 30,000-token budget.
In Malayalam, however, `expand` measured better than `add` (+2.5%).
The guidebook attributes the reversal to base-vocabulary coverage: where the base tokenizer barely covers the script, almost any well-chosen unit helps and there is little for merge training to add.
Compare the methods on your own language rather than assuming an ordering.

## Matching the Initialization Arm

The `init_embeddings` step must know how the tokenizer was built.
Its `arm` parameter accepts `add` or `replace`:

| `extend` `method` | `init_embeddings` `arm` | Reason |
|-------------------|-------------------------|--------|
| `add` | `add` | Rows are appended and base identifiers are unchanged |
| `expand` | `add` | Also append-only, so the same surgery applies |
| `replace` | `replace` | Pruning renumbers identifiers; survivor rows are permuted through `id_remap.json` |

Only the `replace/` output directory contains `id_remap.json`.
Pointing `arm=replace` at an append-style tokenizer has no remap to apply, and pointing `arm=add` at a Replace tokenizer raises the `replace_tokenizer_rejected` error.

## Budget Before Method

The guidebook's central finding for method selection is that the extension budget matters more than the choice between Add and Replace.
In the reported measurements, Add won 11 of 11 matched Add-versus-Replace fertility pairs, but only by 0.016% to 0.30%.
Choose Add for operational compatibility and Replace when every vocabulary row matters, and verify `tokens_spliced` in `summary.json` before comparing any two arms.

## Related Pages

- Procedure: {doc}`../how-to/extend-a-tokenizer`
- Fields: {doc}`../reference/extend-config`
- Source contract: [`src/nemotron/steps/tokenizer_extension/extend/README.md`](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/steps/tokenizer_extension/extend/README.md)
