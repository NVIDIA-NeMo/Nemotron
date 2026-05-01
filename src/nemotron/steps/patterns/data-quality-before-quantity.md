---
id: data-quality-before-quantity
title: "Prioritize data quality before scale"
tags: [curate, data-quality]
triggers:
  - "You are considering collecting more training data to fix weak model behavior."
  - "The current corpus contains duplicates, inconsistent formatting, or noisy labels."
  - "A small clean dataset is competing with a much larger but messy alternative."
steps: [curate/nemo_curator]
confidence: high
---

## When to apply

Use this during data acquisition and curation, especially when teams are tempted to solve every quality problem by adding more rows. For SFT, a smaller high-signal corpus often beats a much larger noisy one.

Apply it when the dataset mixes sources, contains scraped text with inconsistent structure, or includes uncertain prompt/response quality. Noise compounds quickly in instruction tuning.

It also matters when you only have budget for one curation pass. The first pass should improve signal density, not just volume.

## What to do

Define what "good data" means for the target use case before scaling. For example: correct language, correct domain, correct format, safe content, and responses that actually demonstrate the behavior you want the model to copy.

Remove obvious duplicates and near-duplicates early. Repeated low-value examples distort training and create false confidence about dataset size.

Normalize schema and formatting. Mixed role labels, inconsistent system prompts, and malformed conversations create training noise that no optimizer setting will fix.

Sample and review data manually. Even a few hundred inspected rows can reveal systemic issues like template leakage, bad translations, policy violations, or mismatched domains.

Prefer targeted acquisition over bulk scraping when you know the desired behavior. Ten thousand relevant rows are often more useful than a million generic ones.

Track rejection reasons during curation. Knowing why data was filtered helps improve future acquisition and prevents reintroducing the same noise later.

Scale only after the clean subset shows the expected learning signal in baseline experiments. If the clean set cannot move the metric at all, adding more of the same weak signal may not help.

## Exceptions

For large pretraining-style corpora, sheer volume matters more than in narrow SFT tasks, but even then gross duplication and contamination are still harmful.

Do not over-curate into a tiny monoculture dataset. Quality matters, but coverage and diversity still matter once the obvious noise has been removed.

If the user problem is genuinely missing coverage, then more data is the answer — just more relevant data, not indiscriminate volume.

## References

- Most directly relevant to `curate/nemo_curator` and upstream data decisions that feed all later steps.
- This pattern often explains why a "small clean pilot" outperforms a rushed large corpus.
- Pair with eval-bookends so curation improvements are measured, not assumed.
