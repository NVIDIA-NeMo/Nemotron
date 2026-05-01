# Quality And Filtering

MCQ generation uses several quality gates:

- Judgement checks that candidate questions are useful and answerable.
- Semantic deduplication removes near-duplicate questions when configured to do so.
- Distractor validity checks that incorrect options are plausible but not correct.
- Semantic outlier detection catches answer options that are too different from distractors.
- Hallucination filtering checks whether model answers agree with the generated answer.
- Easiness filtering flags questions that too many models answer correctly.

Translation quality uses Curator experimental translation metrics such as SacreBLEU, chrF, and TER. Keep
inline Curator filtering disabled until rows are restored to the benchmark schema unless the run is
intentionally allowed to drop low-quality translated questions with `remove_low_quality`.
