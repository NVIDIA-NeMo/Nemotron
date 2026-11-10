# Optional document rewards

These independent baselines cover transcription, layout, tables, charts and
completion quality. Each returns a dictionary containing `score` in `[0, 1]`
and component measurements. A reference of `None` means that component is not
annotated/applicable; its score is `None`. Invalid generated structures score
zero. Invalid reference tables/charts raise an error so they can be corrected.

The dental recipe uses its exact-page reward. These functions are options for
other document tasks; their weighting is independent of reported exact-match
accuracy.

| Function | Input | Scalar and measurements |
|---|---|---|
| `text_quality(prediction, reference)` | Transcription strings | `max(0, 1-CER)`; raw CER/WER and text precision/recall/F1. Case, digits and punctuation matter. |
| `box_quality(prediction, reference)` | Lists of `[x1, y1, x2, y2]` boxes in the same coordinate units | One-to-one IoU credit divided by both box counts; presence precision/recall/F1 and matched IoU. Extra, missing and duplicate boxes reduce credit. Default matching threshold: IoU 0.5. |
| `table_quality(prediction, reference)` | One HTML table, Markdown table, or rectangular string grid | 80% GriTS-Con + 20% GriTS-Top by default; both metrics returned separately. HTML supports row/column spans. Convert LaTeX to HTML or a grid first. |
| `chart_quality(prediction, reference)` | Canonical Parse `<chart type="...">` targets | Associated numeric/categorical factual F1, with chart type, table presence and metadata. Numeric value tolerance defaults to 5% relative / `1e-8` absolute; units and series/category/axis associations are retained. |
| `completion_quality(text, ...)` | Generated text plus actual EOS/finish metadata or token IDs | Clean EOS completion; token-cap and repetition diagnostics. Unknown EOS status gives no completion credit (`None`) unless a detected cap/loop establishes failure. |

Text inputs should be the extracted transcription, not raw HTML/LaTeX table
serialization. The normalizer handles Parse coordinate/decoder tokens and basic
presentation markup. Supply rich tables and charts to their respective scorers.
Box detection expects annotated empty lists for genuinely empty pages; use
`None` when layout was not annotated.

## Combine the scores

```python
from nemotron.recipes.parse.document_rewards import weighted_reward

# Each value can be a scorer's result dictionary, a scalar, or None.
components = {
    "text": {"score": 0.95},
    "boxes": {"score": 0.80},
    "tables": {"score": 0.90},
    "charts": None,  # No chart annotation on this page.
    "completion": {"score": 1.0},
}
weights = {"text": 0.40, "boxes": 0.20, "tables": 0.20,
           "charts": 0.15, "completion": 0.05}
result = weighted_reward(components, weights)
reward = result["score"]
```

Weights are renormalized over applicable components. Here the chart term is
excluded and the result is approximately `0.9059`. `effective_weights` and
`contributions` explain the total. Weights are example starting points, not a
production reward recipe. Keep individual metrics visible when changing them.

[The runnable example](../examples/weighted_reward.py) calls all five scorers:

```sh
# From the Nemotron checkout, using the isolated Parse runtime:
PARSE_RUNTIME=.nemotron/parse-runtime
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PARSE_RUNTIME/venv/bin/python" -m nemotron.recipes.parse.examples.weighted_reward
```

To use a mixture in NeMo RL, compute these components inside the environment's
scoring function and return `result["score"]` as the scalar reward. Keep labels
in scoring metadata, not in the model prompt. The standalone baseline mixture
does not replace the dental environment automatically.

## Dependencies

Box, table and chart scoring require NumPy and SciPy. Text, completion and
weighted combination use only the Python standard library. If these optional
dependencies are absent from the selected runtime, install them explicitly:

```sh
uv pip install --python "$PARSE_RUNTIME/venv/bin/python" 'numpy>=1.26' 'scipy>=1.11'
```

`table_teds` is an optional alternative table metric. Install its extras before
calling it:

```sh
uv pip install --python "$PARSE_RUNTIME/venv/bin/python" \
  'numpy>=1.26' 'scipy>=1.11' 'apted>=1.0.3' 'python-Levenshtein>=0.25' 'lxml>=5'
```

Table metrics score the grid, excluding captions and inline typography; text
scoring can check captions separately. Mathematical scripts remain content.
TEDS uses normalized HTML with cell headers represented as ordinary cells.
`structure_only=True` excludes cell content. The GriTS scorer uses the reference
grid-alignment algorithm, with optional Numba acceleration when available.
Large tables can be expensive. `max_cells` bounds table-grid expansion (default
10,000 cells); exceeding it fails a prediction or raises for a reference. It
does not substitute an approximate GriTS score.

Chart scoring reuses the training chart parser, fact keys and numeric matcher.
It is a conservative baseline, not the full production chart reward: it does
not perform production scope/orientation recovery, table record matching or
version-specific reward routing. Consistent scope declarations are expected.

## Completion and loop control

Set a finite `max_new_tokens` and keep EOS enabled in the generation backend.
Use actual token IDs/EOS ID or `eos_reached` metadata with `completion_quality`.
A `finish_reason="stop"` alone could mean a stop string, so it is insufficient
proof of natural EOS. Reaching the token cap always scores zero for completion.

The repetition detector checks word n-grams, identical Parse blocks and repeated
token runs. Text/block repetition is compared with the reference; pass
`reference_text` when repeated content is legitimate. Defaults are 24-word
n-grams, four repeats and a 32-token identical-token run.

This is post-generation scoring, not an early-abort callback. A small completion
weight does not guarantee EOS; if clean completion is required, multiply the
mixture by the completion score after supplying actual termination metadata.
NeMo RL's overlong filtering drops truncated samples: dropped samples do not
learn from a completion penalty. Adjust that policy deliberately when using
completion as a training objective. Repetition penalties also change decoding
probabilities; use the framework's corresponding policy/log-probability handling.

Third-party attribution is in [NOTICE](NOTICE).
