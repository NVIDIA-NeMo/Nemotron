# Adding A Signal

How to add a measurement the shipped set does not have — a custom domain filter,
a script-specific heuristic, anything a `DocumentFilter` can express — and make
it namable from a policy.

Four steps, and one thing to know before the first: **subclass `_Base` from
`runtime/signals.py`, not `DocumentFilter` directly.** `_Base` *is* Curator's
`DocumentFilter`, plus one property enforced in `__init_subclass__`: scoring
happens on NFC. Subclassing `DocumentFilter` yourself compiles, runs, and gives
wrong answers on any script with two encodings of the same text — measured on a
correct Vietnamese sentence, `diacritic_ratio` reads 0.3231 in NFC and 0.0615 in
NFD, the same score as text whose diacritics were genuinely stripped. 11.69% of a
20,000-document Vietnamese C4 sample and 28.82% of the Hindi one are not NFC, so
this is the common case rather than an edge one.

| # | Step | Where |
|---|---|---|
| 1 | Write the filter | `runtime/signals.py` |
| 2 | Implement scoring and retention | same class |
| 3 | Register it | `runtime/registry.py` |
| 4 | Name it from a policy | your policy YAML |

---

## 1. Write the filter

```python
# runtime/signals.py

class DomainConfidence(_Base):
    """How confident a domain classifier is that the document is in-domain.

    Subclasses _Base, which is DocumentFilter plus NFC enforcement. Deriving
    from DocumentFilter directly is the one mistake that produces plausible
    numbers rather than an error.
    """

    def __init__(self, min_confidence: float = 0.5, score_field: str = "__domain_score") -> None:
        super().__init__()
        self._cutoff = float(min_confidence)
        self._score_field = score_field
```

A signal that needs language data takes a pack instead of hard-coding anything:
`_PackFilter` is the base for those, and it is what makes the runtime
language-parametric. Nothing under `runtime/` may name a language.

## 2. Implement scoring and retention

Two methods, and they answer different questions. Keeping them separate is what
lets `curate/profile` report a distribution without gating on it:

```python
    def score_document(self, text: str) -> float:
        """The measurement. One number per document, higher meaning more of it.

        Called with NFC text by _Base. Return a float; a document that cannot be
        scored should raise rather than return a sentinel, because profile counts
        scoring failures and a sentinel would land in the distribution.
        """
        ...

    def keep_document(self, score: float) -> bool:
        """The gate. Compare against the threshold this instance was built with."""
        return score >= self._cutoff
```

Rules the rest of the category relies on:

- **`score_document` must not modify the document.** A measurement stage that
  rewrites the text it measures is the defect that truncated 53.4% of one corpus
  while every row count reconciled.
- **The score must be monotone in the direction you declare.** `direction="min"`
  means "keep documents scoring at least the bound"; `max` is the reverse.
  Declaring the wrong direction builds a gate that keeps exactly what it was
  meant to drop, and no test of the pipeline's *shape* catches it.
- **One number, not a tuple.** An interval signal still scores once; the two
  bounds are the gate, not the measurement.

## 3. Register it

```python
# runtime/registry.py

register(
    Signal(
        name="domain_confidence",          # the key a policy will use
        factory=_local("DomainConfidence"),
        direction="min",
        units="ratio",
        grid=Grid(0.0, 1.0, 64),           # what profile sweeps
        threshold_params=("min_confidence",),  # your __init__ parameter names
        requires=(),                       # pack capabilities, if any
        notes="...",                       # any assumption it makes, in the report
    )
)
```

`threshold_params` is the load-bearing field: the bound from the policy is mapped
onto that keyword argument by name. Getting it wrong passes a `max:` bound to a
`min_*` parameter and inverts the gate silently, which is why the mapping is
declared rather than inferred positionally.

**Registration is a code act, and deliberately so.** A config may name a
registered signal and nothing else — it can never name an import path, because a
config is a document people paste between machines, and one that can name a class
is one that can execute arbitrary code on whoever pastes it. That property is
pinned by `test_config_cannot_name_an_import_path`.

So `register()` must run before the step reads its config:

- **In tree** — call it from `runtime/registry.py`, as the shipped signals do.
- **Out of tree** — import your module before invoking the step. There is no
  entry-point hook today; adding one is a change to this contract, not a
  configuration.

Redefining an existing name is refused rather than overwritten. `SIGNALS` is a
dict, so a plain assignment would let a second definition win while reports still
named the first — a number attributed to a measurement that did not produce it.

**If it declares a `requires`, the filter step has to be able to supply it.**
`KNOWN_REQUIREMENTS` in `step.py` is the set it can;
`test_every_requirement_the_registry_declares_is_one_this_step_can_supply` fails
here rather than at someone's pipeline construction.

## 4. Name it from a policy

```yaml
thresholds:
  - {signal: domain_confidence, min: 0.62}
```

Both bounds for an interval signal, one for `min` or `max`. A threshold naming a
signal with neither bound is refused — there is deliberately no fallback to the
signal's `curator_default`, because substituting a shipped default produces a
corpus the author did not ask for and cannot reconstruct.

Then profile it before you gate on it. `curate/profile` sweeps your `grid` and
reports what each point would retain; the threshold becomes executable only when
a person writes an `approve:` block. See the category README.

---

## Domain filters specifically

Two different mechanisms answer to the word "domain", and they are not
interchangeable:

**Curator's `MultilingualDomainClassifier`**, configured by `domains:`. Entries
are **OR** — a document is kept if its label matches any listed domain. Setting
`annotate_domains: true` instead labels every document and drops none, which is
the honest first run: filtering on a first pass drops most of a corpus with
nothing to justify it. See the category README § How The Stages Combine.

**A custom signal**, the four steps above. Use this when you need a score you can
profile and gate on a threshold somebody approved, rather than a classifier's
categorical verdict.

A custom domain filter is the second. If what you want is "keep these labels",
the first already does it and needs no code.
