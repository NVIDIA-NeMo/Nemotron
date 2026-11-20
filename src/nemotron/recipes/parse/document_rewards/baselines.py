"""Text, localization, completion diagnostics and explicit reward weighting."""

from collections import Counter
from difflib import SequenceMatcher
import math
from numbers import Real
import re

from ._normalization import normalize_visible_text, strip_decoder_wrappers


def _visible(text):
    text = strip_decoder_wrappers(text or "")
    text = re.sub(r"<(?:x|y)_[^>]+>|<class_[^>]+>|<tbc>", " ", text)
    return normalize_visible_text(text)


def _distance(reference, prediction):
    """Exact character or word Levenshtein distance using bit vectors."""
    if len(reference) > len(prediction):
        reference, prediction = prediction, reference
    if not reference:
        return len(prediction)
    masks = {}
    for i, token in enumerate(reference):
        masks[token] = masks.get(token, 0) | (1 << i)
    positive, negative, distance = ~0, 0, len(reference)
    last = 1 << (len(reference) - 1)
    for token in prediction:
        equal = masks.get(token, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        distance += bool(plus & last) - bool(minus & last)
        plus, minus = (plus << 1) | 1, minus << 1
        positive, negative = minus | ~(vertical | plus), plus & vertical
    return distance


def text_quality(prediction, reference):
    """Return bounded character accuracy plus raw CER/WER and content F1.

    Inputs are transcription text. Parse coordinate/decoder tokens and basic
    presentation markup are normalized; case, punctuation and digits matter.
    """
    if reference is None:
        return {"score": None}
    pred, ref = _visible(prediction), _visible(reference)
    cer = _distance(ref, pred) / max(1, len(ref))
    pwords, rwords = pred.split(), ref.split()
    wer = _distance(rwords, pwords) / max(1, len(rwords))
    matched = sum(b.size for b in SequenceMatcher(None, ref, pred, autojunk=False).get_matching_blocks())
    precision = matched / len(pred) if pred else float(not ref)
    recall = matched / len(ref) if ref else float(not pred)
    f1 = 2 * matched / (len(pred) + len(ref)) if pred or ref else 1.0
    return dict(score=max(0.0, 1.0 - cer), cer=cer, wer=wer,
                precision=precision, recall=recall, f1=f1)


def _valid_box(box):
    try:
        return (len(box) == 4 and all(isinstance(x, Real) and math.isfinite(x) for x in box)
                and box[2] > box[0] and box[3] > box[1])
    except (TypeError, ValueError):
        return False


def _iou(a, b):
    if not _valid_box(a) or not _valid_box(b):
        return 0.0
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area - intersection)


def box_quality(prediction, reference, *, min_iou=0.5):
    """One-to-one IoU matching of [x1, y1, x2, y2] boxes in common units.

    The scalar is overlap credit divided by both box counts. Missing, extra,
    duplicate and invalid predicted boxes therefore reduce the reward.
    """
    if reference is None:
        return {"score": None}
    if not math.isfinite(min_iou) or not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0, 1]")
    pred, ref = list(prediction) if prediction is not None else [], list(reference)
    if not all(_valid_box(b) for b in ref):
        raise ValueError("Reference boxes must be finite, positive-area xyxy boxes")
    pairs = []
    if pred and ref:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        matrix = np.array([[_iou(p, r) for r in ref] for p in pred])
        pi, ri = linear_sum_assignment(-matrix)
        pairs = [(int(p), int(r), float(matrix[p, r])) for p, r in zip(pi, ri)
                 if matrix[p, r] >= min_iou]
    count, total = len(pairs), len(pred) + len(ref)
    return dict(score=2 * sum(iou for _, _, iou in pairs) / total if total else 1.0,
                precision=count / len(pred) if pred else float(not ref),
                recall=count / len(ref) if ref else float(not pred),
                f1=2 * count / total if total else 1.0,
                matched_iou=sum(iou for _, _, iou in pairs) / count if count else 0.0,
                matched=count, predicted=len(pred), reference=len(ref), matches=pairs)


def completion_quality(text, *, token_ids=None, eos_token_id=None, eos_reached=None,
                       finish_reason=None, reference_text="", ngram_size=24,
                       min_repeats=4, token_run_limit=32):
    """Diagnose completed generation; this function does not stop decoding.

    Supply actual EOS metadata or token IDs. A generic 'stop' reason alone
    cannot distinguish EOS from a custom stop string.
    """
    if min(ngram_size, min_repeats, token_run_limit) < 1:
        raise ValueError("Repetition detector sizes must be positive")
    ids = list(token_ids) if token_ids is not None else []
    ended = bool(ids[-1] == eos_token_id) if ids and eos_token_id is not None else eos_reached
    capped = finish_reason in {"length", "max_tokens"}

    def ngrams(value):
        words = _visible(value[-4096:]).casefold().split()
        return Counter(tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1))

    predicted, expected = ngrams(text), ngrams(reference_text or "")
    repeated = max(predicted.values(), default=0)
    excess = any(count >= min_repeats and count > expected[gram] for gram, count in predicted.items())
    block_pattern = r"<x_[^>]+><y_[^>]+>.*?<x_[^>]+><y_[^>]+><class_[^>]+>"
    blocks = Counter(re.findall(block_pattern, text, re.S))
    ref_blocks = Counter(re.findall(block_pattern, reference_text or "", re.S))
    excess |= any(count >= min_repeats and count > ref_blocks[block] for block, count in blocks.items())
    longest = run = 0
    last = None
    for token in ids:
        run = run + 1 if token == last else 1
        longest = max(longest, run)
        last = token
    looped = excess or longest >= token_run_limit
    score = 0.0 if capped or looped else (float(ended) if ended is not None else None)
    return dict(score=score, eos_reached=ended, capped=capped, looped=bool(looped),
                max_repeated_ngram=repeated, max_identical_token_run=longest)


def weighted_reward(components, weights):
    """Combine named scores in [0, 1], renormalizing over applicable terms.

    A component can be a scorer's result dictionary, a scalar, or None for a
    task that is not annotated/applicable. Weights need not sum to one.
    """
    values, active = {}, {}
    for name, weight in weights.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Weights must be finite and nonnegative")
        value = components[name]
        value = value["score"] if isinstance(value, dict) else value
        if value is None or weight == 0:
            continue
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must have a finite score in [0, 1]")
        values[name], active[name] = float(value), float(weight)
    total = sum(active.values())
    if not total:
        raise ValueError("At least one positive-weight component must be applicable")
    normalized = {name: weight / total for name, weight in active.items()}
    contributions = {name: normalized[name] * value for name, value in values.items()}
    return dict(score=sum(contributions.values()), scores=values,
                effective_weights=normalized, contributions=contributions)
