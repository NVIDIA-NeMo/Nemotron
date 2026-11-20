"""Conservative visible-text normalization shared by reward consumers."""
from __future__ import annotations

import html
import re
import unicodedata

# Protect math before considering Markdown delimiters. An underscore or star
# inside an equation is content, even when it resembles paired emphasis.
MATH_PATTERN = re.compile(
    r'\$\$(.+?)\$\$|(?<![\\$])\$([^$\n]+?)\$(?!\$)|\\\[(.+?)\\\]|\\\((.+?)\\\)',
    re.S,
)
EMPHASIS_PATTERN = re.compile(
    r'(?<![\\*])\*\*(?!\s)([^\n]*?\S)\*\*(?!\*)|'
    r'(?<![\\\w_])__(?!\s)([^\n]*?\S)__(?![\w_])|'
    r'(?<![\\\w*])\*(?!\s)([^*\n]*?\S)\*(?![\w*])|'
    r'(?<![\\\w_])_(?!\s)([^_\n]*?\S)_(?![\w_])|'
    r'(?<![\\~])~~(?!\s)([^\n]*?\S)~~(?!~)'
)


def strip_markdown_emphasis(value: str) -> str:
    def strip(fragment: str) -> str:
        for _ in range(32):
            stripped = EMPHASIS_PATTERN.sub(
                lambda match: next(part for part in match.groups() if part is not None), fragment,
            )
            if stripped == fragment:
                break
            fragment = stripped
        return fragment
    parts = []
    start = 0
    for match in MATH_PATTERN.finditer(value):
        parts.extend((strip(value[start:match.start()]), match.group()))
        start = match.end()
    parts.append(strip(value[start:]))
    return ''.join(parts)


def normalize_unicode(value: str) -> str:
    """Normalize canonical Unicode equivalents without erasing scientific text.

    NFKC changes ``x²`` to ``x2`` and ``H₂O`` to ``H2O``. Those are different
    contents, including in plain OCR, so compatibility folding is inappropriate.
    Entity decoding happens exactly once per call.
    """
    return unicodedata.normalize('NFC', html.unescape(value or ''))


_SCRIPT_MARKUP_RE = re.compile(r'<(sup|sub)\b[^>]*>(.*?)</\1\s*>', re.I | re.S)
_TYPOGRAPHY_RE = re.compile(
    r'</?(?:b|strong|i|em|u|s|strike|del|mark|h[1-6]|code)\b[^>]*>', re.I,
)
_EDGE_TOKEN_RE = re.compile(r'^<(?:/?s|pad|eos|bos)>')
_END_TOKEN_RE = re.compile(r'<(?:/?s|pad|eos|bos)>$')


def preserve_script_markup(value: str) -> str:
    """Keep HTML script content as explicit mathematical script notation.

    The paired form makes ``H<sub>2</sub>O`` distinct from ``H2O`` and gives
    numeric consumers a representation they can interpret without losing the
    exponent. Unclosed markup is left intact for structural diagnostics.
    """
    return _SCRIPT_MARKUP_RE.sub(
        lambda match: ('^' if match.group(1).lower() == 'sup' else '_')
        + '{' + script_body(match.group(2)) + '}', value,
    )


def strip_decoder_wrappers(value: str) -> str:
    """Remove decoder tokens only at completion edges, never inside blocks.

    A leading ``<s>`` with its closing tag before a trailing class token is
    actual strike markup in the class-only dialect, so preserve that pair.
    A complete outer ``<s>...</s>`` pair is treated as a decoder wrapper; an
    unstructured completion alone cannot distinguish that from strike markup.
    Call this on a completion, not on an individual block body.
    """
    value = (value or '').strip()
    while match := _EDGE_TOKEN_RE.match(value):
        token = match.group()
        if token == '<s>' and '</s>' in value[match.end():]:
            depth = 0
            closing = None
            for candidate in re.finditer(r'</?s>', value):
                depth += 1 if candidate.group() == '<s>' else -1
                if depth == 0:
                    closing = candidate
                    break
            if closing is not None and not value[closing.end():].strip():
                value = value[match.end():closing.start()].strip()
                continue
            tail = value[closing.end():] if closing is not None else ''
            if re.search(r'<(?:class|x|y)_', tail):
                break
        value = value[match.end():].lstrip()
    while match := _END_TOKEN_RE.search(value):
        # Preserve an actual strike pair that survived the leading-token rule.
        if match.group() == '</s>' and value.startswith('<s>'):
            break
        value = value[:match.start()].rstrip()
    return value


def _looks_like_math(value: str) -> bool:
    """Conservatively avoid treating a mathematical sign as a list marker."""
    value = value.strip()
    return bool(
        re.match(r'(?:[+−-]\s*)?(?:\d|\.\d)', value)
        or re.search(r'[=<>≤≥≠≈±×÷^_]|\\(?:frac|sum|prod|int|sqrt)\b', value)
        or re.fullmatch(r'[^\W\d_](?:\s*[+−*/-]\s*[^\W\d_])*', value, re.UNICODE)
    )


def _strip_markdown_lines(value: str) -> str:
    def list_marker(match: re.Match[str]) -> str:
        return match.group() if _looks_like_math(match.group(2)) else match.group(2)
    value = re.sub(r'(?m)^\s*#{1,6}\s+', '', value)
    value = re.sub(r'(?m)^[ \t]*([-+*]|\d+[.)])[ \t]+([^\n]+)', list_marker, value)
    return re.sub(r'```(?:[^\n]*)\n|```', '', value)


def normalize_visible_text(
    value: str, *, output_mode: str = 'unspecified', content_class: str | None = None,
) -> str:
    """Normalize rendered prose while preserving math and plain OCR content.

    Lists/headings/fences are rendering syntax only in Markdown mode. Formula
    bodies never use prose Markdown stripping. Even in Markdown Text blocks,
    a line such as ``- x = 10`` keeps its sign. The unspecified mode retains
    legacy paired-emphasis equivalence without guessing that a sign is a list.
    Decoder and coordinate/class tokens belong to the page parser, not here.
    """
    value = normalize_unicode(value)
    value = _TYPOGRAPHY_RE.sub('', value)
    if content_class != 'Formula' and output_mode != 'plain':
        value = strip_markdown_emphasis(value)
        if output_mode == 'markdown':
            parts = []
            start = 0
            for match in MATH_PATTERN.finditer(value):
                parts.extend((_strip_markdown_lines(value[start:match.start()]), match.group()))
                start = match.end()
            parts.append(_strip_markdown_lines(value[start:]))
            value = ''.join(parts)
    return re.sub(r'\s+', ' ', canonical_unicode_scripts(preserve_script_markup(value))).strip()

# Fold only explicit script glyphs, never ordinary scientific text wholesale.
UNICODE_SCRIPT_PATTERNS = (("sup", re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ]+")),
                           ("sub", re.compile(r"[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ]+")))
def script_body(value):
    return unicodedata.normalize('NFKC', value)
def unicode_script_spans(value):
    for kind, pattern in UNICODE_SCRIPT_PATTERNS:
        for match in pattern.finditer(value):
            yield kind, script_body(match.group())
def canonical_unicode_scripts(value):
    for kind, pattern in UNICODE_SCRIPT_PATTERNS:
        value = pattern.sub(lambda m: ('^' if kind == 'sup' else '_')+'{'+script_body(m.group())+'}', value)
    return value
def decoded_cell_text(value):
    # HTMLParser already decoded entities and separated real tags. Its data
    # is literal content: do not parse Markdown or HTML a second time.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', value)).strip()
