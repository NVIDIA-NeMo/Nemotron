"""Content scoring shared by training rewards and held-out evaluation.

Coordinates and markup do not earn content credit. Text CER preserves case,
digits and punctuation after markup removal and whitespace normalization.
"""

from collections import Counter
from html.parser import HTMLParser
import html
import re
import unicodedata

ORDER = "MODBIFL"
ANTERIOR = set(range(6, 12)) | set(range(22, 28))


def clean(text):
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\\(?:textbf|textit|mathrm|textrm|text|thead)\{([^{}]*)\}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def table_rows(text):
    # Accept HTML, LaTeX and Markdown tables.
    if "<table" in text.lower():

        class Reader(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows = []
                self.row = []
                self.cell = None

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self.row = []
                elif tag in ["td", "th"]:
                    self.cell = []

            def handle_data(self, data):
                if self.cell is not None:
                    self.cell.append(data)

            def handle_endtag(self, tag):
                if tag in ["td", "th"] and self.cell is not None:
                    self.row.append(clean(" ".join(self.cell)))
                    self.cell = None
                elif tag == "tr" and self.row:
                    self.rows.append(self.row)

        reader = Reader()
        reader.feed(text)
        yield from reader.rows
        text = re.sub(r"<table\b[^>]*>.*?</table>", "", text, flags=re.S | re.I)
    for line in re.split(r"\\\\|\n", text):
        line = re.sub(r"<[^>]*>", " ", line)
        line = re.sub(r"\\(?:begin|end)\{tabular\}(?:\{[^}]*\})?", "", line)
        line = re.sub(r"\\(?:hline|cline\{[^}]*\})", "", line)
        sep = "&" if "&" in line else "|" if "|" in line else None
        if sep:
            cells = [clean(c).strip(" $") for c in line.strip().strip("|").split(sep)]
            yield cells


def compare(expected, predicted):
    truth = Counter(expected)
    pred = Counter(predicted)
    tp = sum((truth & pred).values())
    nt = sum(truth.values())
    np = sum(pred.values())
    return dict(
        tp=tp,
        expected=nt,
        predicted=np,
        precision=tp / np if np else float(nt == 0),
        recall=tp / nt if nt else float(np == 0),
        f1=2 * tp / (nt + np) if nt + np else 1.0,
        exact=truth == pred,
    )


def allowed(t):
    return ["M", "D", "F", "L", "I"] if t in ANTERIOR else ["M", "D", "B", "L", "O"]


def canonical(codes):
    return "".join(c for c in ORDER if c in codes)


def surface_codes(value, tooth):
    value = value.strip()
    if value.lower() in ["-", "whole", "whole-tooth", "n/a"]:
        return "-"
    names = {
        "mesial": "M",
        "distal": "D",
        "buccal": "B",
        "facial": "F",
        "labial": "F",
        "lingual": "L",
        "palatal": "P",
        "occlusal": "O",
        "incisal": "I",
    }
    codes = [code for word, code in names.items() if re.search(r"\b" + word + r"\b", value, re.I)]
    if not codes:
        compact = re.sub(r"[\s,;/+\-]", "", value.upper())
        if compact and set(compact) <= set("MODBFILP"):
            codes = list(compact)
    # B/F is a common combined charting button: normalize the outer surface to
    # the target tooth's convention. Upper palatal and lingual name the same
    # inward-facing surface. Do not penalize equivalent anatomical notation.
    anterior = tooth in set(range(6, 12)) | set(range(22, 28))
    codes = [
        ("F" if anterior else "B") if c in ["B", "F"] else "L" if c == "P" and tooth <= 16 else c
        for c in codes
    ]
    return canonical(codes) if codes and set(codes) <= set(allowed(tooth)) else "unknown"


def extracted(text):
    result = []
    mapping = None
    history = False
    for cells in table_rows(text):
        headers = {
            k: next((i for i, c in enumerate(cells) if c in names), None)
            for k, names in dict(
                tooth=["tooth", "tooth number", "tooth #", "tooth no"],
                procedure=["procedure", "treatment"],
                surface=["surfaces", "surface"],
            ).items()
        }
        if headers["tooth"] is not None and headers["procedure"] is not None:
            mapping = headers
            history = "date" in cells
            continue
        if history or mapping is None:
            continue

        def get(k):
            i = mapping[k]
            return cells[i] if i is not None and i < len(cells) else "unknown"

        if not re.fullmatch(r"#?\d{1,2}", get("tooth")):
            continue
        tooth = int(get("tooth").lstrip("#"))
        procedure = get("procedure").replace("root-canal", "root canal")
        if not 1 <= tooth <= 32 or procedure not in [
            "filling",
            "crown",
            "extraction",
            "root canal",
        ]:
            continue
        surface = surface_codes(get("surface"), tooth)
        result.append((tooth, procedure, surface))
    # Plain-language extraction earns the same content credit as a Chart table.
    for line in text.splitlines():
        if "|" in line or "&" in line:
            continue
        matches = list(re.finditer(r"\btooth\s*#?\s*(\d{1,2})\b", line, re.I))
        for i, m in enumerate(matches):
            tooth = int(m[1])
            tail = line[m.end() : matches[i + 1].start() if i + 1 < len(matches) else None]
            procedure = re.search(r"\b(filling|crown|extraction|root[ -]canal)\b", tail, re.I)
            if not procedure or not 1 <= tooth <= 32:
                continue
            remainder = tail[procedure.end() :]
            names = re.findall(
                r"\b(?:mesial|distal|buccal|facial|labial|lingual|palatal|occlusal|incisal)\b",
                remainder,
                re.I,
            )
            codes = re.findall(r"\b[MODBFILP]{1,5}\b", remainder)
            value = " ".join(names or codes)
            if re.search(r"\bwhole(?:-tooth)?\b", remainder, re.I):
                value = "-"
            result.append(
                (
                    tooth,
                    procedure[1].lower().replace("root-canal", "root canal"),
                    surface_codes(value, tooth),
                )
            )
    return result


def visible_text(text):
    """Visible full-page text, retaining case, digits and punctuation, not boxes/markup."""
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\\(?:begin|end)\{tabular\}(?:\{[^}]*\})?", " ", text)
    text = re.sub(r"\\(?:hline|cline\{[^}]*\})", " ", text)
    text = re.sub(r"\\(?:textbf|textit|mathrm|textrm|text|thead)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"(?m)^\s*\|?[ :|\-]+\|\s*$", " ", text)
    text = text.replace("|", " ").replace("&", " ").replace("\\\\", " ")
    return " ".join(unicodedata.normalize("NFKC", html.unescape(text)).split())


def edit_distance(reference, prediction):
    """Exact character Levenshtein distance using bit vectors; no new dependency."""
    if not reference:
        return len(prediction)
    masks = {}
    for index, char in enumerate(reference):
        masks[char] = masks.get(char, 0) | (1 << index)
    positive = ~0
    negative = 0
    distance = len(reference)
    last = 1 << (len(reference) - 1)
    for char in prediction:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        distance += bool(plus & last) - bool(minus & last)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = minus | ~(vertical | plus)
        negative = plus & vertical
    return distance


def character_error_rate(reference, prediction):
    return edit_distance(reference, prediction) / max(1, len(reference))


def components(text, row):
    expected = [
        (e["tooth"], e["procedure"], e["surface"])
        for e in row["record"]["odontogram"]["annotations"]
    ]
    predicted = extracted(text)

    def atoms(events):
        return [
            (tooth, code)
            for tooth, procedure, surfaces in events
            if procedure == "filling"
            for code in surfaces
            if code in ORDER
        ]

    cer = character_error_rate(visible_text(row["target"]), visible_text(text))
    patient = row["record"]["patient"]
    visit = row["record"]["visit"]
    medical = row["record"]["medical_history"]
    fields = dict(
        name=patient["name"],
        patient_id=patient["patient_id"],
        dob=patient["dob"],
        phone=patient["phone"],
        address=patient["address"],
        visit_date=visit["date"],
        allergies=medical["allergies"],
        medications=medical["medications"],
    )
    plain = clean(text)
    field_accuracy = sum(clean(v) in plain for v in fields.values()) / len(fields)
    parsed = list(table_rows(text))
    history = row["record"]["treatment_history"]
    history_accuracy = (
        sum(
            any(
                c[:4] == [clean(str(e[k])) for k in ("date", "tooth", "procedure", "fee")]
                for c in parsed
            )
            for e in history
        )
        / len(history)
        if history
        else 1.0
    )
    chart_exact = compare(expected, predicted)["exact"]
    return {
        "page_exact": float(chart_exact and cer == 0.0),
        "chart_exact": float(chart_exact),
        "text_cer": cer,
        "tooth_procedure_f1": compare([e[:2] for e in expected], [e[:2] for e in predicted])["f1"],
        "surface_combination_f1": compare(
            [e for e in expected if e[1] == "filling"], [e for e in predicted if e[1] == "filling"]
        )["f1"],
        "surface_atom_f1": compare(atoms(expected), atoms(predicted))["f1"],
        "patient_fields": field_accuracy,
        "history_rows": history_accuracy,
    }


def exact_page_reward(prediction, reference):
    scores = components(prediction, reference)
    return float(scores["chart_exact"] == 1.0 and scores["text_cer"] == 0.0)


def successful_answer(answer, row):
    # Natural EOS completion is required in reported sampled accuracy.
    return bool(answer["ended"] and exact_page_reward(answer["text"], row))
