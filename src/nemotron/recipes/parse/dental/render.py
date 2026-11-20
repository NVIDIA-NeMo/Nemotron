"""Original fictional records and artwork for the paper-ledger form family."""

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
PROCEDURES = ["filling", "crown", "extraction"]
INK = "#264f49"
ANTERIOR = set(range(6, 12)) | set(range(22, 28))


@lru_cache(maxsize=32)
def font(size, serif=False):
    name = "DejaVuSerif.ttf" if serif else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path(__file__).parent / "fonts" / name), size, layout_engine=ImageFont.Layout.RAQM
    )


def tooth(draw, cx, cy, number, upper, scale=1):
    anterior = number in range(6, 12) or number in range(22, 28)
    width = (
        21 if anterior else (29 if number in [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32] else 25)
    )
    top, bottom = cy - 16, cy + 16
    if upper:
        if anterior:
            draw.polygon(
                [
                    (cx - width / 2, top + 3),
                    (cx - 6, top - 39),
                    (cx, top - 47),
                    (cx + 7, top - 35),
                    (cx + width / 2, top + 3),
                ],
                fill="white",
                outline=INK,
                width=1,
            )
        else:
            draw.polygon(
                [
                    (cx - width / 2, top + 2),
                    (cx - width / 2 + 1, top - 36),
                    (cx - 7, top - 42),
                    (cx - 2, top - 14),
                    (cx + 5, top - 41),
                    (cx + 11, top - 39),
                    (cx + width / 2, top + 2),
                ],
                fill="white",
                outline=INK,
                width=1,
            )
    else:
        if anterior:
            draw.polygon(
                [
                    (cx - width / 2, bottom - 3),
                    (cx - 6, bottom + 39),
                    (cx, bottom + 47),
                    (cx + 7, bottom + 35),
                    (cx + width / 2, bottom - 3),
                ],
                fill="white",
                outline=INK,
                width=1,
            )
        else:
            draw.polygon(
                [
                    (cx - width / 2, bottom - 2),
                    (cx - width / 2 + 1, bottom + 36),
                    (cx - 7, bottom + 42),
                    (cx - 2, bottom + 14),
                    (cx + 5, bottom + 41),
                    (cx + 11, bottom + 39),
                    (cx + width / 2, bottom - 2),
                ],
                fill="white",
                outline=INK,
                width=1,
            )
    draw.rounded_rectangle(
        (cx - width / 2, top, cx + width / 2, bottom), radius=7, fill="white", outline=INK, width=1
    )
    draw.ellipse((cx - 6, cy - 7, cx + 6, cy + 7), outline="#b5c6bd", width=1)
    if not anterior:
        draw.line((cx - width / 2 + 3, cy, cx + width / 2 - 3, cy), fill="#b5c6bd", width=1)
        draw.line((cx, top + 3, cx, bottom - 3), fill="#b5c6bd", width=1)
    return [int(cx - 20), int(cy - 23), int(cx + 20), int(cy + 23)]


def patient_record(rng, pair, split):
    first = rng.choice(
        [
            "Mara",
            "Evan",
            "Nora",
            "Lina",
            "Owen",
            "Elio",
            "Mira",
            "Arlo",
            "Tessa",
            "Finn",
            "Sora",
            "Vera",
        ]
    )
    last = rng.choice(
        [
            "Lintell",
            "Veld",
            "Arden",
            "Neri",
            "Halden",
            "Kellis",
            "Rell",
            "Tavin",
            "Vale",
            "Maren",
            "Selden",
            "Fenn",
        ]
    )
    patient = dict(
        name=first + " " + last,
        patient_id=f"DEMO-{rng.randrange(10000, 99999)}",
        dob=f"{rng.randrange(1965, 2005)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
        phone=f"202-555-01{rng.randrange(100):02d}",
        address=f"{rng.randrange(10, 990)} Example Avenue, Demo City",
    )
    visit = dict(
        date=f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
        provider=rng.choice(["Dr Vale", "Dr Neri", "Dr Arden"]),
    )
    complete = rng.sample(range(1, 33), 3)
    remaining = [t for t in range(1, 33) if t not in complete]
    planned = rng.sample(remaining, rng.randrange(2, 6))
    events = [dict(tooth=t, procedure=rng.choice(PROCEDURES), status="completed") for t in complete]
    events += [dict(tooth=t, procedure=rng.choice(PROCEDURES), status="planned") for t in planned]
    history = [
        dict(
            date=f"2025-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
            tooth=e["tooth"],
            procedure=e["procedure"],
            fee=f"{rng.randrange(10, 75) * 10}.00",
        )
        for e in events[:3]
    ]
    medical = dict(
        allergies=rng.choice(["None recorded", "Latex", "Penicillin"]),
        medications=rng.choice(["None listed", "See intake notes"]),
    )
    notes = rng.choice(
        [
            "Review diagram at the next visit.",
            "Patient prefers morning appointments.",
            "Confirm the charted plan at follow-up.",
        ]
    )
    appointment = dict(
        date=str(date.fromisoformat(visit["date"]) + timedelta(days=14)),
        provider=visit["provider"],
        preference=rng.choice(["Morning", "Afternoon", "Any time"]),
    )
    return dict(
        patient=patient,
        visit=visit,
        medical_history=medical,
        treatment_history=history,
        odontogram=dict(
            numbering="universal", annotations=sorted(events, key=lambda e: e["tooth"])
        ),
        next_appointment=appointment,
        notes=notes,
    )


def regions(cx, cy, t, r=21):
    q = r * 0.42
    left = "D" if t <= 8 or t >= 25 else "M"
    right = "M" if left == "D" else "D"
    outer = "F" if t in ANTERIOR else "B"
    top = outer if t <= 16 else "L"
    bottom = "L" if t <= 16 else outer
    return {
        left: [(cx - r, cy - r), (cx - q, cy - q), (cx - q, cy + q), (cx - r, cy + r)],
        right: [(cx + r, cy - r), (cx + q, cy - q), (cx + q, cy + q), (cx + r, cy + r)],
        top: [(cx - r, cy - r), (cx + r, cy - r), (cx + q, cy - q), (cx - q, cy - q)],
        bottom: [(cx - r, cy + r), (cx + r, cy + r), (cx + q, cy + q), (cx - q, cy + q)],
        ("I" if t in ANTERIOR else "O"): [
            (cx - q, cy - q),
            (cx + q, cy - q),
            (cx + q, cy + q),
            (cx - q, cy + q),
        ],
    }


def crown_view(image, cx, cy, t, event, ink, color):
    layer = Image.new("RGB", (56, 56), "white")
    draw = ImageDraw.Draw(layer)
    mask = Image.new("L", (56, 56), 0)
    md = ImageDraw.Draw(mask)
    anterior = t in ANTERIOR
    molar = t in [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32]
    width = 36 if anterior else 48 if molar else 42
    box = (28 - width / 2, 6, 28 + width / 2, 50)
    md.rounded_rectangle(box, radius=12, fill=255)
    for code, poly in regions(28, 28, t, r=22).items():
        marked = event and event["procedure"] == "filling" and code in event["surface"]
        draw.polygon(poly, fill=color if marked else "white", outline=ink, width=1)
    white = Image.new("RGB", (56, 56), "white")
    white.paste(layer, (0, 0), mask)
    d = ImageDraw.Draw(white)
    d.rounded_rectangle(box, radius=12, outline=ink, width=1)
    if event and event["procedure"] == "crown":
        d.rounded_rectangle(box, radius=12, outline=color, width=3)
    elif event and event["procedure"] == "extraction":
        d.line((7, 6, 49, 50), fill=color, width=3)
        d.line((49, 6, 7, 50), fill=color, width=3)
    image.paste(white, (round(cx - 28), round(cy - 28)))


def draw_chart(image, record, chart_bbox):
    d = ImageDraw.Draw(image)
    x, y, x2, y2 = chart_bbox
    cw = x2 - x
    ch = y2 - y
    ink = "#294b42"
    color = record["palette"]["planned"]
    d.rectangle((x, y, x2, y2), fill="white", outline=ink, width=1)
    d.rectangle((x, y, x2, y + 32), fill="#edf3ee", outline=ink)
    d.text((x + 10, y + 5), "ODONTOGRAM / UNIVERSAL ADULT TEETH 1-32", font=font(16), fill=ink)
    d.text((x + 14, y + 43), "All colored marks: PLANNED procedures", font=font(15), fill=ink)

    def text(px, py, value, size=12):
        d.text((px, py), str(value), font=font(size), fill=ink)

    text(
        x + 14,
        y + 68,
        "Shaded surfaces: filling   |   Cap outline: crown   |   X: extraction   |   Root lines: root canal",
    )
    text(
        x + 14,
        y + 92,
        "M: toward midline   D: away from midline   B/F: cheek/lip   L: tongue   O/I: chewing/biting",
    )
    text(x + 14, y + 116, "Each jaw shows tooth outlines and a separate chewing-surface view.")
    groups = [list(range(1, 17)), list(range(32, 16, -1))]
    events = {e["tooth"]: e for e in record["odontogram"]["annotations"]}
    boxes = {}
    for row, ids in enumerate(groups):
        for j, t in enumerate(ids):
            upper = t <= 16
            cx = x + 32 + j * (cw - 64) / 15
            curve = 0
            cy = y + 225 - curve if upper else y + ch - 88 + curve
            occlusal = y + 292 - curve if upper else y + ch - 142 + curve
            event = events.get(t)
            tooth(d, cx, cy, t, upper)
            if event:
                procedure = event["procedure"]
                if procedure == "filling" and ("B" in event["surface"] or "F" in event["surface"]):
                    d.rounded_rectangle((cx - 9, cy - 12, cx + 9, cy + 12), radius=5, fill=color)
                elif procedure == "crown":
                    d.rounded_rectangle(
                        (cx - 17, cy - 18, cx + 17, cy + 18), radius=7, outline=color, width=3
                    )
                elif procedure == "extraction":
                    top, bottom = (cy - 61, cy + 19) if upper else (cy - 19, cy + 61)
                    d.line((cx - 17, top, cx + 17, bottom), fill=color, width=3)
                    d.line((cx + 17, top, cx - 17, bottom), fill=color, width=3)
                elif procedure == "root canal":
                    direction = -1 if upper else 1
                    d.line((cx, cy - direction * 10, cx, cy + direction * 18), fill=color, width=3)
                    tips = [0] if t in ANTERIOR else [-8, 8]
                    for tip in tips:
                        d.line(
                            (cx, cy + direction * 8, cx + tip, cy + direction * 51),
                            fill=color,
                            width=3,
                        )
            crown_view(image, cx, occlusal, t, event, ink, color)
            text(cx - (5 if t < 10 else 9), cy + 27 if upper else occlusal + 24, t, 14)
            boxes[str(t)] = [
                round(cx - 26),
                round(min(cy - 64, occlusal - 28)),
                round(cx + 26),
                round(max(cy + 64, occlusal + 28)),
            ]
    text(x + 14, y + ch * 0.64, "Patient RIGHT", 12)
    text(x2 - 130, y + ch * 0.64, "Patient LEFT", 12)
    return boxes


def chart_payload(data):
    rows = [["Tooth", "Procedure", "Surfaces"]] + [
        [e["tooth"], e["procedure"], e["surface"]] for e in data["odontogram"]["annotations"]
    ]
    return (
        '<chart type="odontogram">\ntitle: ODONTOGRAM / UNIVERSAL ADULT TEETH 1-32'
        "\nlegend[color]: items=All colored marks: planned procedures"
        "\nlegend[symbol]: items=shaded surfaces: filling ; cap outline: crown ; X: extraction ; root lines: root canal"
        "\nnote: Patient RIGHT / LEFT. M: toward midline; D: away from midline; B/F: cheek/lip; L: tongue; O/I: chewing/biting. Surfaces apply to fillings; - means whole-tooth procedure. Unmarked teeth omitted."
        "\n\n| "
        + " | ".join(rows[0])
        + " |\n| --- | --- | --- |"
        + "".join("\n| " + " | ".join(map(str, r)) + " |" for r in rows[1:])
        + "\n</chart>"
    )


def point(x, y, width, height):
    return "".join(
        "<" + a + "_" + str(round(round(v / s * 1024) / 1024, 4)) + ">"
        for a, v, s in [("x", x, width), ("y", y, height)]
    )


def latex(rows):
    return (
        "\\begin{tabular}{"
        + "l" * len(rows[0])
        + "}\n"
        + "\n".join(" & ".join(map(str, row)) + r"\\" for row in rows)
        + "\n\\end{tabular}"
    )


def render_page(data):
    w, h = 1400, 1100
    ink = "#294b42"
    pale = "#edf3ee"
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    blocks = []

    def text(x, y, value, size=18, fill=ink):
        d.text((x, y), str(value), font=font(size), fill=fill)

    def block(box, value, kind="Text", size=18):
        x, y, x2, y2 = box
        for i, line in enumerate(value.split("\n")):
            text(x, y + i * (size + 7), line, size)
        blocks.append((box, value, kind))

    def panel(box, title):
        x, y, x2, y2 = box
        d.rectangle(box, outline=ink, width=1)
        d.rectangle((x, y, x2, y + 32), fill=pale)
        block((x + 10, y + 5, x2 - 8, y + 29), title, "Section-header", 16)

    block((30, 22, w - 30, 58), "SYNTHETIC DENTAL RECORD", "Page-header", 27)
    p = data["patient"]
    v = data["visit"]
    m = data["medical_history"]
    a = data["next_appointment"]
    # Patient header.
    block(
        (30, 80, w - 30, 175),
        f"Patient: {p['name']}    Record: {p['patient_id']}    DOB: {p['dob']}\nPhone: {p['phone']}    Visit: {v['date']}    Clinician: {v['provider']}\nAddress: {p['address']}",
        size=17,
    )
    cb = (30, 200, 970, 735)
    med = (995, 200, w - 30, 465)
    hist = (30, 760, 970, 1040)
    notes = (995, 490, w - 30, 1040)
    boxes = draw_chart(im, data, cb)
    panel(med, "MEDICAL HISTORY")
    mx, my, mx2, my2 = med
    block(
        (mx + 10, my + 48, mx2 - 8, my2 - 8),
        f"Allergies: {m['allergies']}\nMedications: {m['medications']}",
        size=15,
    )
    panel(hist, "PRIOR VISIT LEDGER (SEPARATE FROM CHART)")
    hx, hy, hx2, hy2 = hist
    rows = [["Date", "Tooth", "Procedure", "Fee"]] + [
        [e["date"], e["tooth"], e["procedure"], e["fee"]] for e in data["treatment_history"]
    ]
    tablebox = (hx + 8, hy + 39, hx2 - 8, hy2 - 12)
    rh = min(43, (hy2 - hy - 47) / 4)
    colwidth = (hx2 - hx - 16) / 4
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            xx = hx + 8 + j * colwidth
            yy = hy + 39 + i * rh
            d.rectangle((xx, yy, xx + colwidth, yy + rh), outline="#9aad9f", width=1)
            text(xx + 4, yy + 6, value, 12 if hx2 - hx < 600 else 16)
    blocks.append((tablebox, latex(rows), "Table"))
    panel(notes, "FOLLOW-UP")
    nx, ny, nx2, ny2 = notes
    lines = [
        f"Next visit: {a['date']}",
        f"Clinician: {a['provider']}",
        f"Preference: {a['preference']}",
    ]
    words = data["notes"].split()
    line = ""
    limit = max(25, int((nx2 - nx - 20) / 8.2))
    for word in words:
        if len(line + " " + word) > limit:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    # Wide encounter footer uses two lines, keeping all text in its box.
    if ny2 - ny < 150:
        lines = [" | ".join(lines[:3]), " ".join(lines[3:])]
    block((nx + 10, ny + 44, nx2 - 10, ny2 - 10), "\n".join(lines), size=15)
    block(
        (30, h - 35, w - 30, h - 12),
        "Fictional patient. Original synthetic artwork. Demonstration only.",
        "Page-footer",
        12,
    )
    blocks.append((cb, chart_payload(data), "Chart"))
    blocks.sort(key=lambda b: (b[0][1], b[0][0]))
    native = "\n\n".join(
        point(b[0], b[1], w, h) + value + point(b[2], b[3], w, h) + "<class_" + kind + ">"
        for b, value, kind in blocks
    )
    return (
        im,
        native,
        dict(
            width=w,
            height=h,
            chart_bbox=list(cb),
            mark_boxes=boxes,
        ),
    )
