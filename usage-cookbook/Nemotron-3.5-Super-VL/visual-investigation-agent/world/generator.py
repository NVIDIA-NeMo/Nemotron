# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate help-desk worlds from a seed, to test whether the agent generalizes beyond the Fernledger months.

A world name is g-<family>-<seed>, for example g-price-1003. The seed fixes everything: the company, its
cloud provider, the model catalog (names, prices, tokens, quality), the ticket types, the months, the volume,
which facts live in the database and which only in documents, the report's page order, the scan quality, the
decoy changes, and what really happened. Generation retries until the spend change is clearly visible.

Families (what really happened):
- model_switch: production moved to a reasoning model that is cheaper per token but uses more tokens
- price: the provider raised the production model's list price
- volume: tickets grew after a launch
- mix: the ticket mix shifted toward costlier ticket types
- volume_price: tickets grew and the price rose
- none: nothing material changed (a few percent of noise and growth)
- retry_bug: a gateway change made requests and input tokens per ticket jump (sealed test set only)

Seed ranges keep the splits apart: dev 1000-1999, validation 2000-2999, test 5000-5999.
"""

import datetime as dt
import random
import re

FAMILIES = ["model_switch", "price", "volume", "mix", "volume_price", "none", "retry_bug"]
SPLITS = {"dev": range(1000, 2000), "validation": range(2000, 3000), "test": range(5000, 6000)}
SEALED_FAMILIES = {"retry_bug"}

COMPANIES = ["Brightwell", "Corvane", "Halcyra", "Quillon", "Teverly", "Pinebrook", "Lumenhall", "Rivelle", "Sablecrest",
             "Tamsin", "Wexford", "Zephyrine", "Ardenvale", "Calloway", "Dorrance", "Elmstead", "Fairhollow", "Glenmore",
             "Harrowgate", "Iverson", "Kestrell", "Lindqvist", "Marrowby", "Northam", "Oakhurst", "Prescott"]
PROVIDERS = ["Orrin AI", "Bastion Compute", "Northgate Models", "Cindral", "Halden Inference",
             "Mirabel Cloud", "Stratum AI"]
FAMILY_NAMES = ["Tallis", "Merrow", "Coralia", "Brannoch", "Ilex", "Sorrel", "Kestrel", "Vantor",
                "Juniper", "Halvard", "Pellam", "Rowan", "Sephra", "Thorne"]
SMALL_NAMES = ["Wren", "Finch", "Plover", "Linnet", "Siskin"]
TICKET_POOL = [  # name, short label, difficulty 0 (easy) to 1 (hard)
    ("Account & password", "Account", 0.1), ("Billing", "Billing", 0.25), ("Order status", "Orders", 0.05),
    ("Returns & refunds", "Returns", 0.3), ("Shipping & delivery", "Shipping", 0.2), ("Plan changes", "Plans", 0.35),
    ("Device setup", "Setup", 0.6), ("Technical troubleshooting", "Technical", 0.85), ("Integrations & API", "Integrations", 0.95),
    ("Outage reports", "Outages", 0.7), ("Data export", "Export", 0.65), ("Security & access", "Security", 0.75)]
COLORS = ["#9CA3AF", "#10B981", "#8E6BD1", "#4C7BD9", "#E8A33D", "#EF4444"]
DECOYS = ["Help-center search index rebuilt (no content change)", "Help-center articles refreshed for policy updates",
          "Help-desk reply tone guidelines updated (prompt wording only)", "Dashboard color theme updated",
          "On-call rotation for the support team changed", "Customer satisfaction survey wording updated",
          "Ticket auto-close timer changed from 72 to 96 hours", "Support macros reorganized (no customer-facing change)",
          "Status page provider migrated", "Agent handoff message reworded"]
LAUNCHES = ["In-app help chat enabled for all mobile app users", "Help widget added to the checkout page",
            "Email support routed to the AI agent", "AI agent enabled for the EU region", "Proactive order-update messages launched"]
MIX_CAUSES = [  # (change-log title, ticket types it plausibly drives, direction of their share)
    ("Self-service password reset launched in the app", ["Account & password"], "down"),
    ("Firmware 5.2 rolled out to all devices", ["Device setup", "Technical troubleshooting"], "up"),
    ("Public API v3 launched (v2 endpoints deprecated)", ["Integrations & API", "Data export"], "up"),
    ("Regional network incident (status page: degraded service)", ["Outage reports", "Technical troubleshooting"], "up"),
    ("Single sign-on enforced for all business accounts", ["Security & access", "Account & password"], "up"),
    ("Order tracking page added to the app", ["Order status", "Shipping & delivery"], "down")]
GENERATOR_VERSION = 2  # 2: mix shifts are driven by a change that plausibly affects the ticket types that shift
RETRY_CAUSES = ["Gateway retry policy changed: max retries 1 -> 4", "Tool-call timeout lowered from 30s to 5s",
                "Conversation memory window raised from 6 to 20 turns"]
SCAN_LEVELS = [  # scan quality: 0 like the Fernledger scan, 1 a worse copy, 2 a poor copy
    {"rotate": 0.6, "blur": 0.2, "contrast": 0.94, "quality": 80},
    {"rotate": 1.0, "blur": 0.4, "contrast": 0.9, "quality": 65},
    {"rotate": 1.6, "blur": 0.6, "contrast": 0.86, "quality": 55}]
STRESS_SCAN_LEVELS = SCAN_LEVELS + [  # 3 and 4: badly skewed and blurred copies, only on request (a -s3 or -s4 world name)
    {"rotate": 2.5, "blur": 0.9, "contrast": 0.8, "quality": 45},
    {"rotate": 3.5, "blur": 1.2, "contrast": 0.75, "quality": 35}]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def split_of(seed):
    return next((name for name, seeds in SPLITS.items() if seed in seeds), None)


def name_for(family, seed, scan=None):
    return f"g-{family}-{seed}" + (f"-s{scan}" if scan is not None else "")


def from_name(name):
    """g-volume_price-2041 -> generate("volume_price", 2041); a suffix -s3 forces scan quality level 3."""
    scan = None
    if re.search(r"-s\d$", name):
        name, scan = name[:-3], int(name[-1])
    family, seed = name[2:name.rindex("-")], int(name[name.rindex("-") + 1:])
    if scan is None:
        return generate(family, seed)
    spec = generate(family, seed)
    return {**spec, "name": name_for(family, seed, scan), "scan": STRESS_SCAN_LEVELS[scan], "scan_level": scan}


def _round_price(v):
    return round(v, 3) if v < 1 else round(v, 2)


def _iso(d):
    return d.isoformat()


def _long(d):
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def generate(family, seed, overrides=None):
    """The world spec for (family, seed). Deterministic; retries internally until the change is visible."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    for attempt in range(200):
        spec = _attempt(family, seed, attempt, overrides or {})
        if spec is not None:
            return spec
    raise RuntimeError(f"could not generate a world for {family} {seed}")


def _attempt(family, seed, attempt, overrides):
    r = random.Random(f"{family}:{seed}:{attempt}")
    company = r.choice(COMPANIES)
    provider = r.choice(PROVIDERS)
    month = r.randint(3, 11)
    first = dt.date(2026, month - 1, 1)
    period = (dt.date(2026, month, 1), dt.date(2026, month, 21))
    compare = (first, dt.date(2026, month - 1, 21))
    last = period[1]

    # Ticket types: some easy, some hard.
    n_types = r.randint(3, 5)
    pool = sorted(TICKET_POOL, key=lambda t: t[2])
    easy, hard = [t for t in pool if t[2] < 0.5], [t for t in pool if t[2] >= 0.5]
    n_hard = r.randint(1, max(1, n_types - 2))
    types = sorted(r.sample(easy, n_types - n_hard) + r.sample(hard, n_hard), key=lambda t: t[2])
    mix_cause = None
    if family == "mix":  # pick the cause first, then make sure the world has the ticket type it drives
        mix_cause = r.choice(MIX_CAUSES)
        driven = next(t for t in TICKET_POOL if t[0] == mix_cause[1][0])
        if driven not in types:
            same = [t for t in types if (t[2] < 0.5) == (driven[2] < 0.5)] or types
            types[types.index(r.choice(same))] = driven
            types = sorted(types, key=lambda t: t[2])
        others = [t for t in types if t[0] not in mix_cause[1]]
        if mix_cause[2] == "up" and not any(t[2] < 0.5 for t in others):
            types[types.index(others[0])] = r.choice([t for t in easy if t not in types])
            types = sorted(types, key=lambda t: t[2])
    names = [t[0] for t in types]
    difficulty = {t[0]: t[2] for t in types}
    weights = [r.uniform(0.6, 1.6) * (1.3 if difficulty[n] < 0.5 else 1.0) for n in names]
    share = {n: round(w / sum(weights), 3) for n, w in zip(names, weights)}
    share[names[-1]] = round(1 - sum(share[n] for n in names[:-1]), 3)

    # Model catalog: small, lite, thinking, standard (production and quality baseline), pro.
    fam_a, fam_b = r.sample(FAMILY_NAMES, 2)
    gen = r.randint(2, 6)
    standard, lite, pro = f"{fam_a} {gen}", f"{fam_a} {gen} Lite", f"{fam_a} {gen} Pro"
    thinking, small = f"{fam_b} {r.randint(1, 4)} Thinking", f"{r.choice(SMALL_NAMES)}-{r.randint(2, 4)}-{r.choice([7, 8, 9])}B-Instruct"
    order = [small, lite, thinking, standard, pro]
    scale = r.uniform(0.7, 1.4)  # keeps every model on the Models tab's price axis
    out_ratio = r.choice([3, 4, 5])
    base_in = {small: 0.065, lite: 0.17, thinking: 0.475, standard: 1.9, pro: 5.7}
    prices = {}
    for m in order:
        inp = _round_price(base_in[m] * scale * r.uniform(0.8, 1.25))
        prices[m] = {"input": inp, "output": _round_price(inp * out_ratio)}

    tokens, pass_pct, requests, eval_tickets = {m: {} for m in order}, {m: {} for m in order}, {}, {}
    reason_mult, input_mult = r.uniform(10, 24), r.uniform(1.6, 3.0)
    lite_ok = {n for n in names if difficulty[n] < 0.5 and r.random() < 0.8}
    if not lite_ok:
        lite_ok = {names[0]}
    for n in names:
        d = difficulty[n]
        inp = round(r.uniform(7_000, 12_000) + d * r.uniform(12_000, 24_000), -2)
        ans = round(r.uniform(450, 800) + d * r.uniform(900, 1_700), -1)
        rsn = round(r.uniform(60, 140) + d * r.uniform(120, 260), -1)
        requests[n] = r.randint(5, 8) + round(d * r.randint(2, 4))
        eval_tickets[n] = int(round(r.uniform(1_200, 3_200), -2))
        tokens[standard][n] = [int(inp), int(ans), int(rsn)]
        tokens[lite][n] = [int(inp), int(round(ans * 0.95)), 0]
        tokens[small][n] = [int(inp), int(round(ans * 0.95)), 0]
        tokens[pro][n] = [int(inp), int(round(ans * 1.1)), int(rsn * 3)]
        tokens[thinking][n] = [int(round(inp * input_mult)), int(round(ans * 1.05)), int(round(ans * reason_mult))]
        base = round(r.uniform(88, 95) - d * r.uniform(12, 22), 1)
        pass_pct[standard][n] = base
        pass_pct[lite][n] = round(base - (r.uniform(0.3, 1.9) if n in lite_ok else r.uniform(4, 22)), 1)
        pass_pct[small][n] = round(max(45.0, base - r.uniform(9, 25)), 1)
        pass_pct[thinking][n] = round(base - (r.uniform(0.2, 1.8) if d < 0.5 else r.uniform(2.5, 8)), 1)
        pass_pct[pro][n] = round(min(98.0, base + r.uniform(1, 7)), 1)
    models = {m: {"slug": m.lower().replace(" ", "-"), "tokens": tokens[m], "pass_pct": pass_pct[m], "color": COLORS[i]}
              for i, m in enumerate(order)}

    # Production: the standard model everywhere, or already routing some easy types to the lite model.
    production = {n: standard for n in names}
    pre_routed = r.random() < 0.25 and family != "model_switch"
    if pre_routed:
        for n in lite_ok:
            production[n] = lite

    # Change log: decoys, one of them close to the real change.
    day_of = lambda m, d: dt.date(2026, m, d)
    change_day = r.randint(3, 12)
    change_date = day_of(month, change_day)
    ids = iter(sorted(r.sample(range(4100, 6900), 12)))
    changes, causal, events = [], [], []
    for title in r.sample(DECOYS, r.randint(2, 4)):
        d = day_of(month - 1, r.randint(5, 26)) if r.random() < 0.5 else day_of(month, r.randint(1, 20))
        changes.append([f"CHG-{next(ids)}", _iso(d), f"{r.randint(8, 22):02d}:{r.choice(['00', '15', '30', '45'])}", title])
    near = change_date + dt.timedelta(days=r.choice([-1, 1]))
    changes.append([f"CHG-{next(ids)}", _iso(near), "15:00", r.choice([t for t in DECOYS if t not in [c[3] for c in changes]])])
    decoys = [c[0] for c in changes]

    base_day = day_of(month - 1, 1)
    k = r.choice([2, 3, 4])
    received = base_day - dt.timedelta(days=r.randint(5, 12))
    schedules = [{"id": f"price-schedule-2026-{month - 1:02d}", "effective": _iso(base_day), "effective_text": _long(base_day),
                  "received": f"{MONTHS[received.month - 1][:3].upper()} {received.day} {received.year}",
                  "supersedes": _long(dt.date(2026 if month - 1 - k > 0 else 2025, (month - 2 - k) % 12 + 1, 1)), "prices": prices,
                  "file": f"price-schedule-2026-{month - 1:02d}.jpg",
                  "title": f"{provider} model price schedule, effective {MONTHS[base_day.month - 1][:3]} 1, 2026 "
                           "(scanned copy received by Finance)"}]
    volume_change = 1.0 + r.uniform(-0.01, 0.04)

    def price_rise(effective):
        factor = r.uniform(1.2, 1.4)
        new = {m: dict(p) for m, p in prices.items()}
        new[standard] = {"input": _round_price(prices[standard]["input"] * factor),
                         "output": _round_price(prices[standard]["input"] * factor * out_ratio)}
        got = effective - dt.timedelta(days=r.randint(4, 10))
        doc = {"id": f"price-schedule-2026-{month:02d}", "effective": _iso(effective), "effective_text": _long(effective),
               "received": f"{MONTHS[got.month - 1][:3].upper()} {got.day} {got.year}", "supersedes": schedules[0]["effective_text"],
               "prices": new, "file": f"price-schedule-2026-{month:02d}.jpg"}
        doc["title"] = f"{provider} model price schedule, effective {MONTHS[effective.month - 1][:3]} {effective.day}, 2026 (scanned copy received by Finance)"
        schedules.append(doc)
        return doc["id"]

    causal_docs = []
    if family == "model_switch":
        cid = [f"CHG-{next(ids)}" for _ in range(3)]
        d0 = change_date
        events.append({"type": "model_switch", "from": standard, "to": thinking, "complete": _iso(d0 + dt.timedelta(days=2)),
                       "rollout": {_iso(d0): 0.25, _iso(d0 + dt.timedelta(days=1)): 0.6}})
        changes += [[cid[0], _iso(d0), "16:00", f"Default help-desk model: {standard} -> {thinking}, rollout to 25% of tickets"],
                    [cid[1], _iso(d0 + dt.timedelta(days=1)), "16:00", f"{thinking} rollout increased to 60% of tickets"],
                    [cid[2], _iso(d0 + dt.timedelta(days=2)), "16:00", f"{thinking} rollout completed (100% of tickets)"]]
        causal += cid
    if family in ("volume", "volume_price"):
        cid = f"CHG-{next(ids)}"
        events.append({"type": "volume", "start": _iso(change_date), "multiplier": r.uniform(1.25, 1.55)})
        changes.append([cid, _iso(change_date - dt.timedelta(days=1)), "22:00", r.choice(LAUNCHES)])
        causal.append(cid)
    if family in ("price", "volume_price"):
        effective = day_of(month, 1) if r.random() < 0.6 or family == "volume_price" else day_of(month, r.randint(5, 12))
        causal_docs.append(price_rise(effective))
    if family == "mix":
        title, affected, direction = mix_cause
        cid = f"CHG-{next(ids)}"
        new_share = dict(share)
        driven = [n for n in names if n in affected]
        rest = [n for n in names if n not in affected]
        if direction == "up":  # the driven types take share from the cheaper types
            donors = [n for n in rest if difficulty[n] < 0.5] or rest
            moved = 0.0
            for n in donors:
                cut = new_share[n] * r.uniform(0.35, 0.55)
                new_share[n] -= cut
                moved += cut
            for n in driven:
                new_share[n] += moved / len(driven)
        else:  # the driven (cheap) types lose share to everything else
            moved = 0.0
            for n in driven:
                cut = new_share[n] * r.uniform(0.5, 0.7)
                new_share[n] -= cut
                moved += cut
            total = sum(new_share[n] for n in rest)
            for n in rest:
                new_share[n] += moved * new_share[n] / total
        events.append({"type": "mix_shift", "start": _iso(change_date), "share": {n: round(v, 4) for n, v in new_share.items()}})
        changes.append([cid, _iso(change_date - dt.timedelta(days=1)), "18:00", title])
        causal.append(cid)
    if family == "retry_bug":
        cid = f"CHG-{next(ids)}"
        hit = [n for n in names if difficulty[n] >= 0.5] or names[-1:]
        events.append({"type": "retry_bug", "start": _iso(change_date), "ticket_types": hit,
                       "input_factor": r.uniform(1.8, 2.6), "requests_factor": r.uniform(2.0, 3.2)})
        changes.append([cid, _iso(change_date), "14:30", r.choice(RETRY_CAUSES)])
        causal.append(cid)
    events.insert(0, {"type": "volume", "start": _iso(day_of(month, 1)), "multiplier": volume_change})
    changes.sort(key=lambda c: (c[1], c[2]))

    route_lines = "".join(f"  - ticket_type: \"{n}\"\n    model: {models[production[n]]['slug']}\n" for n in names if production[n] != standard)
    router = (f"# gateway router config for the help-desk agent (router/helpdesk.yaml)\nrouter: helpdesk-agent\n"
              f"default_model: {models[standard]['slug']}\n"
              + (f"routes:\n{route_lines}" if route_lines else "routes: []          # no per-ticket-type routes; every ticket uses default_model\n")
              + f"fallbacks:\n  - {models[pro]['slug']}\n")
    if family == "model_switch":
        router = router.replace(f"default_model: {models[standard]['slug']}", f"default_model: {models[thinking]['slug']}") \
                       .replace(f"  - {models[pro]['slug']}", f"  - {models[standard]['slug']}")

    eval_published = day_of(month - 1, 1) - dt.timedelta(days=r.randint(3, 20))
    level = r.choices([0, 1, 2], weights=[50, 35, 15])[0]
    scan = SCAN_LEVELS[level]
    spec = {
        "name": name_for(family, seed), "family": family, "seed": seed, "split": split_of(seed), "salt": f"{family}:{seed}:{attempt}",
        "generator": GENERATOR_VERSION,
        "company": company, "provider": provider, "eval_team": f"{company} Eval Team",
        "first_day": _iso(first), "last_day": _iso(last), "today": _iso(last + dt.timedelta(days=1)),
        "period": [_iso(period[0]), _iso(period[1])], "compare": [_iso(compare[0]), _iso(compare[1])],
        "models": models, "ticket_types": names, "short": {t[0]: t[1] for t in types}, "share": share,
        "requests_per_ticket": requests, "eval_tickets": eval_tickets,
        "weekday_tickets": int(round(r.uniform(40_000, 200_000), -3)), "weekend_factor": round(r.uniform(0.45, 0.7), 2),
        "production": production, "baseline_model": standard,
        "baseline_reason": f"{standard} is the baseline approved by the {MONTHS[(month - 4) % 12]} 2026 quality review.",
        "events": events, "changes": changes, "router": router, "price_schedules": schedules,
        "eval": {"in_db": r.random() < 0.5, "id": f"eval-report-2026-{eval_published.month:02d}",
                 "published": f"{MONTHS[eval_published.month - 1][:3]} {eval_published.day}, 2026",
                 "title": f"Help-desk offline eval v4: results by model and ticket type (Eval Team report export, "
                          f"{MONTHS[eval_published.month - 1][:3]} {eval_published.day}, 2026)",
                 "tokens_page": r.choice([1, 2]), "pointer_note": r.random() < 0.6,
                 "replayed": f"{MONTHS[(eval_published.month - 4) % 12]}–{MONTHS[(eval_published.month - 2) % 12]} "
                             f"{2026 if eval_published.month > 1 else 2025}"},
        "attach_overview": r.random() < 0.5, "request": None, "scan": scan, "scan_level": level,
        "truth": {"family": family, "causal_changes": causal, "causal_documents": causal_docs, "decoys": decoys,
                  "pre_routed": pre_routed},
    }
    for key, value in overrides.items():
        spec[key] = value
    return spec if _visible(spec) else None


def _visible(spec):
    """Keep worlds whose spend change matches their family: clear for a real cause, small for none."""
    from . import data

    s = spec
    days = data._days(dt.date.fromisoformat(s["first_day"]), dt.date.fromisoformat(s["last_day"]))
    from .spec import register

    register(s)
    try:
        totals = {"compare": 0.0, "period": 0.0}
        for day in days:
            iso = day.isoformat()
            rows = data.day_rows(s, day)
            spend = sum(r["input_usd"] + r["output_usd"] for r in rows)
            if s["compare"][0] <= iso <= s["compare"][1]:
                totals["compare"] += spend
            if s["period"][0] <= iso <= s["period"][1]:
                totals["period"] += spend
    finally:
        from .spec import _REGISTRY

        _REGISTRY.pop(s["name"], None)
    change = totals["period"] / totals["compare"] - 1
    if s["family"] == "none":
        return 0.0 <= change <= 0.05
    return 0.12 <= change <= 0.65


def pick(split, families, n, start=0):
    """n world names from a split, cycling through the families: deterministic, so every arm sees the same worlds."""
    seeds = list(SPLITS[split])[start:]
    return [name_for(families[i % len(families)], seeds[i]) for i in range(n)]
