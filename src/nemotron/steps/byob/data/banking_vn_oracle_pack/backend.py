"""Deterministic banking_vn backend.

Clock and seed arrive only via RunContext. Pipeline owns reset().
"""

from __future__ import annotations

import copy
import re
from typing import Any

_FIXTURE_SNAPSHOT: dict[str, Any] | None = None
_STATE: dict[str, Any] = {}
_CTX: Any = None
_COUNTERS: dict[str, int] = {"TRF": 0, "TXN": 0, "DSP": 0}

_TOOLS = [
    "get_account_balance",
    "get_card_limit",
    "get_transaction_status",
    "list_recent_transactions",
    "get_transfer_fee",
    "create_transfer",
    "get_vietqr_payment_status",
    "get_dispute_status",
    "create_dispute",
]

_BANK_CODE_RE = re.compile(r"^[0-9]{6}$")
_PREFIX = {
    "account_id": "ACC-",
    "card_id": "CARD-",
    "transaction_id": "TXN-",
    "payment_ref": "VQ-",
    "dispute_id": "DSP-",
}


def list_tools() -> list[str]:
    return list(_TOOLS)


def reset(*, ctx: Any, fixtures: dict | None = None) -> None:
    global _FIXTURE_SNAPSHOT, _STATE, _CTX, _COUNTERS
    if fixtures is not None:
        _FIXTURE_SNAPSHOT = copy.deepcopy(fixtures)
    if _FIXTURE_SNAPSHOT is None:
        raise RuntimeError("banking_vn reset requires fixtures on first call")
    _STATE = copy.deepcopy(_FIXTURE_SNAPSHOT)
    _CTX = ctx
    _COUNTERS = {"TRF": 0, "TXN": 0, "DSP": 0}


def get_state() -> dict:
    return copy.deepcopy(_STATE)


def call_tool(name: str, arguments: dict, *, ctx: Any) -> dict:
    global _CTX
    _CTX = ctx
    handlers = {
        "get_account_balance": _get_account_balance,
        "get_card_limit": _get_card_limit,
        "get_transaction_status": _get_transaction_status,
        "list_recent_transactions": _list_recent_transactions,
        "get_transfer_fee": _get_transfer_fee,
        "create_transfer": _create_transfer,
        "get_vietqr_payment_status": _get_vietqr_payment_status,
        "get_dispute_status": _get_dispute_status,
        "create_dispute": _create_dispute,
    }
    fn = handlers.get(name)
    if fn is None:
        return _err("invalid_argument", field="name", message=f"unknown tool {name!r}")
    return fn(arguments or {})


def _clock() -> str:
    value = getattr(_CTX, "clock", None)
    if value is None:
        return "2026-03-02T09:00:00+07:00"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _next_id(kind: str) -> str:
    _COUNTERS[kind] += 1
    n = _COUNTERS[kind]
    if kind == "TXN":
        return f"TXN-{900000 + n}"
    return f"{kind}-{n:06d}"


def _err(
    code: str,
    *,
    entity: str | None = None,
    id: str | None = None,
    field: str | None = None,
    message: str | None = None,
    detail: dict | None = None,
) -> dict:
    err: dict[str, Any] = {"code": code, "entity": entity, "id": id, "field": field}
    if message is not None:
        err["message"] = message
    if detail is not None:
        err["detail"] = detail
    return {"error": err}


def _require_str(arguments: dict, field: str, *, entity: str | None = None) -> str | dict:
    value = arguments.get(field)
    if not isinstance(value, str) or not value:
        return _err("invalid_argument", entity=entity, field=field, message=f"{field} must be a non-empty string")
    return value


def _check_prefix(value: str, field: str, entity: str) -> dict | None:
    prefix = _PREFIX[field]
    if not value.startswith(prefix):
        return _err(
            "invalid_argument",
            entity=entity,
            id=value,
            field=field,
            message=f"{field} must start with {prefix}",
        )
    return None


def _find(collection: str, key: str, value: str) -> dict | None:
    for row in _STATE.get(collection, []):
        if row.get(key) == value:
            return row
    return None


def _lookup_fee(rail: str, amount_vnd: int) -> dict | None:
    tiers = [t for t in _STATE.get("fee_schedule", []) if t.get("rail") == rail]
    tiers.sort(key=lambda t: int(t["min_amount_vnd"]))
    for i, tier in enumerate(tiers):
        lo = int(tier["min_amount_vnd"])
        hi = tier.get("max_amount_vnd")
        last = i == len(tiers) - 1
        if hi is None:
            if amount_vnd >= lo:
                return tier
        elif last:
            if lo <= amount_vnd <= int(hi):
                return tier
        elif lo <= amount_vnd < int(hi):
            return tier
    return None


def _rail_args(arguments: dict) -> tuple[str, str | None] | dict:
    rail = arguments.get("rail")
    if rail not in ("napas", "internal"):
        return _err("invalid_argument", field="rail", message="rail must be napas or internal")
    bank = arguments.get("to_bank_code", None)
    if rail == "napas":
        if bank is None:
            return _err("invalid_argument", field="to_bank_code", message="to_bank_code required for napas")
        if not isinstance(bank, str) or not _BANK_CODE_RE.fullmatch(bank):
            return _err("invalid_argument", field="to_bank_code", message="to_bank_code must be 6 digits")
        return rail, bank
    if bank is not None:
        return _err("invalid_argument", field="to_bank_code", message="to_bank_code must be absent for internal")
    return rail, None


def _txn_view(row: dict) -> dict:
    out = {
        "transaction_id": row["transaction_id"],
        "account_id": row["account_id"],
        "status": row["status"],
        "direction": row["direction"],
        "amount_vnd": row["amount_vnd"],
        "channel": row["channel"],
        "timestamp": row["timestamp"],
    }
    if row.get("counterparty_name") is not None:
        out["counterparty_name"] = row["counterparty_name"]
    return out


def _get_account_balance(arguments: dict) -> dict:
    account_id = _require_str(arguments, "account_id", entity="accounts")
    if isinstance(account_id, dict):
        return account_id
    bad = _check_prefix(account_id, "account_id", "accounts")
    if bad:
        return bad
    acc = _find("accounts", "account_id", account_id)
    if acc is None:
        return _err("not_found", entity="accounts", id=account_id, field="account_id")
    return {
        "account_id": acc["account_id"],
        "balance_vnd": acc["balance_vnd"],
        "currency": "VND",
        "account_type": acc["account_type"],
    }


def _get_card_limit(arguments: dict) -> dict:
    card_id = _require_str(arguments, "card_id", entity="cards")
    if isinstance(card_id, dict):
        return card_id
    bad = _check_prefix(card_id, "card_id", "cards")
    if bad:
        return bad
    card = _find("cards", "card_id", card_id)
    if card is None:
        return _err("not_found", entity="cards", id=card_id, field="card_id")
    return {
        "card_id": card["card_id"],
        "account_id": card["account_id"],
        "remaining_limit_vnd": card["remaining_limit_vnd"],
        "currency": "VND",
    }


def _get_transaction_status(arguments: dict) -> dict:
    txn_id = _require_str(arguments, "transaction_id", entity="transactions")
    if isinstance(txn_id, dict):
        return txn_id
    bad = _check_prefix(txn_id, "transaction_id", "transactions")
    if bad:
        return bad
    txn = _find("transactions", "transaction_id", txn_id)
    if txn is None:
        return _err("not_found", entity="transactions", id=txn_id, field="transaction_id")
    return _txn_view(txn)


def _list_recent_transactions(arguments: dict) -> dict:
    account_id = _require_str(arguments, "account_id", entity="accounts")
    if isinstance(account_id, dict):
        return account_id
    bad = _check_prefix(account_id, "account_id", "accounts")
    if bad:
        return bad
    limit = arguments.get("limit", 5)
    if not isinstance(limit, int) or limit < 1:
        return _err("invalid_argument", field="limit", message="limit must be a positive integer")
    status = arguments.get("status")
    if status is not None and status not in ("pending", "succeeded", "failed", "reversed"):
        return _err("invalid_argument", field="status", message="invalid status filter")
    rows = [t for t in _STATE.get("transactions", []) if t.get("account_id") == account_id]
    if status is not None:
        rows = [t for t in rows if t.get("status") == status]
    rows.sort(key=lambda t: (t["timestamp"], t["transaction_id"]), reverse=True)
    return {"account_id": account_id, "transactions": [_txn_view(t) for t in rows[:limit]]}


def _get_transfer_fee(arguments: dict) -> dict:
    from_id = _require_str(arguments, "from_account_id", entity="accounts")
    if isinstance(from_id, dict):
        return from_id
    to_num = _require_str(arguments, "to_account_number")
    if isinstance(to_num, dict):
        return to_num
    amount = arguments.get("amount_vnd")
    if not isinstance(amount, int) or amount <= 0:
        return _err("invalid_argument", field="amount_vnd", message="amount_vnd must be a positive integer")
    rail_pair = _rail_args(arguments)
    if isinstance(rail_pair, dict):
        return rail_pair
    rail, _bank = rail_pair
    if _find("accounts", "account_id", from_id) is None:
        return _err("not_found", entity="accounts", id=from_id, field="from_account_id")
    tier = _lookup_fee(rail, amount)
    if tier is None:
        return _err("invalid_argument", field="amount_vnd", message="no fee tier for amount")
    return {
        "fee_vnd": tier["fee_vnd"],
        "currency": "VND",
        "eta_hint": tier["eta_hint"],
        "rail": rail,
        "amount_vnd": amount,
    }


def _create_transfer(arguments: dict) -> dict:
    from_id = _require_str(arguments, "from_account_id", entity="accounts")
    if isinstance(from_id, dict):
        return from_id
    to_num = _require_str(arguments, "to_account_number")
    if isinstance(to_num, dict):
        return to_num
    amount = arguments.get("amount_vnd")
    if not isinstance(amount, int) or amount <= 0:
        return _err("invalid_argument", field="amount_vnd", message="amount_vnd must be a positive integer")
    rail_pair = _rail_args(arguments)
    if isinstance(rail_pair, dict):
        return rail_pair
    rail, bank = rail_pair
    confirm = arguments.get("confirm", False)
    if not isinstance(confirm, bool):
        return _err("invalid_argument", field="confirm", message="confirm must be a boolean")
    memo = arguments.get("memo")

    acc = _find("accounts", "account_id", from_id)
    if acc is None:
        return _err("not_found", entity="accounts", id=from_id, field="from_account_id")
    tier = _lookup_fee(rail, amount)
    if tier is None:
        return _err("invalid_argument", field="amount_vnd", message="no fee tier for amount")
    fee = int(tier["fee_vnd"])

    if not confirm:
        return {"status": "awaiting_confirmation", "transfer_id": None, "transaction_id": None}

    if int(acc["balance_vnd"]) < amount + fee:
        return {
            "status": "rejected_insufficient_funds",
            "transfer_id": None,
            "transaction_id": None,
            "amount_vnd": amount,
            "fee_vnd": fee,
        }

    transfer_id = _next_id("TRF")
    transaction_id = _next_id("TXN")
    acc["balance_vnd"] = int(acc["balance_vnd"]) - amount - fee
    transfer_row: dict[str, Any] = {
        "transfer_id": transfer_id,
        "transaction_id": transaction_id,
        "from_account_id": from_id,
        "to_account_number": to_num,
        "rail": rail,
        "amount_vnd": amount,
        "fee_vnd": fee,
        "status": "succeeded",
    }
    if bank is not None:
        transfer_row["to_bank_code"] = bank
    if memo is not None:
        transfer_row["memo"] = memo
    _STATE.setdefault("transfers", []).append(transfer_row)
    txn_row: dict[str, Any] = {
        "transaction_id": transaction_id,
        "account_id": from_id,
        "direction": "debit",
        "amount_vnd": amount + fee,
        "status": "succeeded",
        "channel": rail,
        "timestamp": _clock(),
        "counterparty_name": None,
    }
    if memo is not None:
        txn_row["memo"] = memo
    _STATE.setdefault("transactions", []).append(txn_row)
    return {
        "status": "succeeded",
        "transfer_id": transfer_id,
        "transaction_id": transaction_id,
        "amount_vnd": amount,
        "fee_vnd": fee,
        "rail": rail,
    }


def _get_vietqr_payment_status(arguments: dict) -> dict:
    ref = _require_str(arguments, "payment_ref", entity="vietqr_payments")
    if isinstance(ref, dict):
        return ref
    bad = _check_prefix(ref, "payment_ref", "vietqr_payments")
    if bad:
        return bad
    row = _find("vietqr_payments", "payment_ref", ref)
    if row is None:
        return _err("not_found", entity="vietqr_payments", id=ref, field="payment_ref")
    out = {
        "payment_ref": row["payment_ref"],
        "status": row["status"],
        "amount_vnd": row["amount_vnd"],
        "merchant_name": row["merchant_name"],
    }
    if row.get("transaction_id") is not None:
        out["transaction_id"] = row["transaction_id"]
    return out


def _get_dispute_status(arguments: dict) -> dict:
    dispute_id = _require_str(arguments, "dispute_id", entity="disputes")
    if isinstance(dispute_id, dict):
        return dispute_id
    bad = _check_prefix(dispute_id, "dispute_id", "disputes")
    if bad:
        return bad
    row = _find("disputes", "dispute_id", dispute_id)
    if row is None:
        return _err("not_found", entity="disputes", id=dispute_id, field="dispute_id")
    return {
        "dispute_id": row["dispute_id"],
        "transaction_id": row["transaction_id"],
        "status": row["status"],
        "reason": row["reason"],
        "opened_at": row["opened_at"],
    }


def _create_dispute(arguments: dict) -> dict:
    txn_id = _require_str(arguments, "transaction_id", entity="transactions")
    if isinstance(txn_id, dict):
        return txn_id
    bad = _check_prefix(txn_id, "transaction_id", "transactions")
    if bad:
        return bad
    reason = arguments.get("reason")
    if reason not in ("goods_not_received", "duplicate_charge", "unauthorized", "amount_mismatch"):
        return _err("invalid_argument", field="reason", message="invalid dispute reason")
    confirm = arguments.get("confirm", False)
    if not isinstance(confirm, bool):
        return _err("invalid_argument", field="confirm", message="confirm must be a boolean")
    note = arguments.get("note")

    txn = _find("transactions", "transaction_id", txn_id)
    if txn is None:
        return _err("not_found", entity="transactions", id=txn_id, field="transaction_id")

    openish = {"open", "in_review", "awaiting_confirmation"}
    for d in _STATE.get("disputes", []):
        if d.get("transaction_id") == txn_id and d.get("status") in openish:
            return _err(
                "already_disputed",
                entity="disputes",
                id=txn_id,
                field="transaction_id",
                message="transaction already has an open dispute",
            )

    if txn.get("status") != "succeeded" or txn.get("direction") != "debit":
        return _err(
            "not_disputable",
            entity="transactions",
            id=txn_id,
            field="transaction_id",
            detail={"status": txn.get("status"), "direction": txn.get("direction")},
        )

    if not confirm:
        return {"status": "awaiting_confirmation", "dispute_id": None}

    dispute_id = _next_id("DSP")
    opened_at = _clock()
    row: dict[str, Any] = {
        "dispute_id": dispute_id,
        "transaction_id": txn_id,
        "status": "open",
        "reason": reason,
        "opened_at": opened_at,
    }
    if note is not None:
        row["note"] = note
    _STATE.setdefault("disputes", []).append(row)
    return {
        "status": "open",
        "dispute_id": dispute_id,
        "transaction_id": txn_id,
        "reason": reason,
        "opened_at": opened_at,
    }
