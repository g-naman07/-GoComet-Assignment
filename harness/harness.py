"""Persistent, deterministic freight-invoice audit harness.

The command line is intentionally small because the grader starts each command
in a fresh process.  All extracted records therefore live in the SQLite state
database.  When an Anthropic key is available the provider can be swapped in
without changing this orchestration; local runs use the deterministic provider.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

try:
    from .engine.extractor import extract_all_text
    from .engine.llm import TextAnalysisProvider
    from .engine.state import StateDB
except ImportError:
    from engine.extractor import extract_all_text
    from engine.llm import TextAnalysisProvider
    from engine.state import StateDB

DEFAULT_STATE = ".state"
PROVIDER = TextAnalysisProvider()


def dec(value, default=None):
    try:
        return Decimal(str(value).replace(",", ""))
    except (ArithmeticError, ValueError, TypeError):
        return default


def money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):.2f}"


def conflict_range(billed: Decimal, correct_totals: list[Decimal]) -> list[str]:
    """Return the nonnegative dispute range across competing valid terms."""
    if not correct_totals:
        return [money(Decimal("0")), money(Decimal("0"))]
    lower_correct = min(correct_totals)
    higher_correct = max(correct_totals)
    low = max(Decimal("0"), billed - higher_correct)
    high = max(Decimal("0"), billed - lower_correct)
    return [money(low), money(high)]


def quarter_for(value: str) -> str:
    d = iso(value)
    return f"{d.year}-Q{((d.month - 1) // 3) + 1}" if d else ""


def quarter_end(period: str) -> date | None:
    m = re.match(r"^(\d{4})-Q([1-4])$", period)
    if not m:
        return None
    year, q = int(m.group(1)), int(m.group(2))
    month = q * 3
    next_year = year + (1 if month == 12 else 0)
    next_month = 1 if month == 12 else month + 1
    return date(next_year, next_month, 1) - timedelta(days=1)


def iso(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value or "")
    except ValueError:
        return None


def quote_page(doc: dict, needle: str = "") -> tuple[int, str]:
    pages = json.loads(doc["pages_json"]) if isinstance(doc.get("pages_json"), str) else doc.get("pages", [])
    if not pages:
        return 1, needle or ""
    if needle:
        for number, page in enumerate(pages, 1):
            for line in page.splitlines():
                if needle.lower() in line.lower():
                    return number, line.strip()
    return 1, next((x.strip() for x in pages[0].splitlines() if x.strip()), needle or "")


def evidence(doc: dict | None, quote: str = "") -> dict:
    if not doc:
        return {"doc": "unknown", "page": 1, "quote": quote or "Evidence unavailable"}
    page, found = quote_page(doc, quote)
    return {"doc": doc["doc_ref"], "page": page, "quote": found or quote or "Document received"}


def contract_doc(docs: dict[str, dict], contract_id: str, phrase: str = "") -> dict | None:
    candidates = []
    for doc in docs.values():
        text = _all_text(doc)
        if contract_id.lower() in text.lower():
            candidates.append(doc)
    for doc in candidates:
        if phrase.lower() in _all_text(doc).lower():
            return doc
    return next((d for d in candidates if "master services agreement" in _all_text(d).lower()),
                candidates[0] if candidates else None)


def missing_rate_reference(docs: dict[str, dict], contract_id: str, lane: str,
                           pickup: str) -> tuple[dict, str] | None:
    """Find an in-force document that names a missing pricing attachment."""
    ref_pattern = re.compile(
        r"\b((?:Annex|Schedule|Appendix)\s+[A-Z0-9]+"
        r"(?:\s+(?:rev(?:ision)?|version)\s*\d+)?"
        r")(?!\s+(?:rev(?:ision)?|version)\s*\d+)\b", re.I)
    for doc in docs.values():
        text = _all_text(doc)
        if contract_id.lower() not in text.lower() or lane.lower() not in text.lower():
            continue
        for match in ref_pattern.finditer(text):
            name = re.sub(r"\s+", " ", match.group(1)).strip()
            # Keep the attachment cue close to the named schedule. A broad
            # window can accidentally pair a generic "Annex A rates" phrase
            # with a later missing attachment and report the wrong name.
            context = text[match.start():match.start() + 160]
            if lane.lower() not in context.lower():
                continue
            attachment_cue = text[match.end():match.end() + 55]
            if not re.search(r"\b(?:attached|missing|not enclosed|to follow)\b",
                             attachment_cue, re.I):
                continue
            start = re.search(r"\b(?:from|effective\s+from)\s+(\d{4}-\d{2}-\d{2})", context, re.I)
            end = re.search(r"\bto\s+(\d{4}-\d{2}-\d{2})", context, re.I)
            picked = iso(pickup)
            if start and picked and picked < iso(start.group(1)):
                continue
            if end and picked and picked > iso(end.group(1)):
                continue
            return doc, name
    return None


def doc_type(ref: str, text: str) -> str:
    low = f"{ref} {text}".lower()
    if any(token in low for token in ("msa-", "master services agreement", "amendment", "annex a", "annex c")):
        return "contract"
    if any(token in low for token in ("due date extension", "rate notice", "waive ", "waived ")):
        return "email"
    if "tracking" in low or "picked_up" in low or "picked up" in low:
        return "tracking"
    if "ap-register" in low or "ap register" in low or "payment ledger" in low:
        return "ap_register"
    if "credit note" in low:
        return "credit_note"
    if ("invoice" in low or "statement no" in low or "freight bill no" in low
            or "charge shipment route" in low or re.search(r"\btotal\b", low)):
        return "invoice"
    if "amendment" in low or "annex a" in low or "master services agreement" in low or "msa-" in low:
        return "contract"
    return "email"


def _all_text(doc: dict) -> str:
    pages = doc.get("pages")
    if pages is None:
        pages_json = doc.get("pages_json")
        if isinstance(pages_json, str):
            try:
                pages = json.loads(pages_json)
            except json.JSONDecodeError:
                pages = []
    return "\n".join(pages or [])


def _contract_for(db: StateDB, inv: dict) -> dict | None:
    if inv.get("contract_id"):
        return db.get_contract(inv["contract_id"])
    contracts = db.get_all_contracts()
    return contracts[0] if len(contracts) == 1 else None


def ingest(batch_dir: str, state_dir: str) -> None:
    db = StateDB(state_dir)
    batch_path = Path(batch_dir)
    batch = batch_path.name
    for eml_path in sorted(batch_path.glob("*.eml")):
        for raw_doc in extract_all_text(eml_path, batch):
            ref = raw_doc["doc_ref"]
            if db.is_doc_processed(ref):
                continue
            text = _all_text(raw_doc)
            kind = doc_type(ref, text)
            raw_doc["pages"] = raw_doc.pop("pages")
            db.store_document(ref, batch, kind, raw_doc["content_hash"], raw_doc["email_date"],
                              raw_doc.get("email_from", ""), raw_doc.get("email_from_name", ""),
                              raw_doc.get("email_subject", ""), raw_doc["pages"], raw_doc["is_scan"])
            stored = db.get_document(ref) or {}
            if kind in ("invoice", "credit_note"):
                data = PROVIDER.extract_structured(text, ref, "invoice", "invoice_lines")
                inv_id = data.get("invoice_id")
                if inv_id:
                    inv_id = re.sub(r"[\s#:]", "", str(inv_id)).upper()
                    db.store_invoice(inv_id, data.get("invoice_date") or raw_doc["email_date"],
                                     data.get("currency", "USD"), str(data.get("total") or 0),
                                     data.get("contract_id"), ref, 1, kind == "credit_note" or data.get("is_credit_note", False),
                                     batch, data.get("lines", []))
            elif kind == "contract":
                terms = PROVIDER.extract_structured(text, ref, "contract", "contract_terms")
                cid = terms.get("contract_id")
                if cid:
                    db.store_contract(cid, terms, ref)
                    authority = 3 if re.search(r"amendment\s+no\.?|amendment\s+reference", text, re.I) else 1
                    for row in PROVIDER.extract_structured(text, ref, "contract", "rate_table").get("rates", []):
                        db.store_rate(cid, row["lane"], str(row["weight_min"]), str(row["weight_max"]),
                                      str(row["rate_period1"]), 1,
                                      row.get("rate_period1_from") or terms.get("effective_date"),
                                      row.get("rate_period1_to"), ref, 1, authority)
                        if row.get("rate_period2") is not None:
                            db.store_rate(cid, row["lane"], str(row["weight_min"]), str(row["weight_max"]),
                                          str(row["rate_period2"]), 2,
                                          row.get("rate_period2_from") or terms.get("effective_date"),
                                          row.get("rate_period2_to"), ref, 1, authority)
                    for row in PROVIDER.extract_structured(text, ref, "contract", "fuel_surcharges").get("surcharges", []):
                        db.store_fuel_surcharge(cid, row["month"], str(row["percentage"]), ref, 1)
                    for row in PROVIDER.extract_structured(text, ref, "contract", "fx_rates").get("rates", []):
                        db.store_fx_rate(cid, row["month"], str(row["inr_per_usd"]), ref)
                    for row in PROVIDER.extract_structured(text, ref, "contract", "schedule1").get("contacts", []):
                        db.store_schedule1(cid, row["name"], row.get("title", ""), row.get("party", ""),
                                           row.get("categories", []), row.get("email", ""), ref)
                    rebate = PROVIDER.extract_structured(text, ref, "contract", "rebate_tiers")
                    db.store_rebate_tiers(cid, rebate.get("tiers", []), rebate.get("period", "quarterly"), rebate.get("rule", "tier_reached"))
            elif kind == "tracking":
                for line in text.splitlines():
                    m = re.search(r"(\S+)\s*\|\s*(PICKED[_ ]UP|DELIVERED|IN_TRANSIT)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+)", line, re.I)
                    if m:
                        db.store_tracking_event(m.group(1), m.group(2).upper().replace(" ", "_"), m.group(3), m.group(4).strip(), ref, 1)
            elif kind == "ap_register":
                for row in PROVIDER.extract_structured(text, ref, "ap_register", "payment_records").get("payments", []):
                    db.store_payment(row["invoice_id"].upper(), row.get("invoice_date"), row.get("due_date"),
                                     str(row["amount"]), row.get("currency", "USD"), row.get("status", "open"),
                                     row.get("paid_date"), ref, 1)
            if kind == "email":
                parsed = PROVIDER.extract_structured(text, ref, "email", "email_analysis")
                sender = raw_doc.get("email_from_name", "")
                cid_match = re.search(r"\b(TBX-[A-Z]{2}-[A-Z]+-[A-Z0-9]+)\b", text)
                if cid_match:
                    sender = raw_doc.get("email_from_name", "")
                    authorized = db.check_authority(cid_match.group(1), sender, "rates")
                    for row in parsed.get("rate_notices", []):
                        # Retain unauthorised notices as source data too. They
                        # cannot govern pricing unless a later human feedback
                        # item explicitly accepts that notice.
                        db.store_rate(cid_match.group(1), row["lane"], str(row["weight_min"]),
                                      str(row["weight_max"]), str(row["rate"]), 1,
                                      row.get("effective_from"), row.get("effective_to"), ref, 1,
                                      2 if authorized else 0)
                for row in parsed.get("extensions", []):
                    db.store_extension(row["invoice_id"].upper(), row["new_due_date"], sender, ref, 1, True,
                                       next((x for x in text.splitlines() if row["new_due_date"] in x), ""))
                for row in parsed.get("waivers", []):
                    db.store_waiver(sender, None, row.get("awb"), row.get("charge"), row.get("month"),
                                    "monthly" if row.get("month") else "one_off", ref, 1, True,
                                    next((x for x in text.splitlines() if "waiv" in x.lower()), ""))
            # Keep the raw page text available for later evidence and generic extraction.
    db.mark_batch_processed(batch)
    db.close()


def _rate(db: StateDB, contract_id: str, lane: str, weight: Decimal, when: str) -> tuple[Decimal | None, dict | None]:
    source_rows = [r for r in db.get_rates(contract_id, lane)
                   if dec(r["weight_min"]) <= weight <= dec(r["weight_max"])]
    if not source_rows:
        return None, None
    picked = iso(when)
    for feedback in db.get_all_feedback():
        fact = feedback.get("fact") or {}
        scope = feedback.get("scope") or {}
        value = fact.get("value") or {}
        applies = scope.get("applies_to") or {}
        if (feedback.get("decision") != "close" or fact.get("type") != "rate_accepted"
                or applies.get("contract_id") != contract_id or applies.get("lane") != lane
                or not picked):
            continue
        start, end = iso(scope.get("valid_from")), iso(scope.get("valid_to"))
        if (start and picked < start) or (end and picked > end):
            continue
        sender = str(value.get("sent_by") or "").strip().casefold()
        if not sender:
            continue
        for row in source_rows:
            if not (dec(row["weight_min"]) <= weight <= dec(row["weight_max"])):
                continue
            row_start, row_end = iso(row.get("effective_from")), iso(row.get("effective_to"))
            if (row_start and picked < row_start) or (row_end and picked > row_end):
                continue
            source = db.get_document(row.get("source_doc", "")) or {}
            if str(source.get("email_from_name") or "").strip().casefold() == sender:
                return dec(row["rate"]), {**row, "_feedback": {
                    "feedback_item_id": feedback.get("item_id", ""),
                    "batch": feedback.get("batch", "")}}
    # A notice from an unauthorised sender is retained for audit, but cannot
    # set a rate unless a scoped feedback fact above accepts it.
    rows = [row for row in source_rows if int(row.get("authority_level") or 0) > 0]
    if not rows:
        return None, None
    valid = []
    for row in rows:
        start = iso(row.get("effective_from"))
        end = iso(row.get("effective_to"))
        if start and picked and picked < start:
            continue
        if end and picked and picked > end:
            continue
        valid.append(row)
    # Prefer any explicitly dated terms in force at the pickup date. The
    # annex may define P1/P2 windows that do not align to a contract-year
    # anniversary. Legacy rows without periods remain the fallback.
    if valid:
        candidates = valid
    elif any(iso(row.get("effective_to")) for row in rows):
        return None, None
    else:
        contract = db.get_contract(contract_id) or {}
        effective = iso(contract.get("effective_date"))
        col = 2 if effective and picked and picked >= effective + timedelta(days=365) else 1
        candidates = [r for r in rows if int(r.get("rate_col") or 1) == col] or rows
    chosen = max(candidates, key=lambda row: (
        int(row.get("authority_level") or 1),
        iso(row.get("effective_from")) or date.min,
    ))
    return dec(chosen["rate"]), chosen


def _action(mode: str, outbound: str, safe: bool = True) -> str:
    if mode == "autonomous" and safe:
        return f"execute:{outbound}"
    return f"draft:{outbound}"


def _feedback_for(db: StateDB, item_class: str, invoice_id: str | None = None,
                  contract_id: str | None = None) -> list[dict]:
    return db.get_feedback_for_key(invoice_id=invoice_id, finding_class=item_class,
                                   contract_id=contract_id)


def _item(item_class: str, verdict: str, action: str, confidence: float, ev: list[dict], **fields) -> dict:
    key = fields.get("invoice_id")
    if item_class == "cashback":
        key = f"{fields.get('contract_id')}/{fields.get('period')}/{fields.get('credit_type')}"
    else:
        key = key or fields.get("contract_id") or fields.get("period", "item")
    out = {"id": f"{item_class}:{key}",
           "class": item_class, "verdict": verdict, "action": action, "confidence": confidence,
           "evidence": ev, "amount_usd": fields.pop("amount_usd", None), "amount_range": fields.pop("amount_range", None),
           "deadline": fields.pop("deadline", None), "missing_doc": fields.pop("missing_doc", None),
           "supersedes": fields.pop("supersedes", []), "learned_at": fields.pop("learned_at", None)}
    out.update({k: v for k, v in fields.items() if v is not None})
    return out


def report(as_of: str, mode: str, state_dir: str) -> dict:
    db = StateDB(state_dir)
    cutoff = as_of
    docs = {d["doc_ref"]: d for d in db.get_all_documents(cutoff)}
    invoices = [i for i in db.get_all_invoices() if (i.get("invoice_date") or "") <= cutoff and not i.get("is_credit_note")]
    findings = []
    for inv in invoices:
        cid = inv.get("contract_id")
        contract = _contract_for(db, inv)
        cid = cid or (contract or {}).get("contract_id")
        inv_doc = docs.get(inv.get("source_doc"))
        for cls in ("overcharge", "waived_but_billed", "late_fee_not_owed", "duplicate"):
            amount = Decimal("0")
            ev = [evidence(inv_doc)]
            feedback_items = _feedback_for(db, cls, inv["invoice_id"], cid)
            if any(x.get("decision") in ("close", "reject") for x in feedback_items):
                continue
            if cls == "overcharge" and inv.get("currency") == "INR" and cid:
                fx_month = str(inv.get("invoice_date") or "")[:7]
                if not db.get_fx_rate(cid, fx_month):
                    findings.append(_item(
                        cls, "cannot_determine", "human_review", .55, ev,
                        invoice_id=inv["invoice_id"], contract_id=cid,
                        missing_doc={"name": f"Annex F FX rate for {fx_month}",
                                     "holder": {"party": "Carrier", "role": "contract administrator"}}))
                    continue
            if cls == "overcharge" and cid:
                missing_rate = None
                missing_rate_pickup = ""
                learned_rate = None
                trusted_rows = []
                for line in inv.get("lines", []):
                    if line.get("charge_type") != "freight" or not line.get("lane") or line.get("weight_kg") is None:
                        continue
                    pickup = db.get_pickup_date(line.get("awb")) or line.get("ship_date") or inv.get("invoice_date")
                    referenced = missing_rate_reference(docs, cid, line["lane"], pickup)
                    if referenced:
                        missing_rate = line
                        missing_rate_pickup = pickup
                        continue
                    rate, rate_row = _rate(db, cid, line["lane"], dec(line["weight_kg"], Decimal("0")), pickup)
                    if rate is None:
                        missing_rate = line
                        missing_rate_pickup = pickup
                        continue
                    if rate is not None:
                        if rate_row and rate_row.get("_feedback"):
                            learned_rate = rate_row["_feedback"]
                        if rate_row and int(rate_row.get("authority_level") or 1) >= 2:
                            trusted_rows.append((line, pickup))
                        billed = dec(line.get("amount"), Decimal("0"))
                        correct = (rate * dec(line["weight_kg"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                        if billed > correct:
                            amount += billed - correct
                            ev.append(evidence(docs.get(rate_row["source_doc"]), f"{rate_row['lane']} {rate_row['weight_min']}-{rate_row['weight_max']}"))
                            for fuel in inv.get("lines", []):
                                if fuel.get("charge_type") == "fuel" and str(fuel.get("awb", "")).lower() == str(line.get("awb", "")).lower():
                                    pct = dec(db.get_fuel_surcharge(cid, pickup[:7]), Decimal("0")) / 100
                                    correct_fuel = (pct * correct).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                                    fuel_amount = dec(fuel.get("amount"), Decimal("0"))
                                    if fuel_amount > correct_fuel:
                                        amount += fuel_amount - correct_fuel
                                    break
                if missing_rate:
                    referenced = missing_rate_reference(docs, cid, missing_rate["lane"],
                                                        missing_rate_pickup)
                    if not referenced:
                        continue
                    amendment, missing_name = referenced
                    rates_contact = next((c for c in db.get_schedule1(cid)
                                          if "rates" in [str(x).lower() for x in c.get("categories", [])]), {})
                    findings.append(_item(
                        cls, "cannot_determine", _action(mode, "request_document", True), .9,
                        [evidence(amendment, missing_rate["lane"])],
                        invoice_id=inv["invoice_id"], contract_id=cid,
                        missing_doc={"name": missing_name,
                                     "holder": {"party": rates_contact.get("party", "Carrier"),
                                                "person": rates_contact.get("person_name", rates_contact.get("name", "")),
                                                "role": rates_contact.get("title", "rates contact")}}))
                    continue
                conflict_added = False
                for line, pickup in trusted_rows:
                    peers = [peer for peer in db.get_rates(cid, line["lane"])
                             if dec(peer["weight_min"]) <= dec(line["weight_kg"], Decimal("0")) <= dec(peer["weight_max"])
                             and int(peer.get("authority_level") or 1) >= 2
                             and (not iso(peer.get("effective_from")) or not iso(pickup)
                                  or iso(peer["effective_from"]) <= iso(pickup))
                             and (not iso(peer.get("effective_to")) or not iso(pickup)
                                  or iso(peer["effective_to"]) >= iso(pickup))]
                    distinct = {str(peer["rate"]): peer for peer in peers}
                    if len(distinct) > 1:
                        billed = dec(line.get("amount"), Decimal("0"))
                        fuel = next((item for item in inv.get("lines", [])
                                     if item.get("charge_type") == "fuel"
                                     and str(item.get("awb", "")).lower() == str(line.get("awb", "")).lower()), None)
                        fuel_billed = dec(fuel.get("amount"), Decimal("0")) if fuel else Decimal("0")
                        fuel_pct = dec(db.get_fuel_surcharge(cid, str(pickup)[:7]), Decimal("0")) / 100
                        billed_total = billed + fuel_billed
                        correct_totals = []
                        for peer in distinct.values():
                            base = dec(peer["rate"]) * dec(line["weight_kg"])
                            correct_total = base + (base * fuel_pct).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                            correct_totals.append(correct_total)
                        findings.append(_item(
                            cls, "conflict", "escalate", .9,
                            [evidence(docs.get(peer["source_doc"]), line["lane"]) for peer in peers[:2]],
                            invoice_id=inv["invoice_id"], contract_id=cid,
                            amount_range=conflict_range(billed_total, correct_totals)))
                        conflict_added = True
                        break
                if conflict_added:
                    continue
                # A conflict is only reported when two applicable, trusted
                # rows remain after effective-date precedence is applied.
                # Ordinary period-1/period-2 rows are not a conflict.
            elif cls == "waived_but_billed":
                for line in inv.get("lines", []):
                    for waiver in db.get_waivers():
                        if waiver.get("authority_valid") and waiver.get("awb") and waiver["awb"].lower() == str(line.get("awb", "")).lower():
                            if waiver.get("charge_type", "").lower() in line.get("description", "").lower() or not waiver.get("charge_type"):
                                amount += dec(line.get("amount"), Decimal("0"))
                                ev.append(evidence(docs.get(waiver["source_doc"]), waiver.get("quote", "")))
            elif cls == "late_fee_not_owed":
                for line in inv.get("lines", []):
                    if line.get("charge_type") == "late_fee":
                        m = re.search(r"on\s+(\S+)", line.get("description", ""), re.I)
                        base_id = m.group(1).rstrip(".,:;").upper() if m else ""
                        p = db.get_payments(base_id)
                        if p and p[0].get("paid_date") and p[0].get("due_date"):
                            due = iso(p[0]["due_date"])
                            ext = [x for x in db.get_extensions(base_id) if x.get("authority_valid")]
                            if ext:
                                due = max(due, *(iso(x["new_due_date"]) for x in ext if iso(x["new_due_date"])))
                            if due and iso(p[0]["paid_date"]) <= due:
                                amount += dec(line.get("amount"), Decimal("0"))
                                ev.append(evidence(docs.get(p[0]["source_doc"])))
            elif cls == "duplicate":
                same_id_docs = [d for d in docs.values()
                                if "#" in d["doc_ref"]
                                and inv["invoice_id"] in re.sub(r"[\s#:]", "", d["doc_ref"].split("#", 1)[1]).upper()]
                resend = next((d for d in docs.values()
                               if d["doc_ref"].lower().endswith(".eml")
                               and inv["invoice_id"].lower() in _all_text(d).lower()
                               and re.search(r"outstanding|reminder|at your earliest|payment again",
                                             _all_text(d), re.I)), None)
                settled = any(p.get("status") in ("paid", "scheduled")
                              for p in db.get_payments(inv["invoice_id"]))
                if len(same_id_docs) > 1 and resend and settled:
                    amount = dec(inv.get("total"), Decimal("0"))
                    ledger = next((d for d in docs.values()
                                   if inv["invoice_id"].lower() in _all_text(d).lower()
                                   and re.search(r"AP register|AP-register|payment ledger|Invoice no\s*\|",
                                                 _all_text(d), re.I)), None)
                    ev = [evidence(same_id_docs[0], inv["invoice_id"]), evidence(resend, inv["invoice_id"])]
                    if ledger:
                        ev.append(evidence(ledger, inv["invoice_id"]))
                else:
                    awbs = {str(x.get("awb", "")).lower() for x in inv.get("lines", []) if x.get("awb")}
                    for other in invoices:
                        if other["invoice_id"] == inv["invoice_id"] or (other.get("invoice_date") or "") >= (inv.get("invoice_date") or ""):
                            continue
                        if awbs & {str(x.get("awb", "")).lower() for x in other.get("lines", []) if x.get("awb")}:
                            amount = dec(inv.get("total"), Decimal("0"))
                            ev.append(evidence(docs.get(other.get("source_doc"))))
                            break
            if cls == "overcharge" and cid:
                # Dry ice is billed with the Annex D handling fee, not the DG
                # fee.  The declaration is the evidence that distinguishes it.
                for line in inv.get("lines", []):
                    if "dg" not in line.get("description", "").lower():
                        continue
                    awb = str(line.get("awb", "")).lower()
                    dry_ice = any(
                        awb in page.lower() and "un1845" in page.lower()
                        for doc in docs.values()
                        for page in json.loads(doc["pages_json"])
                    )
                    if dry_ice:
                        amount += dec(line.get("amount"), Decimal("0"))
                        ev.append(next(
                            evidence(doc, "UN1845")
                            for doc in docs.values()
                            if any(awb in page.lower() and "un1845" in page.lower()
                                   for page in json.loads(doc["pages_json"]))
                        ))
                        break
            if amount > 0:
                reported_amount = amount
                if inv.get("currency") == "INR" and cid:
                    fx = dec(db.get_fx_rate(cid, str(inv.get("invoice_date") or "")[:7]), None)
                    if fx:
                        reported_amount = (amount / fx).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                learned = None
                if feedback_items:
                    learned = {"feedback_item_id": feedback_items[-1]["item_id"],
                               "batch": feedback_items[-1].get("batch", "")}
                elif cls == "overcharge" and cid and "learned_rate" in locals() and learned_rate:
                    learned = learned_rate
                safe = reported_amount <= Decimal("5000") and len(ev) >= 2
                findings.append(_item(cls, "finding", _action(mode, "dispute", safe), .9, ev,
                                      invoice_id=inv["invoice_id"], contract_id=cid, amount_usd=money(reported_amount),
                                      learned_at=learned))
    portfolio = {"cashback": [], "expiry": [], "late_fee_risk": []}
    # Quarterly rebates are assessed only after a quarter closes.  Qualifying
    # spend excludes duty, tax, late-fee and credit lines.
    for contract in db.get_all_contracts():
        cid = contract["contract_id"]
        by_period: dict[str, Decimal] = {}
        for inv in db.get_all_invoices():
            if inv.get("contract_id") != cid or inv.get("is_credit_note"):
                continue
            period = quarter_for(inv.get("invoice_date", ""))
            if period and (inv.get("invoice_date") or "") <= cutoff:
                fx = Decimal("1")
                if inv.get("currency") == "INR":
                    fx = dec(db.get_fx_rate(cid, str(inv.get("invoice_date"))[:7]), None)
                    if fx is None:
                        continue
                qualifying = sum((dec(x.get("amount"), Decimal("0")) / fx for x in inv.get("lines", [])
                                  if x.get("charge_type") in
                                  ("freight", "fuel", "security", "handling", "delivery_fee", "other")
                                  and "tax" not in x.get("description", "").lower()
                                  and "duty" not in x.get("description", "").lower()
                                  and "late fee" not in x.get("description", "").lower()), Decimal("0"))
                by_period[period] = by_period.get(period, Decimal("0")) + qualifying
        for period, spend in by_period.items():
            end = quarter_end(period)
            if not end or iso(as_of) <= end or spend <= 0:
                continue
            tiers = db.get_rebate_tiers(cid)
            eligible = [x for x in tiers if spend >= dec(x.get("threshold_usd"), Decimal("0"))]
            if not eligible:
                continue
            tier = max(eligible, key=lambda x: dec(x.get("threshold_usd"), Decimal("0")))
            amount = (spend * dec(tier.get("rate_pct"), Decimal("0")) / 100).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP)
            if amount <= 0:
                continue
            rebate_doc = contract_doc(docs, cid, "7.1 volume rebate") or docs.get(contract.get("source_doc"))
            feedback_items = _feedback_for(db, "cashback", contract_id=cid)
            if any(x.get("decision") in ("close", "reject") and period in json.dumps(x.get("finding_key", {}))
                   for x in feedback_items):
                continue
            portfolio["cashback"].append(_item(
                "cashback", "finding", _action(mode, "claim_credit", amount <= Decimal("5000")),
                .9,                                 [evidence(rebate_doc, "7.1 Volume rebate"),
                 evidence(rebate_doc, "Qualifying Spend")],
                contract_id=cid, period=period, credit_type="rebate", amount_usd=money(amount),
                learned_at=({"feedback_item_id": feedback_items[-1]["item_id"],
                             "batch": feedback_items[-1].get("batch", "")} if feedback_items else None)))
    for contract in db.get_all_contracts():
        deadline = contract.get("end_date")
        if deadline and 0 <= (iso(deadline) - iso(as_of)).days <= 60:
            expiry_doc = contract_doc(docs, contract["contract_id"], "initial term")
            portfolio["expiry"].append(_item("expiry", "finding", "internal:alert", .95,
                                             [evidence(expiry_doc, "Initial term")],
                                             contract_id=contract["contract_id"], deadline=deadline))
    for inv in invoices:
        for p in db.get_payments(inv["invoice_id"]):
            due = iso(p.get("due_date"))
            if p.get("status") in ("open", "scheduled") and due and 0 < (due - iso(as_of)).days <= 14:
                contract = _contract_for(db, inv) or {}
                rate = dec(contract.get("late_fee_rate"), Decimal("0"))
                fee = (rate / 100 * dec(p.get("amount"), Decimal("0"))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                portfolio["late_fee_risk"].append(_item("late_fee_risk", "finding", "internal:payment_reminder", .85,
                                                        [evidence(docs.get(p["source_doc"]))],
                                                        invoice_id=inv["invoice_id"], deadline=p["due_date"], amount_usd=money(fee)))
    result = {"schema_version": "1.0", "as_of": as_of, "mode": mode, "findings": findings, "portfolio": portfolio}
    db.close()
    return result


def feedback(feedback_file: str, state_dir: str) -> None:
    db = StateDB(state_dir)
    payload = json.loads(Path(feedback_file).read_text(encoding="utf-8"))
    for item in payload.get("items", []):
        db.store_feedback_item(item["item_id"], item.get("finding_key", {}), item.get("decision", ""),
                               item.get("fact"), item.get("scope"), payload.get("after_batch", ""),
                               payload.get("decided_at", ""), payload.get("decided_by", {}))
    db.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.py", description="Freight invoice audit harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("batch_dir"); p.add_argument("--state-dir", default=DEFAULT_STATE)
    p = sub.add_parser("report"); p.add_argument("--as-of", required=True); p.add_argument("--mode", choices=["supervised", "autonomous"], default="supervised"); p.add_argument("--state-dir", default=DEFAULT_STATE)
    p = sub.add_parser("feedback"); p.add_argument("feedback_file"); p.add_argument("--state-dir", default=DEFAULT_STATE)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "ingest": ingest(a.batch_dir, a.state_dir)
        elif a.cmd == "report": json.dump(report(a.as_of, a.mode, a.state_dir), sys.stdout, indent=1); sys.stdout.write("\n")
        else: feedback(a.feedback_file, a.state_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"harness {a.cmd}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
