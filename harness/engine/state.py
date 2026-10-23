"""SQLite-based persistent state for cross-batch memory.

Stores documents, extracted facts, contracts, invoices, payments, findings, feedback.
Each batch is ingested incrementally; content hashing avoids reprocessing.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class StateDB:
    """Persistent state backed by SQLite."""

    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "state.db"
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_ref TEXT PRIMARY KEY,
            batch TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            content_hash TEXT,
            email_date TEXT,
            email_from TEXT,
            email_from_name TEXT,
            email_subject TEXT,
            page_count INTEGER DEFAULT 1,
            is_scan INTEGER DEFAULT 0,
            processed_at TEXT,
            pages_json TEXT
        );

        CREATE TABLE IF NOT EXISTS facts (
            fact_id TEXT PRIMARY KEY,
            fact_type TEXT NOT NULL,
            value_json TEXT NOT NULL,
            source_doc TEXT NOT NULL,
            source_page INTEGER DEFAULT 1,
            quote TEXT,
            doc_date TEXT,
            effective_date TEXT,
            authority_level INTEGER DEFAULT 3,
            confidence REAL DEFAULT 0.9,
            extraction_method TEXT DEFAULT 'text',
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS contracts (
            contract_id TEXT PRIMARY KEY,
            effective_date TEXT,
            end_date TEXT,
            auto_renewal INTEGER DEFAULT 0,
            notice_period_days INTEGER,
            payment_terms_days INTEGER,
            grace_period_days INTEGER DEFAULT 0,
            late_fee_rate TEXT,
            late_fee_compounds INTEGER DEFAULT 0,
            source_doc TEXT,
            terms_json TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS contract_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT NOT NULL,
            lane TEXT NOT NULL,
            weight_min TEXT NOT NULL,
            weight_max TEXT NOT NULL,
            rate TEXT NOT NULL,
            rate_col INTEGER DEFAULT 1,
            effective_from TEXT,
            effective_to TEXT,
            source_doc TEXT,
            source_page INTEGER DEFAULT 1,
            authority_level INTEGER DEFAULT 1,
            amendment_id TEXT
        );

        CREATE TABLE IF NOT EXISTS fuel_surcharges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT NOT NULL,
            month TEXT NOT NULL,
            percentage TEXT NOT NULL,
            source_doc TEXT,
            source_page INTEGER DEFAULT 1,
            UNIQUE(contract_id, month)
        );

        CREATE TABLE IF NOT EXISTS fx_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT NOT NULL,
            month TEXT NOT NULL,
            inr_per_usd TEXT NOT NULL,
            source_doc TEXT,
            UNIQUE(contract_id, month)
        );

        CREATE TABLE IF NOT EXISTS charge_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            aliases_json TEXT
        );

        CREATE TABLE IF NOT EXISTS schedule1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT NOT NULL,
            person_name TEXT NOT NULL,
            title TEXT,
            party TEXT NOT NULL,
            categories_json TEXT NOT NULL,
            email TEXT,
            source_doc TEXT
        );

        CREATE TABLE IF NOT EXISTS invoices (
            invoice_id TEXT PRIMARY KEY,
            invoice_date TEXT,
            currency TEXT DEFAULT 'USD',
            total TEXT,
            contract_id TEXT,
            source_doc TEXT,
            source_page INTEGER DEFAULT 1,
            is_credit_note INTEGER DEFAULT 0,
            batch TEXT,
            lines_json TEXT
        );

        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id TEXT NOT NULL,
            invoice_date TEXT,
            due_date TEXT,
            amount TEXT,
            currency TEXT DEFAULT 'USD',
            status TEXT,
            paid_date TEXT,
            source_doc TEXT,
            source_page INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS waivers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            party TEXT,
            awb TEXT,
            charge_type TEXT,
            month TEXT,
            scope TEXT DEFAULT 'one_off',
            source_doc TEXT,
            source_page INTEGER DEFAULT 1,
            authority_valid INTEGER DEFAULT 0,
            quote TEXT
        );

        CREATE TABLE IF NOT EXISTS extensions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id TEXT NOT NULL,
            new_due_date TEXT NOT NULL,
            person_name TEXT,
            source_doc TEXT,
            source_page INTEGER DEFAULT 1,
            authority_valid INTEGER DEFAULT 0,
            quote TEXT
        );

        CREATE TABLE IF NOT EXISTS amendments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amendment_id TEXT,
            contract_id TEXT NOT NULL,
            effective_date TEXT,
            source_doc TEXT,
            source_page INTEGER DEFAULT 1,
            content_json TEXT
        );

        CREATE TABLE IF NOT EXISTS findings_history (
            id TEXT NOT NULL,
            batch TEXT NOT NULL,
            finding_class TEXT NOT NULL,
            verdict TEXT NOT NULL,
            amount_usd TEXT,
            action TEXT,
            finding_json TEXT NOT NULL,
            created_at TEXT,
            PRIMARY KEY (id, batch)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            item_id TEXT PRIMARY KEY,
            finding_key_json TEXT NOT NULL,
            decision TEXT NOT NULL,
            fact_json TEXT,
            scope_json TEXT,
            batch TEXT,
            decided_at TEXT,
            decided_by_json TEXT,
            processed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS batches_processed (
            batch TEXT PRIMARY KEY,
            processed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS rebate_tiers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT NOT NULL,
            threshold_usd TEXT NOT NULL,
            rate_pct TEXT NOT NULL,
            period_type TEXT DEFAULT 'quarterly',
            tier_rule TEXT DEFAULT 'tier_reached'
        );

        CREATE TABLE IF NOT EXISTS tracking_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            awb TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_date TEXT,
            location TEXT,
            source_doc TEXT,
            source_page INTEGER DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(fact_type);
        CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source_doc);
        CREATE INDEX IF NOT EXISTS idx_rates_contract_lane ON contract_rates(contract_id, lane);
        CREATE INDEX IF NOT EXISTS idx_invoices_contract ON invoices(contract_id);
        CREATE INDEX IF NOT EXISTS idx_payments_invoice ON payments(invoice_id);
        CREATE INDEX IF NOT EXISTS idx_tracking_awb ON tracking_events(awb);
        """)
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Document management ---

    def is_doc_processed(self, doc_ref: str) -> bool:
        row = self._get_conn().execute(
            "SELECT 1 FROM documents WHERE doc_ref = ?", (doc_ref,)
        ).fetchone()
        return row is not None

    def store_document(self, doc_ref: str, batch: str, doc_type: str,
                       content_hash: str, email_date: str, email_from: str,
                       email_from_name: str, email_subject: str,
                       pages: list[str], is_scan: bool = False):
        self._get_conn().execute("""
            INSERT OR REPLACE INTO documents
            (doc_ref, batch, doc_type, content_hash, email_date, email_from,
             email_from_name, email_subject, page_count, is_scan, processed_at, pages_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_ref, batch, doc_type, content_hash, email_date, email_from,
              email_from_name, email_subject, len(pages), int(is_scan),
              datetime.utcnow().isoformat(), json.dumps(pages)))
        self._get_conn().commit()

    def get_document(self, doc_ref: str) -> dict | None:
        row = self._get_conn().execute(
            "SELECT * FROM documents WHERE doc_ref = ?", (doc_ref,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_documents(self, before_date: str | None = None) -> list[dict]:
        """Get all documents, optionally filtered by email_date <= before_date."""
        if before_date:
            rows = self._get_conn().execute(
                "SELECT * FROM documents WHERE email_date <= ? ORDER BY email_date, doc_ref",
                (before_date,)
            ).fetchall()
        else:
            rows = self._get_conn().execute(
                "SELECT * FROM documents ORDER BY email_date, doc_ref"
            ).fetchall()
        return [dict(r) for r in rows]

    def is_batch_processed(self, batch: str) -> bool:
        row = self._get_conn().execute(
            "SELECT 1 FROM batches_processed WHERE batch = ?", (batch,)
        ).fetchone()
        return row is not None

    def mark_batch_processed(self, batch: str):
        self._get_conn().execute(
            "INSERT OR REPLACE INTO batches_processed (batch, processed_at) VALUES (?, ?)",
            (batch, datetime.utcnow().isoformat())
        )
        self._get_conn().commit()

    # --- Contracts ---

    def store_contract(self, contract_id: str, terms: dict, source_doc: str):
        previous = self.get_contract(contract_id) or {}
        merged = dict(previous)
        if "late_fee_rate" in merged and "late_fee_rate_pct" not in merged:
            merged["late_fee_rate_pct"] = merged["late_fee_rate"]
        is_amendment = "amd" in source_doc.lower() or "amendment" in source_doc.lower()
        for key, value in terms.items():
            if value not in (None, "", []):
                if is_amendment and key in ("effective_date", "end_date"):
                    continue
                merged[key] = value
        self._get_conn().execute("""
            INSERT OR REPLACE INTO contracts
            (contract_id, effective_date, end_date, auto_renewal, notice_period_days,
             payment_terms_days, grace_period_days, late_fee_rate, late_fee_compounds,
             source_doc, terms_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (contract_id, merged.get("effective_date"), merged.get("end_date"),
              int(merged.get("auto_renewal", False)), merged.get("notice_period_days"),
              merged.get("payment_terms_days"), merged.get("grace_period_days", 0),
              str(merged.get("late_fee_rate_pct", "")) if merged.get("late_fee_rate_pct") else None,
              int(merged.get("late_fee_compounds", False)),
              source_doc, json.dumps(merged), datetime.utcnow().isoformat()))
        self._get_conn().commit()

    def get_contract(self, contract_id: str) -> dict | None:
        row = self._get_conn().execute(
            "SELECT * FROM contracts WHERE contract_id = ?", (contract_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_contracts(self) -> list[dict]:
        rows = self._get_conn().execute("SELECT * FROM contracts").fetchall()
        return [dict(r) for r in rows]

    # --- Rates ---

    def store_rate(self, contract_id: str, lane: str, weight_min: str, weight_max: str,
                   rate: str, rate_col: int = 1, effective_from: str | None = None,
                   effective_to: str | None = None, source_doc: str = "",
                   source_page: int = 1, authority_level: int = 1,
                   amendment_id: str | None = None):
        self._get_conn().execute("""
            INSERT INTO contract_rates
            (contract_id, lane, weight_min, weight_max, rate, rate_col,
             effective_from, effective_to, source_doc, source_page, authority_level, amendment_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (contract_id, lane, weight_min, weight_max, rate, rate_col,
              effective_from, effective_to, source_doc, source_page, authority_level, amendment_id))
        self._get_conn().commit()

    def get_rates(self, contract_id: str, lane: str) -> list[dict]:
        rows = self._get_conn().execute("""
            SELECT * FROM contract_rates
            WHERE contract_id = ? AND lane = ?
            ORDER BY authority_level, effective_from
        """, (contract_id, lane)).fetchall()
        return [dict(r) for r in rows]

    # --- Fuel surcharges ---

    def store_fuel_surcharge(self, contract_id: str, month: str, percentage: str,
                             source_doc: str = "", source_page: int = 1):
        self._get_conn().execute("""
            INSERT OR REPLACE INTO fuel_surcharges
            (contract_id, month, percentage, source_doc, source_page)
            VALUES (?, ?, ?, ?, ?)
        """, (contract_id, month, percentage, source_doc, source_page))
        self._get_conn().commit()

    def get_fuel_surcharge(self, contract_id: str, month: str) -> str | None:
        row = self._get_conn().execute(
            "SELECT percentage FROM fuel_surcharges WHERE contract_id = ? AND month = ?",
            (contract_id, month)
        ).fetchone()
        return row["percentage"] if row else None

    # --- FX rates ---

    def store_fx_rate(self, contract_id: str, month: str, inr_per_usd: str,
                      source_doc: str = ""):
        self._get_conn().execute("""
            INSERT OR REPLACE INTO fx_rates (contract_id, month, inr_per_usd, source_doc)
            VALUES (?, ?, ?, ?)
        """, (contract_id, month, inr_per_usd, source_doc))
        self._get_conn().commit()

    def get_fx_rate(self, contract_id: str, month: str) -> str | None:
        row = self._get_conn().execute(
            "SELECT inr_per_usd FROM fx_rates WHERE contract_id = ? AND month = ?",
            (contract_id, month)
        ).fetchone()
        return row["inr_per_usd"] if row else None

    # --- Charge codes ---

    def store_charge_code(self, contract_id: str, code: str, name: str, aliases: list[str]):
        self._get_conn().execute("""
            INSERT INTO charge_codes (contract_id, code, name, aliases_json)
            VALUES (?, ?, ?, ?)
        """, (contract_id, code, name, json.dumps(aliases)))
        self._get_conn().commit()

    def get_charge_codes(self, contract_id: str) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM charge_codes WHERE contract_id = ?", (contract_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Schedule 1 ---

    def store_schedule1(self, contract_id: str, name: str, title: str, party: str,
                        categories: list[str], email: str, source_doc: str = ""):
        self._get_conn().execute("""
            INSERT INTO schedule1
            (contract_id, person_name, title, party, categories_json, email, source_doc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (contract_id, name, title, party, json.dumps(categories), email, source_doc))
        self._get_conn().commit()

    def get_schedule1(self, contract_id: str) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM schedule1 WHERE contract_id = ?", (contract_id,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["categories"] = json.loads(d["categories_json"])
            result.append(d)
        return result

    def check_authority(self, contract_id: str, person_name: str, category: str) -> bool:
        """Check if a person has authority for a category on a contract."""
        rows = self.get_schedule1(contract_id)
        for r in rows:
            if r["person_name"].lower() == person_name.lower():
                if category in r["categories"]:
                    return True
        # Also check feedback-granted authority
        return self._check_feedback_authority(contract_id, person_name, category)

    def _check_feedback_authority(self, contract_id: str, person_name: str, category: str) -> bool:
        """Check authority granted via feedback."""
        rows = self._get_conn().execute(
            "SELECT fact_json, scope_json FROM feedback WHERE decision = 'confirm'"
        ).fetchall()
        for r in rows:
            fact = json.loads(r["fact_json"]) if r["fact_json"] else {}
            scope = json.loads(r["scope_json"]) if r["scope_json"] else {}
            if (fact.get("type") == "sender_authority" and
                    fact.get("subject", {}).get("person", "").lower() == person_name.lower()):
                applies = scope.get("applies_to", {})
                if applies.get("contract_id") == contract_id and applies.get("category") == category:
                    return True
        return False

    # --- Invoices ---

    def store_invoice(self, inv_id: str, inv_date: str, currency: str, total: str,
                      contract_id: str | None, source_doc: str, source_page: int,
                      is_credit: bool, batch: str, lines: list[dict]):
        self._get_conn().execute("""
            INSERT OR REPLACE INTO invoices
            (invoice_id, invoice_date, currency, total, contract_id, source_doc,
             source_page, is_credit_note, batch, lines_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (inv_id, inv_date, currency, total, contract_id, source_doc,
              source_page, int(is_credit), batch, json.dumps(lines)))
        self._get_conn().commit()

    def get_invoice(self, inv_id: str) -> dict | None:
        row = self._get_conn().execute(
            "SELECT * FROM invoices WHERE invoice_id = ?", (inv_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["lines"] = json.loads(d["lines_json"]) if d["lines_json"] else []
            return d
        return None

    def get_all_invoices(self) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM invoices ORDER BY invoice_date"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["lines"] = json.loads(d["lines_json"]) if d["lines_json"] else []
            result.append(d)
        return result

    # --- Payments ---

    def store_payment(self, invoice_id: str, invoice_date: str | None, due_date: str | None,
                      amount: str, currency: str, status: str, paid_date: str | None,
                      source_doc: str, source_page: int = 1):
        self._get_conn().execute("""
            INSERT INTO payments
            (invoice_id, invoice_date, due_date, amount, currency, status, paid_date,
             source_doc, source_page)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (invoice_id, invoice_date, due_date, amount, currency, status, paid_date,
              source_doc, source_page))
        self._get_conn().commit()

    def get_payments(self, invoice_id: str) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM payments WHERE invoice_id = ?", (invoice_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_payments(self) -> list[dict]:
        rows = self._get_conn().execute("SELECT * FROM payments").fetchall()
        return [dict(r) for r in rows]

    # --- Waivers ---

    def store_waiver(self, person: str, party: str | None, awb: str | None,
                     charge_type: str | None, month: str | None, scope: str,
                     source_doc: str, source_page: int, authority_valid: bool,
                     quote: str = ""):
        self._get_conn().execute("""
            INSERT INTO waivers
            (person_name, party, awb, charge_type, month, scope, source_doc,
             source_page, authority_valid, quote)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (person, party, awb, charge_type, month, scope, source_doc,
              source_page, int(authority_valid), quote))
        self._get_conn().commit()

    def get_waivers(self) -> list[dict]:
        rows = self._get_conn().execute("SELECT * FROM waivers").fetchall()
        return [dict(r) for r in rows]

    # --- Extensions ---

    def store_extension(self, invoice_id: str, new_due_date: str, person: str | None,
                        source_doc: str, source_page: int, authority_valid: bool,
                        quote: str = ""):
        self._get_conn().execute("""
            INSERT INTO extensions
            (invoice_id, new_due_date, person_name, source_doc, source_page,
             authority_valid, quote)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (invoice_id, new_due_date, person, source_doc, source_page,
              int(authority_valid), quote))
        self._get_conn().commit()

    def get_extensions(self, invoice_id: str) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM extensions WHERE invoice_id = ?", (invoice_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Tracking events ---

    def store_tracking_event(self, awb: str, event_type: str, event_date: str | None,
                             location: str | None, source_doc: str, source_page: int = 1):
        self._get_conn().execute("""
            INSERT INTO tracking_events
            (awb, event_type, event_date, location, source_doc, source_page)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (awb, event_type, event_date, location, source_doc, source_page))
        self._get_conn().commit()

    def get_pickup_date(self, awb: str) -> str | None:
        """Get the pickup date for an AWB from tracking events."""
        row = self._get_conn().execute("""
            SELECT event_date FROM tracking_events
            WHERE awb = ? AND event_type = 'PICKED_UP'
            ORDER BY event_date LIMIT 1
        """, (awb,)).fetchone()
        return row["event_date"] if row else None

    # --- Amendments ---

    def store_amendment(self, amendment_id: str | None, contract_id: str,
                        effective_date: str | None, source_doc: str,
                        source_page: int, content: dict):
        self._get_conn().execute("""
            INSERT INTO amendments
            (amendment_id, contract_id, effective_date, source_doc, source_page, content_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (amendment_id, contract_id, effective_date, source_doc, source_page,
              json.dumps(content)))
        self._get_conn().commit()

    # --- Rebate tiers ---

    def store_rebate_tiers(self, contract_id: str, tiers: list[dict],
                           period_type: str = "quarterly", tier_rule: str = "tier_reached"):
        for tier in tiers:
            self._get_conn().execute("""
                INSERT INTO rebate_tiers
                (contract_id, threshold_usd, rate_pct, period_type, tier_rule)
                VALUES (?, ?, ?, ?, ?)
            """, (contract_id, str(tier["threshold_usd"]), str(tier["rate_pct"]),
                  period_type, tier_rule))
        self._get_conn().commit()

    def get_rebate_tiers(self, contract_id: str) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM rebate_tiers WHERE contract_id = ? ORDER BY threshold_usd",
            (contract_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Findings history ---

    def store_finding(self, finding_id: str, batch: str, finding_class: str,
                      verdict: str, amount_usd: str | None, action: str,
                      finding_dict: dict):
        self._get_conn().execute("""
            INSERT OR REPLACE INTO findings_history
            (id, batch, finding_class, verdict, amount_usd, action, finding_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (finding_id, batch, finding_class, verdict, amount_usd, action,
              json.dumps(finding_dict), datetime.utcnow().isoformat()))
        self._get_conn().commit()

    def get_previous_findings(self) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM findings_history ORDER BY batch, id"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["finding"] = json.loads(d["finding_json"])
            result.append(d)
        return result

    # --- Feedback ---

    def store_feedback_item(self, item_id: str, finding_key: dict, decision: str,
                            fact: dict | None, scope: dict | None, batch: str,
                            decided_at: str, decided_by: dict):
        self._get_conn().execute("""
            INSERT OR REPLACE INTO feedback
            (item_id, finding_key_json, decision, fact_json, scope_json,
             batch, decided_at, decided_by_json, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (item_id, json.dumps(finding_key), decision,
              json.dumps(fact) if fact else None,
              json.dumps(scope) if scope else None,
              batch, decided_at, json.dumps(decided_by),
              datetime.utcnow().isoformat()))
        self._get_conn().commit()

    def get_all_feedback(self) -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM feedback ORDER BY decided_at"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["finding_key"] = json.loads(d["finding_key_json"])
            d["fact"] = json.loads(d["fact_json"]) if d["fact_json"] else None
            d["scope"] = json.loads(d["scope_json"]) if d["scope_json"] else None
            d["decided_by"] = json.loads(d["decided_by_json"]) if d["decided_by_json"] else None
            result.append(d)
        return result

    def get_feedback_for_key(self, invoice_id: str | None = None,
                             finding_class: str | None = None,
                             contract_id: str | None = None) -> list[dict]:
        """Get feedback items matching a finding key."""
        all_fb = self.get_all_feedback()
        matches = []
        for fb in all_fb:
            key = fb["finding_key"]
            if invoice_id is not None:
                if key.get("invoice_id") == invoice_id and (
                        finding_class is None or key.get("class") == finding_class):
                    matches.append(fb)
                continue
            if contract_id is not None and key.get("contract_id") == contract_id and (
                    finding_class is None or key.get("class") == finding_class):
                matches.append(fb)
        return matches
