"""Data models for extracted facts, findings, evidence, contracts, invoices."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional


def norm_invoice_id(s: str) -> str:
    """Normalize invoice ID: uppercase, remove spaces/#/:."""
    return re.sub(r"[\s#:]", "", str(s or "")).upper()


def round_half_up(val: Decimal, places: int = 2) -> Decimal:
    """Round to `places` decimal places, half-up (ties away from zero)."""
    q = Decimal(10) ** -places
    return val.quantize(q, rounding=ROUND_HALF_UP)


def money_str(val: Decimal | str | None) -> str | None:
    """Format as 'NNNNN.NN' string for JSON output."""
    if val is None:
        return None
    d = Decimal(str(val))
    return f"{round_half_up(d):.2f}"


@dataclass
class Evidence:
    """A citation: document, page, and quote."""
    doc: str  # e.g. 'batch_01/0003.eml#TX-US-2032-5394.pdf'
    page: int  # 1-indexed
    quote: str

    def to_dict(self) -> dict:
        return {"doc": self.doc, "page": self.page, "quote": self.quote}


@dataclass
class MissingDoc:
    """A document needed but not held."""
    name: str  # exactly as cited in a held document
    holder: dict  # {"party": ..., "person": ..., "role": ...}

    def to_dict(self) -> dict:
        return {"name": self.name, "holder": self.holder}


@dataclass
class ExtractedFact:
    """A single fact extracted from a document."""
    fact_id: str
    fact_type: str  # 'contract_rate', 'invoice_line', 'payment', 'waiver', 'amendment', etc.
    value: dict  # type-specific structured data
    source_doc: str  # document path
    source_page: int
    quote: str
    doc_date: str | None = None
    effective_date: str | None = None
    authority_level: int = 3  # 1=MSA/amendment, 2=Schedule1 email/feedback, 3=records, 4=other email, 5=quoted/AI
    confidence: float = 0.9
    extraction_method: str = "text"  # 'text', 'llm', 'ocr'
    content_hash: str | None = None


@dataclass
class InvoiceLine:
    """A single line item on an invoice."""
    line_num: int
    awb: str | None
    ship_date: str | None
    description: str
    lane: str | None
    weight: Decimal | None
    weight_unit: str | None
    rate: Decimal | None
    amount: Decimal
    charge_type: str | None = None  # 'freight', 'fuel', 'security', 'delivery_fee', 'late_fee', etc.
    charge_code: str | None = None  # Annex E code


@dataclass
class Invoice:
    """Parsed invoice data."""
    invoice_id: str
    invoice_date: str
    currency: str
    total: Decimal
    lines: list[InvoiceLine] = field(default_factory=list)
    contract_id: str | None = None
    source_doc: str = ""
    source_page: int = 1
    is_credit_note: bool = False


@dataclass
class ContractRate:
    """A rate from a contract or amendment."""
    contract_id: str
    lane: str
    weight_min: Decimal
    weight_max: Decimal
    rate: Decimal
    rate_period: int = 1  # column index for rate period
    effective_from: str | None = None
    effective_to: str | None = None
    source_doc: str = ""
    source_page: int = 1
    authority_level: int = 1


@dataclass
class FuelSurcharge:
    """Monthly fuel surcharge percentage."""
    contract_id: str
    month: str  # YYYY-MM
    percentage: Decimal
    source_doc: str = ""
    source_page: int = 1


@dataclass
class PaymentRecord:
    """A payment from the AP register."""
    invoice_id: str
    invoice_date: str | None
    due_date: str | None
    amount: Decimal
    currency: str
    status: str  # 'paid', 'open', 'scheduled', 'disputed'
    paid_date: str | None = None
    source_doc: str = ""
    source_page: int = 1


@dataclass
class Waiver:
    """A waiver from an authorized person."""
    person: str
    party: str
    awb: str | None
    charge_type: str | None
    month: str | None  # YYYY-MM
    scope: str  # 'one_off', 'monthly'
    source_doc: str = ""
    source_page: int = 1
    authority_valid: bool = False


@dataclass
class ContractTerms:
    """Key terms from a contract/MSA."""
    contract_id: str
    effective_date: str | None
    end_date: str | None
    auto_renewal: bool = False
    notice_period_days: int | None = None
    payment_terms_days: int | None = None
    grace_period_days: int = 0
    late_fee_rate: Decimal | None = None
    late_fee_compounds: bool = False
    rebate_tiers: list[dict] = field(default_factory=list)  # [{"threshold": Decimal, "rate": Decimal}, ...]
    schedule1: dict = field(default_factory=dict)  # {"waivers": {"person": ..., "email": ...}, ...}
    source_doc: str = ""
    fx_rates: dict = field(default_factory=dict)  # {"YYYY-MM": Decimal, ...}
    charge_codes: dict = field(default_factory=dict)  # {"SEC": ["Security surcharge", "Screening fee", ...], ...}
    qualifying_spend_def: str = ""  # description of what counts as qualifying spend
    annual_credits: list[dict] = field(default_factory=list)  # annual improvement credits etc.
    margin_step_down: list[dict] = field(default_factory=list)  # handling margin step-down rules


@dataclass
class Finding:
    """An audit finding to include in the report."""
    id: str
    finding_class: str
    verdict: str  # 'finding', 'no_finding', 'conflict', 'cannot_determine'
    invoice_id: str | None = None
    contract_id: str | None = None
    period: str | None = None
    credit_type: str | None = None
    variant: str | None = None
    amount_usd: Decimal | None = None
    amount_range: tuple[Decimal, Decimal] | None = None
    deadline: str | None = None
    action: str = "none"
    confidence: float = 0.9
    evidence: list[Evidence] = field(default_factory=list)
    missing_doc: MissingDoc | None = None
    supersedes: list[str] = field(default_factory=list)
    retro: bool = False
    learned_at: dict | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "class": self.finding_class,
            "verdict": self.verdict,
            "action": self.action,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
        }
        if self.variant:
            d["variant"] = self.variant
        if self.invoice_id:
            d["invoice_id"] = self.invoice_id
        if self.contract_id:
            d["contract_id"] = self.contract_id
        if self.period:
            d["period"] = self.period
        if self.credit_type:
            d["credit_type"] = self.credit_type
        # Amount
        if self.verdict == "finding" and self.finding_class != "expiry":
            d["amount_usd"] = money_str(self.amount_usd) if self.amount_usd is not None else None
        else:
            d["amount_usd"] = None
        # Range (conflicts)
        if self.verdict == "conflict" and self.amount_range:
            d["amount_range"] = [money_str(self.amount_range[0]), money_str(self.amount_range[1])]
        else:
            d["amount_range"] = None
        # Deadline
        d["deadline"] = self.deadline
        # Missing doc
        d["missing_doc"] = self.missing_doc.to_dict() if self.missing_doc else None
        # Supersedes
        d["supersedes"] = self.supersedes if self.supersedes else []
        # Retro
        if self.retro:
            d["retro"] = True
        # Learned at
        d["learned_at"] = self.learned_at
        return d
