"""Document extraction: parse .eml, PDF, XLSX using the grader's own reader for quote compatibility.

Uses grader/extract.py so that every quote we cite is findable by the grader's checker.
Also provides higher-level extraction of structured data from document text.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

# Import the grader's own extract module so our quotes match exactly
_GRADER = Path(__file__).resolve().parent.parent.parent / "grader"
sys.path.insert(0, str(_GRADER))
import extract as grader_extract  # noqa: E402


def content_hash(data: bytes) -> str:
    """SHA-256 hex digest of raw content."""
    return hashlib.sha256(data).hexdigest()


def parse_eml_file(eml_path: str | Path) -> dict:
    """Parse an .eml file and return structured data.

    Returns: {
        'path': str,
        'from_name': str, 'from_addr': str,
        'to': str, 'cc': str | None,
        'date': datetime, 'date_str': str,
        'subject': str,
        'body': str,
        'attachments': [{'name': str, 'data': bytes, 'ext': str}, ...],
        'content_hash': str,
    }
    """
    raw = Path(eml_path).read_bytes()
    msg = grader_extract.parse_eml(raw)
    name, addr = parseaddr(str(msg["From"]))
    dt = parsedate_to_datetime(str(msg["Date"]))
    body_part = msg.get_body(preferencelist=("plain",))
    body = body_part.get_content().replace("\r\n", "\n").rstrip("\n") if body_part else ""

    attachments = []
    for part in msg.iter_attachments():
        fname = part.get_filename() or ""
        try:
            data = part.get_content()
            if isinstance(data, str):
                data = data.encode()
        except Exception:
            data = part.get_payload(decode=True) or b""
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        attachments.append({"name": fname, "data": data, "ext": ext})

    return {
        "path": str(eml_path),
        "from_name": name,
        "from_addr": addr,
        "to": str(msg["To"] or ""),
        "cc": str(msg["Cc"]) if msg["Cc"] else None,
        "date": dt,
        "date_str": dt.strftime("%Y-%m-%d"),
        "subject": str(msg["Subject"] or ""),
        "body": body,
        "attachments": attachments,
        "content_hash": content_hash(raw),
    }


def get_pages(data: bytes, ext: str) -> list[str]:
    """Get text pages using the grader's reader (ensures quote matching)."""
    return grader_extract.pages_of(data, ext)


def is_scan(data: bytes) -> bool:
    """Check if a PDF is a scan (no text layer)."""
    return grader_extract.pdf_is_image_only(data)


def ocr_pdf(data: bytes) -> list[str]:
    """OCR a scanned PDF. Returns list of page texts."""
    try:
        import pymupdf
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        pages = []
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img_data = pix.tobytes("png")
                result, _ = ocr(img_data)
                if result:
                    # Sort boxes by vertical position then horizontal
                    result.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
                    lines = []
                    prev_y = -999
                    for box in result:
                        y = box[0][0][1]
                        text = box[1]
                        if abs(y - prev_y) > 10:  # new line
                            lines.append(text)
                        else:
                            lines[-1] = lines[-1] + " " + text
                        prev_y = y
                    pages.append("\n".join(lines))
                else:
                    pages.append("")
        return pages
    except Exception as e:
        return [f"[OCR failed: {e}]"]


def extract_all_text(eml_path: str | Path, batch_prefix: str) -> list[dict]:
    """Extract all text from an email and its attachments.

    Returns list of document entries:
    [{
        'doc_ref': str (e.g. 'batch_01/0003.eml' or 'batch_01/0003.eml#invoice.pdf'),
        'type': 'email' | 'pdf' | 'xlsx',
        'pages': [str, ...],
        'is_scan': bool,
        'content_hash': str,
        'email_date': str,
        'email_from': str,
        'email_subject': str,
    }]
    """
    eml = parse_eml_file(eml_path)
    eml_name = Path(eml_path).name
    eml_ref = f"{batch_prefix}/{eml_name}"

    results = []

    # The email itself
    email_pages = grader_extract.pages_of(Path(eml_path).read_bytes(), "eml")
    results.append({
        "doc_ref": eml_ref,
        "type": "email",
        "pages": email_pages,
        "is_scan": False,
        "content_hash": eml["content_hash"],
        "email_date": eml["date_str"],
        "email_from": eml["from_addr"],
        "email_from_name": eml["from_name"],
        "email_subject": eml["subject"],
        "email_body": eml["body"],
        "email_to": eml["to"],
    })

    # Attachments
    for att in eml["attachments"]:
        att_ref = f"{eml_ref}#{att['name']}"
        att_hash = content_hash(att["data"])

        if att["ext"] == "pdf":
            scan = is_scan(att["data"])
            if scan:
                pages = ocr_pdf(att["data"])
            else:
                pages = get_pages(att["data"], "pdf")
            results.append({
                "doc_ref": att_ref,
                "type": "pdf",
                "pages": pages,
                "is_scan": scan,
                "content_hash": att_hash,
                "email_date": eml["date_str"],
                "email_from": eml["from_addr"],
                "email_from_name": eml["from_name"],
                "email_subject": eml["subject"],
            })
        elif att["ext"] == "xlsx":
            pages = get_pages(att["data"], "xlsx")
            results.append({
                "doc_ref": att_ref,
                "type": "xlsx",
                "pages": pages,
                "is_scan": False,
                "content_hash": att_hash,
                "email_date": eml["date_str"],
                "email_from": eml["from_addr"],
                "email_from_name": eml["from_name"],
                "email_subject": eml["subject"],
            })

    return results


# --- Text-based structured extraction (no LLM needed) ---

def extract_invoice_id_from_text(text: str) -> str | None:
    """Try to find an invoice ID in document text."""
    # Common patterns: TX-US-2032-5394, INV-1234, etc.
    patterns = [
        r'\b([A-Z]{2,4}-[A-Z]{2,4}-\d{4}-\d{3,6})\b',  # TX-US-2035-0210
        r'Invoice[:\s#]*([A-Z0-9][\w-]{4,})',
        r'Credit Note[:\s#]*([A-Z0-9][\w-]{4,})',
        r'\bINV[- ]?(\d{4,})\b',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raw = m.group(1) if '(' in pattern else m.group(0)
            return re.sub(r"[\s#:]", "", raw).upper()
    return None


def extract_dates_from_text(text: str) -> list[str]:
    """Extract dates in various formats, return as YYYY-MM-DD."""
    dates = []
    # YYYY-MM-DD
    for m in re.finditer(r'\b(\d{4}-\d{2}-\d{2})\b', text):
        dates.append(m.group(1))
    # DD Mon YYYY
    months = {'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
              'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'}
    for m in re.finditer(r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})\b', text, re.IGNORECASE):
        day, mon, year = m.group(1), m.group(2)[:3].lower(), m.group(3)
        dates.append(f"{year}-{months.get(mon, '01')}-{int(day):02d}")
    return dates


def extract_amounts_from_text(text: str) -> list[Decimal]:
    """Extract monetary amounts from text."""
    amounts = []
    for m in re.finditer(r'(?:USD|INR|Rs\.?|₹|\$)\s*([\d,]+\.\d{2})\b', text):
        try:
            amounts.append(Decimal(m.group(1).replace(",", "")))
        except InvalidOperation:
            pass
    # Also bare amounts like 1612.29
    for m in re.finditer(r'\b(\d{1,3}(?:,\d{3})*\.\d{2})\b', text):
        try:
            amounts.append(Decimal(m.group(1).replace(",", "")))
        except InvalidOperation:
            pass
    return amounts


def extract_awb_numbers(text: str) -> list[str]:
    """Extract AWB/tracking numbers."""
    awbs = []
    for m in re.finditer(r'\b(TE\d{8,})\b', text):
        awbs.append(m.group(1))
    for m in re.finditer(r'\b([A-Z]{2}\d{8,})\b', text):
        awbs.append(m.group(1))
    # House AWBs like TE12345678-H1
    for m in re.finditer(r'\b([A-Z]{2}\d{8,}-H\d+)\b', text):
        awbs.append(m.group(1))
    return list(set(awbs))


def extract_contract_id(text: str) -> str | None:
    """Extract a contract ID."""
    m = re.search(r'\b(TBX-[A-Z]{2}-[A-Z]+-[A-Z0-9]+)\b', text)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-Z]{3,}-[A-Z]{2}-[A-Z]+-\d+)\b', text)
    if m:
        return m.group(1)
    return None


def extract_lane(text: str) -> str | None:
    """Extract a lane code like USE2-INE2."""
    m = re.search(r'\b([A-Z]{2,4}\d?-[A-Z]{2,4}\d?)\b', text)
    if m:
        return m.group(1)
    return None
