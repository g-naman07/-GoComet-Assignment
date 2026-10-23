import json
import tempfile
import unittest
from pathlib import Path

from harness.harness import _action, _all_text, _rate, conflict_range, missing_rate_reference, money
from harness.engine.state import StateDB
from harness.engine.llm import TextAnalysisProvider


class HarnessUnitTests(unittest.TestCase):
    def test_money_uses_half_up(self):
        self.assertEqual(money(__import__("decimal").Decimal("1.005")), "1.01")

    def test_conflict_range_is_billed_minus_competing_correct_amounts(self):
        D = __import__("decimal").Decimal
        self.assertEqual(conflict_range(D("421.62"), [D("400.75"), D("421.62")]),
                         ["0.00", "20.87"])
        self.assertEqual(conflict_range(D("500"), [D("450"), D("470")]),
                         ["30.00", "50.00"])
        self.assertEqual(conflict_range(D("400"), [D("450"), D("470")]),
                         ["0.00", "0.00"])

    def test_missing_attachment_is_detected_from_document_reference_and_effective_date(self):
        docs = {"amendment": {"pages": [
            "Amendment TBX-1. USS2-INN1 rates replaced: Annex A rev 2 (attached) applies "
            "from 2035-06-04 to lane USS2-INN1."]}}
        self.assertEqual(missing_rate_reference(docs, "TBX-1", "USS2-INN1", "2035-06-21"),
                         (docs["amendment"], "Annex A rev 2"))
        self.assertIsNone(missing_rate_reference(docs, "TBX-1", "USS2-INN1", "2035-06-01"))
        self.assertIsNone(missing_rate_reference(docs, "TBX-1", "USC3-USS2", "2035-06-21"))

    def test_missing_attachment_uses_the_named_revision_not_generic_annex_reference(self):
        text = (
            "TBX-1. The Annex A rates for lane(s) USS2-INN1 are deleted and replaced "
            "as follows: Annex A rev 2 (attached) applies from 2035-06-04 "
            "to lane(s) USS2-INN1."
        )
        docs = {"amendment": {"pages": [text]}}
        self.assertEqual(missing_rate_reference(docs, "TBX-1", "USS2-INN1", "2035-07-01"),
                         (docs["amendment"], "Annex A rev 2"))

    def test_monthly_statement_is_extracted_as_invoice_and_qualifying_lines(self):
        parsed = TextAnalysisProvider().extract_structured(
            "STATEMENT OF ACCOUNT TX-ST-203501\nStatement date 2035-02-03 | Due date 2035-03-20\n"
            "Contract reference TBX-US-EXP-01\nCharge | Reference | Quantity | Rate | Amount\n"
            "Monthly account fee | 2035-01 | 1 month | 350.28 | 350.28\n"
            "Total USD | 1426.50",
            "statement.xlsx", "invoice", "invoice_lines")
        self.assertEqual(parsed["invoice_id"], "TX-ST-203501")
        self.assertEqual(parsed["invoice_date"], "2035-02-03")
        self.assertEqual(parsed["lines"][0]["amount"], 350.28)

    def test_rate_annex_periods_are_read_from_their_stated_dates(self):
        parsed = TextAnalysisProvider().extract_structured(
            "Period P1: pickups 2034-10-20 to 2035-04-19\n"
            "Period P2: pickups 2035-04-20 to 2035-10-19\n"
            "USE2-INW2 100-299.5 kg 5.57 5.55",
            "msa.pdf", "contract", "rate_table")
        self.assertEqual(parsed["rates"][0]["rate_period1_to"], "2035-04-19")
        self.assertEqual(parsed["rates"][0]["rate_period2_from"], "2035-04-20")

    def test_inr_fuel_row_uses_its_billed_amount_not_its_base(self):
        parsed = TextAnalysisProvider().extract_structured(
            "INVOICE TXI/35-36/0030\nInvoice date 2035-08-26\nContract reference TBX-IN-EXP-01\n"
            "TX15472186 Fuel surcharge 2035-08-13 INN3-USC3 996531 "
            "231549.12 17.06% 39502.28",
            "invoice.pdf", "invoice", "invoice_lines")
        fuel = next(line for line in parsed["lines"] if line["charge_type"] == "fuel")
        self.assertEqual(fuel["amount"], 39502.28)
        self.assertEqual(fuel["rate"], 17.06)

    def test_stored_document_pages_are_available_to_semantic_search(self):
        self.assertEqual(_all_text({"pages_json": '["Page one", "Page two"]'}),
                         "Page one\nPage two")

    def test_supervised_never_executes(self):
        self.assertEqual(_action("supervised", "dispute", True), "draft:dispute")
        self.assertEqual(_action("autonomous", "dispute", False), "draft:dispute")
        self.assertEqual(_action("autonomous", "dispute", True), "execute:dispute")

    def test_rate_uses_weight_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = StateDB(tmp)
            db.store_contract("TBX-US-EXP-TEST", {"effective_date": "2030-01-01"}, "contract.pdf")
            db.store_rate("TBX-US-EXP-TEST", "AAA1-BBB2", "0", "44.5", "7.73",
                          source_doc="contract.pdf")
            rate, row = _rate(db, "TBX-US-EXP-TEST", "AAA1-BBB2",
                              __import__("decimal").Decimal("44"), "2030-02-01")
            self.assertEqual(rate, __import__("decimal").Decimal("7.73"))
            self.assertEqual(row["source_doc"], "contract.pdf")
            db.close()

    def test_confirmed_rate_notice_applies_only_within_feedback_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = StateDB(tmp)
            db.store_contract("TBX-1", {"effective_date": "2034-01-01"}, "msa.pdf")
            db.store_rate("TBX-1", "AAA1-BBB2", "100", "299.5", "4.47", 2,
                          "2034-01-01", None, "msa.pdf", 1, 1)
            db.store_rate("TBX-1", "AAA1-BBB2", "100", "299.5", "4.79", 1,
                          "2035-03-13", "2035-09-25", "notice.eml", 1, 0)
            db.store_document("notice.eml", "batch_01", "email", "hash", "2035-03-06",
                              "sender@example.test", "Ada Rates", "Rate notice", [], False)
            self.assertEqual(_rate(db, "TBX-1", "AAA1-BBB2", __import__("decimal").Decimal("120"),
                                   "2035-06-01")[0], __import__("decimal").Decimal("4.47"))
            db.store_feedback_item("fb-1", {"invoice_id": "INV-1", "class": "overcharge"},
                                   "close", {"type": "rate_accepted", "value": {"source": "rate notice",
                                   "sent_by": "Ada Rates"}}, {"kind": "standing", "applies_to": {
                                   "contract_id": "TBX-1", "lane": "AAA1-BBB2"}, "valid_from": "2035-03-13",
                                   "valid_to": "2035-09-25"}, "batch_01", "2035-06-01T00:00:00Z", {})
            self.assertEqual(_rate(db, "TBX-1", "AAA1-BBB2", __import__("decimal").Decimal("120"),
                                   "2035-06-01")[0], __import__("decimal").Decimal("4.79"))
            self.assertEqual(_rate(db, "TBX-1", "AAA1-BBB2", __import__("decimal").Decimal("120"),
                                   "2035-10-01")[0], __import__("decimal").Decimal("4.47"))
            db.close()

    def test_feedback_for_invoice_does_not_leak_to_other_invoices_on_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = StateDB(tmp)
            db.store_feedback_item("fb-1", {"invoice_id": "INV-1", "class": "overcharge"},
                                   "close", None, None, "batch_01", "2030-01-01T00:00:00Z", {})
            self.assertEqual(db.get_feedback_for_key(invoice_id="INV-2", finding_class="overcharge",
                                                     contract_id="CONTRACT-1"), [])
            self.assertEqual(len(db.get_feedback_for_key(invoice_id="INV-1", finding_class="overcharge",
                                                         contract_id="CONTRACT-1")), 1)
            db.close()

    def test_report_shape_is_json_serializable(self):
        report = {
            "schema_version": "1.0", "as_of": "2030-01-01",
            "mode": "supervised", "findings": [],
            "portfolio": {"cashback": [], "expiry": [], "late_fee_risk": []},
        }
        self.assertEqual(json.loads(json.dumps(report))["mode"], "supervised")


if __name__ == "__main__":
    unittest.main()
