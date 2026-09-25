"""Use case 10 - Invoice processing decisions (after OCR / extraction).

Replaces: a manual AP clerk check or a brittle rules table.
The state is JSON: Laya reads structured documents, not only free text.
NOTE: the project's own numbers show this workflow needs the fine-tuned 'typed-decisions'
checkpoint (0.804 on invoices); the base checkpoints are near chance on it. This script runs
both so you can see the difference.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QUESTIONS = {
    "po_match": {"type": "noul", "instructions": "Do the invoice amount and vendor match the purchase order?"},
    "possible_duplicate": {"type": "noul", "instructions": "Is this invoice likely a duplicate of a previously paid one?"},
    "discrepancy": {"type": "score", "instructions": "How serious is any discrepancy?",
                    "criteria": ["none", "minor rounding", "material difference", "suspected fraud"]},
    "action": {"type": "choice", "instructions": "What should accounts payable do?",
               "criteria": {"approve": "everything matches, pay it",
                            "hold": "needs clarification before paying",
                            "reject": "wrong, duplicate or fraudulent"}},
}

INVOICES = [
    {"invoice": {"no": "INV-100", "vendor": "Acme Ltd", "amount": 1200.00, "date": "2026-09-01"},
     "purchase_order": {"no": "PO-77", "vendor": "Acme Ltd", "amount": 1200.00},
     "previously_paid": []},
    {"invoice": {"no": "INV-100", "vendor": "Acme Ltd", "amount": 1200.00, "date": "2026-09-14"},
     "purchase_order": {"no": "PO-77", "vendor": "Acme Ltd", "amount": 1200.00},
     "previously_paid": [{"no": "INV-100", "amount": 1200.00, "paid_on": "2026-09-05"}]},
    {"invoice": {"no": "X-9", "vendor": "Acme Ltd.", "amount": 9800.00, "bank_account_changed": True},
     "purchase_order": {"no": "PO-78", "vendor": "Acme Ltd", "amount": 980.00},
     "previously_paid": []},
]

if __name__ == "__main__":
    run_cases("Invoice decisions - base checkpoint", QUESTIONS, INVOICES)
    run_cases("Invoice decisions - typed-decisions checkpoint", QUESTIONS, INVOICES, model="typed-decisions")
