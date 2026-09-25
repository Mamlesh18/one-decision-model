"""Use case 2 - Inbound email routing and threat triage.

Replaces: shared-inbox rules ("if subject contains 'invoice'...") plus a separate spam filter.
The state can be a dict (from / subject / body); Laya reads all fields together.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should receive this email?",
        "criteria": {
            "finance": "invoices, payments, refunds",
            "support": "product problems and how-to questions",
            "sales": "new business, quotes, pricing",
            "hr": "jobs, candidates, employees",
            "legal": "contracts, NDAs, compliance",
        },
    },
    "phishing": {"type": "noul", "instructions": "Does the email try to steal credentials or money (phishing or fraud)?"},
    "spam": {"type": "noul", "instructions": "Is this unsolicited bulk marketing?"},
    "needs_reply_today": {"type": "noul", "instructions": "Does the sender need an answer today?"},
}

EMAILS = [
    {"from": "ap@vendor.com", "subject": "Invoice #8812 overdue",
     "body": "Hello, invoice 8812 is 30 days overdue. Please arrange payment by Friday."},
    {"from": "security@micros0ft-support.co", "subject": "Password expires",
     "body": "Your password expires in 1 hour. Click here and enter your current password to keep access."},
    {"from": "cto@prospect.io", "subject": "Quote for 200 seats",
     "body": "We are evaluating your platform for 200 users. Can you send pricing this week?"},
    {"from": "deals@shopmail.com", "subject": "50% off everything!!!",
     "body": "Biggest sale of the year, unsubscribe at the bottom."},
]

if __name__ == "__main__":
    run_cases("Email routing and threat triage", QUESTIONS, EMAILS)
    # For raw emails, laya.email_state(...) / laya.clean_email_body(...) strip quotes and signatures,
    # and laya.email_questions() is a ready-made preset.
