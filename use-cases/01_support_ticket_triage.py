"""Use case 1 - Support ticket triage.

Replaces: a keyword rule engine, or an LLM prompt that returns JSON for every new ticket.
One call returns intent, urgency, frustration and churn risk, each with a probability.

What you do with it:
    - route the ticket to the right queue (intent)
    - sort the queue (urgency, frustration)
    - alert a retention team (churn_risk)
    - send low-confidence tickets to a person instead of guessing
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases, gate

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "refund": "money back or a duplicate charge reversed",
            "technical_help": "a bug, outage, login or integration problem",
            "billing_question": "a question about an invoice, plan or payment method",
            "cancellation": "wants to cancel or downgrade",
            "information": "general information, pricing or how-to",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["can wait", "should be handled today", "blocking the customer right now"],
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": ["calm", "annoyed", "very angry"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel or move to a competitor?"},
}

TICKETS = [
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    "The dashboard shows a 500 error since this morning, our whole team is blocked!!",
    "Could you tell me if the Pro plan includes SSO?",
    "This is the third time I'm writing. Nobody answers. I'm moving to Zendesk next week.",
]

if __name__ == "__main__":
    results = run_cases("Support ticket triage", QUESTIONS, TICKETS)
    print("\nRouting decisions (auto if confident, else a person):")
    for ticket, r in zip(TICKETS, results):
        a = r["answers"]["intent"]
        target = f"queue:{a['choice']}" if gate(a, 0.70) else "human-review"
        print(f"  {target:<24} <- {ticket[:60]}")
    # Laya also ships this as a preset: import laya; laya.triage_questions()
