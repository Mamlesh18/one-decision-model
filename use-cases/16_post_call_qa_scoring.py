"""Use case 16 - Post-call QA scoring (contact centre quality assurance).

Replaces: supervisors listening to 2% of calls, or an LLM grading every transcript (expensive
at 100% coverage). Laya scores 100% of calls; people review the low-confidence and failing ones.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QA = {
    "greeted_properly": {"type": "noul", "instructions": "Did the agent greet the customer and give their name?"},
    "verified_identity": {"type": "noul", "instructions": "Did the agent verify the customer's identity before discussing the account?"},
    "issue_resolved": {"type": "noul", "instructions": "Was the customer's issue resolved by the end of the call?"},
    "empathy": {"type": "score", "instructions": "How empathetic was the agent?",
                "criteria": ["dismissive", "neutral", "empathetic", "very empathetic"]},
    "compliance_risk": {"type": "noul", "instructions": "Did the agent promise something not allowed, like a guaranteed refund date?"},
    "call_outcome": {"type": "choice", "instructions": "How did the call end?",
                     "criteria": {"resolved": "problem fixed", "follow_up": "ticket or callback created",
                                  "escalated": "transferred to another team", "abandoned": "customer hung up unhappy"}},
}

CALLS = [
    [
        {"speaker": "agent", "text": "Thanks for calling Acme, this is Priya. How can I help?"},
        {"speaker": "customer", "text": "My internet has been down since morning."},
        {"speaker": "agent", "text": "I'm sorry, that's frustrating. Can you confirm your date of birth?"},
        {"speaker": "customer", "text": "12 March 1990."},
        {"speaker": "agent", "text": "Thanks. I've reset your line, can you check now?"},
        {"speaker": "customer", "text": "Yes it works, thank you!"},
    ],
    [
        {"speaker": "agent", "text": "Yeah?"},
        {"speaker": "customer", "text": "I want to know why my bill for account 5521 doubled."},
        {"speaker": "agent", "text": "Don't know. You'll definitely get a full refund tomorrow, bye."},
    ],
]

if __name__ == "__main__":
    run_cases("Post-call QA", QA, CALLS)
