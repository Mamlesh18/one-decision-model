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
        {"role": "agent", "content": "Thanks for calling Acme, this is Priya. How can I help?"},
        {"role": "customer", "content": "My internet has been down since morning."},
        {"role": "agent", "content": "I'm sorry, that's frustrating. Can you confirm your date of birth?"},
        {"role": "customer", "content": "12 March 1990."},
        {"role": "agent", "content": "Thanks. I've reset your line, can you check now?"},
        {"role": "customer", "content": "Yes it works, thank you!"},
    ],
    [
        {"role": "agent", "content": "Yeah?"},
        {"role": "customer", "content": "I want to know why my bill for account 5521 doubled."},
        {"role": "agent", "content": "Don't know. You'll definitely get a full refund tomorrow, bye."},
    ],
]

if __name__ == "__main__":
    run_cases("Post-call QA", QA, CALLS)
