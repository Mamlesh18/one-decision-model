"""Use case 5 - Lead scoring from free text (form fills, emails, CRM notes).

Replaces: manual SDR qualification, or points-based scoring that only looks at form fields.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QUESTIONS = {
    "fit": {"type": "score", "instructions": "How well does this lead fit a B2B product for contact centres?",
            "criteria": ["no fit", "weak fit", "good fit", "ideal customer"]},
    "buying_stage": {"type": "choice", "instructions": "Where is the lead in the buying process?",
                     "criteria": {"researching": "just learning, no timeline",
                                  "evaluating": "comparing vendors, asking for demos or pricing",
                                  "ready_to_buy": "has budget and a timeline, wants a contract",
                                  "not_a_buyer": "student, job seeker, vendor or competitor"}},
    "persona": {"type": "choice", "instructions": "Who is the person?",
                "criteria": {"executive": "C-level or VP", "manager": "team or department lead",
                             "practitioner": "engineer, agent, analyst", "other": "anyone else"}},
    "budget_mentioned": {"type": "noul", "instructions": "Does the lead mention a budget or approved spend?"},
}

LEADS = [
    "I'm VP Customer Ops at a 400-seat BPO. Budget is approved for Q4 and we want a voicebot pilot. Can we sign by November?",
    "Hi, I'm a student writing a thesis on chatbots, can I get a free license?",
    "Team lead here, we're comparing you with two other vendors, could we get a demo next week?",
]

if __name__ == "__main__":
    run_cases("Lead scoring", QUESTIONS, LEADS)
