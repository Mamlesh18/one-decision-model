"""Use case 9 - Patient message triage (clinic inbox).

Replaces: a receptionist reading every portal message first.
SAFETY: this sorts the queue; it never replaces a clinician. Catch emergencies with rules
(keywords like "chest pain") AND the model, and send uncertain items to a person.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QUESTIONS = {
    "urgency": {"type": "score", "instructions": "How urgent is the patient's message medically?",
                "criteria": ["routine, can wait days", "needs a reply within 24 hours",
                             "same day", "emergency, call emergency services now"]},
    "department": {"type": "choice", "instructions": "Who should handle the message?",
                   "criteria": {"front_desk": "appointments, rescheduling, directions",
                                "billing": "bills, insurance, payments",
                                "pharmacy": "prescription refills and medication questions",
                                "nurse": "symptoms or clinical questions"}},
    "needs_clinician": {"type": "noul", "instructions": "Does the message describe symptoms that a clinician should review?"},
}

MESSAGES = [
    "Can I move my appointment on Monday to Wednesday afternoon?",
    "I need a refill of my blood pressure tablets, I have 2 left.",
    "Since an hour I have strong chest pain going into my left arm and I'm sweating.",
    "My insurance was charged twice for the same visit.",
]

if __name__ == "__main__":
    run_cases("Patient message triage (a person always in the loop)", QUESTIONS, MESSAGES)
