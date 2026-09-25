"""Use case 7 - Aspect-based sentiment (one question per aspect).

Replaces: overall-only sentiment models that say "negative" without saying about WHAT.
Score questions give an ordinal level; noul tells you whether a specific complaint is present.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
QUESTIONS = {
    "food": {"type": "score", "instructions": "How does the reviewer feel about the food?", "criteria": LEVELS},
    "service": {"type": "score", "instructions": "How does the reviewer feel about the service and staff?", "criteria": LEVELS},
    "price": {"type": "score", "instructions": "How does the reviewer feel about the price or value?", "criteria": LEVELS},
    "complains_about_waiting": {"type": "noul", "instructions": "Does the review complain about waiting?"},
    "emotion": {"type": "choice", "instructions": "What is the main emotion?",
                "criteria": {"joy": "happy, delighted", "anger": "angry, outraged",
                             "disappointment": "let down, sad", "neutral": "no strong emotion"}},
}

REVIEWS = [
    "The pasta was incredible but we waited 50 minutes and the waiter was rude. Pricey too.",
    "Friendly staff, fair prices, food was just ok.",
]

if __name__ == "__main__":
    run_cases("Aspect-based sentiment", QUESTIONS, REVIEWS)
    print("\nNote: score is the weakest question type per the project (SST-5 0.372). Validate on your data.")
