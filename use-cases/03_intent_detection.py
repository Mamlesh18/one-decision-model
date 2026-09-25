"""Use case 3 - Intent detection for chatbots, IVR / voice menus and search.

Replaces: a trained NLU intent classifier (Rasa, Dialogflow, LUIS) or an LLM call per utterance.
Key difference: intents are defined PER REQUEST, so you change the menu without retraining.
Add an "unclear" option and a confidence gate to decide when to ask a follow-up question.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases, gate

INTENTS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the caller want to do?",
        "criteria": {
            "check_balance": "hear the account balance",
            "block_card": "block or freeze a lost or stolen card",
            "dispute_charge": "complain about a transaction they do not recognise",
            "talk_to_agent": "speak to a human",
            "update_address": "change address or contact details",
            "unclear": "the request is not understandable or matches none of these",
        },
    },
}

UTTERANCES = [
    "yeah hi I think I lost my card somewhere at the mall",
    "how much money do I have left",
    "there's a charge from some shop in Dubai I never went to",
    "agent. AGENT. human please",
    "uh what",
    "I moved last month, new place is on Baker Street",
]

if __name__ == "__main__":
    results = run_cases("Intent detection (voice menu / chatbot)", INTENTS, UTTERANCES)
    print("\nDialogue policy:")
    for u, r in zip(UTTERANCES, results):
        a = r["answers"]["intent"]
        if a["choice"] == "unclear" or not gate(a, 0.60):
            action = "ASK: 'Sorry, could you say that again?'"
        else:
            action = f"GO:  {a['choice']}"
        print(f"  {action:<42} <- {u}")
