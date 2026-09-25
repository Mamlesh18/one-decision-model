"""Use case 14 - Voice / chat: live per-turn context (intent, sentiment, topic, complexity).

After every caller turn, compute a small "context vector" in one forward pass and use it to:
    - pick the prompt / sub-agent for the LLM (topic, intent)
    - switch tone when sentiment drops
    - choose a cheaper or stronger LLM (complexity)
    - log everything for analytics ("shadow mode") and only ACT when confident

Replaces: an extra LLM "analysis" call per turn, or a stack of separate classifiers.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import get_router, show

CONTEXT_Q = {
    "intent": {"type": "choice", "instructions": "What does the caller want in their last turn?",
               "criteria": {"ask_info": "asks a question", "give_info": "provides requested details",
                            "complain": "complains or reports a problem", "confirm": "agrees or confirms",
                            "deny": "refuses or says no", "goodbye": "wants to end the call"}},
    "sentiment": {"type": "score", "instructions": "How does the caller feel right now?",
                  "criteria": ["angry", "frustrated", "neutral", "satisfied"]},
    "topic": {"type": "choice", "instructions": "What is the conversation about?",
              "criteria": {"billing": "charges, refunds, payments", "delivery": "orders, shipping",
                           "account": "login, password, profile", "product": "features, how-to"}},
    "complexity": {"type": "score", "instructions": "How complex is the caller's problem?",
                   "criteria": ["simple, one step", "several steps", "needs an expert or a person"]},
    "wants_human": {"type": "noul", "instructions": "Does the caller ask for a human agent?"},
}

CALL = [
    {"speaker": "assistant", "text": "Hi, how can I help?"},
    {"speaker": "caller", "text": "My parcel was supposed to arrive Monday and it's still not here."},
    {"speaker": "assistant", "text": "Sorry about that. Can I have your order number?"},
    {"speaker": "caller", "text": "It's 7 7 1 2 0. And honestly this is the second time this happens."},
    {"speaker": "assistant", "text": "Thanks. It shows delivered yesterday at 6pm."},
    {"speaker": "caller", "text": "That's wrong, nothing was delivered! Just get me a real person."},
]

CONFIDENT = 0.70

if __name__ == "__main__":
    router = get_router()
    for i in range(1, len(CALL), 2):   # after each caller turn
        history = CALL[: i + 1]
        print(f"\n--- after caller: {history[-1]['text']}")
        r = router.predict(history, CONTEXT_Q)
        show(r)
        a = r["answers"]
        if a["wants_human"]["noul"] >= 0.8 or (a["sentiment"]["score"] < 1.0 and a["sentiment"]["answer_confidence"] >= CONFIDENT):
            print("  => ACTION: escalate / switch to empathetic prompt")
        else:
            print("  => shadow-log only, keep the LLM's current plan")
