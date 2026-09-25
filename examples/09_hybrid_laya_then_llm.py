"""Example 9 - The hybrid pattern: Laya first (System 1), LLM only when needed (System 2).

    user text --> Laya (tens of ms, ~free)
                   |-- confident + simple  --> canned answer / direct action (no LLM)
                   |-- needs generation    --> small LLM with the right prompt
                   '-- hard or low conf    --> big LLM or a person

This is how you cut LLM cost and latency in a bot without losing quality: most turns
in a support bot are simple and repetitive. The LLM calls are stubbed here; plug in yours.
"""
from laya import Router

router = Router()

Q = {
    "intent": {"type": "choice", "instructions": "What does the user want?",
               "criteria": {"greeting": "hello, hi, small talk", "thanks_bye": "thanks or goodbye",
                            "order_status": "where is my order", "store_hours": "opening hours",
                            "complex": "anything that needs explanation, advice or several steps"}},
    "needs_big_model": {"type": "noul", "instructions": "Does answering need careful reasoning or a long explanation?"},
}

CANNED = {"greeting": "Hi! How can I help?", "thanks_bye": "You're welcome, bye!",
          "store_hours": "We're open 9am to 9pm, Monday to Saturday."}


def small_llm(text):  # stub
    return f"[small LLM answers: {text!r}]"


def big_llm(text):  # stub
    return f"[big LLM answers: {text!r}]"


def reply(text):
    a = router.predict(text, Q)["answers"]
    intent = a["intent"]
    if intent["answer_confidence"] >= 0.8 and intent["choice"] in CANNED:
        return "canned", CANNED[intent["choice"]]
    if intent["answer_confidence"] >= 0.8 and intent["choice"] == "order_status":
        return "tool", "[call order API, then template the answer]"
    if a["needs_big_model"]["noul"] >= 0.6 or intent["answer_confidence"] < 0.5:
        return "big-llm", big_llm(text)
    return "small-llm", small_llm(text)


for t in ["hi there", "when do you open on saturday?", "where's my order 5512",
          "My router drops wifi every evening and I already tried resetting, what else can cause this?",
          "thanks, bye"]:
    path, answer = reply(t)
    print(f"{path:<10} {t[:55]:<57} {answer[:60]}")
