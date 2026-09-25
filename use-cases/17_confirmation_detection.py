"""Use case 17 - Voice / chat bots: did the user confirm, deny, or neither?

Every bot asks "Shall I book that for you?" and then has to understand
"yeah go ahead" / "nah" / "hmm let me think" / "yes but make it 5pm".
Keyword lists break on these; an LLM call per yes/no is slow and costly.

NOTE: the project reports that negation can fool the model (issue #377) and that noul
on the English checkpoint can follow the "true/false" labels. That is why this uses a
choice with neutral keys. Validate on your own utterances before relying on it.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import get_router

CONFIRM_Q = {
    "reply": {
        "type": "choice",
        "instructions": "How did the user answer the assistant's yes/no question?",
        "criteria": {
            "A": "agrees, accepts, says go ahead",
            "B": "declines, refuses, says do not do it",
            "C": "agrees but asks to change a detail",
            "D": "unsure, thinking, or asks something else",
        },
    },
}
MEANING = {"A": "CONFIRM", "B": "DENY", "C": "CONFIRM_WITH_CHANGE", "D": "UNCLEAR"}

BOT = "Shall I book the appointment for Thursday at 3pm?"
REPLIES = ["yeah go ahead", "yep", "nah", "no, don't book it", "hmm let me check my calendar",
           "yes but make it 5pm", "sure why not", "what's the address again?"]

if __name__ == "__main__":
    router = get_router()
    for reply in REPLIES:
        state = [{"role": "assistant", "content": BOT}, {"role": "user", "content": reply}]
        a = router.predict(state, CONFIRM_Q, model="english")["answers"]["reply"]
        label = MEANING[a["choice"]] if a["answer_confidence"] >= 0.55 else "UNCLEAR (re-ask)"
        print(f"{reply:<35} -> {label:<22} conf={a['answer_confidence']:.2f}")
