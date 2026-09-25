"""Use case 13 - Voice AI: barge-in / interruption semantics.

The problem: while the bot is speaking, the caller says something. VAD fires. Should the bot stop?
    "uh-huh", "yeah", "okay"            -> backchannel: KEEP speaking
    "no no, that's wrong"               -> disagreement: STOP now and listen
    "wait, what about my other order?"  -> new question: STOP and answer
    TV / cough / the bot's own echo     -> noise: ignore

Replaces: "stop TTS on any voice activity" (bot stops on every 'mm-hmm') or a min-word-count
hack (bot talks over real objections). This mirrors TASA's interrupt handling.
"""
import pathlib, sys, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import get_router

INTERRUPT_Q = {
    "kind": {
        "type": "choice",
        "instructions": "The assistant was speaking when the caller said the last line. What kind of interruption is it?",
        "criteria": {
            "backchannel": "short listening signal like uh-huh, yeah, okay, right, mm",
            "disagreement": "the caller objects, corrects or says the assistant is wrong",
            "new_request": "the caller asks a new question or changes the topic",
            "stop_request": "the caller wants the assistant to stop, wait or be quiet",
            "noise": "not addressed to the assistant: background talk, filler or unrelated words",
        },
    },
}

ACTIONS = {
    "backchannel": "KEEP SPEAKING",
    "disagreement": "CANCEL TTS -> listen",
    "new_request": "CANCEL TTS -> answer new request",
    "stop_request": "CANCEL TTS -> wait",
    "noise": "IGNORE",
}

BOT_SAYING = "Your refund of 49 dollars will reach your card in five to seven business days, and"
CALLER_SAYS = [
    "uh-huh",
    "yeah okay",
    "no no, I paid 94 not 49",
    "wait, what about my second order?",
    "hold on, stop",
    "honey can you turn the TV down",
]

if __name__ == "__main__":
    router = get_router()
    for said in CALLER_SAYS:
        state = [{"speaker": "assistant (speaking, interrupted)", "text": BOT_SAYING},
                 {"speaker": "caller", "text": said}]
        t0 = time.perf_counter()
        a = router.predict(state, INTERRUPT_Q, model="english")["answers"]["kind"]
        ms = (time.perf_counter() - t0) * 1000
        # Be conservative: stopping on a real objection matters more than on a backchannel.
        action = ACTIONS[a["choice"]] if a["answer_confidence"] >= 0.5 else "PAUSE TTS briefly, re-check"
        print(f"{said:<42} {a['choice']:<13} conf={a['answer_confidence']:.2f}  {action:<32} {ms:4.0f}ms")
    print("\nTip: run a cheap word-list check first ('uh-huh', 'yeah') and only call Laya for the rest.")
