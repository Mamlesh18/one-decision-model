"""Use case 12 - Voice AI: semantic end-of-turn detection ("is the caller done talking?").

The problem: VAD only hears silence. People pause mid-sentence ("my account number is... uh...")
so a fixed 700-1000 ms silence timeout either cuts callers off or makes the bot feel slow.

The Laya approach (like the TASA assistant's "Cadence-1"):
    on every STT partial/final -> ask Laya "is the thought complete?"
    - high P(complete)  -> shorten the silence timeout (reply fast)
    - low  P(complete)  -> lengthen it (let them finish)
    - trust the verdict only for ~2 s (freshness window); the transcript moves on.

Replaces: fixed VAD silence timeouts, or a small LLM call per partial (too slow for this).
Limit: Laya reads TEXT only. It does not hear intonation, so combine it with VAD/prosody.
"""
import pathlib, sys, time; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import get_router

TURN_QUESTIONS = {
    "turn_complete": {
        "type": "noul",
        "instructions": "Has the caller finished their thought in the last `caller` line, so the assistant can reply now?",
        "criteria": {"false": "the sentence is cut off, trailing, or they are still listing or spelling something",
                     "true": "a complete request, question or answer the assistant can respond to"},
    },
}

# Simulated STT partials for one caller turn, as they would stream in.
PARTIALS = [
    "I want to",
    "I want to change my",
    "I want to change my delivery address to",
    "I want to change my delivery address to 42 Baker Street",
    "my card number is 4 5 3 2",
    "can you tell me my balance please",
    "yes",
    "so the thing is",
]

DEFAULT_SILENCE_MS = 700
FRESHNESS_S = 2.0


def silence_timeout(p_complete: float) -> int:
    """Map the probability to how long we wait in silence before the bot speaks."""
    if p_complete >= 0.80:
        return 250           # clearly done: answer quickly
    if p_complete <= 0.30:
        return 1500          # clearly mid-thought: give them room
    return DEFAULT_SILENCE_MS


class TurnVerdict:
    """Cache the latest verdict with a timestamp so stale ones are ignored."""

    def __init__(self):
        self.p, self.at = None, 0.0

    def update(self, p: float):
        self.p, self.at = p, time.monotonic()

    def timeout_ms(self) -> int:
        if self.p is None or time.monotonic() - self.at > FRESHNESS_S:
            return DEFAULT_SILENCE_MS
        return silence_timeout(self.p)


if __name__ == "__main__":
    router = get_router()
    verdict = TurnVerdict()
    history = [{"speaker": "assistant", "text": "Hi, this is Acme support. How can I help you today?"}]
    print(f"{'partial transcript':<60} P(done)  wait   latency")
    for text in PARTIALS:
        state = history + [{"speaker": "caller", "text": text}]
        t0 = time.perf_counter()
        r = router.predict(state, TURN_QUESTIONS, model="english")
        ms = (time.perf_counter() - t0) * 1000
        p = r["answers"]["turn_complete"]["noul"]
        verdict.update(p)
        print(f"{text:<60} {p:5.2f}  {verdict.timeout_ms():>5}ms  {ms:5.0f}ms")
    print("\nIf Laya's latency is above your STT partial interval, only score FINAL partials or")
    print("the latest partial (drop stale requests). See examples/07_voice_pipeline_simulation.py")
