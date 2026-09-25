"""Example 7 - Where Laya sits in a real-time voice pipeline (simulated, no audio needed).

    Mic -> VAD -> streaming STT --partials--> [ Laya decisions ] --> dialogue manager -> LLM -> TTS
                                                  |  turn_complete   (when to reply)
                                                  |  interrupt kind  (stop TTS or not)
                                                  |  intent/sentiment (which prompt / escalate)

Rules this example follows (they matter more than the model):
    1. Never block the audio loop: run Laya in a worker thread.
    2. Only score the LATEST partial; drop older in-flight requests (they are stale).
    3. Trust a verdict for a short freshness window (2 s), else fall back to VAD defaults.
    4. Fail open: if Laya is slow or errors, the bot still works on plain VAD timing.
"""
import asyncio
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import get_router

TURN_Q = {"turn_complete": {"type": "noul",
                            "instructions": "Has the caller finished their thought, so the assistant can reply now?"}}

# (seconds since turn start, STT partial) as they would stream in from STT
STREAM = [(0.3, "I"), (0.6, "I need to"), (0.9, "I need to reschedule"),
          (1.3, "I need to reschedule my appointment"), (1.8, "I need to reschedule my appointment to"),
          (2.4, "I need to reschedule my appointment to next Friday morning")]

BUDGET_S = 1.5   # latency budget per decision; ~0.1 s is realistic on GPU, CPU needs more
pool = ThreadPoolExecutor(max_workers=1)
latest = {"seq": 0, "p": None, "at": 0.0}


async def score(seq, text, history):
    loop = asyncio.get_running_loop()
    state = history + [{"role": "caller", "content": text}]
    t0 = time.perf_counter()
    try:
        r = await asyncio.wait_for(
            loop.run_in_executor(pool, lambda: get_router().predict(state, TURN_Q, model="english")),
            timeout=BUDGET_S)
    except Exception as e:  # fail open
        print(f"      laya skipped ({type(e).__name__}), using VAD default")
        return
    if seq != latest["seq"]:
        print(f"      dropped stale result for partial #{seq}")
        return
    latest.update(p=r["answers"]["turn_complete"]["noul"], at=time.monotonic())
    print(f"      #{seq} P(done)={latest['p']:.2f} in {(time.perf_counter() - t0) * 1000:.0f} ms")


async def main():
    get_router().predict("warm up", TURN_Q, model="english")   # load the model before the call starts
    history = [{"role": "assistant", "content": "Hello, how can I help you?"}]
    start = time.monotonic()
    tasks = []
    for seq, (t, text) in enumerate(STREAM, 1):
        await asyncio.sleep(max(0, start + t - time.monotonic()))
        latest["seq"] = seq
        print(f"[{t:.1f}s] STT partial: {text!r}")
        tasks.append(asyncio.create_task(score(seq, text, history)))
    await asyncio.gather(*tasks)

    fresh = latest["p"] is not None and time.monotonic() - latest["at"] < 2.0
    wait = 250 if fresh and latest["p"] >= 0.8 else 1500 if fresh and latest["p"] <= 0.3 else 700
    print(f"\nEnd-of-turn silence timeout for this turn: {wait} ms (VAD default 700 ms)")

asyncio.run(main())
