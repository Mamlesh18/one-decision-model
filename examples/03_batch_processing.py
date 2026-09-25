"""Example 3 - Batch: score many items in shared forward passes (big win on GPU).

Use for backlogs: re-tagging old tickets, QA-scoring yesterday's calls, cleaning a CSV.
"""
import time
from laya import Router

router = Router()
questions = {"sentiment": {"type": "choice", "instructions": "What is the sentiment?",
                           "criteria": {"positive": "happy", "negative": "unhappy", "neutral": "neither"}}}

texts = ["Love it!", "Worst support ever.", "It arrived on Tuesday.", "Not bad at all",
         "I want my money back", "Thanks for the quick fix"] * 5

# One by one
t0 = time.perf_counter()
one_by_one = [router.predict(t, questions) for t in texts]
loop_ms = (time.perf_counter() - t0) * 1000

# Batched through the Router (routes first, groups by checkpoint, keeps input order)
requests = [{"state": t, "questions": questions} for t in texts]
t0 = time.perf_counter()
batched = router.predict_batch(requests, batch_size=16)
batch_ms = (time.perf_counter() - t0) * 1000

print(f"{len(texts)} items: loop {loop_ms:.0f} ms, batch {batch_ms:.0f} ms")
for t, r in list(zip(texts, batched))[:6]:
    print(f"  {r['answers']['sentiment']['choice']:<9} {t}")

agree = sum(a["answers"]["sentiment"]["choice"] == b["answers"]["sentiment"]["choice"]
            for a, b in zip(one_by_one, batched))
print(f"loop vs batch agreement: {agree}/{len(texts)} (tiny float differences can flip near-ties)")
