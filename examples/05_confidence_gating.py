"""Example 5 - Confidence gating: automate the sure cases, send the rest to a person / an LLM.

This is the main reason to use a calibrated decision model instead of a plain LLM:
you get a probability you can put a threshold on.

Which number to gate on:
    answer_confidence  = probability of the answer it picked (same meaning on every type) <- use this
    confidence         = on choice/score it is 1 - normalised entropy (NOT a probability)

The threshold is YOUR policy. Pick it from benchmarks/bench_accuracy.py output on your own
labelled data: "at threshold 0.8 we automate 70% of traffic at 96% accuracy".
"""
from laya import Router

router = Router()
q = {"intent": {"type": "choice", "instructions": "What does the customer want?",
                "criteria": {"refund": "money back", "cancel": "end the subscription",
                             "upgrade": "move to a bigger plan", "other": "anything else"}}}

THRESHOLD = 0.75

for text in ["Refund my last payment please", "cancel my account right now",
             "I'm not sure this plan is right for me", "hmm"]:
    a = router.predict(text, q)["answers"]["intent"]
    if a["answer_confidence"] >= THRESHOLD:
        decision = f"AUTO   -> {a['choice']}"
    else:
        decision = "ESCALATE -> LLM or human (Laya's best guess: %s)" % a["choice"]
    print(f"{text:<40} p={a['answer_confidence']:.2f}  {decision}")

print("\nNote: the project says the shipped checkpoints are over-confident and laya-multilingual")
print("has no fitted temperatures. Measure accuracy per confidence bucket before you pick a threshold.")
