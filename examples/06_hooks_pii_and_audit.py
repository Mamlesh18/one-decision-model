"""Example 6 - Hooks: redact PII before the model sees it, and log every decision.

on_predict_start(ctx): can rewrite ctx.states / ctx.questions, or ctx.skip(results) to use a cache
on_predict_end(ctx)  : sees ctx.results, ctx.model, ctx.elapsed_ms (audit logs, metrics)
"""
import json
import re
from laya import Router

PHONE = re.compile(r"\+?\d[\d \-]{7,}\d")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def redact(ctx):
    def clean(s):
        if isinstance(s, str):
            return EMAIL.sub("<email>", PHONE.sub("<phone>", s))
        if isinstance(s, dict):
            return {k: clean(v) for k, v in s.items()}
        if isinstance(s, list):
            return [clean(v) for v in s]
        return s
    ctx.states = [clean(s) for s in ctx.states]


def audit(ctx):
    for state, res in zip(ctx.states, ctx.results or []):
        record = {"model": ctx.model, "ms": round(ctx.elapsed_ms or 0, 1), "state": state,
                  "answers": {k: v.get("choice", v.get("score", v.get("noul"))) for k, v in res["answers"].items()}}
        print("AUDIT", json.dumps(record, ensure_ascii=False))


router = Router(on_predict_start=redact, on_predict_end=audit)
router.predict("Call me on +91 98765 43210 or mail ravi@example.com, my card was charged twice",
               {"billing": {"type": "noul", "instructions": "Is this a billing problem?"}})
