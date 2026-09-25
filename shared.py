"""Small helpers shared by use-cases/, examples/ and benchmarks/.

Every script in this repo does:

    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from shared import get_router, show

so it can be run directly with `python use-cases/01_support_ticket_triage.py`.

Environment variables:
    LAYA_DEVICE   cpu | cuda | mps   (default: Laya picks)
    LAYA_PRELOAD  1 to load english + multilingual up front (default 0: lazy)
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional


def load_env(path: Optional[str] = None) -> None:
    """Load KEY=VALUE lines from the repo's .env into os.environ (existing variables win)."""
    env = pathlib.Path(path) if path else pathlib.Path(__file__).resolve().parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache(maxsize=1)
def get_router():
    """One Router per process. The first predict() downloads a checkpoint (~1 GB) from Hugging Face."""
    from laya import Router

    device = os.environ.get("LAYA_DEVICE") or None
    router = Router(device=device)
    if os.environ.get("LAYA_PRELOAD") == "1":
        router.preload(["english", "multilingual"])
    return router


def _bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "#" * n + "." * (width - n)


def show(result: Dict[str, Any], elapsed_ms: Optional[float] = None) -> None:
    """Pretty-print one Laya result: every answer, its probabilities and its confidence."""
    routing = result.get("routing") or {}
    head = f"  model={routing.get('model', '?')}"
    if elapsed_ms is not None:
        head += f"  time={elapsed_ms:.0f} ms"
    print(head)
    for qid, a in result["answers"].items():
        conf = a.get("answer_confidence", 0.0)
        if a["type"] == "choice":
            print(f"  {qid:<22} choice = {a['choice']:<18} (conf {conf:.2f})")
            for label, p in sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:4]:
                print(f"      {label:<20} {_bar(p)} {p:.2f}")
        elif a["type"] == "score":
            level = int(round(a["score"]))
            print(f"  {qid:<22} score  = {a['score']:.2f} -> '{a['legend'][str(level)]}' (conf {conf:.2f})")
        else:  # noul = probability the answer is "true / yes"
            verdict = "YES" if a["noul"] >= 0.5 else "no"
            print(f"  {qid:<22} P(yes) = {a['noul']:.2f} {_bar(a['noul'])} {verdict}")


def run_cases(title: str, questions: Dict[str, Any], cases: Iterable[Any], **predict_kwargs) -> list:
    """Run the same questions over several states and print each result."""
    router = get_router()
    print("=" * 78)
    print(title)
    print("=" * 78)
    results = []
    for state in cases:
        preview = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        print(f"\n> {preview[:150]}")
        t0 = time.perf_counter()
        result = router.predict(state, questions, **predict_kwargs)
        show(result, (time.perf_counter() - t0) * 1000)
        results.append(result)
    return results


def gate(answer: Dict[str, Any], threshold: float) -> bool:
    """True when an answer is confident enough to act on automatically.

    Uses `answer_confidence` (probability of the reported answer), the one number that means
    the same thing on choice, score and noul. Fit the threshold on your own labelled data.
    """
    return answer.get("answer_confidence", 0.0) >= threshold
