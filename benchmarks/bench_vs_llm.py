"""Head-to-head: Laya vs the LLM you use today, on the same labelled rows.

Works with any OpenAI-compatible chat endpoint (OpenAI, Azure OpenAI, Groq, vLLM, Ollama, LiteLLM...):

    # bash
    export LLM_BASE_URL=https://api.openai.com/v1  LLM_API_KEY=sk-...  LLM_MODEL=gpt-4o-mini
    # PowerShell
    $env:LLM_BASE_URL="http://localhost:11434/v1"; $env:LLM_MODEL="llama3.1"

    python benchmarks/bench_vs_llm.py --task voice_intent

This answers the real question: "can Laya replace this LLM call?"
Compare accuracy AND p50/p95 latency AND cost per 1k calls. If Laya is within a few points of
accuracy and 5-10x faster, it can take the call; if not, use Laya as a first pass and keep the
LLM for low-confidence items (examples/09_hybrid_laya_then_llm.py).
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from shared import get_router   # noqa: E402
from tasks import TASKS          # noqa: E402
from bench_accuracy import load_rows, predicted   # noqa: E402


def llm_answer(state, spec):
    q = spec["question"]
    if q["type"] == "choice":
        options = "\n".join(f"- {k}: {v}" for k, v in q["criteria"].items())
        instr = f"{q['instructions']}\nOptions:\n{options}\nReply with the option key only."
    else:
        instr = f"{q['instructions']}\nReply with true or false only."
    body = {"model": os.environ["LLM_MODEL"], "temperature": 0,
            "messages": [{"role": "system", "content": "You are a classifier. Output only the answer."},
                         {"role": "user", "content": f"Input:\n{json.dumps(state, ensure_ascii=False)}\n\n{instr}"}]}
    req = urllib.request.Request(os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"].strip().strip(".").strip("`").lower()
    if q["type"] == "noul":
        return text.startswith("true") or text.startswith("yes")
    return next((k for k in q["criteria"] if k.lower() == text), text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="voice_intent")
    ap.add_argument("--file")
    args = ap.parse_args()
    if not os.environ.get("LLM_BASE_URL") or not os.environ.get("LLM_MODEL"):
        sys.exit("set LLM_BASE_URL and LLM_MODEL (and LLM_API_KEY if needed)")

    spec = TASKS[args.task]
    rows = load_rows(args.file or HERE / spec["file"])
    questions = {spec["qid"]: spec["question"]}
    router = get_router()
    router.predict(rows[0]["state"], questions, model=spec.get("model"))  # warm-up

    res = {"laya": {"ok": 0, "lat": []}, "llm": {"ok": 0, "lat": []}}
    for row in rows:
        t0 = time.perf_counter()
        a = router.predict(row["state"], questions, model=spec.get("model"))["answers"][spec["qid"]]
        res["laya"]["lat"].append((time.perf_counter() - t0) * 1000)
        res["laya"]["ok"] += predicted(a) == row["label"]

        t0 = time.perf_counter()
        try:
            pred = llm_answer(row["state"], spec)
        except Exception as e:  # noqa: BLE001
            pred = f"error:{type(e).__name__}"
        res["llm"]["lat"].append((time.perf_counter() - t0) * 1000)
        res["llm"]["ok"] += pred == row["label"]

    n = len(rows)
    print(f"task={args.task} n={n} llm={os.environ['LLM_MODEL']}")
    print(f"{'system':<8} {'accuracy':>9} {'p50 ms':>8} {'p95 ms':>8}")
    for name, r in res.items():
        lat = sorted(r["lat"])
        print(f"{name:<8} {r['ok'] / n:9.3f} {statistics.median(lat):8.0f} {lat[min(n - 1, int(n * .95))]:8.0f}")


if __name__ == "__main__":
    main()
