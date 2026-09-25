"""Head-to-head: Laya vs an LLM on the same labelled rows - accuracy, latency, tokens, cost.

Azure OpenAI (read from the repo's .env, never committed):
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_LLM_DEPLOYMENT, AZURE_OPENAI_API_VERSION
Any other OpenAI-compatible endpoint (OpenAI, Groq, vLLM, Ollama, LiteLLM...):
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
Cost (USD per 1M tokens): LLM_PRICE_INPUT_PER_M, LLM_PRICE_OUTPUT_PER_M

    python benchmarks/bench_vs_llm.py                         # all tasks, 10 repeats
    python benchmarks/bench_vs_llm.py --task voice_intent --repeats 3

Every row is sent `--repeats` times to each system (sequentially, one at a time, keep-alive
connection, after a warm-up), so latency percentiles come from real repeated calls and you
also see whether the LLM gives the same answer every time.
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import statistics
import sys
import time

import httpx

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from shared import get_router, load_env   # noqa: E402
from tasks import TASKS                    # noqa: E402
from bench_accuracy import load_rows, predicted   # noqa: E402


class LLM:
    def __init__(self):
        self.client = httpx.Client(timeout=60)
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            dep = os.environ["AZURE_OPENAI_LLM_DEPLOYMENT"]
            ver = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
            self.url = f"{os.environ['AZURE_OPENAI_ENDPOINT'].rstrip('/')}/openai/deployments/{dep}/chat/completions?api-version={ver}"
            self.headers = {"api-key": os.environ["AZURE_OPENAI_API_KEY"]}
            self.name, self.model = f"azure:{dep}", None
        else:
            self.url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
            self.headers = {"authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}"}
            self.name = self.model = os.environ["LLM_MODEL"]
        self.price_in = float(os.environ.get("LLM_PRICE_INPUT_PER_M", "0"))
        self.price_out = float(os.environ.get("LLM_PRICE_OUTPUT_PER_M", "0"))

    @staticmethod
    def prompt(state, q):
        if q["type"] == "choice":
            options = "\n".join(f"- {k}: {v}" for k, v in q["criteria"].items())
            instr = f"{q['instructions']}\nOptions:\n{options}\nReply with the option key only."
        else:
            crit = q.get("criteria") or {}
            extra = "".join(f"\n- {k}: {v}" for k, v in crit.items())
            instr = f"{q['instructions']}{extra}\nReply with true or false only."
        return [{"role": "system", "content": "You are a classifier. Output only the answer, nothing else."},
                {"role": "user", "content": f"Input:\n{json.dumps(state, ensure_ascii=False)}\n\n{instr}"}]

    def ask(self, state, q):
        body = {"messages": self.prompt(state, q), "temperature": 0, "max_tokens": 10}
        if self.model:
            body["model"] = self.model
        retries = 0
        while True:
            t0 = time.perf_counter()
            r = self.client.post(self.url, headers=self.headers, json=body)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code == 429 and retries < 6:
                retries += 1
                time.sleep(float(r.headers.get("retry-after", 2 ** retries)))
                continue
            r.raise_for_status()
            data = r.json()
            break
        text = data["choices"][0]["message"]["content"].strip().strip(".").strip("`").strip().lower()
        if q["type"] == "noul":
            ans = text.startswith("true") or text.startswith("yes")
        else:
            ans = next((k for k in q["criteria"] if k.lower() == text), text)
        u = data.get("usage", {})
        return ans, ms, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), retries


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def run_task(name, llm, repeats, file=None):
    spec = TASKS[name]
    rows = load_rows(file or HERE / spec["file"])
    q = spec["question"]
    questions = {spec["qid"]: q}
    model = spec.get("model")
    router = get_router()

    router.predict(rows[0]["state"], questions, model=model)   # warm-up: model load
    llm.ask(rows[0]["state"], q)                               # warm-up: TLS + connection

    out = {"laya": {"lat": [], "ok": 0, "n": 0, "answers": []},
           "llm": {"lat": [], "ok": 0, "n": 0, "answers": [], "tin": 0, "tout": 0, "retries": 0, "errors": 0}}
    for i, row in enumerate(rows):
        la, lm = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            r = router.predict(row["state"], questions, model=model)
            out["laya"]["lat"].append((time.perf_counter() - t0) * 1000)
            p = predicted(r["answers"][spec["qid"]])
            la.append(p)
            out["laya"]["ok"] += p == row["label"]
            out["laya"]["n"] += 1

            try:
                ans, ms, tin, tout, retries = llm.ask(row["state"], q)
                out["llm"]["tin"] += tin
                out["llm"]["tout"] += tout
                out["llm"]["retries"] += retries
                out["llm"]["lat"].append(ms)
            except Exception as e:  # noqa: BLE001
                ans = f"error:{type(e).__name__}"
                out["llm"]["errors"] += 1
            lm.append(ans)
            out["llm"]["ok"] += ans == row["label"]
            out["llm"]["n"] += 1
        out["laya"]["answers"].append(la)
        out["llm"]["answers"].append(lm)
        print(f"\r  {name}: row {i + 1}/{len(rows)}", end="", flush=True)
    print()

    summary = {"task": name, "rows": len(rows), "repeats": repeats, "routed_model": model or "auto"}
    for sysname, d in out.items():
        consistent = sum(len({json.dumps(a) for a in ans}) == 1 for ans in d["answers"]) / len(rows)
        s = {"accuracy": d["ok"] / d["n"], "calls": d["n"], "consistency": consistent,
             "p50_ms": statistics.median(d["lat"]), "p95_ms": pct(d["lat"], 0.95),
             "mean_ms": statistics.mean(d["lat"]), "max_ms": max(d["lat"])}
        if sysname == "llm":
            cost = d["tin"] / 1e6 * llm.price_in + d["tout"] / 1e6 * llm.price_out
            s.update(tokens_in=d["tin"], tokens_out=d["tout"], avg_tokens_in=d["tin"] / d["n"],
                     cost_usd=cost, cost_per_1k_usd=cost / d["n"] * 1000,
                     retries_429=d["retries"], errors=d["errors"])
        summary[sysname] = s
    return summary


def print_summary(s, llm_name):
    print(f"\n=== {s['task']}  rows={s['rows']} x repeats={s['repeats']}  (laya checkpoint: {s['routed_model']})")
    print(f"{'system':<26} {'accuracy':>8} {'consist.':>8} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8} {'$/1k calls':>11}")
    for key, label in (("laya", "laya (CPU, local)"), ("llm", llm_name)):
        r = s[key]
        cost = f"{r['cost_per_1k_usd']:.4f}" if key == "llm" else "0 (self-host)"
        print(f"{label:<26} {r['accuracy']:8.3f} {r['consistency']:8.2f} {r['p50_ms']:8.0f} {r['p95_ms']:8.0f} {r['max_ms']:8.0f} {cost:>11}")


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), help="default: all tasks")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--file")
    ap.add_argument("--out", default=str(HERE / "results" / "vs_llm.json"))
    args = ap.parse_args()

    llm = LLM()
    names = [args.task] if args.task else list(TASKS)
    results = []
    for n in names:
        s = run_task(n, llm, args.repeats, args.file)
        print_summary(s, llm.name)
        results.append(s)

    import torch
    report = {"date": dt.datetime.now().isoformat(timespec="seconds"), "llm": llm.name,
              "price_per_m": [llm.price_in, llm.price_out],
              "laya_device": os.environ.get("LAYA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"),
              "torch_threads": torch.get_num_threads(), "results": results}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
