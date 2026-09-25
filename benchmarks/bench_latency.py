"""Latency / throughput benchmark on YOUR hardware.

    python benchmarks/bench_latency.py                   # english + multilingual, default device
    python benchmarks/bench_latency.py --device cuda --runs 50
    python benchmarks/bench_latency.py --models english

Measures:
    A. questions per call  (1, 5, 10) on a short state       -> cost of asking more questions
    B. state length        (short chat turn vs ~400-word doc) -> cost of longer inputs / history
    C. batch throughput    (predict_batch, batch 1 / 8 / 32)  -> items per second for backlogs

Voice AI rule of thumb: the whole "caller stops -> bot starts speaking" budget is ~500-800 ms.
A decision that runs on every STT partial should fit in roughly 50-100 ms p95, so check the
p95 column for case A and B on the machine you will deploy to.
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SHORT = "I was charged twice for my March invoice, please refund the duplicate."
LONG = ("Hello team, " + "we have been customers for three years and the service was mostly fine, "
        "but this month the invoice shows two identical charges and support has not replied. " * 12)


def q(n):
    base = [
        ("dept", {"type": "choice", "instructions": "Which team?",
                  "criteria": {"billing": "payments", "tech": "bugs", "sales": "pricing", "other": "else"}}),
        ("urgent", {"type": "noul", "instructions": "Is it urgent?"}),
        ("frustration", {"type": "score", "instructions": "How frustrated?", "criteria": ["calm", "annoyed", "angry"]}),
        ("refund", {"type": "noul", "instructions": "Does the customer ask for money back?"}),
        ("churn", {"type": "noul", "instructions": "Might the customer leave?"}),
    ]
    return {f"{k}_{i}": v for i in range(n) for k, v in [base[i % len(base)]]}


def timeit(fn, runs, warmup=3):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    xs.sort()
    return {"p50": statistics.median(xs), "p95": xs[min(len(xs) - 1, int(len(xs) * 0.95))], "mean": statistics.mean(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="english,multilingual")
    ap.add_argument("--device", default=None)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from laya import Router

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} threads={torch.get_num_threads()}")
    router = Router(device=device, max_loaded=3)
    report = {"device": device, "results": []}

    for model in args.models.split(","):
        t0 = time.perf_counter()
        router.load(model)
        print(f"\n### {model}  (load {time.perf_counter() - t0:.1f} s)")
        print(f"{'case':<34} {'p50 ms':>8} {'p95 ms':>8}")
        for n in (1, 5, 10):
            r = timeit(lambda: router.predict(SHORT, q(n), model=model), args.runs)
            print(f"{'A  ' + str(n) + ' question(s), short':<34} {r['p50']:8.1f} {r['p95']:8.1f}")
            report["results"].append({"model": model, "case": f"questions={n}", **r})
        for label, text in (("short turn", SHORT), ("~400 words", LONG)):
            r = timeit(lambda: router.predict(text, q(1), model=model), args.runs)
            print(f"{'B  ' + label:<34} {r['p50']:8.1f} {r['p95']:8.1f}")
            report["results"].append({"model": model, "case": f"length={label}", **r})
        for bs in (1, 8, 32):
            reqs = [{"state": SHORT, "questions": q(1), "model": model}] * 64
            t0 = time.perf_counter()
            router.predict_batch(reqs, batch_size=bs)
            dt = time.perf_counter() - t0
            print(f"{'C  batch_size=' + str(bs) + ', 64 items':<34} {64 / dt:8.1f} items/s")
            report["results"].append({"model": model, "case": f"batch={bs}", "items_per_s": 64 / dt})

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
