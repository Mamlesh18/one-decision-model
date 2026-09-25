"""Accuracy + calibration benchmark on labelled data.

    python benchmarks/bench_accuracy.py                      # all tasks in tasks.py
    python benchmarks/bench_accuracy.py --task turn_complete
    python benchmarks/bench_accuracy.py --task voice_intent --model multilingual
    python benchmarks/bench_accuracy.py --task voice_intent --file my_data.jsonl   # your own rows

What it reports per task:
    accuracy              how often the top answer matches the label
    ECE                   calibration error: does "0.9 confident" really mean 90% right? (lower = better)
    coverage @ threshold  if you only auto-handle answers above the threshold, what share of traffic
                          you automate and how accurate that share is  <- use this to pick thresholds
    latency p50 / p95     per request, one at a time
    errors                the rows it got wrong, so you can see WHY

The bundled datasets are tiny (24-30 rows) smoke tests. Use 200+ rows of your own for decisions.
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from shared import get_router   # noqa: E402
from tasks import TASKS          # noqa: E402


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ece(confs, correct, bins=10):
    total, err = len(confs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confs[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(acc - conf)
    return err


def predicted(answer):
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    if answer["type"] == "score":
        return int(round(answer["score"]))
    return answer["choice"]


def run_task(name, spec, model=None, file=None):
    router = get_router()
    rows = load_rows(file or HERE / spec["file"])
    questions = {spec["qid"]: spec["question"]}
    model = model or spec.get("model")

    router.predict(rows[0]["state"], questions, model=model)   # warm-up (model load)

    confs, correct, lat, errors, routed = [], [], [], [], {}
    for row in rows:
        t0 = time.perf_counter()
        r = router.predict(row["state"], questions, model=model)
        lat.append((time.perf_counter() - t0) * 1000)
        a = r["answers"][spec["qid"]]
        pred = predicted(a)
        ok = pred == row["label"]
        confs.append(a["answer_confidence"])
        correct.append(ok)
        routed[r["routing"]["model"]] = routed.get(r["routing"]["model"], 0) + 1
        if not ok:
            errors.append((row["state"], row["label"], pred, a["answer_confidence"]))

    n = len(rows)
    lat.sort()
    print(f"\n=== {name}  (n={n}, routed to {routed})")
    print(f"accuracy   {sum(correct) / n:.3f}")
    print(f"ECE        {ece(confs, correct):.3f}   mean confidence {statistics.mean(confs):.3f}")
    print(f"latency    p50 {lat[n // 2]:.0f} ms   p95 {lat[min(n - 1, int(n * 0.95))]:.0f} ms")
    print("threshold  coverage  accuracy-on-covered")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        kept = [c for c, cf in zip(correct, confs) if cf >= th]
        acc = f"{sum(kept) / len(kept):.3f}" if kept else "  -  "
        print(f"  {th:.1f}      {len(kept) / n:6.1%}    {acc}")
    if errors:
        print("errors (state -> expected / got @ conf):")
        for state, exp, got, c in errors[:10]:
            s = state if isinstance(state, str) else json.dumps(state[-1] if isinstance(state, list) else state,
                                                                ensure_ascii=False)
            print(f"  {s[:70]:<72} {exp} / {got} @ {c:.2f}")
    return {"task": name, "n": n, "accuracy": sum(correct) / n, "ece": ece(confs, correct),
            "p50_ms": lat[n // 2], "routed": routed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), help="default: all tasks")
    ap.add_argument("--model", help="force a checkpoint: english | multilingual | typed-decisions")
    ap.add_argument("--file", help="your own JSONL with the same format as the task's file")
    ap.add_argument("--out", help="write a JSON summary here")
    args = ap.parse_args()

    names = [args.task] if args.task else list(TASKS)
    summary = [run_task(n, TASKS[n], model=args.model, file=args.file) for n in names]
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
