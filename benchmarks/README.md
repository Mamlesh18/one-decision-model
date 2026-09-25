# Benchmarks

Measure Laya on **your** hardware and **your** data. Don't rely on the project's published numbers alone.

| Script | Question it answers |
|--------|---------------------|
| `bench_latency.py` | How fast is it here? (by number of questions, input length, batch size) |
| `bench_accuracy.py` | How accurate is it, is its confidence honest (ECE), and which threshold should I use? |
| `bench_vs_llm.py` | Can it replace the LLM call I make today? (same rows, accuracy and latency side by side) |
| `tasks.py` | Task definitions (question + dataset). Add your own here |
| `data/*.jsonl` | Tiny hand-labelled smoke-test sets (24-30 rows each) |

```bash
python benchmarks/bench_latency.py --runs 30 --out benchmarks/results_latency.json
python benchmarks/bench_accuracy.py --out benchmarks/results_accuracy.json
python benchmarks/bench_accuracy.py --task voice_intent --file path/to/your_labelled.jsonl
python benchmarks/bench_vs_llm.py --task voice_intent       # needs LLM_BASE_URL / LLM_MODEL
```

## Bundled tasks

| Task | What it tests | Type |
|------|---------------|------|
| `voice_intent` | 6 banking IVR intents from spoken-style utterances | choice |
| `turn_complete` | Is the caller's partial transcript a finished thought? | noul |
| `interruption` | backchannel / disagreement / new request / stop, while the bot speaks | choice |
| `confirmation` | yes / no / unclear replies to a yes-no question | choice (neutral A/B/D keys) |
| `multilingual_dept` | billing / technical / delivery in 8 languages (Router picks the checkpoint) | choice |

## Your own dataset

JSONL, one row per line:

```json
{"state": "I lost my card", "label": "block_card"}
{"state": [{"speaker": "assistant", "text": "Shall I book it?"}, {"speaker": "user", "text": "yep"}], "label": "A"}
{"state": {"subject": "Invoice", "body": "..."}, "label": true}
```

Take 200 or more rows from real traffic, labelled by people, and include the hard cases:
negations, sarcasm, code-mixed Hindi-English, and STT errors.

## Published numbers (from the Laya project, not re-measured here)

| | laya (English) | laya-multilingual |
|---|---|---|
| 1 question, T4 GPU | 39.5 ms | 32.8 ms |
| 10 questions, T4 GPU | 158.6 ms | 72.3 ms |
| CPU (preloaded) | 193-464 ms | |
| MASSIVE intent, English | 0.783 | 0.657 |
| MASSIVE intent, 13 other languages | 0.306 | 0.451 |
| typed-decisions zero-shot | 0.362 | 0.352 (fine-tuned checkpoint: 0.766) |
