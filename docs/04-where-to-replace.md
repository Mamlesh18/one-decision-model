# 4. Where can it replace what we run today?

**Test:** is the output one of a **fixed set of answers** (a label, a level, yes/no) that you can
write down in advance? Then Laya is a candidate. If the output is free text, keep the LLM.

## Replace, augment, or keep

| What you run today | Verdict | Why |
|--------------------|---------|-----|
| LLM prompt "classify this into one of X, return JSON" | **Replace** (after benchmarking) | Same job, 5-10x faster, no parsing errors, a probability to threshold on |
| NLU intent classifier (Dialogflow, Rasa, LUIS, Lex) | **Replace or augment** | New intents need no retraining (define them per request). A trained classifier may still win on your exact intents, so compare |
| Keyword / regex rules for routing | **Replace** | Understands paraphrases and 100+ languages. Keep regex for hard safety rules |
| Separate toxicity, spam and sentiment APIs | **Replace** | All in one call, self-hosted, no data leaves your servers |
| Detect language, then translate, then classify | **Replace** | Multilingual checkpoint, no translation step |
| "Which tool / which model?" LLM call in an agent | **Replace** | That's a choice question |
| LLM-as-judge on 100% of transcripts for QA | **Augment** | Laya scores everything; the LLM or a person reviews low-confidence cases |
| Fixed VAD silence timeout for end of turn | **Augment** | Laya adds meaning ("is the sentence finished?") on top of VAD. It can't hear prosody |
| "Stop TTS on any voice activity" barge-in | **Augment** | Separates "uh-huh" from "no, that's wrong" |
| LLM guardrail pre-check | **Augment** | Fast first layer; the project reports 0.698 on held-out prompt injections, so keep other layers |
| LLM answer generation, summaries, RAG answers | **Keep the LLM** | Laya doesn't generate text |
| Entity / slot extraction (dates, amounts, names) | **Keep NER / LLM** | Laya picks among options; it doesn't pull spans out of text |
| Classification with 50+ labels | **Keep, or shortlist** | Accuracy drops (Banking77: 0.425 vs Jev 0.870). Use a two-level question or `predict_shortlist` |
| Complex domain decisions (AP matching, SOC triage) zero-shot | **Only after fine-tuning** | Base checkpoints are near chance there (0.36). Fine-tuned: 0.77 |

## How to switch safely (4 steps)

1. **Shadow mode.** Run Laya next to the current system and log both answers. Change nothing.
2. **Measure.** Use `benchmarks/bench_accuracy.py --file your_data.jsonl` and `bench_vs_llm.py` on 200+ real, labelled rows.
3. **Pick a threshold.** Read the coverage table. For example: "at 0.8 confidence, 70% of traffic is handled at 97% accuracy."
4. **Go hybrid.** Laya takes the confident part and the old system (LLM or a person) takes the rest. Widen Laya's share as the numbers allow.

## Cost sketch

Take 1M decisions a month. With an LLM at about 300 input tokens and a few output tokens per decision, you pay per token and wait 250 ms to 2 s each time.
With Laya on one GPU you pay for the machine and wait about 30-150 ms. Throughput reported by the project is 103-332 questions per second on one T4.
Work out your own numbers with `benchmarks/bench_latency.py`.
