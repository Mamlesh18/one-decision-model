# One Decision Model: evaluating Laya

A test bed for [Laya](https://github.com/NandhaKishorM/laya) (v0.3.20). Laya is an open-source model
that answers **typed questions** (pick one, rate on a scale, yes/no) about any text in a single fast
forward pass, with a **probability** for each answer. It can't generate text, so it can't hallucinate.

```
"I was charged twice, refund me or I cancel"   ->   department = billing   (0.94)
                                                    urgency    = 1.8 / 2
                                                    churn_risk = P(yes) 0.89      ~33 ms on GPU
```

**Laya takes the fast, repetitive decisions; the LLM takes the thinking and writing.**

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt      # Windows (use .venv/bin/python elsewhere)
.venv\Scripts\python.exe examples/01_quickstart.py               # first run downloads ~1 GB
.venv\Scripts\python.exe -m pytest tests -v                      # offline tests, no download
```

## Folders

| Folder | What's inside |
|--------|---------------|
| [`docs/`](docs/) | Plain-language guide: what it is, how it works, how to use it, where it replaces current systems, voice AI, limits, deployment |
| [`use-cases/`](use-cases/) | 18 runnable scenarios, one per file: 11 business workflows and 7 conversational / voice AI |
| [`examples/`](examples/) | 9 short scripts, one API feature each (router, batch, schema, gating, hooks, voice loop, HTTP, hybrid LLM) |
| [`benchmarks/`](benchmarks/) | Latency, accuracy and calibration, plus Laya against your current LLM on labelled data |
| [`tests/`](tests/) | Routing tests (offline), output-contract tests, behaviour checks including known weak spots |

## Read in this order

1. [What is Laya](docs/01-what-is-laya.md)
2. [How it works](docs/02-how-it-works.md)
3. [How to use it: writing questions](docs/03-writing-questions.md)
4. [Where it can replace our current systems](docs/04-where-to-replace.md)
5. [Advantages for conversational and voice AI](docs/05-voice-and-conversational-ai.md), including the TASA assistant explained
6. [Limits and gotchas](docs/06-limits-and-gotchas.md). **Read before production.**
7. [Deploy and fine-tune](docs/07-deploy-and-finetune.md)

## Short answers

**Where can it replace our systems?** Anywhere the output is a fixed label, level or yes/no: LLM
classify-to-JSON prompts, NLU intent classifiers, keyword routing, toxicity and spam APIs,
detect-translate-classify chains, and the "which tool?" step in an agent. It **augments** VAD turn-taking,
barge-in handling, guardrails and QA. It **doesn't replace** answer generation, summaries or slot
extraction.

**Why does it matter for voice AI?** It's fast enough (about 30-100 ms on GPU) to run on every STT partial
and every turn. It gives calibrated probabilities, so you act only when it's confident. The answer is always
one of your options. It handles 100+ languages without translating, and it runs self-hosted with no per-call cost.

**Main caveat:** zero-shot it's good on simple, well-described decisions and weak on complex domain
ones until fine-tuned. Measure on your own data with `benchmarks/` before switching anything.
