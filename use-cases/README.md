# Use cases

One file per use case. Each file says **what it replaces**, defines the questions, runs a few
sample inputs, and shows how the answer turns into an action.

```bash
python use-cases/01_support_ticket_triage.py
```

The first run downloads a checkpoint (about 1 GB) from Hugging Face. Set `LAYA_DEVICE=cuda` if you have a GPU.

## Business / text workflows

| # | File | Replaces | Question types |
|---|------|----------|----------------|
| 01 | `01_support_ticket_triage.py` | keyword rules / LLM-JSON triage | choice, score, noul |
| 02 | `02_email_routing.py` | inbox rules + spam filter | choice, noul |
| 03 | `03_intent_detection.py` | NLU intent classifier (Dialogflow, Rasa, LUIS) | choice + "unclear" |
| 04 | `04_content_moderation.py` | toxicity API + rules | noul, score |
| 05 | `05_lead_scoring.py` | manual qualification | score, choice, noul |
| 06 | `06_agent_tool_routing.py` | "which tool?" LLM call | choice, score, noul |
| 07 | `07_aspect_sentiment.py` | overall-only sentiment | score per aspect |
| 08 | `08_multilingual_intake.py` | detect, translate, classify | Router, 100+ languages |
| 09 | `09_patient_message_triage.py` | receptionist first read | score, choice, noul |
| 10 | `10_invoice_decisions.py` | AP clerk checks (needs fine-tuned model) | JSON state |
| 11 | `11_security_alert_triage.py` | tier-1 SOC first look (needs fine-tuned model) | JSON state |

## Conversational AI and voice AI

| # | File | Problem it solves |
|---|------|-------------------|
| 12 | `12_voice_turn_completion.py` | Is the caller done talking? Adjusts the silence timeout |
| 13 | `13_voice_barge_in_interruptions.py` | "uh-huh" (keep talking) vs "no that's wrong" (stop) |
| 14 | `14_voice_live_call_context.py` | Intent, sentiment, topic, complexity after every turn |
| 15 | `15_llm_prompt_guardrail.py` | Jailbreak, injection and off-topic checks before the LLM |
| 16 | `16_post_call_qa_scoring.py` | QA-score 100% of call transcripts |
| 17 | `17_confirmation_detection.py` | "yeah go ahead" / "nah" / "yes but at 5pm" |
| 18 | `18_outbound_call_outcome.py` | Voicemail, right person, callback, do-not-call |

## Before you trust any of these

These are **zero-shot** demos. The project's own numbers show the base checkpoints are strong
on simple, well-described decisions and weak on complex domain workflows until you fine-tune.
Run `benchmarks/bench_accuracy.py` on 100+ of **your own** labelled examples before you
use any of them in production. See `docs/05-limits-and-gotchas.md`.
