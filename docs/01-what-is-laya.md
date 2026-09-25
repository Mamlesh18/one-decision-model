# 1. What is Laya?

**In one line:** a small, fast model that answers *multiple-choice questions about a piece of text*,
with a probability for each answer. It does not write text.

```
input text  +  your questions                   -->  answers with probabilities
"I was charged twice,       department? billing/technical/other    billing   (0.94)
 refund me or I cancel"     urgency?    0..2                        1.8
                            churn risk? yes/no                      P(yes) = 0.89
```

## The "System 1 / System 2" idea

| | System 1 (Laya) | System 2 (an LLM such as GPT, Claude or Llama) |
|---|---|---|
| Job | Quick judgements: pick, rate, yes/no | Think, explain, write, plan |
| Output | Label + probability | Free text |
| Speed | ~30-40 ms on a T4 GPU, ~0.2-0.5 s on CPU | ~250 ms to several seconds |
| Can it make things up? | No. It can only pick one of the options you give it | Yes |
| Cost | Runs on your own machine, no per-call fee (Apache 2.0) | Per token |
| Confidence | A probability you can set a threshold on | Usually none, or unreliable |

Use Laya for the many small **decisions** in a system and keep the LLM for **generation**.

## Three question types ("primitives")

| Type | Asks | Returns | Example |
|------|------|---------|---------|
| `choice` | Which one of these? | label + probability per label | department, intent, tool to call |
| `score` | How much, on this scale? | expected level (e.g. 1.7 on 0..3) | urgency, frustration, severity |
| `noul` | Yes or no? | P(yes), from 0 to 1 | is it spam? is the turn finished? |

You can ask many questions about the same text in one call.

## The three checkpoints (model weights)

| Name | Base encoder | Size | Context | Use for |
|------|--------------|------|---------|---------|
| `english` (`laya`) | ModernBERT-large | 421M | 512 tokens | English |
| `multilingual` | mmBERT-base | 322M | 1024 (up to 8192) | 100+ languages, about 2x faster |
| `typed-decisions` | ModernBERT-large | 421M | 1024 | fine-tuned for invoice, security, support and agent-trace workflows |

The **Router** picks between `english` and `multilingual` for each request by looking at the
script and language. You never have to choose unless you want to.

## Who made it

It is open source (Apache 2.0) from Convai Innovations: github.com/NandhaKishorM/laya,
Hugging Face `convaiinnovations/laya`. The version tested here is 0.3.20, released September 2026.
