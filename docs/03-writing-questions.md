# 3. How to use it: writing good questions

## Install

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt     # Windows
.venv/bin/python -m pip install -r requirements.txt             # macOS / Linux
```

Python 3.10 or newer. The first `predict` downloads about 1 GB from Hugging Face.

## Minimal call

```python
from laya import Router
router = Router()                       # Router(preload=True) for servers
result = router.predict(state, questions)
```

`state` can be:
* a string: `"I was charged twice"`
* a dict: `{"subject": "...", "body": "..."}`
* a list of turns: `[{"role": "assistant", "content": "..."}, {"role": "caller", "content": "..."}]`

## Question formats

```python
{
  "dept":    {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "invoices, refunds", "tech": "bugs, outages"}},
  "urgency": {"type": "score",  "instructions": "How urgent?",
              "criteria": ["not urgent", "today", "blocking"]},          # index 0 = lowest
  "spam":    {"type": "noul",   "instructions": "Is this spam?"},
  "done":    {"type": "noul",   "instructions": "Has the caller finished?",
              "criteria": {"false": "sentence cut off", "true": "complete request"}},
}
```

## Rules of thumb

1. **Describe every option.** `"billing": "invoices, payments, refunds"` works much better than `"billing": ""`. The model reads the descriptions.
2. **Always add an escape option**, such as `"other"` or `"unclear"`. Without one, the model has to pick something.
3. **Keep choice questions to about 20 options or fewer.** For more, split them into a coarse question and a fine one, or use `laya.predict_shortlist`.
4. **Don't use `yes/no/true/false` as choice keys.** The model can follow the key word instead of the meaning. Use semantic keys or neutral ones (`A`, `B`, `C`).
5. **For yes/no, give `noul` the `true`/`false` descriptions**, or use a 2-option choice with neutral keys. On the English checkpoint, a bare `noul` can be pulled by its labels (#156).
6. **Refer to the field.** "Does `message` mention a deadline?" helps when the state is a dict.
7. **Ask several questions in one call.** One call is cheaper than N calls.
8. **Force the checkpoint when you know the language.** `model="english"` for English voice, since short Latin text can be misdetected.
9. **Negation is a known weak spot.** "I do NOT want to cancel" may still come back as `cancel`. Test it.

## Ready-made presets

```python
import laya
laya.triage_questions()      # support: intent, urgency, frustration, refund, churn
laya.email_questions()       # email categories, phishing...
laya.guard_questions()       # jailbreak, prompt injection, sensitive data, harm
laya.moderation_questions()  # toxicity, harassment, threats
laya.router_questions()      # small vs frontier model routing
```

## Schema mode (drop-in for LLM structured output)

```python
router.decide(text, schema={"type": "object", "properties": {
    "department": {"type": "string", "enum": ["billing", "support"]},
    "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
    "needs_human": {"type": "boolean"}}})
# -> {"department": "billing", "urgency": 2, "needs_human": True}
```

## Other ways to run it

* CLI: `laya "My payment failed twice" --preset triage`
* HTTP: `laya-serve`, then `POST /v1/systemone` (see `examples/08_http_server_client.py`)
* MCP server: `laya-mcp-server`, for Claude Desktop or Cursor
* LangChain / LangGraph: `LayaRouter`, `LayaGuardrail`
