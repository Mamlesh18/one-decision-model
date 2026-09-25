# Laya Playground

A browser UI for trying Laya: type a user query, define the questions (choice / score / noul)
and their options, and see Laya's decision with every probability, confidence and latency.

```bash
.venv\Scripts\python.exe playground\server.py            # Windows
.venv/bin/python playground/server.py                    # macOS / Linux
# then open http://127.0.0.1:8765
```

You can also open `playground/index.html` straight from disk; it calls the server at
`http://127.0.0.1:8765` (change it in Settings, "Server URL").

Options: `--preload` (load english + multilingual at startup), `--device cuda`, `--port 9000`.
The server listens on localhost only.

## What you can do

* **Presets**: support triage, end-of-turn, barge-in, yes/no confirmation, IVR intent,
  live call context, LLM guardrail, multilingual (Hindi). Load one, then edit it.
* **State modes**: plain text, a conversation (`role: text` per line, sent as a list of `{role, content}` turns), or raw JSON.
* **Questions**: add, remove, reorder or duplicate. Switch the type:
  * `choice`: option keys with descriptions
  * `score`: ordered levels, lowest first
  * `noul`: optional false/true descriptions and model-facing labels
* **Settings**:
  * checkpoint: auto, english, multilingual or typed-decisions
  * `max_len` (up to 8192 on multilingual for long documents)
  * an auto-act threshold that marks each answer AUTO or ESCALATE
* **Output**:
  * probability bars per option
  * a gauge for score and noul answers
  * `answer_confidence`, the chosen checkpoint and why, model time vs round trip, and input tokens
  * the raw JSON
* **Route only**: shows which checkpoint would be used, without running the model.
* **Copy as Python**: turns the current setup into a runnable `router.predict(...)` script.
* **History**: the last 30 runs of this session; click one to restore it.

The first call for each checkpoint downloads and loads it, which takes a while. Later calls are fast.
On CPU expect several hundred ms per call.
