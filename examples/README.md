# Examples

Small, focused scripts: one Laya feature each. For business scenarios see `../use-cases/`.

| File | Shows |
|------|-------|
| `01_quickstart.py` | The three question types and the raw result format |
| `02_multilingual_router.py` | How the Router picks a checkpoint, and how to override it |
| `03_batch_processing.py` | `predict_batch` vs a loop |
| `04_schema_decide.py` | JSON schema in, typed values out (replaces LLM "structured output") |
| `05_confidence_gating.py` | Automate confident answers, escalate the rest |
| `06_hooks_pii_and_audit.py` | Redact PII before inference, audit-log every decision |
| `07_voice_pipeline_simulation.py` | Laya inside a streaming STT loop: stale-drop, freshness window, fail-open |
| `08_http_server_client.py` | `laya-serve` + an HTTP client (for non-Python gateways) |
| `09_hybrid_laya_then_llm.py` | Laya first, LLM only when needed |

Run from the repo root: `python examples/01_quickstart.py`
