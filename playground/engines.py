"""Remote decision engines for the comparison view: TypeSafe Jev and Azure OpenAI.

Each engine takes the same (state, questions) Laya takes and returns

    {"engine", "answers" (Laya-shaped), "latency_ms", "tokens_in", "tokens_out", "cost_usd", "error"}

so the page can show them side by side. Keys come from the environment / repo .env and never
reach the browser.

    JEV_API_KEY                         TypeSafe key (https://console.typesafe.ai/keys)
    JEV_MODEL                           default jev-latest
    JEV_PRICE_INPUT_PER_M               default 0.042 (output tokens are free)
    AZURE_OPENAI_ENDPOINT / _API_KEY    Azure OpenAI resource
    AZURE_OPENAI_DEPLOYMENT             (or AZURE_OPENAI_LLM_DEPLOYMENT) e.g. gpt-4.1-mini
    AZURE_OPENAI_API_VERSION            default 2024-10-21
    LLM_PRICE_INPUT_PER_M / _OUTPUT_PER_M   default 0.40 / 1.60 (gpt-4.1-mini list price)
"""
import json
import os
import time

import httpx

_client = httpx.Client(timeout=30)   # keep-alive: repeated calls skip the TLS handshake


def _env(*names, default=""):
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return default


def available():
    return {
        "laya": True,
        "jev": bool(_env("JEV_API_KEY", "TYPESAFE_API_KEY")),
        "azure": bool(_env("AZURE_OPENAI_ENDPOINT") and _env("AZURE_OPENAI_API_KEY")
                      and _env("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_LLM_DEPLOYMENT")),
        "azure_deployment": _env("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_LLM_DEPLOYMENT"),
    }


def prices(engine):
    """USD per 1M tokens (input, output) used for the cost figures."""
    if engine == "jev":
        return float(_env("JEV_PRICE_INPUT_PER_M", default="0.042")), 0.0          # Jev output tokens are free
    if engine == "azure":
        return (float(_env("LLM_PRICE_INPUT_PER_M", default="0.40")),
                float(_env("LLM_PRICE_OUTPUT_PER_M", default="1.60")))
    return 0.0, 0.0


def _result(engine, t0, answers=None, tin=0, tout=0, cost=0.0, error=None, model=None):
    p_in, p_out = prices(engine)
    return {"engine": engine, "model": model, "answers": answers or {}, "latency_ms": (time.perf_counter() - t0) * 1000,
            "tokens_in": tin, "tokens_out": tout, "cost_usd": cost, "error": error,
            "price_in_per_m": p_in, "price_out_per_m": p_out}


# --------------------------------------------------------------------------- Jev
def run_jev(state, questions):
    t0 = time.perf_counter()
    key = _env("JEV_API_KEY", "TYPESAFE_API_KEY")
    if not key:
        return _result("jev", t0, error="JEV_API_KEY is not set")
    body = {"state": state, "model": _env("JEV_MODEL", default="jev-latest"), "questions": questions}
    try:
        for attempt in range(3):
            r = _client.post("https://api.typesafe.ai/v1/systemone", json=body,
                             headers={"authorization": f"Bearer {key}"})
            if r.status_code in (429, 529) and attempt < 2:
                time.sleep(0.5 * 2 ** attempt)
                continue
            break
        if r.status_code != 200:
            detail = r.text[:300]
            try:
                detail = json.dumps(r.json().get("detail", r.json()))[:300]
            except Exception:  # noqa: BLE001
                pass
            return _result("jev", t0, error=f"HTTP {r.status_code}: {detail}")
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return _result("jev", t0, error=f"{type(e).__name__}: {e}")
    answers = data.get("answers", {})
    for a in answers.values():   # Jev omits answer_confidence; use P(reported answer), as Laya does
        if a.get("type") == "noul":
            a["answer_confidence"] = max(a["noul"], 1 - a["noul"])
        elif a.get("probabilities"):
            a["answer_confidence"] = max(a["probabilities"].values())
    u = data.get("usage", {})
    tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
    p_in, p_out = prices("jev")
    cost = tin / 1e6 * p_in + tout / 1e6 * p_out
    return _result("jev", t0, answers, tin, tout, cost, model=data.get("model"))


# --------------------------------------------------------------------------- Azure OpenAI
SYSTEM = ("You are a decision engine. Read the INPUT and answer every question. "
          "Reply with one JSON object mapping each question id to its answer: "
          "for a choice question the exact option key (string); for a score question the level number (integer); "
          "for a yes/no question true or false. Output only the JSON object.")


def _azure_prompt(state, questions):
    lines = []
    for qid, q in questions.items():
        if q["type"] == "choice":
            crit = q["criteria"]
            opts = crit.items() if isinstance(crit, dict) else ((c, c) for c in crit)
            lines.append(f'- "{qid}" (choice): {q["instructions"]}\n' + "\n".join(f'    "{k}": {v}' for k, v in opts))
        elif q["type"] == "score":
            lines.append(f'- "{qid}" (score, integer 0-{len(q["criteria"]) - 1}): {q["instructions"]}\n'
                         + "\n".join(f"    {i}: {lv}" for i, lv in enumerate(q["criteria"])))
        else:
            crit = q.get("criteria") or {}
            extra = "".join(f"\n    {k}: {v}" for k, v in crit.items())
            lines.append(f'- "{qid}" (yes/no, true or false): {q["instructions"]}{extra}')
    return f"INPUT:\n{json.dumps(state, ensure_ascii=False)}\n\nQUESTIONS:\n" + "\n".join(lines)


def _azure_answers(raw, questions):
    """Map the LLM's JSON into Laya-shaped answers. An LLM gives no calibrated probabilities."""
    out = {}
    for qid, q in questions.items():
        v = raw.get(qid)
        if q["type"] == "choice":
            keys = list(q["criteria"]) if isinstance(q["criteria"], dict) else list(q["criteria"])
            match = next((k for k in keys if str(v).strip().lower() == k.lower()), None)
            out[qid] = {"type": "choice", "choice": match if match is not None else str(v), "valid": match is not None}
        elif q["type"] == "score":
            try:
                lv = int(round(float(v)))
                ok = 0 <= lv < len(q["criteria"])
            except (TypeError, ValueError):
                lv, ok = None, False
            out[qid] = {"type": "score", "score": lv, "valid": ok,
                        "legend": {str(i): c for i, c in enumerate(q["criteria"])}}
        else:
            yes = v is True or str(v).strip().lower() in ("true", "yes", "1")
            no = v is False or str(v).strip().lower() in ("false", "no", "0")
            out[qid] = {"type": "noul", "noul": 1.0 if yes else 0.0, "valid": yes or no}
    return out


def run_azure(state, questions):
    t0 = time.perf_counter()
    info = available()
    if not info["azure"]:
        return _result("azure", t0, error="Azure OpenAI is not configured (endpoint / key / deployment)")
    dep = info["azure_deployment"]
    url = (f"{_env('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/deployments/{dep}/chat/completions"
           f"?api-version={_env('AZURE_OPENAI_API_VERSION', default='2024-10-21')}")
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": _azure_prompt(state, questions)}],
            "temperature": 0, "max_tokens": 300, "response_format": {"type": "json_object"}}
    try:
        for attempt in range(3):
            r = _client.post(url, json=body, headers={"api-key": _env("AZURE_OPENAI_API_KEY")})
            if r.status_code == 429 and attempt < 2:
                time.sleep(float(r.headers.get("retry-after", 1)))
                continue
            break
        if r.status_code != 200:
            return _result("azure", t0, error=f"HTTP {r.status_code}: {r.text[:300]}", model=dep)
        data = r.json()
        raw = json.loads(data["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001
        return _result("azure", t0, error=f"{type(e).__name__}: {e}", model=dep)
    u = data.get("usage", {})
    tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    p_in, p_out = prices("azure")
    cost = tin / 1e6 * p_in + tout / 1e6 * p_out
    return _result("azure", t0, _azure_answers(raw, questions), tin, tout, cost, model=dep)
