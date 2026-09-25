"""Laya Playground - local backend for playground/index.html.

    python playground/server.py                  # http://127.0.0.1:8765
    python playground/server.py --preload        # load english + multilingual at startup
    python playground/server.py --port 9000 --device cuda

Open http://127.0.0.1:8765 in a browser, or open playground/index.html directly as a file
(it calls this server; CORS is allowed because the server only listens on localhost).

Deployed (see deploy/README.md), configure with environment variables:
    HOST / PORT             bind address (Docker: 0.0.0.0 and the platform's port)
    LAYA_PRELOAD=1          load english + multilingual at startup
    LAYA_PLAYGROUND_KEY     if set, /api/predict and /api/route require "Authorization: Bearer <key>"
    ALLOWED_ORIGINS         comma list for CORS (default "*")
"""
import argparse
import hmac
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import uvicorn                                                  # noqa: E402
from fastapi import Body, Depends, FastAPI, Header, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware              # noqa: E402
from fastapi.responses import FileResponse                      # noqa: E402

from shared import get_router                                   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
MODELS = {"english", "multilingual", "typed-decisions"}
API_KEY = os.environ.get("LAYA_PLAYGROUND_KEY", "")
MAX_BODY_CHARS = 60_000   # a public endpoint should not accept book-sized states

app = FastAPI(title="Laya Playground")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


def require_key(authorization: str = Header(default="")):
    if not API_KEY:
        return
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), API_KEY):
        raise HTTPException(401, "missing or wrong API key")


def _check_size(body: dict):
    import json
    if len(json.dumps(body, ensure_ascii=False)) > MAX_BODY_CHARS:
        raise HTTPException(413, "request too large")


def _model(value):
    if value in (None, "", "auto"):
        return None
    if value not in MODELS:
        raise HTTPException(422, f"model must be auto or one of {sorted(MODELS)}")
    return value


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/chat")
@app.get("/chat.html")
def chat():
    return FileResponse(HERE / "chat.html")


@app.get("/index.html")
def index_html():
    return FileResponse(HERE / "index.html")


@app.get("/api/health")
def health():
    import laya
    import torch
    router = get_router()
    return {"ok": True, "laya": laya.__version__, "loaded": router.loaded,
            "device": os.environ.get("LAYA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"),
            "threads": torch.get_num_threads()}


@app.post("/api/route", dependencies=[Depends(require_key)])
def route(body: dict = Body(...)):
    _check_size(body)
    t0 = time.perf_counter()
    d = get_router().route(body.get("state"), body.get("questions") or {}, model=_model(body.get("model")))
    return {"routing": dict(d), "route_ms": (time.perf_counter() - t0) * 1000}


@app.post("/api/predict", dependencies=[Depends(require_key)])
def predict(body: dict = Body(...)):
    _check_size(body)
    state, questions = body.get("state"), body.get("questions")
    if state in (None, "", [], {}):
        raise HTTPException(422, "state is empty - type a user query first")
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(422, "add at least one question")
    router = get_router()
    model = _model(body.get("model"))
    max_len = body.get("max_len") or None
    if max_len is not None and not (isinstance(max_len, int) and 64 <= max_len <= 8192):
        raise HTTPException(422, "max_len must be an integer between 64 and 8192")
    loaded_before = set(router.loaded)
    t0 = time.perf_counter()
    try:
        result = router.predict(state, questions, model=model, max_len=max_len)
    except ValueError as e:        # Laya's question validation errors are readable, pass them on
        raise HTTPException(422, str(e))
    elapsed = (time.perf_counter() - t0) * 1000
    result["timing"] = {"server_ms": elapsed,
                        "cold_load": result["routing"]["model"] not in loaded_before}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    ap.add_argument("--device", default=None, help="cpu | cuda | mps")
    ap.add_argument("--preload", action="store_true", default=os.environ.get("LAYA_PRELOAD") == "1",
                    help="load english + multilingual at startup")
    args = ap.parse_args()
    if args.device:
        os.environ["LAYA_DEVICE"] = args.device
    if args.host not in ("127.0.0.1", "localhost") and not API_KEY:
        print("WARNING: listening publicly without LAYA_PLAYGROUND_KEY - anyone can use this server")
    if args.preload:
        os.environ["LAYA_PRELOAD"] = "1"
        print("preloading english + multilingual ...")
        get_router()
    print(f"Laya Playground on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
