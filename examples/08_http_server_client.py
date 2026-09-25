"""Example 8 - Run Laya as a service and call it over HTTP from any language.

Terminal 1 (server):
    pip install "laya[serve]"
    LAYA_PRELOAD=1 laya-serve                 # listens on 0.0.0.0:8000
    #  PowerShell:  $env:LAYA_PRELOAD="1"; laya-serve
    #  Set LAYA_API_KEY=secret to require "Authorization: Bearer secret"
    #  Set LAYA_HOST=127.0.0.1 to keep it off the network while testing.

Terminal 2 (this client):
    python examples/08_http_server_client.py

The endpoint is POST /v1/systemone, the same wire format as TypeSafe's hosted Jev API,
so a Node / Java / Go voice gateway can call it without Python.
"""
import json
import os
import time
import urllib.request

URL = os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("LAYA_API_KEY")

body = {
    "state": {"body": "billed twice, refund please or we cancel"},
    "questions": {
        "dept": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "refunds and charges", "tech": "bugs"}},
        "churn": {"type": "noul", "instructions": "Will the customer cancel?"},
    },
}

req = urllib.request.Request(URL + "/v1/systemone", data=json.dumps(body).encode(),
                             headers={"content-type": "application/json",
                                      **({"authorization": f"Bearer {KEY}"} if KEY else {})})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read())
print(f"round trip {(time.perf_counter() - t0) * 1000:.0f} ms")
print(json.dumps(result, indent=2))

# Health check: GET /health
