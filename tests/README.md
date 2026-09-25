# Tests

```bash
pytest tests -v                                   # fast: routing + repo checks, no download
LAYA_RUN_MODEL_TESTS=1 pytest tests -v            # also loads checkpoints and runs inference
# PowerShell: $env:LAYA_RUN_MODEL_TESTS="1"; pytest tests -v
```

| File | Needs model | What it protects |
|------|-------------|------------------|
| `test_routing_offline.py` | no | Script and language routing, `lang_guess`, `route_batch` order |
| `test_repo_files.py` | no | Every script parses; benchmark labels match their questions |
| `test_model_contract.py` | yes | Output shape your code depends on: keys, probabilities sum to 1, determinism, batch = single |
| `test_behaviour_sanity.py` | yes | Obvious cases still work; documents known weak spots as `xfail` |

Run the model tests after every `pip install -U laya` before you deploy.
