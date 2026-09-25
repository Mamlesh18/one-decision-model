# 7. Deploying and fine-tuning

## Deployment options

| Option | Command | When |
|--------|---------|------|
| In-process Python | `Router(preload=True, device="cuda")` | Your voice or chat service is already Python |
| HTTP service | `pip install "laya[serve]"`, then `LAYA_PRELOAD=1 laya-serve` | Gateway in Node, Java or Go; share one GPU across services |
| Docker | `compose.yaml` / `compose.cuda.yaml` in the upstream repo | Kubernetes or VM deployments |
| ONNX Runtime | `pip install "laya[onnx]"` | CPU inference without PyTorch |
| GPU fast path | `pip install "laya[fast]"`, `laya.load(..., fast=True)` | Lowest latency on NVIDIA (TileLang kernels + CUDA graphs) |

`laya-serve` environment variables: `LAYA_HOST`, `LAYA_PORT`, `LAYA_DEVICE`, `LAYA_PRELOAD`,
`LAYA_MODELS`, `LAYA_THREADS` (CPU: at most the number of physical cores), `LAYA_API_KEY` (bearer auth).
Bind to `127.0.0.1` unless you set an API key.

## Memory

* All three checkpoints together are about 1.16B parameters. By default the Router keeps two resident (english and multilingual).
* `max_loaded=1` saves memory but reloads on every language switch, which takes 7-10 s. Avoid it for voice.
* Servers: `Router(preload=True)` and one warm-up call at startup.

## Fine-tuning (where most of the accuracy comes from)

1. Collect labelled decisions from your domain: state, question and correct answer, a few thousand or more.
2. Run the upstream notebook `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
   (free 2x T4 on Kaggle, about 4-5 h for 4 epochs over about 30k questions).
3. It fits calibration temperatures and pushes to the Hugging Face Hub.
4. Load it: `laya.load("your-org/your-checkpoint")`, or attach it to a Router with `router.attach("english", agent)`.
5. Re-run `benchmarks/bench_accuracy.py`. Evaluate on held-out data, not the training items.

Example from the project: a browser-agent decision head went from 0.10 to 0.66 top-1 after fine-tuning.
