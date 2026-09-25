# Deploying the Laya Playground

## Why it's two pieces

`playground/server.py` can't run on Vercel. Vercel's Python functions have a size limit of about 250 MB,
and this server needs PyTorch (~350 MB of libraries) plus the model weights (804 MB English + 614 MB
multilingual) and about 3 GB of RAM. So:

```
Browser --> Vercel (chat.html, index.html, api/proxy.js) --> Docker backend (server.py + Laya)
            static pages + a tiny proxy that holds           Hugging Face Spaces (free, 16 GB RAM),
            the backend URL and API key as secrets           Azure Container Apps, Cloud Run...
```

## Step 1: deploy the backend (Hugging Face Spaces, free CPU tier)

1. Create a Space at https://huggingface.co/new-space. Pick SDK **Docker** and hardware **CPU basic** (free: 2 vCPU, 16 GB RAM).
2. In the Space: **Settings → Variables and secrets → New secret**: `LAYA_PLAYGROUND_KEY` = a long random string.
3. Push the backend files (PowerShell, from the repo root):

```powershell
git clone https://huggingface.co/spaces/<your-hf-user>/laya-playground ..\laya-space
Copy-Item Dockerfile, requirements-server.txt, .dockerignore, shared.py ..\laya-space\
New-Item -ItemType Directory -Force ..\laya-space\playground | Out-Null
Copy-Item playground\server.py, playground\index.html, playground\chat.html ..\laya-space\playground\
Copy-Item deploy\hf-space-README.md ..\laya-space\README.md
cd ..\laya-space
git add . ; git commit -m "Laya playground backend" ; git push
```

The first build takes about 10 minutes (it installs torch and bakes the weights into the image).
Check it at `https://<your-hf-user>-laya-playground.hf.space/api/health`.
Free Spaces sleep after about 48 h without traffic, and the first request after that takes a minute.

**Other hosts.** The same Dockerfile works anywhere that runs containers. Give it at least 3 GB of RAM and set `LAYA_PLAYGROUND_KEY`.
* Azure Container Apps (you already have Azure):
  `az containerapp up --name laya-playground --resource-group <rg> --source . --ingress external --target-port 7860 --env-vars LAYA_PLAYGROUND_KEY=<key>`
  Then set CPU 2 and memory 4Gi on the app.
* Google Cloud Run: `gcloud run deploy laya-playground --source . --memory 4Gi --cpu 2 --port 7860 --set-env-vars LAYA_PLAYGROUND_KEY=<key>`
* Local check: `docker build -t laya-playground .` then `docker run -p 7860:7860 -e LAYA_PLAYGROUND_KEY=<key> laya-playground`

## Step 2: deploy the frontend (Vercel)

### With the CLI

```powershell
npm i -g vercel
cd playground
vercel login
vercel link                                   # new project; framework "Other"; root = this folder
vercel env add LAYA_BACKEND_URL production    # https://<your-hf-user>-laya-playground.hf.space
vercel env add LAYA_PLAYGROUND_KEY production # the same secret as the backend
vercel --prod
```

### Or from the dashboard

Import the GitHub repo, then:
* **Root Directory:** `playground`
* **Framework Preset:** Other
* **Environment Variables:** `LAYA_BACKEND_URL`, `LAYA_PLAYGROUND_KEY`

Then deploy.

Open `https://<project>.vercel.app/chat` for the chat page, or `/` for the advanced playground.

## Files involved

| File | Where it runs | Purpose |
|------|---------------|---------|
| `Dockerfile`, `requirements-server.txt`, `.dockerignore` | backend | CPU image with english + multilingual weights baked in |
| `playground/server.py` | backend | API; reads `HOST`, `PORT`, `LAYA_PRELOAD`, `LAYA_PLAYGROUND_KEY`, `ALLOWED_ORIGINS` |
| `playground/vercel.json` | Vercel | `/api/*` goes to the proxy; `/chat` serves `chat.html` |
| `playground/api/proxy.js` | Vercel | Forwards `health`, `route` and `predict` to the backend and adds the API key |
| `playground/.vercelignore` | Vercel | Keeps `server.py` off Vercel |

## Expectations

* Latency on free CPU hardware is about 1-3 s per message (like the laptop numbers). On a GPU host, tens of ms.
* The proxy's `maxDuration` is 60 s, which covers a cold start after the Space wakes up. The first message may still time out once; retry it.
* Never put `.env` (Azure keys) in the Space or the Vercel project. The playground doesn't need them.
