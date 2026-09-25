# Laya Playground backend (playground/server.py) - CPU image.
#   docker build -t laya-playground .
#   docker run -p 7860:7860 -e LAYA_PLAYGROUND_KEY=change-me laya-playground
# Runs on Hugging Face Spaces (Docker SDK), Azure Container Apps, Google Cloud Run, Railway, Fly.io...
# Needs about 3 GB of RAM (english + multilingual loaded).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf-cache \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    LAYA_PRELOAD=1

WORKDIR /app

# CPU-only torch first (the default wheel pulls ~2 GB of CUDA libraries we don't need)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

# Bake the english + multilingual weights into the image so a cold start doesn't download ~1.4 GB.
# typed-decisions is left out; it still downloads on first use if someone picks it.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('convaiinnovations/laya', ignore_patterns=['typed-decisions/*'])"

COPY shared.py .
COPY playground/server.py playground/index.html playground/chat.html playground/

# Hugging Face Spaces runs the container as uid 1000
RUN useradd -m -u 1000 app && chown -R app /app
USER app

EXPOSE 7860
CMD ["python", "playground/server.py"]
