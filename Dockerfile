FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "transformers>=4.51.0" \
    "qwen-asr" \
    "vllm>=0.14.0" \
    "huggingface-hub" \
    "soundfile" \
    "librosa" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    "python-multipart" \
    "websockets"

WORKDIR /app

COPY app/ /app/

ENV PYTHONPATH=/app:${PYTHONPATH:-}

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
