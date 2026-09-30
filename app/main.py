"""
Confucius4-R2T2 Multi-Architecture Streaming & One-shot ASR Service
Supports:
  - Automatic Model Download (Hugging Face netease-youdao/Confucius4-R2T2)
  - Automatic Warmup (1s Dummy Infer)
  - Dual Backend Auto-Fallback: vLLM (Streaming LSP) with graceful fallback to Transformers
  - Endpoints:
      GET /healthz
      POST /transcribe (one-shot file inference)
      POST /transcribe/stream (Server-Sent Events streaming)
      WS /ws/stream (Real-time PCM chunk WebSocket streaming)
"""

import asyncio
import io
import json
import logging
import os
import platform
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel

# Disable FlashInfer sampler to avoid nvcc JIT compilation requirement
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Import local r2t2 package
from r2t2 import R2T2ASRModel

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
MODEL_ID = os.environ.get("MODEL_ID", "netease-youdao/Confucius4-R2T2")
ASR_BACKEND = os.environ.get("ASR_BACKEND", "auto").lower()  # auto | vllm | transformers
AUTO_DOWNLOAD = os.environ.get("AUTO_DOWNLOAD", "1") == "1"
AUTO_WARMUP = os.environ.get("AUTO_WARMUP", "1") == "1"
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.35"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("r2t2-service")

STATE = {
    "model": None,
    "backend": None,
    "status": "loading",
    "model_loaded": False,
    "download_seconds": None,
    "load_seconds": None,
    "warmup_seconds": None,
    "gpu": None,
    "arch": platform.machine(),
    "vram_used_gb": 0.0,
    "error": None,
}

INFER_LOCK = threading.Lock()


def _ensure_model_downloaded() -> None:
    """Ensure weights exist in MODEL_DIR, downloading automatically if missing."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    required_files = ["model.safetensors", "config.json"]
    has_weights = all(
        os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files
    )

    if not has_weights:
        if not AUTO_DOWNLOAD:
            raise FileNotFoundError(f"Model files not found in {MODEL_DIR} and AUTO_DOWNLOAD=0")
        log.info("未檢測到完整模型權重，開始自 Hugging Face 自動拉取 %s 到 %s ...", MODEL_ID, MODEL_DIR)
        t0 = time.time()
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        STATE["download_seconds"] = round(time.time() - t0, 1)
        log.info("模型自動下載完成，耗時 %.1f 秒", STATE["download_seconds"])
    else:
        log.info("本地已存在模型權重 (%s)", MODEL_DIR)


def _load_model() -> None:
    """Load model with backend auto-selection and execute startup warmup."""
    if not torch.cuda.is_available():
        STATE["error"] = "CUDA 不可用，本服務需要 GPU"
        STATE["status"] = "error"
        log.error(STATE["error"])
        return

    gpu_name = torch.cuda.get_device_name(0)
    STATE["gpu"] = gpu_name
    log.info("檢測到硬體架構: %s | GPU: %s", STATE["arch"], gpu_name)

    # 1. 確保權重檔存在
    try:
        _ensure_model_downloaded()
    except Exception as exc:
        STATE["error"] = f"模型下載失敗: {exc}"
        STATE["status"] = "error"
        log.exception(STATE["error"])
        return

    # 2. 載入模型
    t_load = time.time()
    model = None
    backend_used = None

    # 若為 auto 或 vllm，先嘗試 vLLM
    if ASR_BACKEND in ("auto", "vllm"):
        log.info("嘗試初始化 vLLM 串流後端 (gpu_memory_utilization=%.2f)...", GPU_MEM_UTIL)
        try:
            model = R2T2ASRModel.LLM(
                model=MODEL_DIR,
                gpu_memory_utilization=GPU_MEM_UTIL,
                max_model_len=8192,
                enforce_eager=True,
                disable_log_stats=True,
            )
            backend_used = "vllm"
            log.info("vLLM 後端載入成功！")
        except Exception as exc:
            log.warning("vLLM 後端初始化失敗 (%s)", exc)
            if ASR_BACKEND == "vllm":
                STATE["error"] = f"指定 vLLM 後端但啟動失敗: {exc}"
                STATE["status"] = "error"
                log.exception(STATE["error"])
                return
            log.info("自動降級切換至 Transformers 後端...")

    # 若 vllm 未載入或指定 transformers
    if model is None:
        try:
            log.info("載入 Transformers 後端 (device_map=cuda, dtype=bfloat16)...")
            model = R2T2ASRModel.from_pretrained(
                MODEL_DIR,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )
            backend_used = "transformers"
            log.info("Transformers 後端載入成功！")
        except Exception as exc:
            STATE["error"] = f"Transformers 後端載入失敗: {exc}"
            STATE["status"] = "error"
            log.exception(STATE["error"])
            return

    STATE["load_seconds"] = round(time.time() - t_load, 1)
    STATE["backend"] = backend_used

    # 3. 自動預熱 (Warmup)
    if AUTO_WARMUP:
        log.info("開始執行啟動自動預熱 (Dummy Infer Warmup)...")
        t_warmup = time.time()
        try:
            dummy_wav = np.zeros(16000, dtype=np.float32)
            with INFER_LOCK:
                if backend_used == "vllm":
                    state = model.init_streaming_state(
                        context="",
                        language="Chinese",
                        unfixed_chunk_num=0,
                        unfixed_token_num=1,
                        chunk_size_sec=0.32,
                    )
                    model.streaming_transcribe(dummy_wav, state, max_new_tokens=4)
                    model.finish_streaming_transcribe(state, max_new_tokens=4)
                else:
                    model.transcribe(audio=[(dummy_wav, 16000)])
            STATE["warmup_seconds"] = round(time.time() - t_warmup, 2)
            log.info("自動預熱完成，耗時 %.2f 秒", STATE["warmup_seconds"])
        except Exception as exc:
            log.warning("預熱失敗 (非致命): %s", exc)

    STATE["vram_used_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 1)
    STATE["model"] = model
    STATE["model_loaded"] = True
    STATE["status"] = "ok"
    log.info(
        "Confucius4-R2T2 服務就緒！後端: %s, 顯存佔用: %.1f GB",
        STATE["backend"],
        STATE["vram_used_gb"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_model, daemon=True).start()
    yield
    STATE["model"] = None


app = FastAPI(
    title="Confucius4-R2T2 Streaming ASR",
    description="網易有道 Confucius4-R2T2 實時真流式語音識別服務（支援最長穩定前綴 LSP 只增不改與自動預熱）",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    backend: Optional[str] = None
    model_loaded: bool
    arch: str
    gpu: Optional[str] = None
    vram_used_gb: float
    download_seconds: Optional[float] = None
    load_seconds: Optional[float] = None
    warmup_seconds: Optional[float] = None
    error: Optional[str] = None


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    cuda = torch.cuda.is_available()
    vram = round(torch.cuda.memory_allocated() / 1024**3, 1) if cuda else 0.0
    return HealthResponse(
        status=STATE["status"],
        backend=STATE["backend"],
        model_loaded=STATE["model_loaded"],
        arch=STATE["arch"],
        gpu=STATE["gpu"],
        vram_used_gb=vram,
        download_seconds=STATE["download_seconds"],
        load_seconds=STATE["load_seconds"],
        warmup_seconds=STATE["warmup_seconds"],
        error=STATE["error"],
    )


def _require_model():
    model = STATE["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=STATE["error"] or "模型正在載入或預熱中，請稍候再試",
        )
    return model


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="音訊檔案 (wav, mp3, m4a, flac 等)"),
    language: Optional[str] = Form(None, description="語言指定 (如 Chinese, English)"),
    context: Optional[str] = Form("", description="熱詞或上下文提示詞"),
):
    """一般離線單檔轉寫"""
    model = _require_model()

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "tmp.wav").suffix, delete=False) as tf:
        tf.write(content)
        tmp_path = tf.name

    try:
        t0 = time.time()
        wav, sr = librosa.load(tmp_path, sr=16000, mono=True)

        def infer():
            with INFER_LOCK:
                results = model.transcribe(
                    audio=[(wav, 16000)],
                    context=context or "",
                    language=[language] if language else None,
                    return_time_stamps=False,
                )
                return results[0].text

        text = await asyncio.to_thread(infer)
        elapsed = round(time.time() - t0, 2)
        return JSONResponse(
            {
                "text": text,
                "elapsed_seconds": elapsed,
                "audio_duration_seconds": round(len(wav) / 16000.0, 2),
                "backend": STATE["backend"],
            }
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/transcribe/stream")
async def transcribe_stream(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    context: Optional[str] = Form(""),
    chunk_ms: int = Form(160),
    lookahead_ms: int = Form(160),
):
    """SSE 模擬分塊即時流式轉寫 (Server-Sent Events)"""
    model = _require_model()
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "tmp.wav").suffix, delete=False) as tf:
        tf.write(content)
        tmp_path = tf.name

    async def sse_generator():
        try:
            wav, _ = librosa.load(tmp_path, sr=16000, mono=True)
            step_samples = int(round(chunk_ms / 1000.0 * 16000))
            lookahead_samples = int(round(lookahead_ms / 1000.0 * 16000))
            chunk_size_sec = (chunk_ms + lookahead_ms) / 1000.0

            if STATE["backend"] == "vllm":
                state = model.init_streaming_state(
                    context=context or "",
                    language=language,
                    unfixed_chunk_num=0,
                    unfixed_token_num=1,
                    chunk_size_sec=chunk_size_sec,
                )
                pos = 0
                is_first = True
                prev_text = ""
                max_tokens = max(1, int((step_samples + lookahead_samples) / 1280))

                while pos < len(wav):
                    if is_first:
                        seg = wav[pos : pos + step_samples + lookahead_samples]
                        is_first = False
                    else:
                        seg = wav[pos : pos + step_samples]
                    pos += len(seg)

                    def step_infer():
                        with INFER_LOCK:
                            _, raw = model.streaming_transcribe(seg, state, max_tokens)
                            return raw.split("|")[0].strip()

                    curr_text = await asyncio.to_thread(step_infer)
                    if curr_text and curr_text != prev_text:
                        delta = curr_text[len(prev_text):] if curr_text.startswith(prev_text) else curr_text
                        if delta:
                            yield f"data: {json.dumps({'type': 'token', 'delta': delta, 'text': curr_text}, ensure_ascii=False)}\n\n"
                            prev_text = curr_text

                def finish_infer():
                    with INFER_LOCK:
                        model.finish_streaming_transcribe(state, max_tokens)
                        return state.text.split("|")[0].strip()

                final_text = await asyncio.to_thread(finish_infer)
                yield f"data: {json.dumps({'type': 'done', 'text': final_text}, ensure_ascii=False)}\n\n"
            else:
                # Transformers fallback
                def full_infer():
                    with INFER_LOCK:
                        return model.transcribe(audio=[(wav, 16000)], language=[language] if language else None)[0].text

                text = await asyncio.to_thread(full_infer)
                yield f"data: {json.dumps({'type': 'done', 'text': text}, ensure_ascii=False)}\n\n"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    真實即時語音 WebSocket 串流端點。
    客戶端持續送入 16kHz 16-bit Mono PCM 二進制音訊 chunk，
    服務端利用 R2T2 LSP 只增不改機制即時推送辨識文字。
    """
    await websocket.accept()
    model = STATE["model"]
    if model is None or STATE["backend"] != "vllm":
        await websocket.send_json(
            {"type": "error", "detail": "vLLM 串流模型尚未就緒"}
        )
        await websocket.close()
        return

    chunk_size_sec = 0.32
    state = model.init_streaming_state(
        context="",
        language="Chinese",
        unfixed_chunk_num=0,
        unfixed_token_num=1,
        chunk_size_sec=chunk_size_sec,
    )

    prev_text = ""
    max_tokens = 4

    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data:
                raw_bytes = data["bytes"]
                # 轉成 float32 16k 音訊
                pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                def run_chunk():
                    with INFER_LOCK:
                        _, text_raw = model.streaming_transcribe(pcm, state, max_tokens)
                        return text_raw.split("|")[0].strip()

                curr_text = await asyncio.to_thread(run_chunk)
                if curr_text and curr_text != prev_text:
                    delta = curr_text[len(prev_text):] if curr_text.startswith(prev_text) else curr_text
                    if delta:
                        await websocket.send_json(
                            {
                                "type": "token",
                                "delta": delta,
                                "text": curr_text,
                                "is_final": False,
                            }
                        )
                        prev_text = curr_text

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("action") == "finish":
                    def run_finish():
                        with INFER_LOCK:
                            model.finish_streaming_transcribe(state, max_tokens)
                            return state.text.split("|")[0].strip()

                    final_text = await asyncio.to_thread(run_finish)
                    await websocket.send_json(
                        {"type": "done", "text": final_text, "is_final": True}
                    )
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("WebSocket 串流異常: %s", exc)
        await websocket.send_json({"type": "error", "detail": str(exc)})
