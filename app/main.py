"""
Confucius4-R2T2 Multi-Architecture Streaming & One-shot ASR Service
Supports:
  - Automatic Model Download (Hugging Face netease-youdao/Confucius4-R2T2)
  - Automatic Warmup (1s Dummy Infer)
  - Dual Backend Auto-Fallback: vLLM with graceful fallback to Transformers;
    streaming works on both
  - Endpoints:
      GET /healthz
      POST /transcribe (one-shot file inference)
      POST /transcribe/stream (long files, one Server-Sent Event per 30 s segment)
      WS /asr_stream_api_v1 (real-time PCM streaming with VAD sentence segmentation)
      WS /ws/stream (minimal real-time PCM streaming, no segmentation)
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from pydantic import BaseModel

# Disable FlashInfer sampler to avoid nvcc JIT compilation requirement
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from audio_io import load_audio
from infer_gate import BATCH, LIVE, InferGate
from r2t2 import R2T2ASRModel
from script_convert import convert_text, resolve_output_script
from stream_session import StreamSession, unfixed_tail
from textproc import AUTO_LANGUAGES, clean_transcript, normalize_language
from vad import load_vad_factory

MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
MODEL_ID = os.environ.get("MODEL_ID", "netease-youdao/Confucius4-R2T2")
ASR_BACKEND = os.environ.get("ASR_BACKEND", "auto").lower()  # auto | vllm | llama | transformers
# auto 的嘗試順序：越前面越快，但越挑裝置；transformers 墊底，有 GPU 就一定能跑。
AUTO_BACKEND_ORDER = ("vllm", "llama", "transformers")
GGUF_DIR = os.environ.get("GGUF_DIR", os.path.join(os.environ.get("MODEL_DIR", "/model"), "gguf"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048"))
AUTO_DOWNLOAD = os.environ.get("AUTO_DOWNLOAD", "1") == "1"
AUTO_WARMUP = os.environ.get("AUTO_WARMUP", "1") == "1"
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.35"))
VAD_DIR = os.environ.get("VAD_DIR", os.path.join(MODEL_DIR, "FireRedVAD"))
# 串流沒指定語言時用的預設值。/asr_stream_api_v1 把 zhen、auto 都當成逐句自動判斷；
# /ws/stream 沒有斷句，zhen 仍以 Chinese 解碼。
STREAM_DEFAULT_LANGUAGE = os.environ.get("STREAM_DEFAULT_LANGUAGE", "zhen")
# 逐句自動判斷時信任的語言，第一個是判出清單外語言時的退路。
# 不放葡萄牙文等：帶口音的華語、台語常被判成那些語言。
STREAM_AUTO_LANGUAGES = os.environ.get(
    "STREAM_AUTO_LANGUAGES", "Chinese,English,Japanese,Korean,Spanish"
)
STREAM_AUTO_CODES = ("zhen",) + AUTO_LANGUAGES
STREAM_SECRET_KEYS = {
    k.strip()
    for k in os.environ.get("STREAM_SECRET_KEYS", "test0102").split(",")
    if k.strip()
}
# 有人在講話時，上傳音檔最多每隔這麼多秒佔用一次推論（一次約一兩秒）。
UPLOAD_MIN_INTERVAL_SECONDS = float(os.environ.get("UPLOAD_MIN_INTERVAL_SECONDS", "5"))
# 同時進行的即時串流上限。超過就拒絕新連線，不要讓所有人一起變慢。
MAX_CONCURRENT_STREAMS = int(os.environ.get("MAX_CONCURRENT_STREAMS", "12"))
# 單次推論卡超過這個秒數，/healthz 就回報 stalled。
STALL_SECONDS = float(os.environ.get("STALL_SECONDS", "60"))

# 上傳的音檔每 30 秒切一段各自辨識：整段丟進去會超過模型的 token 上限，
# 分段也讓其他連線能在段與段之間插隊。與 server.py 的切法一致。
TRANSCRIBE_SEGMENT_SECONDS = 30
TRANSCRIBE_SEGMENT_SAMPLES = TRANSCRIBE_SEGMENT_SECONDS * 16000

STREAM_EOS = "YOUDAO_ONETIME_ASR_STREAM_EOS"
STREAM_RECV_TIMEOUT = 120
MAX_SYSTEM_PROMPT_CHARS = 4000
ERROR_MSG_NO_HEADER = "json header is expected"

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
    "vad": None,
    "active_streams": 0,
}

GATE = InferGate(batch_interval=UPLOAD_MIN_INTERVAL_SECONDS)

# 進行中的上傳辨識：job_id -> 取消旗標。/transcribe/cancel 設旗標，辨識迴圈在段與段之間檢查。
UPLOAD_JOBS: dict = {}


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


def _load_vllm():
    return R2T2ASRModel.LLM(
        model=MODEL_DIR,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=VLLM_MAX_MODEL_LEN,
        enforce_eager=True,
        disable_log_stats=True,
    )


def _load_llama():
    from llama_backend import R2T2LlamaModel, ensure_gguf, import_native

    native_cls = import_native()
    if native_cls is None:
        raise RuntimeError("這個環境沒有編譯好的 llama.cpp 擴充")
    paths = ensure_gguf(GGUF_DIR, AUTO_DOWNLOAD)
    if paths is None:
        raise RuntimeError(f"{GGUF_DIR} 內需要剛好一個解碼器 .gguf 與一個 mmproj .gguf")
    return R2T2LlamaModel.load(MODEL_DIR, paths[0], paths[1], native_cls)


def _load_transformers():
    return R2T2ASRModel.from_pretrained(MODEL_DIR, device_map="cuda", dtype=torch.bfloat16)


BACKEND_LOADERS = {
    "vllm": _load_vllm,
    "llama": _load_llama,
    "transformers": _load_transformers,
}


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

    # 2. 載入模型：auto 依序嘗試，哪個後端在這台裝置上起得來就用哪個
    t_load = time.time()
    model = None
    backend_used = None
    candidates = AUTO_BACKEND_ORDER if ASR_BACKEND == "auto" else (ASR_BACKEND,)
    failures = []
    for name in candidates:
        loader = BACKEND_LOADERS.get(name)
        if loader is None:
            failures.append(f"{name}: 不支援的後端名稱")
            continue
        try:
            log.info("嘗試載入 %s 後端...", name)
            model = loader()
            backend_used = name
            log.info("%s 後端載入成功！", name)
            break
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            log.warning("%s 後端無法使用 (%s)", name, exc)

    if model is None:
        STATE["error"] = "沒有可用的推論後端 — " + "；".join(failures)
        STATE["status"] = "error"
        log.error(STATE["error"])
        return

    STATE["load_seconds"] = round(time.time() - t_load, 1)
    STATE["backend"] = backend_used

    # 3. 自動預熱 (Warmup)
    if AUTO_WARMUP:
        log.info("開始執行啟動自動預熱 (Dummy Infer Warmup)...")
        t_warmup = time.time()
        try:
            dummy_wav = np.zeros(16000, dtype=np.float32)
            with GATE.run():
                state = model.init_streaming_state(
                    context="",
                    language="Chinese",
                    unfixed_chunk_num=0,
                    unfixed_token_num=1,
                    chunk_size_sec=0.32,
                )
                model.streaming_transcribe_no_reset(dummy_wav, state, 4)
                model.transcribe(audio=[(dummy_wav, 16000)], language=["Chinese"])
            STATE["warmup_seconds"] = round(time.time() - t_warmup, 2)
            log.info("自動預熱完成，耗時 %.2f 秒", STATE["warmup_seconds"])
        except Exception as exc:
            log.warning("預熱失敗 (非致命): %s", exc)

    STATE["vad"] = load_vad_factory(VAD_DIR, AUTO_DOWNLOAD)
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
    # 這個行程在 GPU 上的實際總佔用（含 llama.cpp 與 PyTorch 保留未用的空間）；
    # vram_used_gb 只算 PyTorch 正在用的，曾經藏住 2.6 GB 的浪費。
    vram_process_gb: Optional[float] = None
    download_seconds: Optional[float] = None
    load_seconds: Optional[float] = None
    warmup_seconds: Optional[float] = None
    error: Optional[str] = None
    streaming: bool = False
    vad_loaded: bool = False
    active_streams: int = 0
    inference: Optional[dict] = None


def _process_vram_gb() -> Optional[float]:
    """GPU memory held by this process as the driver sees it, or None if unknown."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pid = str(os.getpid())
    mib = [int(m) for p, m in (line.split(",") for line in out.splitlines() if "," in line) if p.strip() == pid]
    return round(sum(mib) / 1024, 2) if mib else None


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    cuda = torch.cuda.is_available()
    vram = round(torch.cuda.memory_allocated() / 1024**3, 1) if cuda else 0.0
    inference = GATE.snapshot()
    status = STATE["status"]
    if status == "ok" and inference["busy_seconds"] > STALL_SECONDS:
        status = "stalled"
    return HealthResponse(
        status=status,
        backend=STATE["backend"],
        model_loaded=STATE["model_loaded"],
        arch=STATE["arch"],
        gpu=STATE["gpu"],
        vram_used_gb=vram,
        vram_process_gb=_process_vram_gb() if cuda else None,
        download_seconds=STATE["download_seconds"],
        load_seconds=STATE["load_seconds"],
        warmup_seconds=STATE["warmup_seconds"],
        error=STATE["error"],
        streaming=STATE["model_loaded"],
        vad_loaded=STATE["vad"] is not None,
        active_streams=STATE["active_streams"],
        inference=inference,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return healthz()


def _web_dir() -> str:
    web_dir = os.environ.get("WEB_DIR", "/web")
    if not os.path.exists(web_dir):
        local_web = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
        if os.path.exists(local_web):
            web_dir = local_web
    return web_dir


@app.get("/")
def serve_index():
    index_file = os.path.join(_web_dir(), "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "ok", "service": "Confucius4-R2T2 ASR Service"}


@app.get("/llms.txt")
def serve_llms_txt():
    path = os.path.join(_web_dir(), "llms.txt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="llms.txt not found")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


if os.path.isdir(os.path.join(_web_dir(), "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(_web_dir(), "assets")), name="web_assets")


def _require_model():
    model = STATE["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=STATE["error"] or "模型正在載入或預熱中，請稍候再試",
        )
    return model


def _resolve_language(model, language: Optional[str], default: Optional[str] = None) -> Optional[str]:
    """Client language code -> model language name, None meaning auto-detect."""
    lang = normalize_language(language, default)
    if lang is not None and lang not in model.get_supported_languages():
        raise ValueError(f"Unsupported language: {language}")
    return lang


def _stream_languages(model, header: dict) -> Tuple[Optional[str], List[str]]:
    """Header -> (forced language, auto-detect candidates); exactly one is set."""
    raw = header.get("language") or STREAM_DEFAULT_LANGUAGE
    if str(raw).strip().lower() not in STREAM_AUTO_CODES:
        return _resolve_language(model, raw), []
    codes = header.get("languages")
    if not codes:
        # 伺服器設定的清單只留這個模型支援的，不讓設定檔害每條連線都失敗。
        supported = set(model.get_supported_languages())
        default = [normalize_language(c) for c in STREAM_AUTO_LANGUAGES.split(",") if c.strip()]
        return None, list(dict.fromkeys(c for c in default if c in supported)) or ["Chinese"]
    if isinstance(codes, str):
        codes = codes.split(",")
    if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
        raise ValueError("languages must be a list of language codes")
    candidates = []
    for code in codes:
        lang = _resolve_language(model, code) if code.strip() else None
        if lang is None:
            raise ValueError(f"languages cannot contain {code!r}")
        if lang not in candidates:
            candidates.append(lang)
    return None, candidates


def _batch_language(model, language: Optional[str]) -> Optional[str]:
    try:
        return _resolve_language(model, language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _output_script(value: Optional[str]) -> str:
    try:
        return resolve_output_script(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class CancelRequest(BaseModel):
    job_id: str


async def _save_upload(file: UploadFile) -> str:
    content = await file.read()
    suffix = Path(file.filename or "tmp.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(content)
        return tf.name


def _upload_priority(wav: np.ndarray) -> int:
    """Short clips are someone waiting for an answer; long files are background work.

    一段以內的音檔（30 秒內）通常是互動式的批次辨識，和串流一樣排在前面；
    要切成多段的長音檔才當成背景工作，有人在講話時讓路。
    """
    return LIVE if len(wav) <= TRANSCRIBE_SEGMENT_SAMPLES else BATCH


def _transcribe_segment(
    model, wav: np.ndarray, context: str, lang: Optional[str], priority: int = BATCH
) -> str:
    with GATE.run(priority):
        results = model.transcribe(
            audio=[(wav, 16000)],
            context=context,
            language=[lang] if lang else None,
            return_time_stamps=False,
        )
    return results[0].text


def _segments(wav: np.ndarray):
    for start in range(0, max(len(wav), 1), TRANSCRIBE_SEGMENT_SAMPLES):
        yield start, wav[start : start + TRANSCRIBE_SEGMENT_SAMPLES]


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="音訊檔案 (wav, mp3, m4a, flac 等)"),
    language: Optional[str] = Form(None, description="語言指定 (如 Chinese, English, zh, zhen；不填為自動判斷)"),
    context: Optional[str] = Form("", description="熱詞或上下文提示詞 (如：請使用台灣繁體中文)"),
    output_script: Optional[str] = Form(None, description="輸出文字：simplified（預設）或 traditional（台灣繁體）"),
):
    """一般離線單檔轉寫 (支援防跳針清洗與自訂上下文/熱詞)；超過 30 秒自動分段，各段以換行串接"""
    model = _require_model()
    lang_param = _batch_language(model, language)
    script = _output_script(output_script)
    tmp_path = await _save_upload(file)

    try:
        t0 = time.time()
        wav = await asyncio.to_thread(load_audio, tmp_path)
        raw_parts, clean_parts = [], []
        priority = _upload_priority(wav)
        for _, segment in _segments(wav):
            raw = await asyncio.to_thread(
                _transcribe_segment, model, segment, context or "", lang_param, priority
            )
            clean = convert_text(clean_transcript(raw), script)
            if clean:
                raw_parts.append(raw)
                clean_parts.append(clean)
        raw_text = "\n".join(raw_parts)
        clean_text = "\n".join(clean_parts)
        elapsed = time.time() - t0
        duration = round(len(wav) / 16000.0, 2)
        return JSONResponse(
            {
                "status": "success",
                "text": clean_text,
                "raw_text": raw_text if raw_text != clean_text else None,
                "elapsed_seconds": round(elapsed, 2),
                "audio_duration_seconds": duration,
                "backend": STATE["backend"],
                # server.py 的欄位名稱，兩套服務的用戶端可以共用
                "duration_sec": duration,
                "cost_ms": round(elapsed * 1000, 1),
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
    output_script: Optional[str] = Form(None),
):
    """長音檔逐段辨識 (Server-Sent Events)：start → 每 30 秒一則 segment → done（或 cancelled）"""
    model = _require_model()
    lang_param = _batch_language(model, language)
    script = _output_script(output_script)
    tmp_path = await _save_upload(file)
    try:
        wav = await asyncio.to_thread(load_audio, tmp_path)
    except Exception as exc:
        log.exception("Audio decode error: %s", exc)
        raise HTTPException(status_code=400, detail="無法讀取音訊檔案，請確認檔案格式後重試。")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def event(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def sse_generator():
        duration_sec = len(wav) / 16000.0
        total = max(1, (len(wav) + TRANSCRIBE_SEGMENT_SAMPLES - 1) // TRANSCRIBE_SEGMENT_SAMPLES)
        started_at = time.time()
        job_id = uuid.uuid4().hex
        priority = _upload_priority(wav)
        cancel = threading.Event()
        UPLOAD_JOBS[job_id] = cancel
        completed = 0
        try:
            yield event(
                {
                    "type": "start",
                    "job_id": job_id,
                    "duration_sec": round(duration_sec, 2),
                    "total_segments": total,
                    "segment_seconds": TRANSCRIBE_SEGMENT_SECONDS,
                }
            )
            texts = []
            for index, (start, segment) in enumerate(_segments(wav), start=1):
                if cancel.is_set():
                    yield event({"type": "cancelled", "completed_segments": completed})
                    return
                try:
                    raw = await asyncio.to_thread(
                _transcribe_segment, model, segment, context or "", lang_param, priority
            )
                except Exception as exc:
                    log.exception("Audio segment %s/%s failed: %s", index, total, exc)
                    yield event({"type": "error", "message": str(exc)})
                    return
                text = convert_text(clean_transcript(raw), script)
                if text:
                    texts.append(text)
                start_sec = start / 16000.0
                yield event(
                    {
                        "type": "segment",
                        "index": index,
                        "total": total,
                        "start_sec": round(start_sec, 2),
                        "end_sec": round(min(duration_sec, start_sec + len(segment) / 16000.0), 2),
                        "text": text,
                    }
                )
                completed = index
            yield event(
                {
                    "type": "done",
                    "text": "\n".join(texts),
                    "duration_sec": round(duration_sec, 2),
                    "elapsed_seconds": round(time.time() - started_at, 2),
                    "total_segments": total,
                }
            )
        finally:
            # 用戶端中途斷線時產生器會在這裡被關閉，同樣要把工作移除
            UPLOAD_JOBS.pop(job_id, None)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/transcribe/cancel")
async def transcribe_cancel(request: CancelRequest):
    """停止一個進行中的 /transcribe/stream；目前這一段辨識完就停，已完成的段落保留。"""
    cancel = UPLOAD_JOBS.get(request.job_id)
    if cancel is None:
        return JSONResponse(
            {"status": "error", "message": "Transcription job is no longer active"},
            status_code=404,
        )
    cancel.set()
    return {"status": "cancelling", "job_id": request.job_id}


def _init_stream_state(model, context: str, language: Optional[str], chunk_size_sec: float):
    # tokenizer 不能跨執行緒同時使用，建立 state 也要排隊。
    with GATE.run():
        return model.init_streaming_state(
            context=context,
            language=language,
            unfixed_chunk_num=0,
            unfixed_token_num=1,
            chunk_size_sec=chunk_size_sec,
        )


def _stream_context(header: dict) -> str:
    """Build the decoding context (smoothing hint + hotwords) from a stream header."""
    parts = []
    if header.get("smooth", False):
        parts.append("Smooth the text")
    system_prompt = header.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    if len(system_prompt) > MAX_SYSTEM_PROMPT_CHARS:
        raise ValueError(f"system_prompt must be at most {MAX_SYSTEM_PROMPT_CHARS} characters")
    if system_prompt.strip():
        parts.append(system_prompt.strip())
    return "\n".join(parts)


def _redact_header(header: dict) -> dict:
    redacted = dict(header)
    if "secret_key" in redacted:
        redacted["secret_key"] = "[redacted]"
    if "system_prompt" in redacted:
        prompt = redacted["system_prompt"]
        redacted["system_prompt"] = "[redacted]"
        redacted["system_prompt_chars"] = len(prompt.strip()) if isinstance(prompt, str) else 0
    return redacted


def _pcm_from_bytes(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


@app.websocket("/asr_stream_api_v1")
async def asr_stream_api_v1(websocket: WebSocket):
    """
    即時語音串流（有道 ws_server 協議）。
    第一則訊息是 JSON header，之後持續送 16kHz 16-bit mono PCM，
    以文字訊息 YOUDAO_ONETIME_ASR_STREAM_EOS 結束。
    每句結束時回傳 reset=true，並附上整句重新辨識的 final_text。
    """
    await websocket.accept()
    model = STATE["model"]
    if model is None:
        await websocket.close(code=1011, reason="Server model not ready")
        return

    async def send(payload: dict) -> None:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    try:
        header = json.loads(
            await asyncio.wait_for(websocket.receive_text(), STREAM_RECV_TIMEOUT)
        )
        if not isinstance(header, dict) or "requestId" not in header:
            raise ValueError(ERROR_MSG_NO_HEADER)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        return
    except (ValueError, KeyError):
        await send({"status": "error", "msg": ERROR_MSG_NO_HEADER})
        await websocket.close()
        return

    request_id = header["requestId"]
    log.info("stream header=%s", _redact_header(header))
    if header.get("secret_key") not in STREAM_SECRET_KEYS:
        await websocket.close(code=4401, reason="Unauthorized")
        return
    try:
        context = _stream_context(header)
        language, auto_languages = _stream_languages(model, header)
        output_script = resolve_output_script(header.get("output_script"))
    except ValueError as exc:
        await send({"status": "error", "requestId": f"{request_id}", "msg": str(exc)})
        await websocket.close()
        return

    if STATE["active_streams"] >= MAX_CONCURRENT_STREAMS:
        log.warning("拒絕串流連線：已達同時串流上限 (%d)", MAX_CONCURRENT_STREAMS)
        await send(
            {
                "status": "error",
                "requestId": f"{request_id}",
                "msg": "server busy: max concurrent streams reached",
            }
        )
        # 1013 = Try Again Later，與 server.py 相同
        await websocket.close(code=1013, reason="Server busy: max streaming capacity reached")
        return

    vad_factory = STATE["vad"]
    use_vad = bool(header.get("use_vad", True)) and vad_factory is not None
    session = StreamSession(
        model,
        GATE,
        request_id=request_id,
        context=context,
        language=language,
        vad=vad_factory.new() if use_vad else None,
        final_pass=bool(header.get("final_pass", True)),
        output_script=output_script,
        auto_languages=auto_languages,
    )

    eos, stop = object(), object()
    queue: asyncio.Queue = asyncio.Queue()

    async def receiver() -> None:
        try:
            while True:
                message = await asyncio.wait_for(websocket.receive(), STREAM_RECV_TIMEOUT)
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await queue.put(message["bytes"])
                elif message.get("text") == STREAM_EOS:
                    await queue.put(eos)
                    break
        except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await queue.put(stop)

    STATE["active_streams"] += 1
    recv_task = asyncio.create_task(receiver())
    close_code = 1000
    try:
        await send(
            {
                "status": "connected",
                "requestId": f"{request_id}",
                "msg": "",
                "active_connections": STATE["active_streams"],
                "language": language or "auto",
                "languages": auto_languages,
                "vad": use_vad,
                "output_script": output_script,
            }
        )
        carry = b""
        first_audio = True
        while True:
            # 一次取走所有已到的音訊：推論落後時併成較大的 chunk 追進度，不讓延遲越積越多。
            items = [await queue.get()]
            while not queue.empty():
                items.append(queue.get_nowait())
            raw = b"".join(i for i in items if isinstance(i, bytes))
            if raw:
                if first_audio:
                    first_audio = False
                    if raw[0:4] == b"RIFF" and raw[8:12] == b"WAVE":
                        raw = raw[44:]
                raw = carry + raw
                carry = raw[len(raw) - (len(raw) % 2):]
                msgs = await asyncio.to_thread(session.feed, _pcm_from_bytes(raw[: len(raw) - len(carry)]))
                for msg in msgs:
                    await send(msg)
                if not msgs:
                    await send({})
            if any(i is eos for i in items):
                await send(await asyncio.to_thread(session.finish))
                break
            if any(i is stop for i in items):
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as exc:
        log.exception("requestId=%s 串流異常: %s", request_id, exc)
        # 讓用戶端分得出「伺服器出錯」和「正常結束」：先送錯誤訊息，再用 1011 關閉。
        close_code = 1011
        try:
            await send({"status": "error", "requestId": f"{request_id}", "msg": f"inference failed: {exc}"})
        except Exception:
            pass
    finally:
        STATE["active_streams"] -= 1
        recv_task.cancel()
        try:
            await websocket.close(code=close_code)
        except RuntimeError:
            pass


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, language: Optional[str] = None):
    """
    精簡版即時串流：持續送入 16kHz 16-bit Mono PCM，回傳只增不改的文字。
    不做 VAD 斷句；送 {"action": "finish"} 結束。需要斷句與熱詞請用 /asr_stream_api_v1。
    """
    await websocket.accept()
    model = STATE["model"]
    if model is None:
        await websocket.send_json({"type": "error", "detail": "模型尚未就緒"})
        await websocket.close()
        return
    try:
        lang_param = _resolve_language(model, language, STREAM_DEFAULT_LANGUAGE)
    except ValueError as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close()
        return

    state = await asyncio.to_thread(_init_stream_state, model, "", lang_param, 0.32)
    sent = ""
    max_tokens = 4

    try:
        while True:
            data = await websocket.receive()
            if data["type"] == "websocket.disconnect":
                break
            if data.get("bytes") is not None:
                pcm = _pcm_from_bytes(data["bytes"][: len(data["bytes"]) // 2 * 2])

                def run_chunk(pcm=pcm):
                    with GATE.run():
                        _, fixed = model.streaming_transcribe_no_reset(pcm, state, max_tokens)
                        return fixed.split("|")[0]

                fixed = await asyncio.to_thread(run_chunk)
                if len(fixed) > len(sent):
                    delta, sent = fixed[len(sent):], fixed
                    await websocket.send_json(
                        {
                            "type": "token",
                            "delta": delta,
                            "text": sent,
                            "is_final": False,
                        }
                    )

            elif data.get("text") is not None:
                msg = json.loads(data["text"])
                if msg.get("action") == "finish":
                    def run_finish():
                        with GATE.run():
                            fixed = model.finish_streaming_transcribe_no_reset(state, max_tokens)
                            return (fixed.split("|")[0] + unfixed_tail(state)).strip()

                    final_text = await asyncio.to_thread(run_finish)
                    await websocket.send_json(
                        {"type": "done", "text": final_text, "is_final": True}
                    )
                    await websocket.close()
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("WebSocket 串流異常: %s", exc)
        await websocket.send_json({"type": "error", "detail": str(exc)})
