# coding=utf-8
# Confucius4-R2T2 Streaming ASR & Web Caption Server
# Based on NetEase Youdao Confucius4-R2T2 ws_server.py
# Enhanced with configurable GPU memory utilization, max_model_len, enforce_eager=True, and web frontend serving.

import os
import sys
import time
import json
import uuid
import string
import re
import asyncio
import logging
import argparse
import traceback
import numpy as np

# Ensure local package is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFUCIUS_DIR = os.path.join(CURRENT_DIR, "Confucius4-R2T2")
if CONFUCIUS_DIR not in sys.path:
    sys.path.insert(0, CONFUCIUS_DIR)

from sanic import Sanic, Request, Websocket, response
from sanic.worker.manager import WorkerManager
from sanic.worker.process import WorkerProcess

# Common Chinese and English punctuation characters for hallucination detection
_PUNCT_CHARS = "，。！？、；：,.!?;:~…·\"'()（）《》—-"
_PUNCT_RE = re.compile(r"[%s\s]+" % re.escape(_PUNCT_CHARS))

def _normalize_for_pattern(s: str) -> str:
    return _PUNCT_RE.sub("", s).lower()

def detect_and_fix_repetitions(text, threshold=5):
    def fix_char_repeats(s, thresh):
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i:i+count])
                i += count
        return ''.join(res)

    def fix_pattern_repeats(s, thresh, max_len=20):
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i:i+k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx:start_idx+k] != pattern:
                        valid = False
                        break
                if valid:
                    total_rep = thresh
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index:end_index+k] == pattern:
                        total_rep += 1
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return ''.join(result)

    text_raw = text
    text = fix_char_repeats(text_raw, threshold)
    text = fix_pattern_repeats(text, threshold)
    return text

def detect_hallucination(requestId: str, text: str, pattern_repeat_thresh: int = 5, max_pattern_len: int = 50, tail_check_len: int = 256):
    if not text:
        return False, ""
    text = text[-tail_check_len:]
    n = len(text)
    for k in range(1, max_pattern_len + 1):
        if n < k * pattern_repeat_thresh:
            continue
        pattern = text[-k:]
        if pattern.strip(_PUNCT_CHARS + " \t") == "":
            continue
        ok = True
        for r in range(1, pattern_repeat_thresh):
            if text[-(r + 1) * k:-r * k] != pattern:
                ok = False
                break
        if ok:
            return True, f"tail_pattern:'{pattern}'x{pattern_repeat_thresh}+"

    norm_tail = _normalize_for_pattern(text)
    n2 = len(norm_tail)
    for k in range(3, max_pattern_len + 1):
        if n2 < k * pattern_repeat_thresh:
            continue
        pattern = norm_tail[-k:]
        ok = True
        for r in range(1, pattern_repeat_thresh):
            if norm_tail[-(r + 1) * k:-r * k] != pattern:
                ok = False
                break
        if ok:
            return True, f"tail_pattern_norm:'{pattern}'x{pattern_repeat_thresh}+"
    return False, ""

# Patch WorkerProcess
original_init = WorkerProcess.__init__
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.daemon = False
WorkerProcess.__init__ = patched_init

# Logger setup
logger = logging.getLogger("r2t2_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(_handler)
logger.propagate = False

# Sanic App
Sanic.START_METHOD_SET = True
Sanic.start_method = "fork"
app = Sanic("confucius4_r2t2_server")
app.config.WEBSOCKET_PING_INTERVAL = None
app.config.WEBSOCKET_PING_TIMEOUT = None
WorkerManager.THRESHOLD = 300

YOUDAO_ONETIME_ASR_EOS_STRING = "YOUDAO_ONETIME_ASR_STREAM_EOS"
SAMPLING_RATE = 16000
CHUNK_ASR_SECONDS = 0.16
CHUNK_ASR_SIZE = int(CHUNK_ASR_SECONDS * SAMPLING_RATE)
STEP_MS = int(CHUNK_ASR_SECONDS * 1000)
LOOKAHEAD_MS = 160
UNFIX_TOKEN_NUM = 1
MAX_TOKENS = 10
ERROR_MSG_NO_HEADER = "json header is expected"
MAX_SYSTEM_PROMPT_CHARS = 4000

def resolve_system_prompt(header):
    if "system_prompt" not in header:
        return ""
    system_prompt = header["system_prompt"]
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    if len(system_prompt) > MAX_SYSTEM_PROMPT_CHARS:
        raise ValueError(f"system_prompt must be at most {MAX_SYSTEM_PROMPT_CHARS} characters")
    return system_prompt.strip()

def redact_stream_request_header(header):
    redacted = dict(header)
    if "secret_key" in redacted:
        redacted["secret_key"] = "[redacted]"
    if "system_prompt" in redacted:
        system_prompt = header["system_prompt"]
        redacted["system_prompt"] = "[redacted]"
        redacted["system_prompt_chars"] = len(system_prompt.strip()) if isinstance(system_prompt, str) else 0
    return redacted

def resolve_qwen_context(header, smooth):
    context_parts = []
    if smooth:
        context_parts.append("Smooth the text")
    system_prompt = resolve_system_prompt(header)
    if system_prompt:
        context_parts.append(system_prompt)
    return "\n".join(context_parts)

asr_model = None
stream_vad = None

@app.listener('before_server_start')
async def initialize_models(app):
    global stream_vad, asr_model
    from r2t2 import R2T2ASRModel
    from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

    logger.info(f"Worker (pid: {os.getpid()}) initializing models...")
    
    # 1. Load VAD
    vad_config = FireRedStreamVadConfig(
        use_gpu=True,
        smooth_window_size=5,
        speech_threshold=0.4,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=60,
        chunk_max_frame=30000
    )
    if os.path.exists(args.vad_model_path):
        stream_vad = FireRedStreamVad.from_pretrained(args.vad_model_path, vad_config)
        logger.info("✅ VAD model initialized successfully")
    else:
        logger.warning(f"⚠️ VAD model path not found: {args.vad_model_path}, continuing without VAD")

    # 2. Load ASR Model via vLLM with eager mode (instant start, no graph compilation timeout)
    logger.info(f"Loading ASR model from {args.asr_model_path} with gpu_util={args.gpu_mem_util}, max_len={args.max_model_len}, eager=True")
    asr_model = R2T2ASRModel.LLM(
        model=args.asr_model_path,
        gpu_memory_utilization=float(args.gpu_mem_util),
        max_model_len=int(args.max_model_len),
        enforce_eager=True,
        max_new_tokens=1,
    )
    logger.info("✅ ASR model initialized successfully")

    # Warmup
    logger.info("Warming up ASR model...")
    _sr = SAMPLING_RATE
    _step = int(round(STEP_MS / 1000.0 * _sr))
    _lookhead = int(round(LOOKAHEAD_MS / 1000.0 * _sr))
    for _warmup_lang in ("Chinese", "English"):
        _warmup_state = asr_model.init_streaming_state(
            language=_warmup_lang,
            unfixed_chunk_num=0,
            unfixed_token_num=UNFIX_TOKEN_NUM,
            chunk_size_sec=CHUNK_ASR_SECONDS,
        )
        _warmup_max_new_tokens = max(1, int((_step + _lookhead) / 1280))
        if _warmup_lang == "Chinese":
            _warmup_max_new_tokens = 2 * _warmup_max_new_tokens
        for _ in range(2):
            _dummy = np.zeros(CHUNK_ASR_SIZE, dtype=np.float32)
            asr_model.streaming_transcribe(_dummy, _warmup_state, _warmup_max_new_tokens)
        asr_model.finish_streaming_transcribe(_warmup_state, _warmup_max_new_tokens)
    logger.info("✅ ASR model warmup complete")

def read_pcm(pcm_byte_data, is_wav=False):
    pcm_data = np.frombuffer(pcm_byte_data, dtype=np.int16, offset=44 if is_wav else 0)
    res = pcm_data / (2 ** 15)
    return res.astype(np.float32)

def if_contains_wav_header(byte_array):
    if len(byte_array) < 12: return False
    return byte_array[0:4] == b'RIFF' and byte_array[8:12] == b'WAVE'

def validate_header(header):
    return "requestId" in header

def remove_punctuation(text):
    cn_punct = "？！＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。〃〄々〆〇〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
    translator = str.maketrans('', '', string.punctuation + cn_punct)
    return text.translate(translator)

def split_text_to_tokens(text: str) -> list:
    text = remove_punctuation(text)
    tokens = []
    pattern = re.compile(r'[a-zA-Z]+|[^a-zA-Z]')
    for match in pattern.finditer(text):
        token = match.group()
        if token.strip():
            tokens.append(token)
    return tokens

def is_last_token_chinese(new_asr_tokens: list) -> bool:
    if not new_asr_tokens:
        return False
    last = new_asr_tokens[-1]
    for ch in last:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

active_connections = 0
secret_key_list = ["test0102"]

# Static Web UI routes
WEB_DIR = os.path.join(CURRENT_DIR, "web")

@app.route("/")
async def handle_index(request: Request):
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return await response.file(index_path)
    return response.text("Confucius4-R2T2 service running. Web UI not found.", status=404)

@app.route("/health")
async def handle_health(request: Request):
    return response.json({
        "status": "healthy",
        "service": "Confucius4-R2T2",
        "active_connections": active_connections,
        "gpu_mem_util": args.gpu_mem_util,
        "model_loaded": asr_model is not None
    })

# Streaming WebSocket v1
@app.websocket("/asr_stream_api_v1")
async def asr_stream_api_v1(request: Request, ws: Websocket):
    global active_connections
    if asr_model is None:
        await ws.close(code=1011, reason="Server model not ready")
        return

    audio_buf = np.ndarray(shape=(0,), dtype=np.float32)
    header_raw = await ws.recv()
    header = json.loads(header_raw)
    logger.info(f"header={redact_stream_request_header(header)}")

    is_first_seg = True
    tmp_audio_pointer = 0
    requestId = None
    total_secs = 0
    last_fixed_asr_text = ""
    no_new_text_chunks = 0
    counted = False

    _STOP = object()
    recv_queue: asyncio.Queue = asyncio.Queue()
    RECV_TIMEOUT = 120

    async def receiver():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    break
                await recv_queue.put(data)
                if isinstance(data, str) and data == YOUDAO_ONETIME_ASR_EOS_STRING:
                    break
        except Exception as e:
            logger.info(f"requestId={requestId} receiver exit: {e}")
        finally:
            await recv_queue.put(_STOP)

    async def processor():
        global active_connections
        nonlocal audio_buf, tmp_audio_pointer, is_first_seg, total_secs
        nonlocal last_fixed_asr_text, counted, requestId, no_new_text_chunks

        try:
            if not validate_header(header):
                await ws.send(json.dumps({"status": "error", "msg": ERROR_MSG_NO_HEADER}))
                await ws.close()
                return

            requestId = header.get("requestId", None)
            use_vad = header.get("use_vad", True)
            language = header.get("language", "zhen")
            if language == "zhen":
                language = None
            secret_key = header.get("secret_key", None)
            smooth = header.get("smooth", False)

            if secret_key is None or secret_key not in secret_key_list:
                await ws.close(code=4401, reason="Unauthorized")
                return

            context = resolve_qwen_context(header, smooth)
            active_connections += 1
            counted = True
            await ws.send(json.dumps({
                "status": "connected",
                "requestId": f"{requestId}",
                "msg": "",
                "active_connections": active_connections
            }))

            sr = SAMPLING_RATE
            step = int(round(STEP_MS / 1000.0 * sr))
            lookahead = int(round(LOOKAHEAD_MS / 1000.0 * sr))

            asr_state = asr_model.init_streaming_state(
                context=context,
                language=language,
                unfixed_chunk_num=0,
                unfixed_token_num=UNFIX_TOKEN_NUM,
                chunk_size_sec=CHUNK_ASR_SECONDS,
            )

            if use_vad and stream_vad is not None:
                stream_vad.reset()

            max_new_tokens = max(1, int((step + lookahead) / 1280))
            max_new_tokens_floor = min(32, max(4, 2 * int(step / 1280)))
            is_first = True
            total_secs = LOOKAHEAD_MS / 1000.0
            is_halluc = False
            halluc_reason = ""
            total_new_asr_tokens = []
            last_fixed_asr_text = ""
            ws_closed = False

            while True:
                data = await recv_queue.get()
                if data is _STOP:
                    break

                recv_time = time.time()

                if isinstance(data, str):
                    if data == YOUDAO_ONETIME_ASR_EOS_STRING:
                        first_max_new_tokens = max(1, int((step + lookahead) / 1280))
                        text = asr_model.finish_streaming_transcribe_no_reset(
                            asr_state, first_max_new_tokens
                        )
                        text = text.split("|")[0]
                        if len(text) > len(last_fixed_asr_text):
                            new_asr_text = text[len(last_fixed_asr_text):]
                        else:
                            new_asr_text = ""
                        out_msg = {"text": new_asr_text, "reset": True}
                        out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                        await ws.close()
                        break
                    else:
                        continue

                if is_first_seg:
                    is_wav = if_contains_wav_header(data)
                    audio_seg = read_pcm(data, is_wav)
                    is_first_seg = False
                else:
                    audio_seg = read_pcm(data)

                audio_buf = np.concatenate((audio_buf, audio_seg))
                audio_buf_len = len(audio_buf)
                if is_first and audio_buf_len < int(step + lookahead):
                    await ws.send(json.dumps({}))
                    continue

                #-------- Iterate over audio_buf --------#
                while audio_buf_len - tmp_audio_pointer >= CHUNK_ASR_SIZE:
                    if is_first:
                        asr_state.chunk_size_sec = (step + lookahead) / sr
                        chunk_size_samples = int(round(float(asr_state.chunk_size_sec) * sr))
                        asr_state.chunk_size_samples = max(1, chunk_size_samples)
                        audio_chunk = audio_buf[tmp_audio_pointer : tmp_audio_pointer + step + lookahead]
                        is_first = False
                        total_secs += (step + lookahead) / sr
                        tmp_audio_pointer = step + lookahead
                    else:
                        asr_state.chunk_size_sec = CHUNK_ASR_SECONDS
                        chunk_size_samples = int(round(float(asr_state.chunk_size_sec) * sr))
                        asr_state.chunk_size_samples = max(1, chunk_size_samples)
                        audio_chunk = audio_buf[tmp_audio_pointer : tmp_audio_pointer + CHUNK_ASR_SIZE]
                        tmp_audio_pointer += CHUNK_ASR_SIZE
                        total_secs += CHUNK_ASR_SECONDS

                    # Streaming VAD
                    speech_ended_in_this_chunk = False
                    vad_cost_ms = 0.0
                    if use_vad and stream_vad is not None:
                        audio_chunk_int16 = (audio_chunk * 32768.0).clip(-32768, 32767).astype(np.int16)
                        t0 = time.time()
                        chunk_results = stream_vad.detect_chunk(audio_chunk_int16)
                        for r in chunk_results:
                            if r.is_speech_end:
                                speech_ended_in_this_chunk = True
                        vad_cost_ms = round((time.time() - t0) * 1000, 1)

                    t0 = time.time()
                    text, fixed_asr_text = asr_model.streaming_transcribe_no_reset(
                        audio_chunk, asr_state, int(max_new_tokens), False
                    )
                    fixed_asr_text = fixed_asr_text.split("|")[0]
                    asr_cost_ms = round((time.time() - t0) * 1000, 1)

                    if len(fixed_asr_text) > len(last_fixed_asr_text):
                        new_asr_text = fixed_asr_text[len(last_fixed_asr_text):]
                        last_fixed_asr_text = fixed_asr_text
                        out_msg = {"text": new_asr_text, "asr_cost_ms": asr_cost_ms, "reset": False}
                        new_asr_tokens = split_text_to_tokens(new_asr_text)
                        for asr_word in new_asr_tokens:
                            total_new_asr_tokens.append(asr_word)
                            if len(total_new_asr_tokens) > MAX_TOKENS:
                                total_new_asr_tokens.pop(0)
                        max_new_tokens = max(1, int(step / 1280))
                    else:
                        out_msg = {"text": "", "asr_cost_ms": asr_cost_ms, "reset": False}
                        if not is_last_token_chinese(total_new_asr_tokens):
                            max_new_tokens = max_new_tokens + 0.5
                        else:
                            max_new_tokens = max(1, int(step / 1280))

                    if is_last_token_chinese(total_new_asr_tokens):
                        max_new_tokens = 2 * max_new_tokens
                    max_new_tokens = min(max_new_tokens_floor, max_new_tokens)

                    # Hallucination detection
                    cur_halluc, cur_reason = detect_hallucination(requestId, last_fixed_asr_text)
                    if cur_halluc:
                        is_halluc = True
                        halluc_reason = cur_reason

                    # Speech end / VAD pause
                    if speech_ended_in_this_chunk and use_vad:
                        seg_final_text = asr_state.text.split("|")[0]
                        asr_state = asr_model.init_streaming_state(
                            context=context,
                            language=language,
                            unfixed_chunk_num=0,
                            unfixed_token_num=UNFIX_TOKEN_NUM,
                            chunk_size_sec=CHUNK_ASR_SECONDS,
                        )
                        if len(seg_final_text) > len(last_fixed_asr_text):
                            seg_new_asr_text = seg_final_text[len(last_fixed_asr_text):]
                        else:
                            seg_new_asr_text = ""
                        out_msg = {"text": seg_new_asr_text, "reset": True, "asr_cost_ms": asr_cost_ms}
                        last_fixed_asr_text = ""

                    out_msg["total_cost_ms"] = round(vad_cost_ms + asr_cost_ms, 1)
                    out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                    try:
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                    except Exception as e:
                        ws_closed = True
                        break

                if ws_closed:
                    break
                if len(audio_buf) != 0:
                    audio_buf = audio_buf[tmp_audio_pointer:]
                tmp_audio_pointer = 0

                did_reset = False
                if is_halluc:
                    logger.info(f"requestId={requestId}: hallucination reset, reason={halluc_reason}")
                    text = asr_model.finish_streaming_transcribe_no_reset(
                        asr_state, max(1, int((step + lookahead) / 1280))
                    )
                    text = text.split("|")[0]
                    if len(text) > len(last_fixed_asr_text):
                        new_asr_text = text[len(last_fixed_asr_text):]
                    else:
                        new_asr_text = ""
                    out_msg = {"text": new_asr_text, "reset": True}
                    asr_state = asr_model.init_streaming_state(
                        context=context,
                        language=language,
                        unfixed_chunk_num=0,
                        unfixed_token_num=UNFIX_TOKEN_NUM,
                        chunk_size_sec=CHUNK_ASR_SECONDS,
                    )
                    did_reset = True

                if did_reset:
                    total_secs = 0.0
                    is_halluc = False
                    halluc_reason = ""
                    out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}
                    last_fixed_asr_text = ""
                    try:
                        await ws.send(json.dumps(out_str, ensure_ascii=False))
                    except Exception:
                        break

        except Exception as e:
            logger.exception(f"requestId={requestId} error: {e}")
        finally:
            if counted:
                active_connections -= 1
            try:
                await ws.close()
            except Exception:
                pass

    await asyncio.gather(receiver(), processor())

def args_parser():
    parser = argparse.ArgumentParser(description='Confucius4-R2T2 Server')
    parser.add_argument('-p', '--port', default="8040", dest="port", help="Server port")
    parser.add_argument('-ip', '--ip', default="0.0.0.0", dest="ip", help="Server bind IP")
    parser.add_argument(
        '-m', '--asr_model_path',
        default=os.environ.get("ASR_MODEL_PATH", os.path.join(CURRENT_DIR, "models", "Confucius4-R2T2")),
        dest="asr_model_path",
        help="Path or HF repo id of Confucius4-R2T2"
    )
    parser.add_argument(
        '--vad_model_path',
        default=os.path.join(CONFUCIUS_DIR, "checkpoints", "vad", "Stream-VAD"),
        dest="vad_model_path",
        help="Path to FireRedVAD Stream-VAD"
    )
    parser.add_argument('--gpu_mem_util', default=os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.40"), dest="gpu_mem_util", help="vLLM GPU memory utilization")
    parser.add_argument('--max_model_len', default=os.environ.get("VLLM_MAX_MODEL_LEN", "2048"), dest="max_model_len", help="vLLM max model len")
    return parser.parse_args()

args = args_parser()

if __name__ == '__main__':
    logger.info(f"Starting Confucius4-R2T2 Server on {args.ip}:{args.port}")
    app.run(host=args.ip, port=int(args.port), single_process=True)
