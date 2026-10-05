# Confucius4-R2T2 Multi-Architecture Streaming ASR & Real-time Subtitles Service

網易有道開源之實時真流式語音大模型 **Confucius4-R2T2**（基於 Qwen3-ASR-1.7B），具備最長穩定前綴（Longest Stable Prefix, LSP）只增不改、不回改、不幻覺補全之特性。

本專案封裝為跨架構 Docker 容器化服務與原生 Systemd 部署方案，相容於 **x86_64（RTX 4000 Ada / RTX A6000 等）** 與 **aarch64（ARM64, GB10 等）**，並提供開箱即用的現代化**網頁收音與即時字幕前端系統**。

---

## 🌐 線上運行實例與外部存取

#### 1. 外部存取網址
- **網頁字幕操作介面**：[https://asr.5gao.ai/](https://asr.5gao.ai/)（或相容舊域名 [https://147.5gao.ai/](https://147.5gao.ai/)）
- **內網直接存取**：`http://10.9.0.35:8040/`
- **健康檢查端點**：`https://asr.5gao.ai/health`（或 `https://147.5gao.ai/health`）
  ```json
  {
    "status": "healthy",
    "service": "Confucius4-R2T2",
    "active_connections": 0,
    "queue_size": 0,
    "gpu_mem_util": "0.40",
    "model_loaded": true
  }
  ```

#### 2. 即時字幕網頁介面特色
- **Web Audio 串流收音**：前端透過標準 `AudioContext` 自動降取樣至 16kHz 16-bit Mono PCM，以 160ms 緩衝區切片封裝發送。
- **雙向 WebSocket (WSS)**：自動偵測 https 協定升級為 `wss://147.5gao.ai/asr_stream_api_v1`，全鏈路加密且具備重連機制。
- **即時字幕卡片與波形**：整合即時麥克風音量波形動態視覺化、字元漸進式增量辨識、重複字過濾。
- **OBS 直播推流覆蓋模式**：支援一鍵切換純黑/去背高對比字卡模式，方便串接直播或螢幕擷取。
- **熱詞與語言設定**：支援中文/英文切換，可自訂業務熱詞（Hotwords）提升專有名詞準確率。
- **預設授權金鑰**：`test0102`（網頁已自動填入）。

#### 3. 系統服務常駐狀態
- **Systemd 服務名稱**：`confucius4-r2t2.service`（已設置為開機自啟）
- **服務目錄**：`/home/david/r2t2-service/`
- **GPU 資源隔離**：配置 `gpu_memory_utilization=0.40`（佔用約 6.8 GB VRAM），與主機上原本運行的 IndexTTS (5.7GB) 及 CosyVoice3 (5.3GB) 完美共存，無任何顯存衝突或 OOM。

---

## 🌟 核心特色

1. **整合式現代即時字幕 Web UI**（`web/index.html`）：
   - **Web Audio 串流收音**：瀏覽器原生麥克風自動重取樣為 16kHz 16-bit Mono PCM，自適應動態緩衝區。
   - **音訊動態波形視覺化**：HTML5 Canvas 即時繪製聲波振幅，清楚掌握收音狀態。
   - **OBS 直播推流覆蓋模式**：一鍵切換純黑高對比/去背字卡模式，無縫串接 OBS / vMix 等推流軟體。
   - **字幕歷史歸檔**：即時字幕卡片增量上屏，自動分句歸檔歷史記錄，支援一鍵複製與 TXT 下載。
   - **雙向 WebSocket (WSS)**：完整適配 SSL/TLS 加密反向代理，連線中斷自動提示。

2. **防靜音重複刷屏與幻覺機制**：
   - **純 Delta 字元增量推送**：服務端嚴格計算新增確認文字（`fixed_text[len(last_fixed):]`），靜音或未說話時零輸出，徹底解決網頁反覆刷屏重複累積文字之 Bug。
   - **整合 FireRedVAD 串流斷句**：精準追蹤語音活動，當說話人停頓（約 700ms）時自動觸發 `speech_end`，提交當前句子並徹底重設 ASR 解碼上下文（清空 KV-Cache），避免自迴歸 LLM 注意力陷入循環。
   - **重複模式防禦 (`detect_hallucination`)**：即時監測連續重複詞綴，偵測到死循環立即自動截斷並恢復乾淨狀態。

3. **啟動加速與虛擬化硬體避坑**：
   - **vLLM Eager Mode 快速初始化**：啟用 `enforce_eager=True` 規避圖編譯，啟動預熱時間自 120 秒縮減至 **4 秒**。
   - **QEMU CPU 無 AVX 崩潰相容性防護**：自動隔離非必要之 `dynet38`/`nagisa` C++ 擴展，解決虛擬機執行 Qwen-ASR 時遭遇 `SIGILL (status=4/ILL)` 的致命問題。
   - **記憶體動態分配**：配置 `max_model_len=2048` 與 `gpu_memory_utilization=0.40`，預留足夠 KV Cache 空間，可與其他大型模型（如 IndexTTS、CosyVoice）和平共存。

4. **多協議接口**：
   - HTTP 狀態檢查：`GET /healthz` 或 `GET /health`
   - 即時字幕網頁介面：`GET /`
   - HTTP 一次性檔案轉寫：`POST /transcribe`
   - HTTP SSE 逐段檔案轉寫：`POST /transcribe/stream`，每段完成即回傳字幕
   - 停止指定音檔辨識：`POST /transcribe/cancel`，傳入 SSE `start` 事件的 `job_id`
   - WebSocket 串流介面：`WS /asr_stream_api_v1`（VAD 斷句、熱詞）與 `WS /ws/stream`（精簡版）

5. **兩種後端都能串流、可同時多路**：
   - 串流不再限定 vLLM：`app/main.py` 在 Transformers 後端（例如 aarch64 的 NVIDIA GB10）同樣提供 `/asr_stream_api_v1`。
   - 所有模型呼叫（批次與串流）共用同一個排隊鎖並在執行緒中執行，多路連線互相輪流、不會卡死事件迴圈；推論落後時自動併塊追進度。
   - 每條連線有獨立的 VAD 狀態；偵測到語音才開始解碼，靜音不佔 GPU。
   - 每句結束時附上整句重新辨識的 `final_text`，修正串流邊聽邊出字造成的句首雜字與同音誤判。
   - 不必先指定語言：串流逐句自動判斷語言（中、英、日、韓、西），句末回報判出的語言，下一句起串流跟著切換。

---

## 📁 專案目錄結構

```text
confucius4-r2t2-asr/
├── app/
│   ├── main.py              # FastAPI 容器服務入口
│   ├── stream_session.py    # 單一串流的狀態機（VAD 斷句、增量解碼、句末重辨識）
│   ├── infer_gate.py        # 模型呼叫排隊鎖與推論狀態
│   ├── vad.py               # 每條連線獨立的串流 VAD
│   ├── textproc.py          # 語言代碼對應與文字清理
│   └── r2t2/                # R2T2 串流推論核心引擎模組
├── tests/                   # pytest
├── web/
│   ├── index.html           # 語音辨識工作台
│   └── assets/              # 前端樣式與互動程式
├── deploy/
│   ├── systemd/
│   │   └── confucius4-r2t2.service  # Linux 原生 Systemd 常駐守護進程配置
│   └── nginx/
│       └── 147.5gao.ai.conf # Nginx SSL + WSS 反向代理範例配置
├── compose.yaml             # Docker Compose 雙容器編排 (API + Nginx)
├── Dockerfile               # 跨架構 CUDA 容器構建檔
├── requirements.txt         # 容器映像的相依套件（不含 vLLM）
├── nginx.template           # 容器版 Nginx 模板
├── server.py                # Sanic + vLLM 專用高效能串流服務端
└── README.md
```

---

## 🚀 部署方式

### 方法 A：Docker Compose 容器化部署（推薦）

映像跑的是 `app/main.py`，內含 llama.cpp 與 Transformers 兩種後端，x86_64 與 aarch64（NVIDIA GB10）用同一份 `Dockerfile` build。啟動時自動選用（`ASR_BACKEND=auto`）：有編好的 llama.cpp 與 GGUF 模型就用 llama.cpp，否則退回 Transformers。llama.cpp 是在 build 時從原始碼編的，不依賴任何平台專屬的預編譯套件；映像不含 vLLM。

```bash
# 1. 相依套件約 3 GB，先下到 wheels/；Dockerfile 只讀這個目錄，不連網
#    （Python 版本要與映像相同，所以在 python:3.10 容器裡下載）
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/p python:3.10-slim \
  pip download -r /p/requirements.txt setuptools wheel -d /p/wheels \
    --extra-index-url https://download.pytorch.org/whl/cu130

# 2. 建立這台機器的設定（對外埠、模型目錄等；.env 不進版控）
cp .env.example .env

# 3. build 與啟動分開做；build 很吃磁碟 I/O，一次只跑一個
docker compose build r2t2-api
docker compose up -d

# 觀察啟動日誌與自動預熱
docker compose logs -f r2t2-api
```

- 模型放在宿主目錄，由 `.env` 的 `MODEL_HOST_DIR`、`VAD_HOST_DIR`、`GGUF_HOST_DIR` 指定（預設 `./models/...`）；目錄是空的且 `AUTO_DOWNLOAD=1` 時會自動下載（ASR 約 4 GB、GGUF 約 2.4 GB、VAD 約 2 MB）。
- 第一次 build 會拉 CUDA devel 映像（約 7 GB）並編譯 llama.cpp（GB10 上約 10 分鐘）。不需要 llama.cpp 時加 `WITH_LLAMA=0`（`.env` 或 `--build-arg`）跳過，服務會改用 Transformers。
- `app/` 與 `web/` 以唯讀方式掛進容器，改了程式只要 `docker compose restart r2t2-api`，不必重 build；改 `requirements.txt` 才需要。
- GPU 以 CDI 模式掛載（`/var/run/cdi/nvidia.yaml`）。主機若只有 legacy nvidia runtime，把 `compose.yaml` 的 `driver: cdi` 與 `device_ids` 改成 `driver: nvidia`、`count: all`。
- 容器以宿主帳號執行（`HOST_UID`／`HOST_GID`，預設 1000）。

- 容器將對外服務於 `:8803`（可於 `.env` 中調整 `PUBLIC_PORT`）。`.env` 的所有可用變數與預設值見 `.env.example`。
- 打開瀏覽器訪問 `http://<主機IP>:8803/` 即可進入**即時字幕操作介面**。

---

### 方法 B：Linux 原生 Systemd 服務部署（高效能生產主機）

1. **建立 Python 虛擬環境並安裝依賴**：
   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate
   uv pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
   uv pip install "vllm>=0.14.0" fireredvad sanic soundfile librosa "opencc-python-reimplemented>=0.1.7"
   # 不用 vLLM（例如 aarch64 / 顯存吃緊）改跑 app/main.py 時：
   # uv pip install "transformers>=4.51.0" fireredvad "fastapi>=0.115" "uvicorn[standard]" python-multipart "opencc-python-reimplemented>=0.1.7"
   uv pip uninstall nagisa dynet38 # 排除 QEMU CPU 缺少 AVX 時的 SIGILL 崩潰
   ```

2. **配置 Systemd 服務**：
   ```bash
   sudo cp deploy/systemd/confucius4-r2t2.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now confucius4-r2t2
   ```

3. **配置 Nginx 反向代理（支援 HTTPS / WSS）**：
   參考 `deploy/nginx/147.5gao.ai.conf`，確保加入 WebSocket 升級標頭：
   ```nginx
   proxy_http_version 1.1;
   proxy_set_header Upgrade $http_upgrade;
   proxy_set_header Connection "upgrade";
   proxy_read_timeout 3600s;
   ```

### 方法 C：GitHub Actions + GHCR 自動部署

推送至 `main` 後，`.github/workflows/deploy-ghcr.yml` 會建立版本映像並發布至 GHCR，再由正式主機上的 `r2t2-deploy` self-hosted runner 拉取程式檔、更新 `/home/david/r2t2-service`，並重啟 `confucius4-r2t2.service`。部署完成條件為 `http://127.0.0.1:8040/health` 回傳成功；健康檢查逾時會還原上一份 `server.py` 與 `web/`。

部署 runner 僅處理 `main` push 與手動觸發工作，不接收 Pull Request 工作。新主機設定 runner 時，需在該 repo 的 **Settings → Actions → Runners → New self-hosted runner** 註冊 Linux x64 runner，並加上 `r2t2-deploy` label；映像標籤包含 `latest` 與 `sha-<commit>`。

---

## 📡 外部 App 調用 API 指南 (External Application Integration)

外部應用程式（如 Python 腳本、Node.js 服務、iOS/Android App、桌面軟體、自製網頁）可依據需求選擇以下兩種方式調用本 ASR 服務：

---

### 模式一：HTTP REST 音訊檔案轉寫 API（適合錄音檔/一次性辨識）

上傳已錄製之音訊檔案（支援 WAV、MP3、FLAC、M4A、OGG 等格式，單檔上限 200 MB）。長音檔每 30 秒重新建立辨識狀態；`/transcribe/stream` 會逐段傳回字幕與音訊時間，前端可取消單一辨識工作。

- **端點 URL**：`https://147.5gao.ai/transcribe`（或內網 `http://10.9.0.35:8040/transcribe`）
- **HTTP 方法**：`POST`
- **Content-Type**：`multipart/form-data`
- **參數**：
  - `file` (必填)：音訊二進制檔案，最大 200 MB
  - `language` (選填)：`zhen`（中英混合）、`Chinese`（中文）、`English`（英文）
  - `output_script` (選填)：`traditional`（繁體中文，前端預設）或 `simplified`（簡體中文；REST 呼叫預設）
  - `context` (選填)：自訂提示詞或上下文熱詞

長音檔使用 `POST /transcribe/stream`。服務每 30 秒回傳一個 `segment` 事件；`start` 事件包含 `job_id`。取消時以 JSON 呼叫 `POST /transcribe/cancel`：`{"job_id":"<job_id>"}`。已完成的字幕片段會保留。

#### 1. cURL 呼叫範例
```bash
curl -X POST https://147.5gao.ai/transcribe \
  -F "file=@your_audio.wav" \
  -F "language=zhen" \
  -F "output_script=traditional"
```

響應範例 (JSON)：
```json
{
  "status": "success",
  "text": "之前有顧客自己帶酒水也沒加收錢或者不讓喝",
  "duration_sec": 6.74,
  "cost_ms": 3773.3
}
```

#### 2. Python (requests) 呼叫範例
```python
import requests

url = "https://147.5gao.ai/transcribe"
files = {"file": open("sample.wav", "rb")}
data = {"language": "zhen"}

response = requests.post(url, files=files, data=data)
print(response.json())
# {"status": "success", "text": "...", "duration_sec": 6.74, "cost_ms": 3773.3}
```

---

### 模式二：WebSocket 即時真流式語音辨識 API（推薦！極致低延遲）

外部 App 透過 WebSocket 建立長連線，持續送入麥克風或音訊切片（16kHz 16-bit Mono PCM，每 160ms 一個 Chunk），服務端即時（~30ms）回傳確認的文字增量（Delta），並透過 VAD 自動換句。

- **WebSocket URL**：`wss://147.5gao.ai/asr_stream_api_v1`（內網：`ws://10.9.0.35:8040/asr_stream_api_v1`）
- **音訊格式**：16kHz, 16-bit, Mono PCM raw binary（每切片 160ms = 2,560 samples = 5,120 bytes）
- **授權金鑰**：`test0102`

#### 通訊流程：
1. **建立 WebSocket 連線**。
2. **傳送 Handshake JSON**：
   ```json
   {
     "requestId": "488fbe6c-8fe8-442a-a925-fb355a153406",
     "language": "zhen",
     "output_script": "traditional",
     "use_vad": true,
     "secret_key": "test0102"
   }
   ```
3. **持續發送 PCM 二進制 Chunk**：每 160ms 傳送 5,120 bytes 的二進制數據。
4. **即時接收服務端 JSON 響應**：
   ```json
   {
     "status": "success",
     "requestId": "488fbe6c-8fe8-442a-a925-fb355a153406",
     "msg": {
       "text": "即時辨識新增字詞",
       "reset": false,
       "asr_cost_ms": 32.5,
       "total_cost_ms": 35.1
     }
   }
   ```
   > 說明：
   > - `msg.text`：本次 Chunk **新增之確認文字（增量 Delta）**，靜音時為空字串，前端直接累加即可。
   > - `msg.reset`：當說話人停頓（約 700ms）時觸發 VAD 斷句，回傳 `true`，代表當前句子結束，客戶端可進行換行存檔並重置當前句暫存。
5. **結束傳輸**：發送結束字串 `"YOUDAO_ONETIME_ASR_STREAM_EOS"` 或主動關閉連線。

#### 1. Python 完整即時串流呼叫腳本
專案已提供可直接執行的示範腳本：[`examples/client_stream_demo.py`](./examples/client_stream_demo.py)
```bash
# 安裝依賴
pip install websockets

# 串流傳送音訊檔並即時印出逐字字幕
python examples/client_stream_demo.py --audio test.wav --url wss://147.5gao.ai/asr_stream_api_v1
```

核心實作代碼片段：
```python
import asyncio
import json
import uuid
import websockets

async def run_asr_stream():
    url = "wss://147.5gao.ai/asr_stream_api_v1"
    async with websockets.connect(url) as ws:
        # 1. 握手
        await ws.send(json.dumps({
            "requestId": str(uuid.uuid4()),
            "language": "zhen",
            "use_vad": True,
            "secret_key": "test0102"
        }))

        # 2. 接收即時文字協程
        async def on_receive():
            full_sentence = ""
            async for msg in ws:
                data = json.loads(msg)
                msg_body = data.get("msg", {})
                delta = msg_body.get("text", "")
                if delta:
                    full_sentence += delta
                    print(f"\r[即時字幕]: {full_sentence}", end="", flush=True)
                if msg_body.get("reset"):
                    print(f"\n[句結歸檔]: {full_sentence}")
                    full_sentence = ""

        recv_task = asyncio.create_task(on_receive())

        # 3. 串流送入 PCM 音訊 (每 160ms 送 5120 bytes)
        with open("sample_16k.pcm", "rb") as f:
            while chunk := f.read(5120):
                await ws.send(chunk)
                await asyncio.sleep(0.16)

        await ws.send("YOUDAO_ONETIME_ASR_STREAM_EOS")
        await asyncio.sleep(1.0)
        recv_task.cancel()

asyncio.run(run_asr_stream())
```

#### 2. Node.js (ws) 呼叫範例
```javascript
import WebSocket from 'ws';
import fs from 'fs';

const ws = new WebSocket('wss://147.5gao.ai/asr_stream_api_v1');

ws.on('open', () => {
  // 1. 握手
  ws.send(JSON.stringify({
    requestId: 'app-client-' + Date.now(),
    language: 'zhen',
    use_vad: true,
    secret_key: 'test0102'
  }));

  // 2. 串流發送 PCM 音訊
  const stream = fs.createReadStream('audio.pcm', { highWaterMark: 5120 });
  stream.on('data', (chunk) => {
    ws.send(chunk);
  });
  stream.on('end', () => {
    ws.send('YOUDAO_ONETIME_ASR_STREAM_EOS');
  });
});

let currentSentence = '';
ws.on('message', (data) => {
  const res = JSON.parse(data.toString());
  if (res.msg) {
    if (res.msg.text) {
      currentSentence += res.msg.text;
      process.stdout.write(`\r[字幕] ${currentSentence}`);
    }
    if (res.msg.reset) {
      console.log(`\n[完成] ${currentSentence}`);
      currentSentence = '';
    }
  }
});
```

---

### 模式三：服務健康檢查 API (`GET /health`)

```bash
curl https://asr.5gao.ai/health
```
響應：
```json
{
  "status": "healthy",
  "service": "Confucius4-R2T2",
  "active_connections": 0,
  "queue_size": 0,
  "gpu_mem_util": "0.40",
  "model_loaded": true,
  "inference": {
    "waiting": 0,
    "busy_seconds": 0.0,
    "last_success_seconds_ago": 0.18,
    "max_wait_seconds": 1.834
  }
}
```
`inference` 反映模型是否真的在動：`busy_seconds` 是目前這次推論已經跑了多久，超過 60 秒時 `status` 會變成 `stalled`（`server.py` 同時回 HTTP 503）；`waiting` 是正在排隊的呼叫數。

`app/main.py` 的 `GET /healthz` 另外回報 `backend`、`arch`、`gpu`、`vram_used_gb`、`streaming`、`vad_loaded`、`active_streams`，正常時 `status` 為 `ok`、載入中為 `loading`。

### `app/main.py` 的串流協議細節 (`WS /asr_stream_api_v1`)
以下是 `app/main.py`（容器／GB10 部署，預設 `:8803`）的行為；`server.py` 用同一協議，差異列在最後。
- **握手訊息 (JSON Header)**：
  ```json
  {
    "requestId": "uuid-here",
    "language": "zhen",
    "use_vad": true,
    "secret_key": "test0102",
    "system_prompt": "鶴記企業、沉水泵、EUBL",
    "final_pass": true
  }
  ```
  | 欄位 | 必填 | 說明 |
  | :--- | :--- | :--- |
  | `requestId` | 是 | 任意識別字串，回應會原樣帶回 |
  | `secret_key` | 是 | 需在 `STREAM_SECRET_KEYS` 內，否則以 4401 關閉連線 |
  | `language` | 否 | 不填、`zhen` 或 `auto`：逐句自動判斷語言（見下方說明）。填 `zh`、`en`、`ja`、`Chinese`、`English`、`Cantonese`… 則固定用該語言解碼，不做判斷 |
  | `languages` | 否 | 自動判斷時信任的語言清單，例如 `["zh", "en", "es"]`；預設為 `STREAM_AUTO_LANGUAGES`（Chinese、English、Japanese、Korean、Spanish）。指定了具體 `language` 時忽略 |
  | `use_vad` | 否 | 預設 `true`，以 VAD 自動斷句 |
  | `system_prompt` | 否 | 熱詞或上下文提示，最多 4000 字 |
  | `smooth` | 否 | `true` 時要求模型輸出較通順的文字 |
  | `final_pass` | 否 | 預設 `true`，句末整句重新辨識並放在 `final_text` |
  | `output_script` | 否 | `simplified`（預設）或 `traditional`（台灣繁體，OpenCC `s2twp`）。繁體時每一步都以整句累積的文字轉換再送出新增部分，被拆在兩則訊息裡的詞（软、件）仍會轉成慣用語（軟體）；`final_text` 以整句轉換，為定稿 |

  握手成功後伺服器回 `{"status": "connected", "language": "auto", "languages": ["Chinese", "English", ...], "vad": true, ...}`（指定語言時 `language` 為該語言、`languages` 為空）；header 有誤則回 `{"status": "error", "msg": "..."}` 並關閉連線。
- **自動判斷語言**：
  模型串流時的語言判斷只看開頭不到一秒的音訊，定下來就不再改，常判錯（帶口音的華語、台語曾被當成葡萄牙文），所以不直接用。實際做法：
  1. 每一句的串流字幕用「目前語言」解碼，連線開始時是清單的第一個（預設 Chinese）。
  2. 句末整句重新辨識時讓模型自行判斷，`final_text` 就是這次的結果，判出的語言放在句末訊息的 `language`。模型常把華語標成 English，所以語言以實際寫出的文字為準：有韓文字母為 Korean、有假名為 Japanese、有漢字為 Chinese。
  3. 判出清單外的語言時不採信，改用清單第一個語言重新辨識這一句。
  4. 判出的語言從下一句起成為串流語言。判出 English 時仍以 Chinese 串流（Chinese 解碼會照實輸出英文，中英混講才不會來回切換）；短於 1.5 秒的句子不拿來切換。

  所以講者換語言時，換過去的第一句串流字幕會是亂的，那句的 `final_text` 仍正確，下一句起串流就跟上。2026-10-05 在 .37 以 edge-tts 合成的中、英、西、日、韓、中英混講各一句實測，六句的 `final_text` 與 `language` 全部正確；固定 Chinese 時日文被翻成中文、韓文串流時被翻成中文。
- **音訊傳輸 (Binary Chunks)**：
  客戶端每 160ms 發送 16kHz 16-bit Mono PCM raw binary（5,120 bytes / 2,560 samples）。
- **服務端即時響應 (JSON)**：
  ```json
  {
    "status": "success",
    "requestId": "uuid-here",
    "msg": {
      "text": "即時辨識新增字詞",
      "reset": false,
      "asr_cost_ms": 32.5,
      "total_cost_ms": 35.1
    }
  }
  ```
  `text` 是只增不改的新增文字，同一句內依序串接即可。當說話者停頓約 0.6 秒時，VAD 判定語句結束，回傳句末訊息：
  ```json
  {
    "status": "success",
    "requestId": "uuid-here",
    "msg": {
      "text": "",
      "reset": true,
      "final_text": "整句重新辨識後的文字",
      "final_pass": true,
      "language": "Chinese",
      "asr_cost_ms": 310.2,
      "total_cost_ms": 314.9
    }
  }
  ```
  **請以 `final_text` 作為這一句的定稿。** 串流是邊聽邊出字，已送出的字無法收回，句首偶爾會多出雜字或聽錯同音字；`final_text` 是整句聽完後一次辨識的結果；若一次辨識漏掉串流已經送出的句尾，會自動接回。`final_pass` 為 `false` 表示這句沒有重新辨識（單句超過 30 秒、或因重複幻覺被強制斷句），此時 `final_text` 就是串流累積的文字。`language` 是這句的語言：自動判斷時為判出的語言，指定語言時就是該語言，沒有重新辨識時為當時的串流語言。
- **結束串流**：
  送出文字訊息 `YOUDAO_ONETIME_ASR_STREAM_EOS`（全大寫，原樣送出，不是 JSON）。伺服器會回最後一則 `reset: true` 訊息後關閉連線。連續 120 秒沒有收到資料也會關閉。
- **連線數上限**：
  同時進行的串流達到 `MAX_CONCURRENT_STREAMS`（預設 12）時，新連線會收到 `{"status": "error", "msg": "server busy: max concurrent streams reached"}`，並以關閉碼 `1013` 結束。
- **串流優先於上傳**：
  上傳音檔的每一段（30 秒音訊）要佔用推論一兩秒，期間所有串流的字幕都會頓住。所以有人在講話時，上傳最多每 `UPLOAD_MIN_INTERVAL_SECONDS`（預設 5 秒）跑一段；等待時間依上一段花的時間計算（4 倍，上限即此值），沒人講話時上傳一段接一段，不受限制。30 秒以內的單段音檔視為互動式辨識，和串流同等優先，不受這個限制。
- **推論出錯**：
  伺服器先送 `{"status": "error", "requestId": "...", "msg": "inference failed: ..."}`，再以關閉碼 `1011` 結束連線；正常結束的關閉碼是 `1000`。
- **同時多路**：
  多條連線共用一個模型、輪流推論。Transformers 後端在 GB10 上單路每步約 0.1 秒；三路同時講話時斷句約晚 0.3–1.7 秒，五路約晚 1.4–3.3 秒（2026-10-01 實測，每路都在同一時間講話的最壞情況）。
- `server.py`（Sanic + vLLM 專用）提供同一協議，但沒有 `final_text`，也沒有逐句自動判斷：`zhen` 以 Chinese 解碼。

### `app/main.py` 的一次性轉寫 (`POST /transcribe`)
```bash
curl -F file=@audio.wav -F language=zhen -F context="鶴記企業、沉水泵" http://localhost:8803/transcribe
```
`language` 可省略（自動判斷）或使用與串流相同的代碼；不支援的語言回 400。回應的 `text` 已去除模型的斷句標記 `|` 與重複幻覺，若與原始輸出不同，原文放在 `raw_text`。超過 30 秒的音檔每 30 秒切一段各自辨識，各段以換行串接。回應同時帶有 `server.py` 的欄位名稱（`duration_sec`、`cost_ms`），兩套服務的用戶端可以共用。

### 長音檔逐段辨識 (`POST /transcribe/stream`)
`app/main.py` 與 `server.py` 回傳相同的 Server-Sent Events：先一則 `start`（`job_id`、`duration_sec`、`total_segments`、`segment_seconds`），每辨識完 30 秒回一則 `segment`（`index`、`total`、`start_sec`、`end_sec`、`text`），最後一則 `done`（全文 `text`）；失敗時回 `error`。網頁的上傳進度就是讀這些事件。兩個上傳端點都接受 `output_script`。

要中途停止，把 `start` 事件的 `job_id` 送到 `POST /transcribe/cancel`（JSON：`{"job_id": "..."}`）。`app/main.py` 會在目前這一段辨識完後停下並回 `cancelled`（`completed_segments`），已完成的段落保留；`server.py` 則在目前的解碼步驟後停下。

---

## ⚙️ 環境變數設定 (`compose.yaml` / `.env`)

| 變數名稱 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `PUBLIC_PORT` | `8803` | 對外公開監聽端口（Nginx Reverse Proxy） |
| `ASR_BACKEND` | `auto` | 後端推論引擎：`auto`（依序嘗試 `vllm` → `llama` → `transformers`，哪個在這台裝置起得來就用哪個）或指定其中之一；指定的後端起不來時服務回報錯誤，不會偷偷換別的 |
| `AUTO_DOWNLOAD` | `1` | 若 `/model` 權重不存在，自動自 Hugging Face 下載 |
| `AUTO_WARMUP` | `1` | 服務啟動後自動以 1 秒靜音推論預熱 CUDA kernel 與注意力快取 |
| `GPU_MEMORY_UTILIZATION` | `0.40` | vLLM 顯存分配佔比（留有充裕 KV cache，避免 OOM） |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | 設為 `0` 避開缺少 nvcc 時 FlashInfer sampling 之 JIT 編譯需求 |
| `VAD_DIR` | `$MODEL_DIR/FireRedVAD` | 串流 VAD 模型目錄（容器內固定為 `/vad`）；不存在且 `AUTO_DOWNLOAD=1` 時自動下載。載入失敗時串流照常運作但不自動斷句 |
| `MODEL_HOST_DIR` | `./models/Confucius4-R2T2` | 掛進容器 `/model` 的宿主目錄（僅 compose） |
| `GGUF_DIR` | `$MODEL_DIR/gguf` | llama.cpp 後端的 GGUF 目錄（容器內固定為 `/gguf`），需剛好一個解碼器 `.gguf` 與一個 `mmproj` `.gguf` |
| `GGUF_HOST_DIR` | `./models/Confucius4-R2T2-GGUF` | 掛進容器 `/gguf` 的宿主目錄（僅 compose） |
| `WITH_LLAMA` | `1` | build 時是否編譯 llama.cpp 後端（僅 compose build） |
| `VLLM_MAX_MODEL_LEN` | `2048` | vLLM 後端的最大序列長度 |
| `VAD_HOST_DIR` | `./models/FireRedVAD` | 掛進容器 `/vad` 的宿主目錄（僅 compose） |
| `STREAM_DEFAULT_LANGUAGE` | `zhen` | 串流 header 沒帶 `language` 時的預設值；`zhen`／`auto` 為逐句自動判斷（`/ws/stream` 沒有斷句，`zhen` 仍以 Chinese 解碼） |
| `STREAM_AUTO_LANGUAGES` | `Chinese,English,Japanese,Korean,Spanish` | 自動判斷時信任的語言，第一個是判出清單外語言時的退路 |
| `STREAM_SECRET_KEYS` | `test0102` | 串流允許的 `secret_key`，逗號分隔 |
| `STALL_SECONDS` | `60` | 單次推論超過此秒數，`/healthz` 回報 `stalled` |
| `MAX_CONCURRENT_STREAMS` | `12` | 同時進行的即時串流上限，超過時以關閉碼 1013 拒絕新連線 |
| `UPLOAD_MIN_INTERVAL_SECONDS` | `5` | 有人在講話時，上傳音檔最多每隔幾秒佔用一次推論 |

---

## 🏗️ 高並發架構演進與微批優先級排程 (Concurrency Architecture & Evolution Roadmap)

在生產環境的多用戶並發情境中（例如多位使用者同時開啟即時字幕網頁、同時有外部系統透過 HTTP `/transcribe` 上傳長音檔），大模型 ASR 服務常面臨執行緒安全與延遲競爭的嚴峻考驗。本節詳解系統的並發架構、死鎖排查成果、當前採用的微批調度機制，以及未來的架構演進評估。

> **實測（2026-10-02，內網直連 `10.9.0.35:8040`，RTX 4000 Ada，vLLM）**：下文的「每步約 15ms」是設計時的估計，不是量測值。實際單路串流每步（160ms 音訊）約 137–156 ms，也就是一個人講話就接近用滿；三路同時講話時每步約 390 ms、五路約 665 ms、十路約 1.4 秒，進度會持續落後，音訊送完 15 秒後連線被關閉，尚未算完的句子不會輸出。排程器解決的是「不再死鎖」，不是「多路都能即時」。`app/main.py` 另有落後時併塊與靜音不解碼，數字見下方「`app/main.py` 的串流協議細節」。

### 1. vLLM V1 底層 IPC 死鎖根因剖析 (Root-Cause Diagnosis)

在原生架構中，若採用多執行緒（如 Python `asyncio.to_thread`）直接並發調用 `asr_model.streaming_transcribe`，服務在高負載或多請求交疊時會瞬間凍結（死鎖無回應）。經 GDB 堆疊回溯排查，根因如下：
- **vLLM V1 EngineCore 跨進程架構**：`vllm.entrypoints.llm.LLM` 為離線批處理設計，主行程透過單一 ZeroMQ IPC Pipe 與底層 `EngineCore` 子進程進行通訊與輸出拉取（`outputs_queue.get()`）。
- **非執行緒安全 (Thread-Unsafe)**：多個 Python 執行緒並發進入 `LLM.generate()` 時，會同時競爭並拉取同一條 IPC Pipe 的回傳資料，引發資料錯位與 Linux 底層 `pthread_mutex / futex wait` 互鎖死鎖，導致 Sanic 事件迴圈與推論核心徹底失去回應。

```
[多執行緒並發直接調用 (舊架構 - 致命缺陷)]
HTTP Request A  ──> thread-1 ─┐
HTTP Request B  ──> thread-2 ─┼─> LLM.generate() ──> [單一 ZeroMQ Pipe 競爭衝突] ──> ❌ 致命死鎖 (Futex Wait)
WebSocket Chunk ──> thread-3 ─┘
```

---

### 2. 方案 A：微批優先級排程隊列 (Micro-Batch Priority Pipeline Scheduler - 當前實作)

為徹底根除死鎖並保障即時語音辨識的毫秒級低延遲，本專案重構了推論調度核心，採用**「微批優先級排程隊列 (InferenceScheduler)」**：

```
[InferenceScheduler 微批雙軌排程 (方案 A - 當前架構)]

WebSocket 實時語音 ──────> priority=0 (高優先級) ──┐
                                                    ├─> [ PriorityQueue ] ──> [ 單一專屬 GPU Worker 線程 ] ──> vLLM Engine
HTTP 檔案轉寫 (320ms切片) ─> priority=1 (普通優先級) ─┘                               (100% 執行緒安全，零 IPC 競爭)
```

#### 核心設計重點：
1. **單一專屬 GPU 推論工作線程 (`ASRInferenceWorker`)**：
   - 所有 GPU 推論任務一律進入排程隊列，由單一背景工作線程依序執行。徹底消滅多線程直接搶佔 vLLM IPC 的死鎖條件，確保 100% 穩定。
2. **優先級雙軌排程 (`queue.PriorityQueue`)**：
   - **優先級 0 (高優先級 - 實時串流)**：所有 WebSocket 即時語音切片（`priority=0`）享有無條件優先插隊權。
   - **優先級 1 (普通優先級 - 批次檔案)**：HTTP REST `/transcribe` 與 `/transcribe/stream` 的音訊切片使用 `priority=1`。
3. **時間片微切片交錯推論 (Time-sliced Interleaving)**：
   - 上傳 200MB 的長音檔時，系統並不會一次性霸佔 GPU 數十秒；而是切分為 **320ms 小音訊切片**。
   - 每個 320ms 切片推論僅需約 15ms。在此期間，若 WebSocket 實時說話音訊抵達，將以 `priority=0` 在**下一個 15ms 間隙立即插隊**。
   - **成效**：即使後台正在處理 1 小時的錄音檔轉寫，前端即時說話字幕依然維持 **15~30ms 極致即時響應**，兩者互不干擾、永不卡死。
4. **非同步橋接與安全回調**：
   - 使用 `task.loop.call_soon_threadsafe(_safe_resolve / _safe_reject)` 將 GPU 推論結果回傳給 Sanic 協程，完全不阻塞主事件迴圈。
5. **即時監控可觀測性**：
   - `GET /health` 端點新增 `"queue_size"` 欄位，實時反映當前等待 GPU 推論的任務深度。

---

### 3. 未來進階演化路徑評估 (Future Evolution Roadmap)

隨著後續業務場景擴展，可進一步按以下路徑無縫升級：

| 方案 | 架構類型 | 適用場景 | 優勢 | 資源消耗與複雜度 |
| :--- | :--- | :--- | :--- | :--- |
| **方案 A (當前實作)** | **微批優先級排程隊列** | 中小型並發、直播字幕 + 偶發檔案轉寫 | 單一實例顯存佔用低 (6.8GB)，即時插隊保證低延遲，零死鎖風險 | 顯存極省 (40% GPU)，架構精簡 |
| **方案 B** | **進程級雙實例物理隔離** | 高頻率、大體量長音檔轉寫 + 7x24 直播 | 物理隔離：檔案轉寫即便負載 100% 也不消耗直播實例算力 | 需啟動兩個服務實例 (8040 + 8041)，顯存需劃分 (如 25% + 15%)，Nginx 需配置路徑分流 |
| **方案 C** | **原生真非同步合批引擎 (AsyncLLMEngine)** | 百人以上大型會議同傳、SaaS 多租戶高並發 | 支援 Continuous Dynamic Batching，多路音訊在同一個 Forward Pass 合批推論，GPU 核心利用率最大化 | 需將 R2T2 聲學編碼器與底層 vLLM 的解碼層深度重構為原生非同步驅動 |

#### 方案 B 實作藍圖（雙實例物理隔離）：
- **Instance 1 (`:8040`)**：`--gpu_mem_util 0.25`，專門承載 `wss://` 即時串流字幕。
- **Instance 2 (`:8041`)**：`--gpu_mem_util 0.15`，專門承載 `/transcribe` 批次檔案上傳。
- **Nginx 反向代理動態分流**：
  ```nginx
  location /asr_stream_api_v1 {
      proxy_pass http://127.0.0.1:8040; # 專屬實時串流
  }
  location /transcribe {
      proxy_pass http://127.0.0.1:8041; # 專屬檔案轉寫
  }
  ```

#### 方案 C 實作藍圖（vLLM AsyncLLMEngine）：
- 將 `app/r2t2/` 串流狀態機適配為非同步生成器（`async for output in engine.generate(...)`）。
- 讓多個並發連線的 token 生成自然合批（Continuous Batching），適合高密度企業級部署。

---

## 🔬 端側推論評估（瀏覽器完全本機運算可行性分析）

針對**「是否能完全由使用者瀏覽器本機端運算（WebGPU / ONNX Web / Wasm）？」**之技術評估：

1. **模型量級與特徵抽取瓶頸**：
   - Confucius4-R2T2 採用 **Qwen3-ASR-1.7B** 語言骨幹 + 聲學特徵編碼器（Audio Encoder）+ 具備時序因果快取的真流式注意力架構。
   - 即使以 INT4 量化，模型權重體積仍達 **~1.1 GB - 1.4 GB**，使用者首次開啟網頁需承受龐大的模型下載頻寬與等待時間。
2. **WebGPU 顯存與運算效能限制**：
   - 1.7B 參數量在瀏覽器 WebGPU 環境下推論，需佔用使用者端至少 2.5 GB - 3.5 GB 的專屬 GPU 記憶體。在普通辦公電腦、行動裝置（手機/平板）或整合式內顯上，極易發生 WebGL/WebGPU context lost 或記憶體耗盡（OOM）崩潰。
   - R2T2 要求每 **160ms chunk** 必須在 **40ms 內** 完成自迴歸解碼，才不致於累積音訊延遲。目前瀏覽器 WebGPU 僅在高階獨顯（如 RTX 3060 以上）或 Apple Silicon (M1/M2/M3) 能勉強達標，跨裝置相容性極低。
3. **結論與最佳架構建議**：
   - **當前最佳實踐（本專案架構）**：**「輕量網頁端（Web Audio PCM 採樣與波形渲染） + WebSocket 低延遲串流 + 雲端/伺服器 GPU (vLLM 加速推論）」**。
   - **優勢**：
     - 使用者端**零下載、即開即用**，即使低階手機也能秒開。
     - 伺服器端透過 vLLM PagedAttention + 嚴格增量 Delta 串流，推論延遲控制在 **30ms 左右**，兼具精準度、極致即時性與高相容性。

---

## 📝 變更日誌

詳細版本變更與升級歷史請參閱 [CHANGELOG.md](./CHANGELOG.md)。
