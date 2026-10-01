# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **一般／深色模式與字幕工作台改版**：新增主題切換，OBS 空白狀態不再顯示無法操作的提示。
- **長音檔逐段字幕**：新增 `POST /transcribe/stream`，每 30 秒建立獨立辨識狀態，逐段回傳字幕、音訊時間與進度。
- **前端資產拆分**：將樣式與互動程式移至 `web/assets/app.css` 與 `web/assets/app.js`，由 Sanic `/assets` 路由提供。

### Changed
- **長音檔上下文管理**：`POST /transcribe` 與串流上傳流程改為每 30 秒重設辨識狀態，避免整段音檔累積超過模型 token 上限。
- **音檔容量與等待時間**：網頁單檔上限提高至 200 MB；Nginx 與 Sanic 設定同步支援大型上傳及長時間辨識。
- **產品標題**：網頁頁籤與頁首改為 `333-R2T2-ASR`。

## [1.3.0] - 2026-10-01

### Added
- **Transformers 後端也能串流**：`app/r2t2/r2t2_asr.py` 的串流解碼不再限定 vLLM，aarch64 / 顯存吃緊的主機（如 NVIDIA GB10）可直接串流。
- **llama.cpp 後端與自動選擇**（`app/llama_backend.py`）：聲學編碼器留在 PyTorch、逐字生成交給 llama.cpp，不需要 vLLM。`ASR_BACKEND=auto` 依序嘗試 `vllm` → `llama` → `transformers`。Docker 映像在 build 時從原始碼編譯 llama.cpp，x86_64 與 aarch64 共用同一份 `Dockerfile`。
- **`app/main.py` 提供 `WS /asr_stream_api_v1`**：與 `server.py` 相同的有道協議（JSON header、PCM、`YOUDAO_ONETIME_ASR_STREAM_EOS`），支援 VAD 斷句、熱詞（`system_prompt`）、`smooth`。
- **句末 `final_text`**：每句結束（`reset: true`）時整句重新辨識一次，修正串流邊聽邊出字造成的句首雜字與誤判；header 可用 `final_pass: false` 關閉。
- **`/healthz` 回報推論狀態**：`inference`（排隊數、目前推論已跑多久、上次成功距今多久）、`active_streams`、`vad_loaded`；推論卡住超過 `STALL_SECONDS` 時 `status` 變為 `stalled`。`server.py` 的 `/health` 同步加入，卡住時回 503。
- 環境變數 `VAD_DIR`、`STREAM_DEFAULT_LANGUAGE`、`STREAM_SECRET_KEYS`、`STALL_SECONDS`。
- `tests/`（pytest）：語言對應、排隊鎖、串流 session、滑動視窗裁切，以及兩套服務、README、網頁之間的協議一致性檢查。
- `deploy/systemd/confucius-r2t2.user.service`：GB10 上以 systemd user unit 執行 `app/main.py` 的設定。

### Changed
- **所有模型呼叫共用一個排隊鎖並在執行緒中執行**（`app/main.py` 與 `server.py`）：多路串流與批次請求輪流推論，不再阻塞事件迴圈。
- **每條連線獨立的 VAD 狀態**：只共用模型權重。有 VAD 時偵測到語音才開始解碼，靜音不佔 GPU。
- **推論落後時自動併塊**：一步最多處理 1.28 秒音訊；滑動視窗改以實際樣本數裁切，分塊大小可變。
- **串流預設語言由自動判斷改為 Chinese**：`zhen`（中英混講）對應成 Chinese，夾雜的英文照實輸出；要自動判斷須明確送 `auto`。網頁語言選單同步調整。
- `/transcribe`、`/transcribe/stream`、`/ws/stream` 接受相同的語言代碼（`zhen`、`zh`、`en`…），不支援的語言回 400 而非 500。
- `/transcribe` 的 `text` 去除模型輸出的 `|` 標記。
- **`app/main.py` 跟上新版網頁**：提供 `/assets`、`POST /transcribe/stream` 改為與 `server.py` 相同的逐段事件（`start`／`segment`／`done`），`/transcribe` 同樣每 30 秒分段並補上 `duration_sec`、`cost_ms` 欄位。原本的 token 增量 SSE 已移除。
- `server.py` 的 `/transcribe`、`/transcribe/stream` 每一步解碼都經過排隊鎖。
- **上傳音檔改用 soundfile + soxr 解碼**（`app/audio_io.py`）：輸出與 `librosa.load` 相同，但省掉它每個行程第一次呼叫的初始化（GB10 約 6 秒、A4000 主機約 27 秒），服務重啟後的第一個上傳不再卡住。
- 網頁在句末採用 `final_text` 歸檔。
- **Docker 映像改為 `builder` → `runner` 兩階段、以 `nvidia/cuda:13.0.3-cudnn-runtime` 為底**：原本的 `pytorch/pytorch` 基底只有 amd64，在 aarch64 的 GB10 上無法使用。相依套件改由 `requirements.txt` 加本機 `wheels/` 安裝，不含 vLLM；GPU 改用 CDI 掛載；模型改掛宿主目錄（`MODEL_HOST_DIR`、`VAD_HOST_DIR`）；容器以宿主帳號執行。

### Fixed
- **llama.cpp 原生擴充在多路並行時讀到損毀的 embedding**（`native_ext.cpp`）：放掉 GIL 之後才讀取 numpy 陣列，其他 Python 執行緒活動時會出現 `cannot create std::vector larger than max_size()` 或 segfault。
- **高負載併塊時可能吞掉一整句**：VAD 改為永遠逐塊執行，只合併解碼。
- **`server.py` 的 `POST /transcribe` 同時請求會讓服務卡死**：多個執行緒同時呼叫同一個 vLLM 引擎。現在與串流共用排隊鎖，且每一步解碼之間會讓出鎖，長音檔不會讓串流停擺。
- **README 與範例腳本的串流結束字串寫錯**（寫成小寫的 `youdao_onetime_asr_eos_string`，伺服器不認得）。
- **`server.py` 同時多路串流互相卡住**：推論原本直接在事件迴圈上同步執行，且多條連線共用同一個 VAD 實例，導致多數句子等不到 `reset`。
- **自動語言模式串流停在第一個字**：模型回 `language None` 時語言標籤沒有帶回前綴，每一步都把 token 額度花在重寫標籤上。
- **串流不指定語言時漂到其他語言**（帶口音的華語被轉成葡萄牙文）。
- `/ws/stream` 結束時未送出 close frame。

### Known gaps
- 已實測的組合：GB10（aarch64）的 Docker 映像（llama.cpp、Transformers）；RTX A4000（x86_64）的 Docker 映像（llama.cpp、Transformers）以及直接執行的 `server.py` 與 `app/main.py` 搭 vLLM 0.14。正式的 vLLM 主機（RTX 4000 Ada）尚未部署這一版。
- vLLM 0.14 在 GB10 上無法啟動（CUDA 12 版本，且編譯工具不支援該晶片）；在 A4000 上需要 `gpu_memory_utilization` 約 0.50（約 6.2 GB），實測沒有比 Transformers 快。
- `server.py` 沒有 `final_text`。
## [1.2.0] - 2026-09-30

### Added
- **HTTP REST 音訊轉寫端點 (`POST /transcribe`)**：
  - 於 `server.py` 實作音訊檔案上傳介面，支援 WAV, MP3, FLAC, M4A, OGG 等多種格式，外部 App 可直接發起單次 HTTP 請求獲取轉寫文字。
- **外部應用程式即時串流示範腳本 (`examples/client_stream_demo.py`)**：
  - 提供開箱即用的 Python 客戶端範例，示範建立 WebSocket 連線、發送握手 JSON、串流 PCM 音訊切片（160ms chunks）及接收即時增量字幕。
- **外部 App 調用 API 指南**：
  - 於 `README.md` 詳細補充 HTTP REST 與 WebSocket 的通訊協定規範、cURL、Python 及 Node.js 呼叫代碼範例。
- **端點與運行狀態資訊**：
  - 於 `README.md` 標註外部存取網址（`https://147.5gao.ai/`）、即時字幕網頁介面特色與 GPU 顯存隔離（40% VRAM 與 TTS 服務共存）狀態。

---

## [1.1.0] - 2026-09-30

### Added
- **即時字幕 Single Page Application (SPA)** (`web/index.html`)：
  - 原生 Web Audio API 即時收音與高品質線性降採樣（任何瀏覽器取樣率轉為 16kHz 16-bit Mono PCM）。
  - HTML5 Canvas 即時聲波振幅（Waveform）視覺化。
  - OBS / vMix 直播透明覆蓋模式（去背、無背景、高對比字卡）。
  - 歷史字幕卡片滾動歸檔，支援一鍵複製與 TXT / JSON 導出。
  - 完整 SSL / WSS 雙向連線狀態與延遲指示燈。
- **反幻覺與熔斷機制 (`server.py`)**：
  - 實作 N-gram 重複詞元偵測演算法 (`detect_hallucination`)，當自迴歸模型陷入死循環時主動中斷並重置 ASR State。
  - 整合 `FireRedStreamVad` 語音活動檢測，將停頓靜音閥值調優為 `min_silence_frame=60`（約 700ms），自動標記 `speech_end` 並換句。
- **生產環境部署模組 (`deploy/`)**：
  - 原生 Systemd 守護進程配置 (`deploy/systemd/confucius4-r2t2.service`)，預載虛擬環境與顯存參數。
  - Nginx 反向代理配置範例 (`deploy/nginx/147.5gao.ai.conf`)，支援 HTTPS 憑證與 WebSocket 升級通道。
- **R2T2 核心演算法模組** (`app/r2t2/`)：
  - 補齊並整合 upstream 缺失之 `r2t2_asr.py` 核心演算法模組。

### Changed
- **修正字幕重複刷屏 Bug**：
  - 由原先累積文字全量廣播改為嚴格的 **Delta 增量流式發送**（`fixed_text[len(last_fixed):]`），靜音無聲時保持 0 輸出，徹底解決前端文字無限重複累積問題。
- **vLLM 啟動與效能調優**：
  - 預設啟用 `enforce_eager=True`，避開昂貴的 CUDA Graph 編譯，服務預熱時間自 120 秒縮減至 **4 秒**。
  - 限制 `gpu_memory_utilization=0.40`（約 6.8GB VRAM），實現與 IndexTTS、CosyVoice 等大型服務在一張 RTX 4000 Ada GPU 上穩定共存。
- **更新 Docker 編排與服務掛載**：
  - `compose.yaml` 掛載最新 `web/` 目錄並支援健康檢查別名 `GET /health` 與 `GET /healthz`。

### Fixed
- **QEMU CPU 無 AVX 指令集導致 SIGILL 崩潰問題**：
  - 移除不相容之 `nagisa` 與 `dynet38` C++ 擴充依賴，確保在虛擬化與各類雲端環境下執行 Qwen-ASR 模型不觸發非法指令（Illegal Instruction）。

---

## [1.0.0] - 2026-09-30

### Added
- 初始化專案結構，封裝 Confucius4-R2T2（基於 Qwen3-ASR-1.7B）流式推論服務。
- 提供 FastAPI 服務入口 (`app/main.py`) 與 WebSocket 串流接口 (`/ws/stream`, `/asr_stream_api_v1`)。
- 提供 Dockerfile 與 Docker Compose 雙容器環境（FastAPI + Nginx）。
- 提供自動下載權重與啟動預熱（Warmup）腳本。
