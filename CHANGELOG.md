# Changelog

All notable changes to this project will be documented in this file.
Versions are categorized by date (`YYYY-MM-DD`).

---

## [dev 分支，尚未併入 main]

### Added
- **Transformers 後端也能串流**：`app/r2t2/r2t2_asr.py` 的串流解碼不再限定 vLLM，aarch64 / 顯存吃緊的主機（如 NVIDIA GB10）可直接串流。
- **llama.cpp 後端與自動選擇**（`app/llama_backend.py`）：聲學編碼器留在 PyTorch、逐字生成交給 llama.cpp，不需要 vLLM。`ASR_BACKEND=auto` 依序嘗試 `vllm` → `llama` → `transformers`。Docker 映像在 build 時從原始碼編譯 llama.cpp，x86_64 與 aarch64 共用同一份 `Dockerfile`。
- **`app/main.py` 提供 `WS /asr_stream_api_v1`**：與 `server.py` 相同的有道協議（JSON header、PCM、`YOUDAO_ONETIME_ASR_STREAM_EOS`），支援 VAD 斷句、熱詞（`system_prompt`）、`smooth`。
- **句末 `final_text`**：每句結束（`reset: true`）時整句重新辨識一次，修正串流邊聽邊出字造成的句首雜字與誤判；header 可用 `final_pass: false` 關閉。
- **`/healthz` 回報推論狀態**：`inference`（排隊數、目前推論已跑多久、上次成功距今多久）、`active_streams`、`vad_loaded`；推論卡住超過 `STALL_SECONDS` 時 `status` 變為 `stalled`。`server.py` 的 `/health` 同步加入，卡住時回 503。
- 環境變數 `VAD_DIR`、`STREAM_DEFAULT_LANGUAGE`、`STREAM_SECRET_KEYS`、`STALL_SECONDS`。
- `tests/`（pytest）：語言對應、排隊鎖、串流 session、滑動視窗裁切，以及兩套服務、README、網頁之間的協議一致性檢查。
- `deploy/systemd/confucius-r2t2.user.service`：GB10 上以 systemd user unit 執行 `app/main.py` 的設定。

- **`app/main.py` 支援繁體輸出與停止上傳辨識**：`output_script`（串流 header 與兩個上傳端點）、`POST /transcribe/cancel` 與 `start` 事件的 `job_id`，與 `server.py` 相同。串流的繁體轉換以整句累積文字進行，被拆在兩則訊息裡的詞仍會轉成台灣慣用語；`final_text` 以整句轉換。

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
- **llama.cpp 後端遇到被拆成多個 token 的字（如「絞」）時整條串流被關閉**：生成長度剛好切在字的中間時，原生擴充把不完整的 UTF-8 轉成字串會直接出錯。改為寬鬆轉換，與其他後端一致。
- **串流推論出錯時用戶端分不出是錯誤還是正常結束**：現在先送 `status: "error"` 訊息，再以關閉碼 1011 結束。
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

---

## [2026-10-02]

### Added
- **專業英文規格文件 (`llms.txt`)**：
  - 依據 [llmstxt.org](https://llmstxt.org) 規範重寫專案根目錄 `llms.txt` 與前端 `web/llms.txt`。
  - 詳細記錄架構設計、HTTP REST (`/transcribe`)、長音檔 SSE 分段串流 (`/transcribe/stream`)、任務取消 (`/transcribe/cancel`)、低延遲 WebSocket (`/asr_stream_api_v1`) 通訊協定規範。
  - 詳列二進位音訊規範（16kHz, 16-bit signed integer, Mono, 160ms/5120 bytes chunk）以及 cURL、Python (requests/websockets)、JavaScript/TypeScript (Fetch/SSE) 整合範例。
- **全新獨立網域名稱與 SSL (`asr.5gao.ai`)**：
  - 在伺服器 Nginx 配置全新獨立站點 `asr.5gao.ai`，取得並配置 Let's Encrypt 自動續期 SSL 憑證。
  - 保留原有 `147.5gao.ai` 雙網域並行運作，對外提供穩定的 HTTPS 與 WSS 雙通道。
- **CI/CD 自動化部屬管線 (GitHub Actions & Self-hosted Runner)**：
  - 於主機 `10.9.0.35` 註冊系統級守護進程 `actions.runner.CreateIntelligens-confucius4-r2t2-asr.virtualhumantest-r2t2.service`。
  - 建立 `.github/workflows/deploy-ghcr.yml` 與 `Dockerfile.ghcr`，提交至 `main` 分支時自動建置映像發布至 GHCR，並由主機自動拉取部署、平滑重啟 `confucius4-r2t2.service` 與執行 `/health` 健康檢查，異常時具備自動回滾能力。
- **品牌識別與 Open Graph 社交卡片**：
  - 於網頁底部加入「技術提供 david888.com | llms.txt」精緻連結（OBS 直播模式下自動隱藏）。
  - 設計並產出標準 1200×630 尺寸 Open Graph 社群分享圖 (`web/assets/og-image.png`，524KB) 與社交 Meta 標籤。

### Fixed
- **全端點 CORS 與 OPTIONS Preflight 跨域支援**：
  - 實作 Sanic `MethodNotAllowed` (405) 例外攔截器，當收到任意路徑之 `OPTIONS` 預檢請求時，統一回傳 `HTTP 204 No Content` 並附加完整 CORS 標頭 (`Access-Control-Allow-Origin: *`、`Access-Control-Allow-Methods: GET, POST, OPTIONS, PUT, DELETE`)。
  - 針對 `/transcribe`、`/transcribe/stream`、`/transcribe/cancel`、`/health` 端點明確支援 `OPTIONS` 方法，徹底解決 Vue/React 等第三方瀏覽器應用程式發起跨網域請求時被瀏覽器攔截的 CORS 阻擋問題。
- **語音模型語系代碼正規化 (Language Normalization)**：
  - 實作 `normalize_asr_language()` 模組，將前端與 API 傳入之 `auto`、`zh`、`en`、`zhen` 等代碼自動正規化為 Qwen-ASR 底層相容名稱，防止模型拋出 `ValueError: Unsupported language` 崩潰。

### Changed
- **前端工作台視覺淨化**：
  - 移除標頭中冗贅之文字（「語音工作台 / 語音工作區...」），提升整體版面質感與工藝標準（Craft Floor）。

## [2026-10-01]

### Added
- **長音檔逐段字幕串流 (`POST /transcribe/stream`)**：
  - 每 30 秒建立獨立辨識狀態，透過 Server-Sent Events (SSE) 逐段回傳字幕文字、音訊時間戳與整體進度。
- **繁體中文輸出選項**：
  - 整合 OpenCC 繁體中文轉換庫，支援簡體與台灣繁體中文切換，即時串流與音檔辨識皆可套用。
- **轉寫任務取消機制 (`POST /transcribe/cancel`)**：
  - 提供 `task_id` 取消正在進行之長音檔轉寫分段，已轉寫完成部分保留輸出。
- **深色模式與多主題切換**：
  - 支援系統主題同步與深淺切換，OBS 模式下空白狀態優化。
- **前端模組資產獨立化**：
  - 將樣式與邏輯拆分至 `web/assets/app.css` 與 `web/assets/app.js`，交由 Sanic `/assets` 靜態路由分發。

### Changed
- **長音檔上下文記憶體管理**：
  - 音檔分段解碼改為每 30 秒自動重設 ASR state，避免單一任務 token 累積超過模型上限。
- **大型上傳支援**：
  - 單檔音訊上傳上限提升至 200 MB，Nginx 與 Sanic 設定超時時間同步拉長至 1800 秒。
- **品牌名稱統一**：
  - 網頁標題與品牌識別統一為 `333-R2T2-ASR`。

---

## [2026-09-30]

### Added
- **HTTP REST 音訊轉寫端點 (`POST /transcribe`)**：
  - 於 `server.py` 實作音訊檔案上傳介面，支援 WAV, MP3, FLAC, M4A, OGG 等多種格式。
- **即時字幕 Single Page Application (SPA)** (`web/index.html`)：
  - 原生 Web Audio API 即時收音與降採樣（16kHz 16-bit Mono PCM）。
  - HTML5 Canvas 即時聲波振幅（Waveform）視覺化。
  - OBS / vMix 直播透明覆蓋模式（去背、無背景、高對比字卡）。
  - 歷史字幕滾動歸檔，支援一鍵複製與 TXT / JSON 導出。
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
