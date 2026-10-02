# Changelog

All notable changes to this project will be documented in this file.
Versions are categorized by date (`YYYY-MM-DD`).

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

---

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
