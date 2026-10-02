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
- **微批優先級排程調度器 (Option A: FairInferenceScheduler with Deficit Round-Robin)**：
  - 徹底重構 GPU 推論排程架構，以單一專屬推論工作線程 (`ASRInferenceWorker`) 搭配 DRR 虧額輪轉雙隊列（`MAX_LIVE_BURST=6`），徹底消除多線程直接調用 vLLM 底層 IPC ZeroMQ 造成的跨進程 Futex 互鎖死鎖，同時杜絕大檔轉寫在持續直播負載下的飢餓風險。
  - 雙軌優先級：WebSocket 實時語音享受最高優先級 (`priority=0`)，HTTP 大檔轉寫微切片使用普通優先級 (`priority=1`)，每 6 個實時切片強制保證調度 1 個檔案切片，維持 15~30ms 極致即時響應且永不飢餓。
  - 斷線任務跳過 (`cancelled_event`)：客戶端斷線時工作線程自動跳過 GPU 推論，避免浪費顯存算力。
  - VAD 異步化與安全隔離：將 FireRedVAD 檢測移至 `asyncio.to_thread` 執行緒池，徹底解放 Sanic 主事件迴圈；克隆失敗時安全關閉 VAD，杜絕跨連線特徵快取污染。
  - 連續說話長度溢出防護 (`MAX_CONTINUOUS_SPEECH_SEC=25.0`)：超過 25 秒無停頓自動觸發語意分段重設，防止超出 vLLM `max_model_len=2048` 崩潰。
  - 串流連線上限門禁 (`MAX_CONCURRENT_STREAMS=12`)：超載時回傳 1013 Server Busy，防止單卡排隊延遲雪崩。
  - 雙向任務生命週期監控：以 `asyncio.wait(..., return_when=FIRST_COMPLETED)` 監管 `receiver` 與 `processor`，徹底消除客戶端提早退出時的協程與 Socket 洩漏。
  - 防禦性關閉與佇列清理：`_dispatch_to_loop` 防止 Event Loop 關閉時 Worker 線程拋出異常死亡；`stop()` 自動拒絕殘留任務。
  - 健壯性 PCM 解碼：`read_pcm` 自動防範奇數長度與截斷二進位音訊封包。
  - 於 `GET /health` 端點新增 `"queue_size"` 即時排隊深度監控。
  - 於 `README.md` 詳盡記錄高並發架構演進評估（方案 A 微批調度、方案 B 進程級雙實例隔離、方案 C 原生 AsyncLLMEngine 動態合批）。
- **品牌識別與 Open Graph 社交卡片**：
  - 於網頁底部加入「技術提供 david888.com | llms.txt」精緻連結（OBS 直播模式下自動隱藏）。
  - 設計並產出標準 1200×630 尺寸 Open Graph 社群分享圖 (`web/assets/og-image.png`，524KB) 與社交 Meta 標籤。

### Fixed
- **vLLM 推論線程競爭死鎖與事件循環阻塞修復 (Inference Thread Lock & Event Loop Decoupling)**：
  - 診斷出當 HTTP 轉寫請求 (`/transcribe`、`/transcribe/stream`) 與 WebSocket 實時流 (`/asr_stream_api_v1`) 同時並發調用時，因底層 vLLM V1 引擎 `LLM.generate` 缺乏線程安全保護，導致多線程同時向 EngineCore IPC 佇列發送請求並在 `outputs_queue.get()` 相互競爭搶奪輸出，造成跨進程死鎖、Sanic 主線程陷入 futex 阻塞，導致整機連線超時假死。
  - 引入全域互斥鎖 `ASR_INFER_LOCK` 嚴格序列化每次 `generate` 推論，長音檔以 0.32 秒極小粒度推論並釋放鎖，使 WebSocket 即時語音能以 ~15ms 級微小延遲交錯運算，兼顧即時性與高並發安全。
  - 將 WebSocket 串流之推論步驟移入 `asyncio.to_thread` 異步執行緒池，徹底解放 Sanic 主事件循環，確保 `/health` 與連線管理永遠流暢無阻塞。
  - 實作每連線獨立之 `FireRedStreamVad` 狀態複製 (`conn_vad`)，杜絕多連線並行時語音活動檢測特徵快取污染。
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
