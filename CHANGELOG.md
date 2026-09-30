# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

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
