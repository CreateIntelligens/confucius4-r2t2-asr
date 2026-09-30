# Confucius4-R2T2 Multi-Architecture Streaming ASR & Real-time Subtitles Service

網易有道開源之實時真流式語音大模型 **Confucius4-R2T2**（基於 Qwen3-ASR-1.7B），具備最長穩定前綴（Longest Stable Prefix, LSP）只增不改、不回改、不幻覺補全之特性。

本專案封裝為跨架構 Docker 容器化服務與原生 Systemd 部署方案，相容於 **x86_64（RTX 4000 Ada / RTX A6000 等）** 與 **aarch64（ARM64, GB10 等）**，並提供開箱即用的現代化**網頁收音與即時字幕前端系統**。

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
   - HTTP SSE 模擬流式轉寫：`POST /transcribe/stream`
   - WebSocket 串流介面：`WS /ws/stream` 與 `WS /asr_stream_api_v1`

---

## 📁 專案目錄結構

```text
confucius4-r2t2-asr/
├── app/
│   ├── main.py              # FastAPI 容器服務入口
│   └── r2t2/                # R2T2 串流推論核心引擎模組
├── web/
│   └── index.html           # 現代化即時字幕 Single Page Application (SPA)
├── deploy/
│   ├── systemd/
│   │   └── confucius4-r2t2.service  # Linux 原生 Systemd 常駐守護進程配置
│   └── nginx/
│       └── 147.5gao.ai.conf # Nginx SSL + WSS 反向代理範例配置
├── compose.yaml             # Docker Compose 雙容器編排 (API + Nginx)
├── Dockerfile               # 跨架構 CUDA 容器構建檔
├── nginx.template           # 容器版 Nginx 模板
├── server.py                # Sanic + vLLM 專用高效能串流服務端
└── README.md
```

---

## 🚀 部署方式

### 方法 A：Docker Compose 容器化部署（推薦）

```bash
# 啟動容器
docker compose up -d --build

# 觀察啟動日誌與自動預熱
docker compose logs -f r2t2-api
```

- 容器將對外服務於 `:8803`（可於 `.env` 中調整 `PUBLIC_PORT`）。
- 打開瀏覽器訪問 `http://<主機IP>:8803/` 即可進入**即時字幕操作介面**。

---

### 方法 B：Linux 原生 Systemd 服務部署（高效能生產主機）

1. **建立 Python 虛擬環境並安裝依賴**：
   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate
   uv pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
   uv pip install "vllm>=0.14.0" fireredvad sanic soundfile librosa
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

---

## 📡 API 與 WebSocket 介面說明

### 1. 健康檢查 (`GET /healthz` 或 `GET /health`)
```bash
curl http://localhost:8803/healthz
```
響應：
```json
{
  "status": "ok",
  "backend": "vllm",
  "model_loaded": true,
  "arch": "x86_64",
  "gpu": "NVIDIA RTX 4000 Ada Generation",
  "vram_used_gb": 6.8,
  "error": null
}
```

### 2. 即時語音 WebSocket 串流 (`WS /asr_stream_api_v1`)
- **握手訊息 (JSON Header)**：
  ```json
  {
    "requestId": "uuid-here",
    "language": "zhen",
    "use_vad": true,
    "secret_key": "test0102"
  }
  ```
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
  當說話者停頓約 0.7 秒時，VAD 判定語句結束，將回傳 `{"reset": true}`，前端自動換行並歸檔歷史記錄。

---

## ⚙️ 環境變數設定 (`compose.yaml` / `.env`)

| 變數名稱 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `PUBLIC_PORT` | `8803` | 對外公開監聽端口（Nginx Reverse Proxy） |
| `ASR_BACKEND` | `auto` | 後端推論引擎：`auto` (優先 vLLM，失敗自動降級 Transformers) / `vllm` / `transformers` |
| `AUTO_DOWNLOAD` | `1` | 若 `/model` 權重不存在，自動自 Hugging Face 下載 |
| `AUTO_WARMUP` | `1` | 服務啟動後自動以 1 秒靜音推論預熱 CUDA kernel 與注意力快取 |
| `GPU_MEMORY_UTILIZATION` | `0.40` | vLLM 顯存分配佔比（留有充裕 KV cache，避免 OOM） |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | 設為 `0` 避開缺少 nvcc 時 FlashInfer sampling 之 JIT 編譯需求 |

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

