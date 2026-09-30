# Confucius4-R2T2 Multi-Architecture Streaming ASR Service

網易有道開源之實時真流式語音大模型 **Confucius4-R2T2**（基於 Qwen3-ASR-1.7B），具備最長穩定前綴（Longest Stable Prefix, LSP）只增不改、不回改、不幻覺補全之特性。

本專案封裝為跨架構 Docker 容器化服務，相容於 **x86_64（AMD64, RTX A6000 / NVIDIA GPUs）** 與 **aarch64（ARM64, GB10 等）**。

## 🌟 核心特色

1. **自動下載 Model**（`AUTO_DOWNLOAD=1`）：
   - 首次啟動若檢測到權重不存在，自動自 Hugging Face (`netease-youdao/Confucius4-R2T2`) 拉取 3.8 GB 權重，無須手動搬運。
2. **自動預熱**（`AUTO_WARMUP=1`）：
   - 容器啟動於背景自動送入 1 秒 Dummy 音訊完成 CUDA kernel、音訊編碼器與注意力快取預熱，首呼客戶端享受毫秒級穩態延遲。
3. **跨架構雙後端無縫回退**（`ASR_BACKEND=auto`）：
   - 優先啟用 **vLLM 串流後端**（支援 LSP 只增不改）。
   - 若底層 GPU 驅動或架構（如新一代 Blackwell sm_121）尚未原生適配 Triton JIT 時，**自動優雅降級為 Transformers 後端**，保障服務 100% 穩定在線不崩潰。
4. **多協議接口**：
   - HTTP 狀態檢查：`GET /healthz`
   - HTTP 一次性檔案轉寫：`POST /transcribe`
   - HTTP SSE 模擬流式轉寫：`POST /transcribe/stream`
   - WebSocket 真實 PCM 即時串流：`WS /ws/stream`

## 🚀 部署與啟動

```bash
docker compose up -d --build
docker compose logs -f r2t2-api
```

對外端口預設為 `:8803`。

## 📡 API 測試範例

### 1. 健康檢查
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
  "gpu": "NVIDIA RTX A6000",
  "vram_used_gb": 3.8,
  "download_seconds": 95.2,
  "load_seconds": 22.4,
  "warmup_seconds": 1.1,
  "error": null
}
```

### 2. 音檔轉錄 (HTTP POST)
```bash
curl -X POST http://localhost:8803/transcribe \
  -F "file=@audio.wav"
```

### 3. SSE 流式轉錄 (Server-Sent Events)
```bash
curl -N -X POST http://localhost:8803/transcribe/stream \
  -F "file=@audio.wav"
```

### 4. WebSocket 真實即時流式轉錄 (PCM Chunk)
連線至 `ws://localhost:8803/ws/stream`，客戶端持續傳送 16kHz 16-bit Mono PCM raw binary，辨識結果以 JSON 即時增量推送：
```json
{"type": "token", "delta": "智慧", "text": "智慧醫療", "is_final": false}
```
音訊結束時送入 `{"action": "finish"}` 即可獲得最終定稿 `{"type": "done", "text": "...", "is_final": true}`。

---

## ⚙️ 環境變數設定 (`compose.yaml` / `.env`)

| 變數名稱 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `PUBLIC_PORT` | `8803` | 對外公開監聽端口（Nginx Reverse Proxy） |
| `ASR_BACKEND` | `auto` | 後端推論引擎：`auto` (優先 vLLM，失敗自動降級 Transformers) / `vllm` / `transformers` |
| `AUTO_DOWNLOAD` | `1` | 若 `/model` 權重不存在，自動自 Hugging Face 下載 |
| `AUTO_WARMUP` | `1` | 服務啟動後自動以 1 秒靜音推論預熱 CUDA kernel 與注意力快取 |
| `GPU_MEMORY_UTILIZATION` | `0.35` | vLLM 顯存分配佔比（A6000 48GB 約分配 16.8GB，留有足夠 KV cache） |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | 設為 `0` 避開缺少 nvcc 時 FlashInfer sampling 之 JIT 編譯需求 |

---

## 🚚 遷移至 `10.9.0.37` (ARM64 GB10) 部署指南

本專案之 `Dockerfile` 基於標準 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`，已包含跨架構相容設計：
1. 將本目錄打包或 scp 至 `10.9.0.37`：
   ```bash
   scp -r confucius_r2t2_deploy <user>@10.9.0.37:~/confucius-r2t2-asr
   ```
2. 在 `10.9.0.37` 上直接執行一鍵構建與啟動：
   ```bash
   cd ~/confucius-r2t2-asr
   docker compose up -d --build
   ```
3. 容器將自動感測 ARM64（`aarch64`）硬體架構，自 Hugging Face 下載權重並完成開機自動預熱，對外同樣服務於 `:8803`。
