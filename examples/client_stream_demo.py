#!/usr/bin/env python3
# coding=utf-8
"""
Confucius4-R2T2 Streaming ASR Client Example
示範外部應用程式如何透過 WebSocket (WSS) 串流發送音訊切片並即時獲取字幕文字。
"""

import sys
import json
import time
import uuid
import asyncio
import wave
import argparse
import ssl

try:
    import websockets
except ImportError:
    print("請先安裝 websockets: pip install websockets")
    sys.exit(1)

YOUDAO_ONETIME_ASR_EOS_STRING = "YOUDAO_ONETIME_ASR_STREAM_EOS"

async def stream_audio_file(ws_url: str, wav_path: str, secret_key: str = "test0102", language: str = "zhen"):
    # 設定 SSL 忽略自主簽名（若使用標準憑證可省略）
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    print(f"[*] 連線至服務端: {ws_url} ...")
    async with websockets.connect(ws_url, ssl=ssl_context if ws_url.startswith("wss://") else None) as ws:
        request_id = str(uuid.uuid4())
        
        # 1. 發送握手 JSON 標頭
        header = {
            "requestId": request_id,
            "language": language,
            "use_vad": True,
            "secret_key": secret_key
        }
        await ws.send(json.dumps(header))
        print(f"[+] 握手成功，requestId: {request_id}")

        # 背景接收辨識結果
        async def receive_results():
            sentence_acc = ""
            try:
                async for message in ws:
                    data = json.loads(message)
                    if data.get("status") == "success":
                        msg = data.get("msg", {})
                        delta = msg.get("text", "")
                        reset = msg.get("reset", False)
                        cost_ms = msg.get("total_cost_ms", msg.get("asr_cost_ms", 0))

                        if delta:
                            sentence_acc += delta
                            # 即時輸出增量文字
                            print(f"\r[即時字幕] {sentence_acc} ({cost_ms}ms)", end="", flush=True)
                        
                        if reset and sentence_acc:
                            print(f"\n[句結存檔] {sentence_acc}")
                            sentence_acc = ""
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"\n[!] 接收錯誤: {e}")

        recv_task = asyncio.create_task(receive_results())

        # 2. 讀取 WAV 音訊並每 160ms 發送一個 Chunk
        with wave.open(wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            
            assert sample_rate == 16000, f"目前僅支援 16kHz 音訊，檔案為 {sample_rate}Hz"
            assert channels == 1, "目前僅支援單聲道 (Mono) 音訊"
            assert sampwidth == 2, "目前僅支援 16-bit PCM"

            # 160ms = 2560 samples = 5120 bytes
            chunk_size = 2560 * 2 
            
            print(f"[*] 開始串流傳送音訊: {wav_path} ...")
            while True:
                chunk = wf.readframes(2560)
                if not chunk:
                    break
                await ws.send(chunk)
                await asyncio.sleep(0.16) # 模擬真實 160ms 時間步

        # 3. 發送音訊結束訊號
        print("\n[*] 音訊傳送完畢，等待最終結果...")
        await ws.send(YOUDAO_ONETIME_ASR_EOS_STRING)
        await asyncio.sleep(1.0)
        recv_task.cancel()
        print("[✓] 辨識流程完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confucius4-R2T2 ASR WebSocket Client")
    parser.add_argument("--url", default="wss://147.5gao.ai/asr_stream_api_v1", help="WebSocket URL")
    parser.add_argument("--audio", required=True, help="16kHz 16-bit Mono WAV 檔案路徑")
    parser.add_argument("--key", default="test0102", help="授權金鑰")
    parser.add_argument("--lang", default="zhen", choices=["zhen", "Chinese", "English"], help="辨識語言")
    args = parser.parse_args()

    asyncio.run(stream_audio_file(args.url, args.audio, args.key, args.lang))
