// --- 狀態管理 ---
let isRecording = false;
let ws = null;
let audioContext = null;
let mediaStream = null;
let audioProcessor = null;
let chunkCounter = 0;

let currentSegmentText = "";
let historyRecords = [];
let isUploadingAudio = false;

// DOM 元素
const recordBtn = document.getElementById('record-btn');
const recordText = document.getElementById('record-text');
const uploadBtn = document.getElementById('upload-btn');
const audioFileInput = document.getElementById('audio-file');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const themeLabel = document.getElementById('theme-label');
const wsDot = document.getElementById('ws-dot');
const wsStatus = document.getElementById('ws-status');
const sourceStatus = document.getElementById('source-status');
const captionBox = document.getElementById('caption-box');
const asrCostEl = document.getElementById('asr-cost');
const totalCostEl = document.getElementById('total-cost');
const chunkCountEl = document.getElementById('chunk-count');
const langSelect = document.getElementById('lang-select');
const contextInput = document.getElementById('context-input');
const secretKeyInput = document.getElementById('secret-key');
const micSelect = document.getElementById('mic-select');
const historyList = document.getElementById('history-list');
const historyCount = document.getElementById('history-count');
const historyEmpty = document.getElementById('history-empty');
const copyBtn = document.getElementById('copy-btn');
const exportBtn = document.getElementById('export-btn');
const clearBtn = document.getElementById('clear-btn');
const obsToggle = document.getElementById('obs-toggle');
const toastEl = document.getElementById('toast');
const uploadProgress = document.getElementById('upload-progress');
const uploadStatus = document.getElementById('upload-status');
const uploadProgressLabel = document.getElementById('upload-progress-label');
const uploadProgressTrack = document.getElementById('upload-progress-track');
const uploadProgressFill = document.getElementById('upload-progress-fill');
const uploadProgressMeta = document.getElementById('upload-progress-meta');
const canvas = document.getElementById('visualizer');
const canvasCtx = canvas.getContext('2d');

// 視波器變數
let analyserNode = null;
let animFrameId = null;

// 提示通知
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(() => toastEl.classList.remove('show'), 2500);
}

function applyTheme(theme, persist = true) {
  const isDark = theme === 'dark';
  document.documentElement.dataset.theme = isDark ? 'dark' : 'light';
  themeToggle.setAttribute('aria-pressed', String(isDark));
  themeToggle.setAttribute('aria-label', isDark ? '切換一般模式' : '切換深色模式');
  themeLabel.textContent = isDark ? '一般模式' : '深色模式';
  themeIcon.innerHTML = isDark
    ? '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"/>'
    : '<path d="M20.2 15.4A8.4 8.4 0 0 1 8.6 3.8 8.5 8.5 0 1 0 20.2 15.4Z"/>';
  if (persist) {
    try { localStorage.setItem('r2t2-theme', isDark ? 'dark' : 'light'); } catch (err) {}
  }
  resetVisualizer();
}

let savedTheme = 'light';
try {
  const storedTheme = localStorage.getItem('r2t2-theme');
  if (storedTheme === 'dark' || storedTheme === 'light') savedTheme = storedTheme;
} catch (err) {}
applyTheme(savedTheme, false);
themeToggle.addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
});

function showCaptionStatus(message) {
  const status = document.createElement('span');
  status.className = 'caption-empty';
  status.textContent = message;
  captionBox.replaceChildren(status);
}

function formatAudioTime(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours
    ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
    : `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function setUploadProgress(progress, label, detail) {
  const percent = Math.max(0, Math.min(100, progress));
  uploadProgress.hidden = false;
  uploadStatus.textContent = label;
  uploadProgressLabel.textContent = `${Math.round(percent)}%`;
  uploadProgressTrack.setAttribute('aria-valuenow', String(Math.round(percent)));
  uploadProgressFill.style.transform = `scaleX(${percent / 100})`;
  uploadProgressMeta.textContent = detail;
}

async function transcribeUploadedAudio(file) {
  if (isRecording || isUploadingAudio) {
    showToast('請先停止收音，再上傳音檔');
    audioFileInput.value = '';
    return;
  }
  if (file.size > 200000000) {
    showToast('音檔超過 200 MB，請選擇較小的檔案');
    audioFileInput.value = '';
    return;
  }

  isUploadingAudio = true;
  uploadBtn.disabled = true;
  recordBtn.disabled = true;
  sourceStatus.textContent = '音檔轉寫中';
  setUploadProgress(0, '正在準備音檔', `${file.name} · 每 30 秒辨識一段`);
  showCaptionStatus('音檔已上傳，完成的字幕片段會逐段顯示…');

  const formData = new FormData();
  formData.append('file', file, file.name);
  if (langSelect.value !== 'zhen') formData.append('language', langSelect.value);
  if (contextInput.value.trim()) formData.append('context', contextInput.value.trim());

  let savedSegments = 0;
  let totalSegments = 0;
  let durationSec = 0;
  let completed = false;

  try {
    const response = await fetch('/transcribe/stream', { method: 'POST', body: formData });
    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      throw new Error(result.message || result.detail || `伺服器回應 ${response.status}`);
    }
    if (!response.body) throw new Error('瀏覽器無法讀取逐段辨識結果，請重新整理後再試。');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';

    function handleEventBlock(block) {
      const data = block.split(/\r?\n/)
        .filter(line => line.startsWith('data:'))
        .map(line => line.slice(5).trim())
        .join('\n');
      if (!data) return;
      const event = JSON.parse(data);

      if (event.type === 'start') {
        totalSegments = event.total_segments || 1;
        durationSec = Number(event.duration_sec) || 0;
        setUploadProgress(0, `準備辨識 ${totalSegments} 段音訊`, `音檔長度 ${formatAudioTime(durationSec)} · 每段 30 秒`);
      } else if (event.type === 'segment') {
        totalSegments = event.total || totalSegments || 1;
        const percent = event.index / totalSegments * 100;
        const range = `${formatAudioTime(event.start_sec)}–${formatAudioTime(event.end_sec)}`;
        setUploadProgress(percent, `已辨識 ${event.index} / ${totalSegments} 段`, `${range} / ${formatAudioTime(durationSec)}`);
        const text = (event.text || '').trim();
        if (text) {
          captionBox.textContent = text;
          commitHistorySegment(text, range);
          savedSegments++;
        }
      } else if (event.type === 'done') {
        completed = true;
        setUploadProgress(100, '音檔辨識完成', `${totalSegments} 段 · 音訊 ${formatAudioTime(event.duration_sec)}`);
        sourceStatus.textContent = '音檔已完成';
        if (!savedSegments) showCaptionStatus('音檔已處理，沒有辨識到語音。');
        showToast(savedSegments ? `辨識完成，共 ${savedSegments} 段字幕` : '音檔已處理，沒有辨識到語音');
      } else if (event.type === 'error') {
        throw new Error(event.message || '音檔片段辨識失敗');
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const blocks = pending.split(/\r?\n\r?\n/);
      pending = blocks.pop() || '';
      blocks.forEach(handleEventBlock);
    }
    pending += decoder.decode();
    if (pending.trim()) handleEventBlock(pending);
    if (!completed) throw new Error('辨識連線中斷；已完成的字幕片段已保留。');
  } catch (err) {
    console.error('音檔辨識失敗:', err);
    sourceStatus.textContent = '音檔辨識中斷';
    showCaptionStatus(savedSegments
      ? `辨識中斷，已完成的 ${savedSegments} 段字幕仍保存在右側紀錄。`
      : `音檔辨識失敗：${err.message || '請更換檔案後重試。'}`);
    uploadStatus.textContent = '辨識中斷';
    uploadProgressMeta.textContent = err.message || '已完成的字幕片段已保留。';
    showToast(`音檔辨識失敗：${err.message || '請稍後重試'}`);
  } finally {
    isUploadingAudio = false;
    uploadBtn.disabled = false;
    recordBtn.disabled = false;
    audioFileInput.value = '';
    if (!isRecording && !completed && sourceStatus.textContent === '音檔轉寫中') sourceStatus.textContent = '待命';
  }
}

uploadBtn.addEventListener('click', () => audioFileInput.click());
audioFileInput.addEventListener('change', () => {
  const file = audioFileInput.files && audioFileInput.files[0];
  if (file) transcribeUploadedAudio(file);
});

// 列出音訊設備
async function loadAudioDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(d => d.kind === 'audioinput');
    micSelect.innerHTML = '<option value="">預設麥克風</option>';
    audioInputs.forEach((device, index) => {
      const opt = document.createElement('option');
      opt.value = device.deviceId;
      opt.textContent = device.label || `麥克風 ${index + 1}`;
      micSelect.appendChild(opt);
    });
  } catch (err) {
    console.warn('無法枚舉音訊設備:', err);
  }
}
navigator.mediaDevices?.addEventListener('devicechange', loadAudioDevices);
loadAudioDevices();

function visualizerColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// 視波器繪製
function drawVisualizer() {
  if (!analyserNode) return;
  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyserNode.getByteTimeDomainData(dataArray);

  canvas.width = canvas.offsetWidth * window.devicePixelRatio;
  canvas.height = canvas.offsetHeight * window.devicePixelRatio;
  canvasCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

  const width = canvas.offsetWidth;
  const height = canvas.offsetHeight;

  canvasCtx.fillStyle = visualizerColor('--wave-bg');
  canvasCtx.fillRect(0, 0, width, height);

  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = isRecording ? visualizerColor('--accent') : visualizerColor('--wave-line');
  canvasCtx.beginPath();

  const sliceWidth = width * 1.0 / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0;
    const y = v * height / 2;

    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }
    x += sliceWidth;
  }

  canvasCtx.lineTo(width, height / 2);
  canvasCtx.stroke();

  if (isRecording) {
    animFrameId = requestAnimationFrame(drawVisualizer);
  }
}

// 重設畫布
function resetVisualizer() {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  canvasCtx.fillStyle = visualizerColor('--wave-bg');
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
  canvasCtx.strokeStyle = visualizerColor('--wave-line');
  canvasCtx.beginPath();
  canvasCtx.moveTo(0, canvas.height / 2);
  canvasCtx.lineTo(canvas.width, canvas.height / 2);
  canvasCtx.stroke();
}
resetVisualizer();

// --- 音訊重採樣至 16000Hz PCM Int16 ---
function resampleAndConvertToInt16(audioBuffer, inputSampleRate, targetSampleRate = 16000) {
  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.round(audioBuffer.length / ratio);
  const result = new Int16Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const origIndex = i * ratio;
    const indexFloor = Math.floor(origIndex);
    const indexCeil = Math.min(audioBuffer.length - 1, indexFloor + 1);
    const fraction = origIndex - indexFloor;
    const sample = (audioBuffer[indexFloor] * (1 - fraction) + audioBuffer[indexCeil] * fraction);
    const clamped = Math.max(-1, Math.min(1, sample));
    result[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF;
  }
  return result;
}

// 切換 OBS 模式
obsToggle.addEventListener('click', () => {
  const isObsMode = document.body.classList.toggle('obs-mode');
  obsToggle.setAttribute('aria-pressed', String(isObsMode));
  showToast(isObsMode ? 'OBS 字幕模式已開啟 · 按 Esc 返回控制台' : '已返回控制台');
});

// 建立 UUID
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// 開始錄音
async function startRecording() {
  try {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/asr_stream_api_v1`;

    wsStatus.textContent = '連線中...';
    wsDot.className = 'status-dot';

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = async () => {
      wsDot.className = 'status-dot connected';
      wsStatus.textContent = '服務已連線';
      sourceStatus.textContent = '即時收音中';

      // 發送 Handshake JSON Header
      const reqId = generateUUID();
      const header = {
        channels: 1,
        sample_rate: 16000,
        requestId: reqId,
        language: langSelect.value,
        use_vad: true,
        secret_key: secretKeyInput.value.trim() || 'test0102',
        mode: 'slow',
        system_prompt: contextInput.value.trim()
      };
      ws.send(JSON.stringify(header));

      // 啟動瀏覽器麥克風
      const deviceId = micSelect.value ? { exact: micSelect.value } : undefined;
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: deviceId,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(mediaStream);

      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 512;
      source.connect(analyserNode);

      // 160ms = 2560 samples at 16000Hz
      const TARGET_CHUNK_SAMPLES = 2560;
      const inputSampleRate = audioContext.sampleRate;

      audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);
      let pcmBufferAccumulator = [];

      audioProcessor.onaudioprocess = (e) => {
        if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
        const channelData = e.inputBuffer.getChannelData(0);

        const int16Data = resampleAndConvertToInt16(channelData, inputSampleRate, 16000);

        for (let i = 0; i < int16Data.length; i++) {
          pcmBufferAccumulator.push(int16Data[i]);
        }

        while (pcmBufferAccumulator.length >= TARGET_CHUNK_SAMPLES) {
          const chunk = new Int16Array(pcmBufferAccumulator.slice(0, TARGET_CHUNK_SAMPLES));
          pcmBufferAccumulator = pcmBufferAccumulator.slice(TARGET_CHUNK_SAMPLES);
          ws.send(chunk.buffer);
          chunkCounter++;
          chunkCountEl.textContent = chunkCounter;
        }
      };

      source.connect(audioProcessor);
      audioProcessor.connect(audioContext.destination);

      isRecording = true;
      recordBtn.classList.add('recording');
      recordText.textContent = '停止收音';
      sourceStatus.textContent = '即時收音';
      wsDot.className = 'status-dot active';
      wsStatus.textContent = '即時推論中';
      currentSegmentText = "";
      captionBox.innerHTML = '<span class="caption-empty">聆聽中...</span>';

      drawVisualizer();
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status === 'connected') {
          console.log('R2T2 handshake confirmed:', data);
          return;
        }

        if (data.status === 'success' && data.msg) {
          const newText = data.msg.text || '';
          const reset = data.msg.reset || false;

          if (data.msg.asr_cost_ms !== undefined) {
            asrCostEl.textContent = Number(data.msg.asr_cost_ms).toFixed(1);
          }
          if (data.msg.total_cost_ms !== undefined) {
            totalCostEl.textContent = Number(data.msg.total_cost_ms).toFixed(1);
          }

          if (newText) {
            currentSegmentText += newText;
            captionBox.innerHTML = `${escapeHtml(currentSegmentText)}<span class="incremental"></span>`;
          }

          if (reset && currentSegmentText.trim()) {
            commitHistorySegment(currentSegmentText);
            currentSegmentText = "";
            captionBox.innerHTML = '<span class="caption-empty">聆聽中...</span>';
          }
        } else if (data.status === 'error') {
          console.error('Server error:', data.msg);
          showToast('辨識錯誤: ' + data.msg);
        }
      } catch (e) {
        console.warn('非 JSON 訊息:', event.data);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket 錯誤:', err);
      showToast('WebSocket 連線異常');
      stopRecording();
    };

    ws.onclose = () => {
      wsDot.className = 'status-dot';
      wsStatus.textContent = '連線關閉';
      stopRecording();
    };

  } catch (err) {
    console.error('開啟麥克風失敗:', err);
    showToast('無法存取麥克風: ' + err.message);
    stopRecording();
  }
}

// 停止錄音
function stopRecording() {
  if (!isRecording && !ws) return;
  isRecording = false;

  recordBtn.classList.remove('recording');
  recordText.textContent = '開始收音';
  sourceStatus.textContent = '待命';
  wsDot.className = 'status-dot';
  wsStatus.textContent = '已停止';

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send("YOUDAO_ONETIME_ASR_STREAM_EOS");
    setTimeout(() => {
      try { ws.close(); } catch(e){}
      ws = null;
    }, 500);
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (audioProcessor) {
    try { audioProcessor.disconnect(); } catch(e){}
    audioProcessor = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    try { audioContext.close(); } catch(e){}
    audioContext = null;
  }

  if (currentSegmentText.trim()) {
    commitHistorySegment(currentSegmentText);
    currentSegmentText = "";
  }

  cancelAnimationFrame(animFrameId);
  resetVisualizer();
}

// 寫入歷史紀錄
function commitHistorySegment(text, timeOverride = null) {
  if (!text || !text.trim()) return;
  const now = new Date();
  const timeStr = timeOverride || (now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0').slice(0, 2));

  const record = { time: timeStr, text: text.trim() };
  historyRecords.unshift(record);
  historyEmpty.hidden = true;

  const itemEl = document.createElement('div');
  itemEl.className = 'history-item';
  itemEl.innerHTML = `
    <span class="history-time">[${timeStr}]</span>
    <span class="history-content">${escapeHtml(text.trim())}</span>
  `;
  historyList.prepend(itemEl);
  historyCount.textContent = historyRecords.length;
}

function escapeHtml(str) {
  return str.replace(/[&<>"']/g, function(m) {
    return ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    })[m];
  });
}

// 按鈕監聽
recordBtn.addEventListener('click', () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// 複製文字
copyBtn.addEventListener('click', () => {
  if (historyRecords.length === 0) {
    showToast('目前尚無記錄');
    return;
  }
  const fullText = historyRecords.map(r => `[${r.time}] ${r.text}`).reverse().join('\n');
  navigator.clipboard.writeText(fullText).then(() => {
    showToast('已複製全部字幕到剪貼簿');
  });
});

// 匯出 TXT
exportBtn.addEventListener('click', () => {
  if (historyRecords.length === 0) {
    showToast('目前尚無記錄');
    return;
  }
  const fullText = historyRecords.map(r => `[${r.time}] ${r.text}`).reverse().join('\n');
  const blob = new Blob([fullText], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `r2t2_subtitles_${new Date().toISOString().slice(0, 10)}.txt`;
  a.click();
});

// 清空紀錄
clearBtn.addEventListener('click', () => {
  historyRecords = [];
  historyList.innerHTML = '';
  historyCount.textContent = '0';
  historyEmpty.hidden = false;
  showCaptionStatus('已清空紀錄。可開始收音或上傳音檔。');
  showToast('紀錄已清空');
});

// 空白鍵快捷鍵切換錄音
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.body.classList.contains('obs-mode')) {
    document.body.classList.remove('obs-mode');
    obsToggle.setAttribute('aria-pressed', 'false');
    showToast('已返回控制台');
    return;
  }
  if (e.code === 'Space' && e.target === document.body) {
    e.preventDefault();
    recordBtn.click();
  }
});
