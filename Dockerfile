# app/main.py（FastAPI）的映像：llama.cpp 與 Transformers 兩種後端，啟動時自動選用。
# 不含 vLLM，原因見 requirements.txt。
#
# base 用 CUDA runtime 而不是 pytorch 官方映像：後者只有 amd64，在 aarch64 的
# GB10 上跑不起來；torch 直接裝 cu130 wheel，同一份 Dockerfile 兩種架構都能 build。
# cudnn 在 aarch64 的 torch wheel 不透過 pip 帶，靠 base 的 cudnn-runtime 變體。
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu22.04
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04

# llama.cpp 解碼後端：在這裡從原始碼編，產物才會符合這台的 CPU 架構與映像裡的 Python。
# repo 內附的預編譯檔是照別台機器的指令集編的，換一台就可能 Illegal instruction。
# 不需要這個後端時用 --build-arg WITH_LLAMA=0 跳過（省下 devel 映像與約 10 分鐘編譯），
# 服務會自動改用 Transformers。
FROM ${CUDA_DEVEL_IMAGE} AS llama-builder
ARG WITH_LLAMA=1
# 與 app/r2t2_llama/README.md 釘的版本一致
ARG LLAMA_CPP_COMMIT=ad6c66839af3c5646fba8c6c2e2087a1e4e38948

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ git ca-certificates python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pybind11

COPY app/r2t2_llama/CMakeLists.txt app/r2t2_llama/native_ext.cpp /src/r2t2_llama/
# GGML_NATIVE=OFF：不針對 build 這台的 CPU 最佳化，映像搬到同架構的別台也能跑；
# 運算都在 GPU 上，CPU 指令集對速度沒有影響。
RUN mkdir -p /opt/r2t2_native/lib \
    && if [ "$WITH_LLAMA" = "1" ]; then \
        git init -q /src/llama.cpp \
        && git -C /src/llama.cpp fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_COMMIT" \
        && git -C /src/llama.cpp checkout -q FETCH_HEAD \
        && cmake -S /src/r2t2_llama -B /build \
            -DLLAMA_CPP_DIR=/src/llama.cpp \
            -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
        && cmake --build /build -j 8 \
        && cp /src/r2t2_llama/native/qwen3asr_native*.so /opt/r2t2_native/ \
        && cp -a /build/bin/lib*.so* /opt/r2t2_native/lib/ ; \
    fi

FROM ${CUDA_BASE_IMAGE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# 相依套件約 3 GB。wheel 先下到宿主的 wheels/（見 README），build 時以 bind mount
# 讀進來，網路慢或中斷都不必整層重來。--no-index 是刻意的：只給 --find-links 的話
# pip 仍會挑 index 上的同版本重新下載。缺件就讓它明確失敗。
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=bind,source=wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements.txt

FROM ${CUDA_BASE_IMAGE} AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=llama-builder /opt/r2t2_native /opt/r2t2_native
ENV PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/r2t2_native/lib:${LD_LIBRARY_PATH}

WORKDIR /app
COPY app/ /app/
COPY web/ /web/

# 服務以宿主 uid 執行而 venv 屬 root：會寫快取的函式庫都導到 /tmp。
# NUMBA_CACHE_DIR 沒設的話 librosa 的 numba JIT 寫不進套件目錄，librosa.load 會直接失敗。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/opt/r2t2_native \
    HOME=/tmp \
    HF_HOME=/tmp/hf \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
