# Qwen3-TTS Docker Image
# Based on NVIDIA CUDA for GPU acceleration

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Qwen3-TTS and dependencies
RUN pip install -U qwen-tts
RUN pip install -U flash-attn --no-build-isolation
RUN pip install websockets soundfile numpy scipy aiohttp

# Create app directory
WORKDIR /app

# Copy server files
COPY server.py /app/server.py
COPY voices/ /app/voices/

# Expose WebSocket port
EXPOSE 8765

# Set default command
CMD ["python", "server.py"]
