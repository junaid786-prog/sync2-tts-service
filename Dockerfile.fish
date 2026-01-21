# Sync2 Fish Speech TTS Service - GPU Docker Image
# Using OpenAudio S1-mini (formerly Fish Speech) - most natural open-source TTS

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create app user (non-root for security)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.fish.txt ./requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Clone Fish Speech repository for local inference
RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app/fish-speech-repo && \
    cd /app/fish-speech-repo && \
    pip install --no-cache-dir -e . || pip install --no-cache-dir .

# Install huggingface_hub for model download
RUN pip install --no-cache-dir huggingface_hub[cli]

# Create directories for voices, cache, and models
RUN mkdir -p /app/voices /app/cache /app/models /app/checkpoints \
    && chown -R appuser:appuser /app

# Copy Fish Speech application code
COPY fish_speech/ ./fish_speech/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Download OpenAudio S1-mini model (smaller, faster)
RUN huggingface-cli download fishaudio/openaudio-s1-mini --local-dir /app/checkpoints/openaudio-s1-mini || true

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Run the server
CMD ["python", "-m", "fish_speech.server"]
