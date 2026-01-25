#!/bin/bash
# Faster Whisper STT Installation Script
# For AWS EC2 with NVIDIA GPU

set -e

echo "=============================================="
echo "Faster Whisper STT Installation"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda first."
    echo "Run: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh"
    exit 1
fi

# Create conda environment
echo "Creating conda environment: faster-whisper-stt"
conda create -n faster-whisper-stt python=3.11 -y || true

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate faster-whisper-stt

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Faster Whisper
echo "Installing Faster Whisper..."
pip install faster-whisper

# Install other dependencies
echo "Installing dependencies..."
pip install websockets numpy scipy

# Install Silero VAD (will download on first use)
echo "Pre-downloading Silero VAD..."
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True)"

# Download Whisper model
echo "Pre-downloading Whisper model (medium)..."
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "To start the server manually:"
echo "  conda activate faster-whisper-stt"
echo "  python server.py"
echo ""
echo "To install as systemd service:"
echo "  sudo cp faster-whisper-stt.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable faster-whisper-stt"
echo "  sudo systemctl start faster-whisper-stt"
echo ""
