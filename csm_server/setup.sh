#!/bin/bash
# CSM-1B TTS Server Setup Script for AWS g5.xlarge (A10G 24GB)

set -e

echo "=== CSM-1B TTS Server Setup ==="

# Update system
echo "Updating system packages..."
sudo yum update -y || sudo apt update -y

# Install system dependencies
echo "Installing system dependencies..."
sudo yum install -y git wget ffmpeg || sudo apt install -y git wget ffmpeg

# Create directory
mkdir -p ~/csm-tts-server
cd ~/csm-tts-server

# Setup conda environment
echo "Setting up conda environment..."
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
fi

# Create environment
conda create -n csm python=3.10 -y || true
conda activate csm

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Clone CSM repository for generator
echo "Cloning CSM repository..."
if [ ! -d "csm" ]; then
    git clone https://github.com/SesameAILabs/csm.git
    cd csm
    pip install -e .
    cd ..
fi

# Download model weights
echo "Downloading CSM-1B model..."
python -c "
from huggingface_hub import snapshot_download
import os

# Check if already downloaded
if not os.path.exists('./models/csm-1b'):
    print('Downloading CSM-1B from HuggingFace...')
    snapshot_download(
        repo_id='sesame/csm-1b',
        local_dir='./models/csm-1b',
        local_dir_use_symlinks=False
    )
    print('Download complete!')
else:
    print('Model already exists')
"

# Verify GPU
echo "Verifying GPU..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== Setup Complete ==="
echo "To start the server:"
echo "  conda activate csm"
echo "  python server.py"
echo ""
