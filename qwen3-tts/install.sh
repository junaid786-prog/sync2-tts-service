#!/bin/bash
# Qwen3-TTS Installation Script
# Run on TTS server (EC2 instance with GPU)

set -e

echo "=========================================="
echo "Qwen3-TTS Installation"
echo "=========================================="

# Create directory
mkdir -p ~/qwen3-tts
cd ~/qwen3-tts

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "❌ No NVIDIA GPU detected. Qwen3-TTS requires GPU."
    exit 1
fi

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
fi

# Create conda environment
echo "Creating conda environment..."
conda create -n qwen3-tts python=3.12 -y
source ~/miniconda/etc/profile.d/conda.sh
conda activate qwen3-tts

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Qwen3-TTS
echo "Installing Qwen3-TTS..."
pip install -U qwen-tts

# Install FlashAttention (optional but recommended)
echo "Installing FlashAttention..."
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation || echo "FlashAttention installation failed, continuing..."

# Install additional dependencies
echo "Installing additional dependencies..."
pip install websockets soundfile numpy scipy aiohttp

# Create voices directory
mkdir -p ~/qwen3-tts/voices

# Create systemd service file
echo "Creating systemd service..."
sudo tee /etc/systemd/system/qwen3-tts.service > /dev/null << 'EOF'
[Unit]
Description=Qwen3-TTS WebSocket Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/qwen3-tts
Environment="PATH=/home/ec2-user/miniconda/envs/qwen3-tts/bin:/usr/local/bin:/usr/bin:/bin"
Environment="QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
Environment="QWEN_TTS_VOICE=Vivian"
Environment="QWEN_TTS_LANGUAGE=English"
Environment="WEBSOCKET_PORT=8765"
ExecStart=/home/ec2-user/miniconda/envs/qwen3-tts/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo "✅ Qwen3-TTS Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy server.py to ~/qwen3-tts/"
echo "2. Start the service: sudo systemctl start qwen3-tts"
echo "3. Enable on boot: sudo systemctl enable qwen3-tts"
echo "4. Check logs: sudo journalctl -u qwen3-tts -f"
echo ""
echo "WebSocket endpoint: ws://$(curl -s ifconfig.me):8765/tts/stream"
