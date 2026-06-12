#!/bin/bash
# ============================================================================
# Environment Setup Script for LaViRA on Unitree G1
# ============================================================================
# Run this script on the G1's onboard computer (or connected development PC)
# to install all required dependencies.
#
# Prerequisites:
#   - Ubuntu 20.04 or 22.04
#   - Python 3.8+
#   - Network connection to G1 robot
#
# Usage:
#   sudo bash scripts/setup_g1_env.sh
# ============================================================================

set -e

echo "============================================"
echo "  LaViRA G1 Environment Setup"
echo "============================================"

# 1. System Dependencies
echo "[1/5] Installing system dependencies..."
apt-get update
apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    v4l-utils \
    udev

# 2. Orbbec SDK (pyorbbecsdk)
echo "[2/5] Installing Orbbec SDK (pyorbbecsdk)..."
echo "  NOTE: pyorbbecsdk must be installed from the Orbbec SDK release page."
echo "  Download the wheel matching your Python version and platform from:"
echo "    https://github.com/orbbec/pyorbbecsdk"
echo "  Then install with: pip3 install pyorbbecsdk-*.whl"
echo "  Skipping automatic install — install manually."

# 3. Python Dependencies
echo "[3/5] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install opencv-python numpy Pillow
pip3 install openai requests flask flask-socketio
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install pyyaml colorama imageio
pip3 install gevent gevent-websocket simplejpeg
pip3 install faster-whisper

# 4. Unitree SDK2 Python
echo "[4/5] Installing Unitree SDK2 Python (unitree_sdk2py)..."
if ! pip3 show unitree-sdk2py > /dev/null 2>&1; then
    cd /tmp
    if [ ! -d "unitree_sdk2_python" ]; then
        git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
    fi
    cd unitree_sdk2_python
    pip3 install -e .
    echo "Unitree SDK2 Python installed."
else
    echo "Unitree SDK2 Python already installed."
fi

# 5. Verify Installation
echo "[5/5] Verifying installation..."
python3 -c "
import cv2; print(f'  OpenCV: {cv2.__version__}')
import numpy; print(f'  NumPy: {numpy.__version__}')
import torch; print(f'  PyTorch: {torch.__version__}')
import openai; print(f'  OpenAI: {openai.__version__}')
try:
    import pyorbbecsdk; print('  pyorbbecsdk (Orbbec): OK')
except ImportError: print('  pyorbbecsdk: Not available (install separately from Orbbec release)')
try:
    import unitree_sdk2py; print('  Unitree SDK2: OK')
except ImportError: print('  Unitree SDK2: Not available')
"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Set camera serials (from udevadm / device labels):"
echo "     export ORBBEC_FRONT_SERIAL='CPC7B5300XXX'"
echo "     export ORBBEC_LEFT_SERIAL='CPC7B5300XXX'"
echo "     export ORBBEC_RIGHT_SERIAL='CPC7B5300XXX'"
echo "     export ORBBEC_REAR_SERIAL='CPC7B5300XXX'"
echo ""
echo "  2. Configure the LLM endpoint (local llama.cpp or remote API):"
echo "     export LA_BASE_URL='http://localhost:8000/v1'"
echo "     export VA_BASE_URL='http://localhost:8000/v1'"
echo "     # For remote API endpoints also set LA_API_KEY / VA_API_KEY"
echo ""
echo "  3. Start iPlanner server:"
echo "     ./scripts/run_iplanner_server.sh"
echo ""
echo "  4. Run navigation:"
echo "     ./scripts/run_navigation.sh 'Go to the kitchen'"
echo ""
