#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh — Sets up a Python environment for training an LLM from scratch
# with an NVIDIA GPU. Run this ON YOUR OWN MACHINE (the one with the GPU) —
# it needs internet access and a real CUDA-capable device to do anything useful.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# ==============================================================================
set -euo pipefail

ENV_NAME="llm-train"
PYTHON_VERSION="3.11"

echo "=== 1. Checking for NVIDIA driver ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Install the NVIDIA driver for your GPU"
    echo "before continuing, or this environment will fall back to CPU-only torch."
else
    nvidia-smi
fi

echo ""
echo "=== 2. Detecting CUDA driver version to pick a matching PyTorch build ==="
CUDA_TAG="cu124"   # default: CUDA 12.4 wheels, works with driver >= 550
if command -v nvidia-smi &> /dev/null; then
    DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
    echo "Driver reports max supported CUDA: ${DRIVER_CUDA:-unknown}"
    # Adjust the tag below if you need an older/newer build, e.g. cu121, cu128
fi
echo "Using PyTorch wheel index: $CUDA_TAG"

echo ""
echo "=== 3. Creating conda environment '$ENV_NAME' (Python $PYTHON_VERSION) ==="
if command -v conda &> /dev/null; then
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "conda not found — falling back to a plain venv instead."
    python3 -m venv "$ENV_NAME"
    # shellcheck disable=SC1091
    source "$ENV_NAME/bin/activate"
fi

echo ""
echo "=== 4. Installing PyTorch with CUDA support ($CUDA_TAG) ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ""
echo "=== 5. Installing the rest of the training stack ==="
pip install -r requirements.txt

echo ""
echo "=== 6. Verifying the install ==="
python scripts/verify_setup.py

echo ""
echo "Setup complete. Activate this environment in future sessions with:"
if command -v conda &> /dev/null; then
    echo "  conda activate $ENV_NAME"
else
    echo "  source $ENV_NAME/bin/activate"
fi
