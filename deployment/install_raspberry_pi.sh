#!/usr/bin/env bash
# ==============================================================================
# SafeStep: Raspberry Pi 4 Model B (4GB RAM) Automated Installation Script
# Target OS: Raspberry Pi OS (64-bit Bookworm / Bullseye)
# Hardware:
#   - Raspberry Pi Camera Module connected via CSI (Picamera2 / libcamera)
#   - HC-SR04 Ultrasonic Sensor: GPIO23 (TRIG), GPIO24 (ECHO via 1kΩ/2kΩ divider)
#   - Earphones (3.5mm jack) or USB Audio Output
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv_rpi"

echo "=================================================================="
echo "  SafeStep: Raspberry Pi 4 Model B Installation & Setup"
echo "  Project Root: ${PROJECT_DIR}"
echo "=================================================================="

echo "[1/6] Updating apt package index and installing system & hardware packages..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-picamera2 \
    python3-libcamera \
    python3-rpi.gpio \
    python3-gpiozero \
    rpicam-apps \
    espeak-ng \
    espeak \
    alsa-utils \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    ffmpeg

echo "[2/6] Ensuring current user (${USER}) belongs to gpio, video, and audio groups..."
sudo usermod -aG gpio,video,audio "${USER}" || true

echo "[3/6] Creating virtual environment (--system-site-packages for Picamera2 & libcamera)..."
python3 -m venv --system-site-packages "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[4/6] Installing Python packages optimized for ARM64 Raspberry Pi 4..."
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"
pip install RPi.GPIO gpiozero

echo "[5/6] Activating Raspberry Pi deployment configuration profile..."
export SAFESTEP_ENV=rpi
if ! grep -q "SAFESTEP_ENV=rpi" "${HOME}/.bashrc" 2>/dev/null; then
    echo 'export SAFESTEP_ENV=rpi' >> "${HOME}/.bashrc"
fi

echo "[6/6] Running initial Raspberry Pi hardware diagnostics..."
python3 "${PROJECT_DIR}/hardware/raspberry_pi_setup.py"

echo ""
echo "=================================================================="
echo "  SafeStep Installation Complete on Raspberry Pi 4!"
echo ""
echo "  Next Steps on the Raspberry Pi:"
echo "    1. Activate environment: source ${VENV_DIR}/bin/activate"
echo "    2. Test CSI Camera:      python3 hardware/test_picamera2_hw.py"
echo "    3. Test HC-SR04 Sensor:  python3 hardware/test_hcsr04_hw.py"
echo "    4. Test Earphone Audio:  python3 hardware/test_audio_hw.py --speak"
echo "    5. Launch SafeStep UI:   SAFESTEP_ENV=rpi streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
echo "=================================================================="
