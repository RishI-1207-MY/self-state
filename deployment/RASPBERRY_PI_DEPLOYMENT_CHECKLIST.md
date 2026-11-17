# SafeStep: Raspberry Pi 4 Model B Hardware Setup & Deployment Checklist

> **Current Parallel Workflow Status**:
> - **Windows Laptop Development & AI Training**: **COMPLETED & VERIFIED**
> - **Physical Raspberry Pi 4 Hardware Execution**: **PENDING PHYSICAL EXECUTION ON PI** (All hardware test scripts are created, verified in dry-run/laptop mode, and explicitly marked `[ ] PENDING` below until executed on the physical Raspberry Pi 4).

---

## 1. Target Hardware Specification

- **Compute Board**: Raspberry Pi 4 Model B (4GB RAM) running Raspberry Pi OS (64-bit Bookworm recommended)
- **Camera**: Raspberry Pi Camera Module connected via **CSI ribbon port** (`Picamera2` / `libcamera`)
- **Ultrasonic Sensor**: HC-SR04 Ultrasonic Rangefinder
  - **VCC**: Physical Pin 2 (`5V`)
  - **TRIG**: Physical Pin 16 (`BCM GPIO 23`)
  - **ECHO**: Physical Pin 18 (`BCM GPIO 24`) **via 1kΩ / 2kΩ voltage divider** (steps 5V ECHO pulse down to safe `3.3V` logic level)
  - **GND**: Physical Pin 6 (`GND`)
- **Audio Output**: Wired Earphones (3.5mm TRRS jack) or USB Audio Adapter (`espeak-ng` + `ALSA` / `pyttsx3`)

### HC-SR04 Voltage Divider Schematic (5V -> 3.3V on GPIO24)

```text
HC-SR04 VCC  ----------------------> Pin 2  (5V)
HC-SR04 TRIG ----------------------> Pin 16 (BCM GPIO 23)
HC-SR04 ECHO ----[ 1kΩ ]----+------> Pin 18 (BCM GPIO 24)
                            |
                         [ 2kΩ ]
                            |
HC-SR04 GND  ---------------+------> Pin 6  (GND)
```

---

## 2. Deployment & Verification Checklist

### Phase A: Windows Laptop Software, Training & Packaging (Completed)
- [x] **A1. Modular SafeStep Codebase**: Created all AI modules (`object_detection.py`, `ocr_module.py`, `currency_detection.py`, `distance_sensor.py`, `audio_feedback.py`, `camera.py`, `navigation_engine.py`) and accessible Streamlit UI (`app.py`).
- [x] **A2. Hardware Isolation**: Isolated Raspberry Pi GPIO (`hardware/gpio_config.py`, `hardware/raspberry_pi_setup.py`) and CSI Picamera2 (`hardware/picamera2_interface.py`) so Windows laptop development never requires Raspberry Pi packages.
- [x] **A3. Separate Environment Configurations**:
  - `config.windows.yaml`: Windows development profile (`backend: opencv`, `distance_sensor.mode: simulation`).
  - `config.rpi.yaml`: Raspberry Pi 4 deployment profile (`backend: picamera2`, `distance_sensor.mode: hardware`, `img_size: 320`, `prefer_onnx_runtime: true`).
- [x] **A4. Custom Datasets & Zero-Leakage Splits**:
  - Dataset A (`object_detection`): 2,250 usable images, 6,613 bounding boxes across 12 classes (`PASSED`).
  - Dataset B (`currency_detection`): 2,580 usable images, 3,432 bounding boxes across 6 INR denominations (`PASSED`).
- [x] **A5. External Model Training & Raspberry Pi Optimization**:
  - Trained custom YOLOv8n Object Detection (`mAP@50 = 0.9393`) and Currency Detection (`mAP@50 = 0.9148`).
  - Exported FP32 ONNX (`best.onnx`) and quantized INT8 ONNX (`best_rpi_int8.onnx`) via `training/optimize_for_raspberry_pi.py`.
- [x] **A6. SSH/SCP Deployment Archive Built**:
  - Created `deployment/deploy_to_pi.py` and `deployment/deploy_to_pi.ps1` and generated `deployment/safestep_rpi_bundle.tar.gz`.

---

### Phase B: Raspberry Pi OS & Physical Wiring Setup (User Parallel Tasks — Pending)
- [ ] **B1. Flash & Boot Raspberry Pi OS (64-bit)** on Raspberry Pi 4 Model B (4GB RAM) and enable SSH (`sudo raspi-config` -> *Interface Options* -> *SSH*).
- [ ] **B2. Connect Raspberry Pi Camera Module to CSI Port** (with power off, blue backing/contacts facing the Ethernet/USB ports side as applicable to your Pi 4 CSI connector).
- [ ] **B3. Wire HC-SR04 Ultrasonic Sensor** to `5V` (Pin 2), `GND` (Pin 6), `GPIO23` (Pin 16) for `TRIG`, and `GPIO24` (Pin 18) through the **1kΩ + 2kΩ voltage divider** for `ECHO`.
- [ ] **B4. Connect Earphones or USB Audio Output** to the Raspberry Pi 4.

---

### Phase C: SSH/SCP Transfer & Automated Installation on Raspberry Pi (Pending)
- [ ] **C1. Transfer SafeStep & Optimized Weights from Windows Laptop to Raspberry Pi**:
  Run from PowerShell on your Windows laptop once the Pi is reachable on your network:
  ```powershell
  cd c:\Users\rishi\OneDrive\Desktop\v2\SafeStep
  .\deployment\deploy_to_pi.ps1 -PiHost <RASPBERRY_PI_IP> -PiUser pi -RunInstall
  ```
  *(Or run `python deployment/deploy_to_pi.py --host <RASPBERRY_PI_IP> --user pi --run-install`)*
- [ ] **C2. Verify Installation Script Execution (`deployment/install_raspberry_pi.sh`)** on Raspberry Pi:
  Installs `python3-picamera2`, `rpicam-apps`, `python3-rpi.gpio`, `espeak-ng`, `alsa-utils`, and creates `.venv_rpi` with `--system-site-packages`.

---

### Phase D: Independent Physical Hardware Verification on Raspberry Pi 4 (Pending)
> **Important**: These tests are currently marked **`PENDING_PHYSICAL_PI_EXECUTION`** in `evaluation/results/hw_test_*.json`. Run each command below on the physical Raspberry Pi 4 and confirm status changes to **`PASSED_ON_HARDWARE`**:

- [ ] **D1. Picamera2 CSI Camera Hardware Test (`PENDING_PHYSICAL_PI_EXECUTION`)**:
  ```bash
  source ~/SafeStep_Deploy/SafeStep/.venv_rpi/bin/activate
  python3 hardware/test_picamera2_hw.py --frames 10
  ```
  - Expected Output: `"status": "PASSED_ON_HARDWARE"`, saves `logs/picamera2_hardware_test.jpg`.

- [ ] **D2. HC-SR04 Ultrasonic Sensor Hardware Test (`PENDING_PHYSICAL_PI_EXECUTION`)**:
  ```bash
  python3 hardware/test_hcsr04_hw.py --samples 10
  ```
  - Expected Output: `"status": "PASSED_ON_HARDWARE"`, `"is_simulated": false`, prints 10 real distance measurements in meters from `GPIO23`/`GPIO24`.

- [ ] **D3. Earphones / USB Audio Hardware Test (`PENDING_PHYSICAL_PI_EXECUTION`)**:
  ```bash
  python3 hardware/test_audio_hw.py --speak
  ```
  - Expected Output: `"status": "PASSED_ON_PI_HARDWARE"`, speaks all 4 priority alerts clearly through earphones/USB audio.

- [ ] **D4. Full End-to-End SafeStep Launch on Raspberry Pi 4 (`PENDING_PHYSICAL_PI_EXECUTION`)**:
  ```bash
  SAFESTEP_ENV=rpi streamlit run app.py --server.port 8501 --server.address 0.0.0.0
  ```
