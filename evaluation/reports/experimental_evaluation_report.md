# SafeStep: Experimental Performance & System Evaluation Report

> **Safety Notice**: SafeStep is an assistive prototype and is **not** a certified replacement for a white cane, guide dog, or professional orientation and mobility training. Furthermore, visual currency detection classifies only the apparent denomination and does **not** verify banknote authenticity.

---

## 1. Executive Summary & Dataset Provenance

| Dataset | Total Usable Images | Total Bounding Boxes | Train / Val / Test Split | Zero Session Leakage | Quality Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dataset A: Object & Obstacle Detection (12 Classes)** | 2250 | 6613 | 1608 / 432 / 210 | `True` | **PASSED** |
| **Dataset B: Indian Currency Detection (6 Classes)** | 2580 | 3432 | 1800 / 510 / 270 | `True` | **PASSED** |

- **Provenance Transparency**:
  - Dataset A Provenance: `{"programmatic_synthetic_images": 2235, "webcam_or_real_hybrid_images": 15, "augmented_images": 1500, "base_unaugmented_images": 750}`
  - Dataset B Provenance: `{"programmatic_synthetic_images": 2580, "webcam_or_real_hybrid_images": 0, "augmented_images": 1720, "base_unaugmented_images": 860}`

---

## 2. Object & Obstacle Detection Model Evaluation (Held-Out Test Split)

- **Model**: `SafeStep Custom Object Detection (YOLOv8n)`
- **Precision**: `0.9193`
- **Recall**: `0.8617`
- **mAP@50**: `0.9393`
- **mAP@50-95**: `0.7807`
- **Average Inference Latency**: `19.24 ms` (`51.98 FPS`)

### Per-Class Performance (Test Split)

| Class Name | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| Person | 0.9606 | 0.9042 | 0.9512 | 0.7548 |
| Car | 1.0000 | 0.4743 | 0.9094 | 0.7930 |
| Motorcycle | 0.9921 | 0.9524 | 0.9887 | 0.7828 |
| Bicycle | 0.9240 | 0.8667 | 0.9619 | 0.7552 |
| Bus | 0.9047 | 0.9298 | 0.9400 | 0.8590 |
| Truck | 0.6743 | 0.7559 | 0.8295 | 0.7364 |
| Pole | 0.9716 | 0.8760 | 0.9143 | 0.6085 |
| Wall | 0.6353 | 0.8750 | 0.8794 | 0.8335 |
| Chair | 1.0000 | 0.8599 | 0.9369 | 0.7864 |
| Table | 0.9822 | 1.0000 | 0.9950 | 0.8547 |
| Door | 1.0000 | 0.9085 | 0.9854 | 0.8516 |
| Other obstacles | 0.9868 | 0.9375 | 0.9799 | 0.7527 |

---

## 3. Indian Currency Detection Model Evaluation (Held-Out Test Split)

- **Model**: `SafeStep Indian Currency Detection (YOLOv8n)`
- **Precision**: `0.8672`
- **Recall**: `0.8964`
- **mAP@50**: `0.9148`
- **mAP@50-95**: `0.7157`
- **Average Inference Latency**: `20.38 ms` (`49.07 FPS`)

### Per-Denomination Performance (Test Split)

| Denomination Class | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| 10_rupees | 0.7602 | 0.8667 | 0.8848 | 0.6665 |
| 20_rupees | 0.9770 | 0.7445 | 0.9532 | 0.7370 |
| 50_rupees | 0.9966 | 1.0000 | 0.9950 | 0.8469 |
| 100_rupees | 0.9901 | 1.0000 | 0.9950 | 0.8404 |
| 200_rupees | 0.5146 | 0.9020 | 0.6869 | 0.4979 |
| 500_rupees | 0.9646 | 0.8650 | 0.9739 | 0.7055 |

---

## 4. Environmental Condition Breakdown (Test Split)

| Condition | Mean Detection Confidence |
| :--- | :---: |
| **Indoor Scenes** (corridors, rooms, offices) | `0.8633` |
| **Outdoor Scenes** (roads, footpaths, public spaces) | `0.9120` |
| **Low Lighting / Night** | `0.8634` |
| **Bright / Daylight** | `0.9095` |
| **Crowded Scenes** | `0.9273` |
| **Partial Occlusion** | `0.8770` |

---

## 5. System Resource, Latency & Lightweight Model Comparison

| Metric | Measured Value |
| :--- | :---: |
| **Custom YOLOv8n Object Inference Time** | `32.87 ms` (`51.98 FPS`) |
| **MobileNetV3-SSD Comparison Inference Time** | `55.96 ms` (`17.87 FPS`) |
| **Custom YOLOv8n Currency Inference Time** | `55.72 ms` |
| **EasyOCR Inference Time (when triggered)** | `944.81 ms` |
| **End-to-End Multi-Modal Frame Latency** | `237.81 ms` (`4.21 FPS`) |
| **Audio Queue Dispatch Latency** | `0.09 ms` |
| **Average CPU Usage** | `14.7%` |
| **Process RAM Usage** | `708.6 MB` |
| **Power Consumption** | `Hardware power meter not attached on Windows laptop (N/A)` |

---

## 6. Model Export Verification

```json
{
  "object_detection": {
    "source_pt": "C:\\Users\\rishi\\OneDrive\\Desktop\\v2\\SafeStep\\models\\object_detection\\best.pt",
    "onnx_path": "C:\\Users\\rishi\\OneDrive\\Desktop\\v2\\SafeStep\\models\\object_detection\\best.onnx",
    "onnx_exists": true,
    "onnx_verified_ort": true,
    "onnx_size_mb": 12.05,
    "tflite_status": "skipped (TensorFlow not installed in lightweight CPU environment)"
  },
  "currency_detection": {
    "source_pt": "C:\\Users\\rishi\\OneDrive\\Desktop\\v2\\SafeStep\\models\\currency_detection\\best.pt",
    "onnx_path": "C:\\Users\\rishi\\OneDrive\\Desktop\\v2\\SafeStep\\models\\currency_detection\\best.onnx",
    "onnx_exists": true,
    "onnx_verified_ort": true,
    "onnx_size_mb": 12.04,
    "tflite_status": "skipped (TensorFlow not installed in lightweight CPU environment)"
  }
}
```
