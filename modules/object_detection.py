"""Real-time Object and Obstacle Detection Module for SafeStep.

Features:
- Loads custom-trained Ultralytics YOLO weights (`models/object_detection/best.pt`)
  with graceful fallback to pretrained `yolov8n.pt` if custom training has not yet run
- Optional lightweight MobileNet-SSD / Torchvision SSDLite320_MobileNet_V3_Large
  comparison backend (`backend="mobilenet_ssd"`)
- Frame preprocessing, confidence thresholding, and Non-Maximum Suppression (NMS)
- Cross-frame centroid/IoU object tracking to suppress repeated voice alerts
- Field-of-view spatial localization (`left`, `ahead`, `right`) and proximity estimation
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from utils.logger import get_logger, get_project_root
from utils.preprocessing import validate_image

logger = get_logger("SafeStep.ObjectDetection")

SAFESTEP_OBJECT_CLASSES: Dict[int, str] = {
    0: "Person",
    1: "Car",
    2: "Motorcycle",
    3: "Bicycle",
    4: "Bus",
    5: "Truck",
    6: "Pole",
    7: "Wall",
    8: "Chair",
    9: "Table",
    10: "Door",
    11: "Other obstacles",
}

COCO_FALLBACK_MAP: Dict[int, str] = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    56: "Chair",
    60: "Table",
}

HAZARD_CLASSES = {"Person", "Car", "Motorcycle", "Bicycle", "Bus", "Truck", "Pole", "Other obstacles"}


class SimpleDetectionTracker:
    """Lightweight IoU + centroid tracker to track objects across consecutive frames."""

    def __init__(self, max_disappeared: int = 15, iou_threshold: float = 0.35) -> None:
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        if inter <= 0:
            return 0.0
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return inter / float(area_a + area_b - inter)

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign persistent `track_id` and `is_new_track` flags to detections."""
        used_tracks = set()
        for det in detections:
            best_tid = None
            best_iou = 0.0
            for tid, trk in self.tracks.items():
                if tid in used_tracks or trk["class_name"] != det["class_name"]:
                    continue
                iou = self._compute_iou(det["bbox"], trk["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_threshold:
                self.tracks[best_tid]["bbox"] = det["bbox"]
                self.tracks[best_tid]["disappeared"] = 0
                self.tracks[best_tid]["age"] += 1
                det["track_id"] = best_tid
                det["is_new_track"] = False
                used_tracks.add(best_tid)
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    "bbox": det["bbox"],
                    "class_name": det["class_name"],
                    "disappeared": 0,
                    "age": 1,
                }
                det["track_id"] = tid
                det["is_new_track"] = True
                used_tracks.add(tid)

        stale_ids = []
        for tid, trk in self.tracks.items():
            if tid not in used_tracks:
                trk["disappeared"] += 1
                if trk["disappeared"] > self.max_disappeared:
                    stale_ids.append(tid)
        for tid in stale_ids:
            del self.tracks[tid]

        return detections


class ObjectDetectionModule:
    """Real-time obstacle and object detector with YOLO and optional MobileNet-SSD comparison."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        img_size: int = 416,
        backend: str = "yolo",
    ) -> None:
        self.root = get_project_root()
        self.weights_path = (
            Path(weights_path)
            if weights_path
            else self.root / "models" / "object_detection" / "best.pt"
        )
        self.fallback_weights = self.root / "models" / "pretrained" / "yolov8n.pt"
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.img_size = int(img_size)
        self.backend = backend.lower()

        self.model: Any = None
        self.mobilenet_model: Any = None
        self.model_loaded = False
        self.using_custom_weights = False
        self.tracker = SimpleDetectionTracker()
        self.last_inference_ms = 0.0
        self.fps = 0.0

        self.load_model()

    def load_model(self) -> bool:
        """Load the custom YOLO model or optional MobileNet-SSD model."""
        if self.backend == "mobilenet_ssd":
            return self._load_mobilenet_ssd()

        try:
            from ultralytics import YOLO

            if self.weights_path.exists():
                self.model = YOLO(str(self.weights_path))
                self.using_custom_weights = True
                self.model_loaded = True
                logger.info("Loaded custom SafeStep object detection model: %s", self.weights_path)
                return True
            if self.fallback_weights.exists():
                self.model = YOLO(str(self.fallback_weights))
                self.using_custom_weights = False
                self.model_loaded = True
                logger.info("Loaded fallback YOLO weights: %s", self.fallback_weights)
                return True

            # Allow Ultralytics to initialize/download yolov8n into pretrained dir if needed
            self.fallback_weights.parent.mkdir(parents=True, exist_ok=True)
            self.model = YOLO(str(self.fallback_weights))
            self.using_custom_weights = False
            self.model_loaded = True
            return True
        except Exception as exc:
            logger.error("Failed to load YOLO object detection model: %s", exc)
            self.model_loaded = False
            return False

    def _load_mobilenet_ssd(self) -> bool:
        """Load lightweight MobileNetV3-SSD from torchvision for benchmark comparison."""
        try:
            import torch
            import torchvision.models.detection as tv_det

            self.mobilenet_model = tv_det.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
            self.mobilenet_model.eval()
            self.model_loaded = True
            logger.info("Loaded MobileNetV3-SSD comparison model.")
            return True
        except Exception as exc:
            logger.warning("MobileNet-SSD load failed (%s); falling back to YOLO.", exc)
            self.backend = "yolo"
            return self.load_model()

    @staticmethod
    def _estimate_position(x1: int, x2: int, frame_w: int) -> str:
        cx = (x1 + x2) / 2.0
        ratio = cx / max(1, frame_w)
        if ratio < 0.35:
            return "left"
        if ratio > 0.65:
            return "right"
        return "ahead"

    def detect(
        self, frame: np.ndarray, confidence_threshold: float | None = None
    ) -> Dict[str, Any]:
        """Run object detection on a single BGR frame and return structured results."""
        if not validate_image(frame):
            return {
                "detections": [],
                "num_objects": 0,
                "hazards": [],
                "inference_ms": 0.0,
                "fps": 0.0,
                "error": "Invalid input frame",
            }

        conf_thresh = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.confidence_threshold
        )
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        raw_detections: List[Dict[str, Any]] = []

        if not self.model_loaded:
            self.load_model()

        if self.backend == "mobilenet_ssd" and self.mobilenet_model is not None:
            import torch

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                outputs = self.mobilenet_model([tensor])[0]
            boxes = outputs["boxes"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()
            for box, score, lbl in zip(boxes, scores, labels):
                if float(score) < conf_thresh:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                cls_name = COCO_FALLBACK_MAP.get(int(lbl) - 1, "Other obstacles")
                pos = self._estimate_position(x1, x2, w)
                raw_detections.append(
                    {
                        "class_id": int(lbl),
                        "class_name": cls_name,
                        "confidence": round(float(score), 4),
                        "bbox": [x1, y1, x2, y2],
                        "position": pos,
                        "is_hazard": cls_name in HAZARD_CLASSES,
                    }
                )
        elif self.model is not None:
            results = self.model.predict(
                frame,
                conf=conf_thresh,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
            )
            names_map = getattr(self.model, "names", SAFESTEP_OBJECT_CLASSES)
            for res in results:
                if res.boxes is None:
                    continue
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    if self.using_custom_weights:
                        cls_name = SAFESTEP_OBJECT_CLASSES.get(cls_id, str(names_map.get(cls_id, "Other obstacles")))
                    else:
                        cls_name = COCO_FALLBACK_MAP.get(cls_id, str(names_map.get(cls_id, "Other obstacles")))
                    pos = self._estimate_position(x1, x2, w)
                    raw_detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": [x1, y1, x2, y2],
                            "position": pos,
                            "is_hazard": cls_name in HAZARD_CLASSES,
                        }
                    )

        tracked = self.tracker.update(raw_detections)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.last_inference_ms = round(elapsed_ms, 2)
        self.fps = round(1000.0 / max(1.0, elapsed_ms), 2)

        hazards = [d for d in tracked if d.get("is_hazard")]
        return {
            "detections": tracked,
            "num_objects": len(tracked),
            "hazards": hazards,
            "inference_ms": self.last_inference_ms,
            "fps": self.fps,
            "backend": self.backend,
            "using_custom_weights": self.using_custom_weights,
        }
