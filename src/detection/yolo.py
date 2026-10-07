"""Detector de personas con YOLO (Ultralytics), clase persona solamente."""

from __future__ import annotations

import threading
import time

import numpy as np

from src.detection.base import FrameResult


PERSON_CLASS = 0


class YoloPersonDetector:
    """YOLO restringido a la clase persona; se carga y precalienta una vez."""

    def __init__(self, model: str, confidence: float = 0.40,
                 image_size: int = 640, device: str = "cpu",
                 weak_confidence: float = 0.15) -> None:
        from ultralytics import YOLO  # import diferido: es pesado

        self._model = YOLO(model)
        self._confidence = confidence
        self._weak_confidence = min(weak_confidence, confidence)
        self._image_size = image_size
        self._device = device
        self._lock = threading.Lock()
        self.detect(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> FrameResult:
        with self._lock:
            prediction = self._model.predict(
                frame, conf=self._weak_confidence, imgsz=self._image_size,
                classes=[PERSON_CLASS], device=self._device, verbose=False,
            )[0]
        strong: list[tuple[float, tuple[int, int, int, int]]] = []
        weak: list[tuple[float, tuple[int, int, int, int]]] = []
        for box in prediction.boxes:
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            (strong if confidence >= self._confidence else weak).append(
                (confidence, (x1, y1, x2, y2)))
        return FrameResult(time.monotonic(),
                           tuple(c for c, _ in strong), tuple(b for _, b in strong),
                           tuple(c for c, _ in weak), tuple(b for _, b in weak))
