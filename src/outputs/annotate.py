"""Cajas y etiquetas sobre las fotos guardadas."""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.base import FrameResult


def annotate(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    image = frame.copy()
    for (x1, y1, x2, y2), confidence in zip(result.boxes, result.confidences):
        cv2.rectangle(image, (x1, y1), (x2, y2), (50, 220, 70), 2)
        cv2.putText(image, f"persona {confidence:.0%}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 220, 70), 2)
    return image
