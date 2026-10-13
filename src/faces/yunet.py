"""Detector de caras YuNet (OpenCV Zoo): ~75 mil parámetros, 5 puntos faciales."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.faces.base import Face


class YuNetDetector:
    def __init__(self, model: Path, score: float = 0.80, nms: float = 0.30) -> None:
        self._detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score, nms, 5000)

    def detect(self, image: np.ndarray) -> list[Face]:
        height, width = image.shape[:2]
        if width < 16 or height < 16:
            return []
        self._detector.setInputSize((width, height))
        _, rows = self._detector.detect(image)
        if rows is None:
            return []
        faces = []
        for row in rows:
            landmarks = tuple((float(row[i]), float(row[i + 1])) for i in range(4, 14, 2))
            faces.append(Face(float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                              landmarks, float(row[14]), row.astype(np.float32)))
        return faces
