"""Reconocedor SFace (OpenCV Zoo): huella de 128 valores; es la parte cara (~36 ms)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.faces.base import Face


class SFaceRecognizer:
    name = "sface_2021dec"

    def __init__(self, model: Path) -> None:
        self._recognizer = cv2.FaceRecognizerSF.create(str(model), "")

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray | None:
        try:
            aligned = self._recognizer.alignCrop(image, face.raw)
            vector = np.asarray(self._recognizer.feature(aligned), np.float32).reshape(-1)
        except cv2.error:
            return None
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else None
