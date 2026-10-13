"""Interfaces del reconocimiento facial.

- ``FaceDetector.detect(imagen) → [Face]``: dónde hay caras (hoy YuNet).
- ``FaceRecognizer.embed(imagen, cara) → huella``: vector normalizado que
  identifica a la persona (hoy SFace). Dos huellas se comparan con el
  producto punto (similitud coseno).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Face:
    x: float
    y: float
    width: float
    height: float
    landmarks: tuple[tuple[float, float], ...]   # ojo der., ojo izq., nariz, boca der., boca izq.
    score: float
    raw: np.ndarray                               # fila original del detector (para alinear)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.x + self.width), int(self.y + self.height))

    def shifted(self, dx: float, dy: float) -> "Face":
        """La misma cara en coordenadas de una imagen mayor (recorte → imagen)."""
        raw = self.raw.copy()
        raw[0:14:2] += dx
        raw[1:14:2] += dy
        return Face(self.x + dx, self.y + dy, self.width, self.height,
                    tuple((x + dx, y + dy) for x, y in self.landmarks), self.score, raw)


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Face]:
        ...


class FaceRecognizer(Protocol):
    name: str

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray | None:
        """Huella normalizada (norma 1) o None si no se pudo calcular."""
