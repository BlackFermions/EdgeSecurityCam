"""Interfaz del detector de personas y su resultado.

Cualquier detector (YOLO con PyTorch hoy; ONNX o una NPU mañana) debe
devolver un ``FrameResult`` desde ``detect(frame)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FrameResult:
    at: float                                       # time.monotonic()
    confidences: tuple[float, ...]                  # detecciones fuertes
    boxes: tuple[tuple[int, int, int, int], ...]
    # Detecciones débiles: solo mantienen viva una trayectoria existente;
    # nunca crean personas ni cuentan para el veredicto (idea de ByteTrack).
    weak_confidences: tuple[float, ...] = ()
    weak_boxes: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def people(self) -> int:
        return len(self.confidences)


class PersonDetector(Protocol):
    def detect(self, frame: np.ndarray) -> FrameResult:
        """Personas en el fotograma: fuertes (cuentan) y débiles (solo mantienen)."""
