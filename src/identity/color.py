"""Huella de apariencia clásica: histograma Hue-Saturation del torso."""

from __future__ import annotations

import cv2
import numpy as np

from src.tracking.geometry import Box


def color_signature(frame: np.ndarray | None, box: Box) -> np.ndarray | None:
    """Histograma Hue-Saturation del torso, o None si no hay color útil.

    Se usa la franja central (20–60% de la altura, 60% del ancho) para evitar
    fondo, cabeza y piernas. Con IR, oscuridad o colores apagados casi ningún
    píxel tiene saturación y la apariencia no se usa.
    """
    if frame is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    tx1, tx2 = int(max(0, x1 + 0.2 * bw)), int(min(width, x2 - 0.2 * bw))
    ty1, ty2 = int(max(0, y1 + 0.2 * bh)), int(min(height, y1 + 0.6 * bh))
    if tx2 - tx1 < 6 or ty2 - ty1 < 6:
        return None
    hsv = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    if cv2.countNonZero(mask) < 0.15 * mask.size:
        return None
    histogram = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, 1.0, 0.0, cv2.NORM_L1)
    return histogram


def appearance_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """1 = mismo color, 0 = nada en común; None si alguno no tiene color."""
    if a is None or b is None:
        return None
    return 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def blend_signature(old: np.ndarray | None, new: np.ndarray | None,
                    weight: float = 0.3) -> np.ndarray | None:
    if new is None:
        return old
    if old is None:
        return new
    mixed = (1 - weight) * old + weight * new
    cv2.normalize(mixed, mixed, 1.0, 0.0, cv2.NORM_L1)
    return mixed




class ColorHistogramAppearance:
    """Implementación de ``AppearanceModel`` con el histograma de color."""

    name = "color_torso"

    def signature(self, frame: np.ndarray | None, box: Box) -> np.ndarray | None:
        return color_signature(frame, box)

    def similarity(self, a, b) -> float | None:
        return appearance_similarity(a, b)

    def blend(self, old, new, weight: float = 0.3):
        return blend_signature(old, new, weight)
