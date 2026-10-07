"""Puerta de movimiento: diferencia de fotogramas a baja resolución."""

from __future__ import annotations

import cv2
import numpy as np

from src.tracking.flow import FLOW_SCALE
from src.tracking.geometry import Box


class MotionGate:
    """Diferencia de fotogramas a baja resolución.

    Indica si hay movimiento fuera de las cajas seguidas (posible persona
    nueva) y si las personas seguidas se mueven.
    """

    def __init__(self, threshold: int = 25, min_fraction: float = 0.004,
                 margin: float = 0.15) -> None:
        self.threshold = threshold
        self.min_fraction = min_fraction
        self.margin = margin
        self._previous: np.ndarray | None = None

    def update(self, gray: np.ndarray, boxes: list[Box]) -> tuple[bool, bool]:
        """Devuelve (movimiento fuera de las cajas, movimiento dentro)."""
        small = cv2.GaussianBlur(cv2.resize(gray, (160, 90),
                                            interpolation=cv2.INTER_AREA), (5, 5), 0)
        previous, self._previous = self._previous, small
        if previous is None:
            return False, False
        mask = cv2.absdiff(small, previous) > self.threshold
        inside = np.zeros_like(mask)
        sx = 160 / (gray.shape[1] / FLOW_SCALE)
        sy = 90 / (gray.shape[0] / FLOW_SCALE)
        for x1, y1, x2, y2 in boxes:
            mx, my = (x2 - x1) * self.margin, (y2 - y1) * self.margin
            inside[max(0, int((y1 - my) * sy)):int((y2 + my) * sy) + 1,
                   max(0, int((x1 - mx) * sx)):int((x2 + mx) * sx) + 1] = True
        minimum = self.min_fraction * mask.size
        outside_count = int((mask & ~inside).sum())
        inside_count = int((mask & inside).sum())
        return outside_count >= minimum, inside_count >= minimum / 2
