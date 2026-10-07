"""Flujo óptico Lucas-Kanade con escala (Median Flow)."""

from __future__ import annotations

import cv2
import numpy as np

from src.tracking.geometry import Box, box_from, center, size


FLOW_SCALE = 0.5            # el flujo y la puerta trabajan a media resolución


class FlowTracker:
    """Sigue una caja moviendo puntos característicos con Lucas-Kanade.

    Trabaja sobre imágenes en gris a escala ``FLOW_SCALE``. Usa verificación
    ida y vuelta: los puntos que no regresan a su origen se descartan. Estima
    desplazamiento (mediana) y escala (mediana de cambios de distancia al
    centro de los puntos, como Median Flow). Si quedan muy pocos puntos (zona
    oscura, oclusión) la trayectoria se marca perdida y YOLO decide.
    """

    MAX_POINTS = 30
    MIN_POINTS = 5
    MAX_FB_ERROR = 1.5
    MAX_SCALE_STEP = 0.08       # cambio de tamaño máximo por fotograma

    def __init__(self, gray: np.ndarray, box: Box) -> None:
        self.box = box
        self.lost = False
        self.moved = 0.0
        self._points = self._seed(gray, box)
        self._initial = len(self._points)
        if self._initial < self.MIN_POINTS:
            self.lost = True

    @staticmethod
    def _seed(gray: np.ndarray, box: Box) -> np.ndarray:
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = (int(round(v * FLOW_SCALE)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return np.empty((0, 1, 2), np.float32)
        points = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], FlowTracker.MAX_POINTS,
                                         0.01, 3)
        if points is None:
            return np.empty((0, 1, 2), np.float32)
        return points.astype(np.float32) + np.float32([x1, y1])

    def update(self, previous: np.ndarray, current: np.ndarray) -> bool:
        """Mueve y escala la caja; devuelve False si la trayectoria se perdió."""
        if self.lost or len(self._points) == 0:
            self.lost = True
            return False
        params = dict(winSize=(15, 15), maxLevel=2)
        new, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, self._points,
                                                  None, **params)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new,
                                                        None, **params)
        error = np.linalg.norm((self._points - back).reshape(-1, 2), axis=1)
        good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                & (error < self.MAX_FB_ERROR))
        if good.sum() < max(self.MIN_POINTS, 0.3 * self._initial):
            self.lost = True
            return False
        old_points = self._points.reshape(-1, 2)[good]
        new_points = new.reshape(-1, 2)[good]
        shift = np.median(new_points - old_points, axis=0)
        old_spread = np.linalg.norm(old_points - np.median(old_points, axis=0), axis=1)
        new_spread = np.linalg.norm(new_points - np.median(new_points, axis=0), axis=1)
        valid = old_spread > 2.0
        scale = float(np.median(new_spread[valid] / old_spread[valid])) if valid.any() else 1.0
        scale = float(np.clip(scale, 1 - self.MAX_SCALE_STEP, 1 + self.MAX_SCALE_STEP))
        dx, dy = float(shift[0]) / FLOW_SCALE, float(shift[1]) / FLOW_SCALE
        cx, cy = center(self.box)
        width, height = size(self.box)
        self.box = box_from(cx + dx, cy + dy, width * scale, height * scale)
        self.moved = float(np.hypot(dx, dy))
        self._points = new_points.reshape(-1, 1, 2)
        return True
