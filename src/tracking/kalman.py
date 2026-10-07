"""Filtro de Kalman de velocidad constante para el centro de cada caja."""

from __future__ import annotations

import numpy as np

from src.tracking.geometry import Box, box_from, center, size


class MotionModel:
    """Kalman de velocidad constante sobre el centro de la caja.

    El tamaño se suaviza aparte (media móvil). Las mediciones del flujo óptico
    se consideran más ruidosas que las de YOLO.
    """

    FLOW_NOISE = 8.0            # px, desviación de la medición por flujo
    YOLO_NOISE = 3.0            # px, desviación de la medición por YOLO
    ACCELERATION = 300.0        # px/s², cambios de velocidad de una persona

    def __init__(self, box: Box, now: float) -> None:
        cx, cy = center(box)
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([25.0, 25.0, 400.0, 400.0])
        self.width, self.height = size(box)
        self.t = now

    def predict(self, now: float) -> None:
        dt = now - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.ACCELERATION ** 2
        self.t = now

    def update(self, box: Box, now: float, noise: float, size_weight: float) -> None:
        self.predict(now)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        z = np.array(center(box))
        S = H @ self.P @ H.T + np.eye(2) * noise ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        width, height = size(box)
        self.width += size_weight * (width - self.width)
        self.height += size_weight * (height - self.height)

    def box(self) -> Box:
        return box_from(self.x[0], self.x[1], self.width, self.height)

    def extrapolate(self, now: float, horizon: float = 1.5) -> Box:
        """Posición esperada sin modificar el estado. La velocidad se aplica
        como máximo ``horizon`` s: una persona oculta suele detenerse."""
        dt = min(max(now - self.t, 0.0), horizon)
        return box_from(self.x[0] + self.x[2] * dt, self.x[1] + self.x[3] * dt,
                         self.width, self.height)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))
