"""Dónde buscar la cara y si vale la pena reconocerla (todo casi gratis).

- ``head_band``: franja superior de la caja de la persona, a todo su ancho.
  Con dos personas de altura parecida en una misma caja, las dos caras caen
  en la franja (y YuNet encuentra 2: caja unida).
- ``face_quality``: tamaño, frontalidad (simetría de ojos y nariz), nitidez
  (varianza del Laplaciano) y brillo. SFace, lo caro, solo corre con caras
  que pasan estos filtros.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.faces.base import Face


def head_band(box: tuple[float, float, float, float], fraction: float,
              min_width_fraction: float = 0.0) -> tuple[int, int, int, int]:
    """Franja superior de la caja. Su altura es ``fraction`` del alto, pero
    al menos ``min_width_fraction`` del ancho: en personas sentadas o
    cortadas por el borde de la imagen la caja es baja y el 30 % del alto
    cortaría la cabeza; la cabeza escala con el cuerpo, no con lo visible.
    (Medido el 2026-10-13: persona cortada por el borde inferior, franja del
    30 % por debajo de los ojos.)"""
    x1, y1, x2, y2 = box
    height = max((y2 - y1) * fraction, (x2 - x1) * min_width_fraction)
    return int(x1), int(y1), int(x2), int(min(y2, y1 + height))


def clamp_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))


def expand_box(box, fraction: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * fraction, (y2 - y1) * fraction
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


@dataclass(frozen=True)
class FaceQuality:
    size_px: float
    frontal: float          # 1 = de frente, 0 = de perfil
    sharpness: float
    brightness: float

    def problems(self, min_px: float, min_frontal: float, min_sharpness: float,
                 min_brightness: float) -> list[str]:
        issues = []
        if self.size_px < min_px:
            issues.append("pequeña")
        if self.frontal < min_frontal:
            issues.append("de perfil")
        if self.sharpness < min_sharpness:
            issues.append("borrosa")
        if self.brightness < min_brightness:
            issues.append("oscura")
        return issues


def frontality(face: Face) -> float:
    (rx, _), (lx, _), (nx, _) = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    half_eyes = abs(lx - rx) / 2
    if half_eyes < 1:
        return 0.0
    offset = abs(nx - (rx + lx) / 2) / half_eyes
    return float(np.clip(1.0 - offset, 0.0, 1.0))


def face_quality(image: np.ndarray, face: Face) -> FaceQuality:
    x1, y1, x2, y2 = clamp_box(face.box, image.shape[1], image.shape[0])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return FaceQuality(face.width, frontality(face), 0.0, 0.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Nitidez a escala fija: comparable entre caras de distinto tamaño.
    normalized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    return FaceQuality(float(face.width), frontality(face), sharpness, float(gray.mean()))
