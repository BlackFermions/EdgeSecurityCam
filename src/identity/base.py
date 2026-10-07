"""Interfaz de la identidad corporal (apariencia).

Cualquier implementación (histograma de color hoy; Re-ID con OSNet mañana)
debe ofrecer estos tres métodos. El tracker solo depende de esta interfaz.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from src.tracking.geometry import Box


class AppearanceModel(Protocol):
    name: str

    def signature(self, frame: np.ndarray | None, box: Box) -> Any | None:
        """Huella de la persona en esa caja, o None si no se puede calcular."""

    def similarity(self, a: Any | None, b: Any | None) -> float | None:
        """1 = misma apariencia, 0 = nada en común; None si falta alguna huella."""

    def blend(self, old: Any | None, new: Any | None, weight: float = 0.3) -> Any | None:
        """Actualiza la huella acumulada de una trayectoria."""
