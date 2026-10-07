"""Asignación óptima entre trayectorias y detecciones."""

from __future__ import annotations

import numpy as np


def assign(score: np.ndarray) -> list[tuple[int, int]]:
    """Asignación óptima (húngara) que maximiza la afinidad total."""
    if score.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:                       # respaldo: voraz por afinidad
        pairs, used_rows, used_cols = [], set(), set()
        for flat in np.argsort(-score, axis=None):
            row, col = np.unravel_index(flat, score.shape)
            if score[row, col] <= 0:
                break
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            pairs.append((int(row), int(col)))
        return pairs
    rows, cols = linear_sum_assignment(-score)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if score[r, c] > 0]
