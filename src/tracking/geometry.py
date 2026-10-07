"""Geometría de cajas (x1, y1, x2, y2 en píxeles)."""

from __future__ import annotations

Box = tuple[float, float, float, float]       # x1, y1, x2, y2 en píxeles


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def size(box: Box) -> tuple[float, float]:
    return max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)


def box_from(cx: float, cy: float, width: float, height: float) -> Box:
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
