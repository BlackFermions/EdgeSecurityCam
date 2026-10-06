"""Detector YOLO de personas y veredicto de cada alarma.

El uso del detector durante una alarma (sesión, seguimiento y salvaguardas)
está en ``src.analysis``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np


PERSON_CLASS = 0
SUB_STREAM_PATH = "/stream2"


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


@dataclass
class VerificationSummary:
    camera: str
    verdict: str | None = None          # "confirmed", "discarded" o None
    max_people: int = 0
    best_confidence: float = 0.0
    frames: int = 0
    first_person_after: float | None = None     # s desde el inicio de la alarma
    snapshot: Path | None = None
    error: str | None = None
    alarm_id: str | None = None


class VerificationTracker:
    """Decide persona confirmada o descartada a partir de inferencias sucesivas.

    Confirma con ``confirm_frames`` fotogramas seguidos con persona; descarta si
    pasan ``decide_seconds`` sin confirmar. Una alarma descartada puede pasar a
    confirmada si alguien aparece después (alarmas largas).
    """

    def __init__(self, camera: str, started_at: float, confirm_frames: int = 2,
                 decide_seconds: float = 4.0) -> None:
        if confirm_frames < 1 or decide_seconds <= 0:
            raise ValueError("parámetros de verificación no válidos")
        self.summary = VerificationSummary(camera)
        self.started_at = started_at
        self.confirm_frames = confirm_frames
        self.decide_seconds = decide_seconds
        self._streak = 0
        self.best_frame_key: tuple[int, float] = (-1, 0.0)

    def add(self, result: FrameResult) -> str | None:
        """Registra una inferencia; devuelve el veredicto si acaba de cambiar."""
        summary = self.summary
        summary.frames += 1
        summary.max_people = max(summary.max_people, result.people)
        if result.confidences:
            summary.best_confidence = max(summary.best_confidence, max(result.confidences))
            if summary.first_person_after is None:
                summary.first_person_after = max(0.0, result.at - self.started_at)
        self._streak = self._streak + 1 if result.people else 0
        if summary.verdict != "confirmed" and self._streak >= self.confirm_frames:
            summary.verdict = "confirmed"
            return "confirmed"
        return self.check_timeout(result.at)

    def check_timeout(self, now: float) -> str | None:
        if self.summary.verdict is None and now - self.started_at >= self.decide_seconds:
            self.summary.verdict = "discarded"
            return "discarded"
        return None

    def is_better_frame(self, result: FrameResult) -> bool:
        """La mejor foto es la de más personas y, a igualdad, mayor confianza."""
        key = (result.people, max(result.confidences, default=0.0))
        if key > self.best_frame_key:
            self.best_frame_key = key
            return True
        return False


class PersonDetector:
    """YOLO restringido a la clase persona; se carga y precalienta una vez."""

    def __init__(self, model: str, confidence: float = 0.40,
                 image_size: int = 640, device: str = "cpu",
                 weak_confidence: float = 0.15) -> None:
        from ultralytics import YOLO  # import diferido: es pesado

        self._model = YOLO(model)
        self._confidence = confidence
        self._weak_confidence = min(weak_confidence, confidence)
        self._image_size = image_size
        self._device = device
        self._lock = threading.Lock()
        self.detect(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> FrameResult:
        with self._lock:
            prediction = self._model.predict(
                frame, conf=self._weak_confidence, imgsz=self._image_size,
                classes=[PERSON_CLASS], device=self._device, verbose=False,
            )[0]
        strong: list[tuple[float, tuple[int, int, int, int]]] = []
        weak: list[tuple[float, tuple[int, int, int, int]]] = []
        for box in prediction.boxes:
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            (strong if confidence >= self._confidence else weak).append(
                (confidence, (x1, y1, x2, y2)))
        return FrameResult(time.monotonic(),
                           tuple(c for c, _ in strong), tuple(b for _, b in strong),
                           tuple(c for c, _ in weak), tuple(b for _, b in weak))


def rtsp_url(host: str, username: str, password: str, port: int = 554,
             path: str = SUB_STREAM_PATH) -> str:
    return (f"rtsp://{quote(username, safe='')}:{quote(password, safe='')}"
            f"@{host}:{port}{path}")


def annotate(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    image = frame.copy()
    for (x1, y1, x2, y2), confidence in zip(result.boxes, result.confidences):
        cv2.rectangle(image, (x1, y1), (x2, y2), (50, 220, 70), 2)
        cv2.putText(image, f"persona {confidence:.0%}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 220, 70), 2)
    return image
