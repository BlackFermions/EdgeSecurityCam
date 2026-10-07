"""Veredicto de cada alarma: persona confirmada o descartada."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.detection.base import FrameResult


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

    Confirma con ``confirm_frames`` fotogramas seguidos con persona. Descarta si
    pasan ``decide_seconds`` sin ver a nadie; si YOLO ya vio a alguien al menos
    una vez (p. ej. con poca luz no lo ve en dos fotogramas seguidos), espera
    hasta ``seen_decide_seconds`` antes de descartar, para no dar un veredicto
    que luego haya que corregir. Una alarma descartada aún puede pasar a
    confirmada si alguien aparece después (alarmas largas).
    """

    def __init__(self, camera: str, started_at: float, confirm_frames: int = 2,
                 decide_seconds: float = 4.0, seen_decide_seconds: float = 8.0) -> None:
        if confirm_frames < 1 or not 0 < decide_seconds <= seen_decide_seconds:
            raise ValueError("parámetros de verificación no válidos")
        self.summary = VerificationSummary(camera)
        self.started_at = started_at
        self.confirm_frames = confirm_frames
        self.decide_seconds = decide_seconds
        self.seen_decide_seconds = seen_decide_seconds
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
        limit = (self.seen_decide_seconds if self.summary.first_person_after is not None
                 else self.decide_seconds)
        if self.summary.verdict is None and now - self.started_at >= limit:
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
