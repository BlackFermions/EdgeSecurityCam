"""Verificación periódica en reposo: YOLO mira aunque la cámara no avise.

Cada ``interval`` segundos sin sesión abierta se toma un fotograma y pasa por
YOLO; si hay una persona, el nodo abre una sesión (alarma tipo ``reposo``).
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class IdleChecker(threading.Thread):
    """Cada ``interval`` s sin sesión abierta, un fotograma pasa por YOLO.

    - ``is_busy(camera)``: True si ya hay una sesión (no hace falta mirar).
    - ``on_person(camera, result)``: hay una persona con confianza suficiente.
    - ``cooldown``: tras un disparo, espera antes de volver a disparar; evita
      repetir sesiones por un objeto quieto que YOLO confunda con una persona.
    - ``on_status(camera, message)``: fallo al obtener fotogramas (una vez por
      caída) y su recuperación.
    """

    def __init__(self, camera: str, source, detector,
                 is_busy: Callable[[str], bool],
                 on_person: Callable[[str, object], None],
                 on_status: Callable[[str, str], None] = lambda *_: None,
                 interval: float = 10.0, cooldown: float = 60.0,
                 min_confidence: float = 0.50) -> None:
        super().__init__(name=f"reposo-{camera}", daemon=True)
        self.camera = camera
        self._source = source
        self._detector = detector
        self._is_busy = is_busy
        self._on_person = on_person
        self._on_status = on_status
        self.interval = interval
        self.cooldown = cooldown
        self.min_confidence = min_confidence
        self._halt = threading.Event()
        self._cooldown_until = 0.0
        self._failing_since: float | None = None
        self.checks = 0
        self.triggers = 0

    def check_once(self, now: float | None = None) -> bool:
        """Una verificación. Devuelve True si disparó una sesión."""
        now = time.monotonic() if now is None else now
        if now < self._cooldown_until or self._is_busy(self.camera):
            return False
        frame = self._source.grab()
        if frame is None:
            if self._failing_since is None:
                self._failing_since = now
                self._on_status(self.camera, "verificación en reposo: no se pudo "
                                             "obtener un fotograma de la cámara")
            return False
        if self._failing_since is not None:
            self._on_status(self.camera, "verificación en reposo recuperada tras "
                                         f"{now - self._failing_since:.0f} s")
            self._failing_since = None
        self.checks += 1
        result = self._detector.detect(frame)
        if not any(confidence >= self.min_confidence for confidence in result.confidences):
            return False
        if self._is_busy(self.camera):            # llegó una alarma mientras tanto
            return False
        self.triggers += 1
        self._cooldown_until = now + self.cooldown
        self._on_person(self.camera, result)
        return True

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            try:
                self.check_once()
            except Exception as error:            # noqa: BLE001 - el hilo no debe morir
                self._on_status(self.camera, f"verificación en reposo: error "
                                             f"{type(error).__name__}")

    def request_stop(self) -> None:
        self._halt.set()
