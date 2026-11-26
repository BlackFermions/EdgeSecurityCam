"""Trabajador del rostro: corre aparte para no frenar a YOLO ni al seguimiento.

La sesión pide "intenta reconocer a estas personas"; el trabajador abre el
vídeo HD una vez (ráfaga), toma la posición ACTUAL de cada persona en el
substream (pudo moverse mientras se abría el vídeo) e intenta todas con ese
mismo fotograma. Respeta un presupuesto de reconocimientos SFace por segundo.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

from src.faces.engine import FaceEngine


class FaceWorker(threading.Thread):
    def __init__(self, camera: str, engine: FaceEngine, grabber,
                 current_boxes: Callable[[set[str]], dict[str, tuple]],
                 sub_size: Callable[[], tuple[int, int] | None], params: dict) -> None:
        super().__init__(name=f"rostro-{camera}", daemon=True)
        self.camera = camera
        self._engine = engine
        self._grabber = grabber
        self._current_boxes = current_boxes
        self._sub_size = sub_size
        self._params = params
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._halt = threading.Event()
        self.results: queue.Queue = queue.Queue()      # (track_id, FaceAttempt, hd_ms)
        self._tokens = params["presupuesto_sface_s"]
        self._last_refill = time.monotonic()
        self.grabs = 0
        self.grab_failures = 0

    def request(self, track_id: str) -> None:
        with self._lock:
            self._pending.add(track_id)
        self._wake.set()

    def _refill(self) -> None:
        now = time.monotonic()
        rate = self._params["presupuesto_sface_s"]
        self._tokens = min(rate, self._tokens + (now - self._last_refill) * rate)
        self._last_refill = now

    def run(self) -> None:
        while not self._halt.is_set():
            self._wake.wait(0.5)
            self._wake.clear()
            with self._lock:
                pending, self._pending = self._pending, set()
            if not pending or self._halt.is_set():
                continue
            self._refill()
            if self._tokens < 1:
                with self._lock:                      # sin presupuesto: más tarde
                    self._pending |= pending
                self._halt.wait(1.0 / self._params["presupuesto_sface_s"])
                self._wake.set()
                continue
            frame, seconds = self._grabber.grab()
            self.grabs += 1
            size = self._sub_size()
            boxes = self._current_boxes(pending)
            if frame is None or size is None:
                self.grab_failures += 1
                for track_id in pending:
                    self.results.put((track_id, None, seconds * 1000))
                continue
            for track_id in pending:
                box = boxes.get(track_id)
                if box is None:                        # ya no se sigue
                    continue
                self._refill()
                attempt = self._engine.analyze(frame, box, size, self._params)
                if attempt.counts:
                    self._tokens -= 1
                self.results.put((track_id, attempt, seconds * 1000))

    def request_stop(self) -> None:
        self._halt.set()
        self._wake.set()
