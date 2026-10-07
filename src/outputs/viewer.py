"""Ventana de depuración: muestra en vivo lo que decide el análisis.

Las sesiones (otros hilos) publican el fotograma ya anotado; el hilo
principal es el único que llama a ``cv2.imshow``, como exige HighGUI.

Fuera de las sesiones el análisis no abre el vídeo (reposo). Para que la
ventana muestre igual la cámara, la vista abre su propia conexión RTSP: solo
existe con ``--ver`` y no cambia el comportamiento del análisis.

OpenCV no dibuja acentos: los textos sobre la imagen van sin tildes.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import cv2
import numpy as np

from src.sources.video import open_rtsp


GREEN = (60, 210, 60)        # posición corregida por YOLO en este fotograma
CYAN = (230, 200, 40)        # posición estimada con flujo óptico
RED = (40, 40, 230)          # trayectoria perdida
YELLOW = (40, 220, 240)
WHITE = (240, 240, 240)
GRAY = (150, 150, 150)       # en gracia: oculta en el interior o saliendo
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(image: np.ndarray, text: str, origin: tuple[int, int], color,
           scale: float = 0.5) -> None:
    (width, height), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = origin
    cv2.rectangle(image, (x, y - height - 6), (x + width + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y - 3), FONT, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, tracks, reason: str | None, motion_outside: bool,
                 stats, alarm_active: bool, verdict_pending: bool) -> np.ndarray:
    image = frame.copy()
    for track in tracks:
        lost = track.flow is not None and track.flow.lost
        color = RED if lost else (GREEN if reason is not None else CYAN)
        source = "perdida" if lost else ("YOLO" if reason is not None else "flujo")
        if track.exit_kind == "interior":          # oculta: se sigue esperando
            color, source = GRAY, "oculta"
        elif track.exit_kind == "borde":           # en gracia junto al borde
            color, source = GRAY, "saliendo"
        x1, y1, x2, y2 = (int(v) for v in track.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1 if track.exit_kind else 2)
        _label(image, f"{track.code.rsplit('-', 2)[-2]}-{track.code.rsplit('-', 1)[-1]}"
                      f" {track.confidence:.2f} {source}", (x1, max(16, y1)), color, 0.45)

    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 24), (0, 0, 0), -1)
    state = ("verificando" if verdict_pending else
             "alarma activa" if alarm_active else "siguiendo")
    cv2.putText(image, f"{datetime.now():%H:%M:%S}  {state}  personas {len(tracks)}  "
                       f"fotogramas {stats.frames}  YOLO {stats.yolo_runs}  "
                       f"sin YOLO {stats.savings:.0%}",
                (6, 17), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
    if reason is not None:
        _label(image, f"YOLO: {reason}", (6, height - 8), YELLOW, 0.55)
    else:
        _label(image, "flujo optico (sin YOLO)", (6, height - 8), CYAN, 0.55)
    if motion_outside:
        cv2.rectangle(image, (1, 25), (width - 2, height - 2), RED, 2)
        _label(image, "movimiento nuevo", (width - 160, height - 8), RED, 0.5)
    return image


class CameraPreview(threading.Thread):
    """Lectura continua del substream solo para mostrarlo en la ventana."""

    def __init__(self, camera: str, url: str) -> None:
        super().__init__(name=f"vista-{camera}", daemon=True)
        self._url = url
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._at = 0.0

    def latest(self, max_age: float = 3.0) -> np.ndarray | None:
        with self._lock:
            if self._frame is None or time.monotonic() - self._at > max_age:
                return None
            return self._frame

    def run(self) -> None:
        while not self._halt.is_set():
            capture = open_rtsp(self._url)
            while not self._halt.is_set() and capture.isOpened():
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                with self._lock:
                    self._frame, self._at = frame, time.monotonic()
            capture.release()
            self._halt.wait(2.0)

    def request_stop(self) -> None:
        self._halt.set()


class LiveViewer:
    """Una ventana por cámara; se refresca desde el hilo principal."""

    def __init__(self, cameras: list[str], urls: dict[str, str] | None = None,
                 scale: float = 1.5) -> None:
        self._cameras = cameras
        self._scale = scale
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray | None] = {camera: None for camera in cameras}
        self._idle_since: dict[str, float] = {camera: time.monotonic() for camera in cameras}
        self._previews = {camera: CameraPreview(camera, url)
                          for camera, url in (urls or {}).items()}
        for preview in self._previews.values():
            preview.start()

    def publish(self, camera: str, image: np.ndarray) -> None:
        with self._lock:
            self._frames[camera] = image

    def idle(self, camera: str) -> None:
        with self._lock:
            self._frames[camera] = None
            self._idle_since[camera] = time.monotonic()

    def _placeholder(self, camera: str) -> np.ndarray:
        """Reposo: la cámara en vivo (si hay vista previa) con el aviso."""
        seconds = time.monotonic() - self._idle_since[camera]
        preview = self._previews.get(camera)
        live = preview.latest() if preview is not None else None
        if live is None:
            image = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(image, "conectando con la camara...", (40, 180),
                        FONT, 0.7, WHITE, 1, cv2.LINE_AA)
        else:
            image = live.copy()
        height, width = image.shape[:2]
        cv2.rectangle(image, (0, 0), (width, 24), (0, 0, 0), -1)
        cv2.putText(image, f"{datetime.now():%H:%M:%S}  REPOSO: YOLO dormido, esperando "
                           f"alarma de la camara ({seconds:.0f} s)",
                    (6, 17), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
        _label(image, "q o Esc: cerrar", (6, height - 8), (160, 160, 160), 0.5)
        return image

    def refresh(self) -> bool:
        """Dibuja las ventanas; devuelve False si el usuario pidió cerrar."""
        for camera in self._cameras:
            with self._lock:
                image = self._frames[camera]
            if image is None:
                image = self._placeholder(camera)
            if self._scale != 1.0:
                image = cv2.resize(image, None, fx=self._scale, fy=self._scale,
                                   interpolation=cv2.INTER_LINEAR)
            title = f"CamDetector {camera}"
            cv2.imshow(title, image)
        key = cv2.waitKey(40) & 0xFF
        if key in (ord("q"), 27):
            return False
        for camera in self._cameras:
            try:
                if cv2.getWindowProperty(f"CamDetector {camera}", cv2.WND_PROP_VISIBLE) < 1:
                    return False
            except cv2.error:
                return False
        return True

    def close(self) -> None:
        for preview in self._previews.values():
            preview.request_stop()
        cv2.destroyAllWindows()
