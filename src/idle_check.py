"""Verificación periódica en reposo: YOLO mira aunque la cámara no avise.

La detección de la cámara falla en zonas oscuras o con personas quietas: si
la cámara no avisa, el análisis no se despierta. Para no depender solo de
ella, cada ``interval`` segundos (sin sesión abierta) se toma **un**
fotograma y se pasa por YOLO. Si aparece una persona, el nodo abre una sesión
como si hubiera llegado una alarma (tipo ``reposo``).

El fotograma se obtiene de la forma más barata disponible:

1. foto suelta ONVIF (``GetSnapshotUri``): una petición HTTP, sin vídeo;
2. respaldo: abrir el RTSP, leer unos fotogramas y cerrarlo.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

import cv2
import numpy as np

from src.camera_events import CameraEventError, OnvifEventClient


MEDIA_NS = "http://www.onvif.org/ver10/media/wsdl"
SCHEMA_NS = "http://www.onvif.org/ver10/schema"


class SnapshotSource:
    """Obtiene un fotograma suelto de la cámara: foto ONVIF o, si no, RTSP."""

    def __init__(self, host: str, username: str, password: str, rtsp_url: str,
                 port: int = 80, timeout: float = 6.0) -> None:
        self.host = host
        self._username = username
        self._password = password
        self._rtsp_url = rtsp_url
        self._media_url = f"http://{host}:{port}/onvif/Media"
        self._timeout = timeout
        self._snapshot_uri: str | None = None
        self._snapshot_checked = False
        self.method: str | None = None          # "onvif" o "rtsp" (último usado)

    # --- foto ONVIF -------------------------------------------------------

    def _discover_snapshot_uri(self) -> str | None:
        client = OnvifEventClient(self.host, self._username, self._password)
        root = client.call(self._media_url, f'<GetProfiles xmlns="{MEDIA_NS}"/>')
        tokens = [element.attrib.get("token") for element in root.iter(f"{{{MEDIA_NS}}}Profiles")]
        for token in tokens:
            if not token:
                continue
            reply = client.call(
                self._media_url,
                f'<GetSnapshotUri xmlns="{MEDIA_NS}"><ProfileToken>{token}</ProfileToken>'
                "</GetSnapshotUri>")
            uri = next((element.text for element in reply.iter(f"{{{SCHEMA_NS}}}Uri")
                        if element.text), None)
            if uri:
                return uri.strip()
        return None

    def _fetch_snapshot(self, uri: str) -> np.ndarray | None:
        manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        manager.add_password(None, uri, self._username, self._password)
        opener = urllib.request.build_opener(urllib.request.HTTPDigestAuthHandler(manager),
                                             urllib.request.HTTPBasicAuthHandler(manager))
        with opener.open(uri, timeout=self._timeout) as response:
            data = response.read(8_000_000)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return image

    # --- respaldo RTSP ------------------------------------------------------

    def _grab_rtsp(self, frames: int = 8) -> np.ndarray | None:
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000",
        )
        capture = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)
        image = None
        try:
            # Los primeros fotogramas pueden llegar incompletos hasta el
            # siguiente fotograma clave: se toma el último de unos pocos.
            for _ in range(frames):
                ok, frame = capture.read()
                if ok and frame is not None:
                    image = frame
        finally:
            capture.release()
        return image

    def grab(self) -> np.ndarray | None:
        if not self._snapshot_checked:
            self._snapshot_checked = True
            try:
                self._snapshot_uri = self._discover_snapshot_uri()
            except (CameraEventError, OSError):
                # La cámara no respondió: se vuelve a preguntar la próxima vez.
                # Solo se renuncia a la foto ONVIF si respondió sin ofrecerla.
                self._snapshot_uri = None
                self._snapshot_checked = False
        if self._snapshot_uri:
            try:
                image = self._fetch_snapshot(self._snapshot_uri)
            except (urllib.error.URLError, OSError, cv2.error):
                image = None
            if image is not None:
                self.method = "onvif"
                return image
        image = self._grab_rtsp()
        self.method = "rtsp" if image is not None else None
        return image


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
