"""Fotograma suelto de la cámara: foto ONVIF (GetSnapshotUri) o RTSP."""

from __future__ import annotations

import urllib.error
import urllib.request

import cv2
import numpy as np

from src.sources.camera_events import CameraEventError, OnvifEventClient
from src.sources.video import open_rtsp


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
        capture = open_rtsp(self._rtsp_url)
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
