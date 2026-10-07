"""Vídeo RTSP de la cámara."""

from __future__ import annotations

import os
from urllib.parse import quote

import cv2


SUB_STREAM_PATH = "/stream2"
FFMPEG_OPTIONS = "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000"


def rtsp_url(host: str, username: str, password: str, port: int = 554,
             path: str = SUB_STREAM_PATH) -> str:
    return (f"rtsp://{quote(username, safe='')}:{quote(password, safe='')}"
            f"@{host}:{port}{path}")




def open_rtsp(url: str) -> cv2.VideoCapture:
    """Abre un RTSP por TCP con tiempos de espera acotados."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", FFMPEG_OPTIONS)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
