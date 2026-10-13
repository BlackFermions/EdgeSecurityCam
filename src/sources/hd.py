"""Fotograma del vídeo principal (HD) en ráfaga.

La foto ONVIF de la W51-TY sale solo a 640x360 y su resolución no se puede
cambiar (firmware: res_cnt = 0). La cara se toma del vídeo principal
(2880x1616), abierto solo cuando alguien necesita reconocimiento: abrirlo y
obtener el primer fotograma cuesta ~1 s y ~0,35 s de CPU; tenerlo abierto
costaría ~90 % de un núcleo (medido el 2026-10-13).
"""

from __future__ import annotations

import time

import numpy as np

from src.sources.video import open_rtsp


class HdFrameGrabber:
    def __init__(self, url: str, max_reads: int = 30) -> None:
        self._url = url
        self._max_reads = max_reads

    def grab(self) -> tuple[np.ndarray | None, float]:
        """Abre el vídeo HD, toma el primer fotograma completo y lo cierra.

        Devuelve (fotograma o None, segundos que tardó).
        """
        started = time.monotonic()
        capture = open_rtsp(self._url)
        frame = None
        try:
            for _ in range(self._max_reads):
                ok, image = capture.read()
                if ok and image is not None:
                    frame = image
                    break
        finally:
            capture.release()
        return frame, time.monotonic() - started
