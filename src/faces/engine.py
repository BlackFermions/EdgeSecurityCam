"""Motor de rostro: de la caja de una persona (substream) a una identidad.

Pasos de un intento, del más barato al más caro:

1. convertir la caja del substream a coordenadas del fotograma HD (con margen,
   porque la persona pudo moverse entre ambos fotogramas);
2. recortar la franja de la cabeza y buscar caras con YuNet (unos ms);
3. dos o más caras en la franja → caja unida: no se reconoce;
4. calidad de la mejor cara (tamaño, frontalidad, nitidez, brillo);
5. solo si pasa: SFace (~36 ms) y comparación con la galería.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from src.faces.base import FaceDetector, FaceRecognizer
from src.faces.gallery import Gallery, Match
from src.faces.quality import FaceQuality, clamp_box, expand_box, face_quality, head_band


@dataclass(frozen=True)
class FaceAttempt:
    result: str                  # coincide / bajo_umbral / ambiguo / sin_cara / calidad / union
    quality: FaceQuality | None
    faces_in_band: int
    match: Match | None
    problems: tuple[str, ...]
    crop: np.ndarray | None      # recorte de la cara (para calibrar)
    elapsed_ms: float

    @property
    def counts(self) -> bool:
        """¿Se llegó a comparar con la galería? (cuenta para votos e intentos)."""
        return self.match is not None


class FaceEngine:
    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer,
                 gallery: Gallery) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.gallery = gallery
        self._lock = threading.Lock()    # los modelos de OpenCV no son seguros entre hilos

    def analyze(self, hd_frame: np.ndarray, sub_box, sub_size: tuple[int, int],
                params: dict, crop_margin: float = 0.15) -> FaceAttempt:
        started = time.perf_counter()
        height, width = hd_frame.shape[:2]
        sx, sy = width / sub_size[0], height / sub_size[1]
        x1, y1, x2, y2 = sub_box
        person = expand_box((x1 * sx, y1 * sy, x2 * sx, y2 * sy), crop_margin)
        bx1, by1, bx2, by2 = clamp_box(head_band(person, params["franja_cabeza"],
                                                 params["franja_min_ancho"]), width, height)
        band = hd_frame[by1:by2, bx1:bx2]

        def done(result, quality=None, faces=0, match=None, problems=(), crop=None):
            return FaceAttempt(result, quality, faces, match, tuple(problems), crop,
                               (time.perf_counter() - started) * 1000)

        if band.size == 0:
            return done("sin_cara")
        with self._lock:
            faces = [f for f in self.detector.detect(band) if f.score >= params["yunet_puntuacion"]]
        if not faces:
            return done("sin_cara")
        if len(faces) >= 2:
            return done("union", faces=len(faces))
        # La cara se pasa a coordenadas del fotograma HD completo: la calidad,
        # el recorte y el alineamiento de SFace usan la imagen entera, no la
        # franja (que puede cortar la barbilla).
        face = faces[0].shifted(bx1, by1)
        if face.y + face.height >= by2 - 2 and by2 < height:
            # La cara toca el borde inferior de la franja (persona mirando
            # hacia abajo, 2026-10-13): se vuelve a buscar con la franja
            # ampliada para tener la cara completa.
            ey2 = min(height, int(face.y + face.height * 2.2))
            larger = hd_frame[by1:ey2, bx1:bx2]
            with self._lock:
                again = [f for f in self.detector.detect(larger)
                         if f.score >= params["yunet_puntuacion"]]
            if again:
                cx, cy = face.x + face.width / 2, face.y + face.height / 2
                face = min((f.shifted(bx1, by1) for f in again),
                           key=lambda f: abs(f.x + f.width / 2 - cx) + abs(f.y + f.height / 2 - cy))
        quality = face_quality(hd_frame, face)
        fx1, fy1, fx2, fy2 = clamp_box(face.box, width, height)
        crop = hd_frame[fy1:fy2, fx1:fx2].copy()
        problems = quality.problems(params["cara_min_px"], params["frontal_min"],
                                    params["nitidez_min"], params["brillo_min"])
        if face.y + face.height > height - 2 or face.y < 1:
            problems.append("cortada")
        if problems:
            return done("calidad", quality, 1, None, problems, crop)
        with self._lock:
            vector = self.recognizer.embed(hd_frame, face)
        if vector is None:
            return done("calidad", quality, 1, None, ("no alineable",), crop)
        match = self.gallery.match(vector, params["umbral"], params["margen_segundo"])
        return done(match.reason, quality, 1, match, (), crop)
