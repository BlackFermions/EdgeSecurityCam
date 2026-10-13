"""Galería de personas conocidas: varias huellas por persona.

A diferencia de CameraCaptor (una plantilla promedio por persona), se guarda
una huella por foto y la similitud con una persona es la MÁXIMA entre sus
huellas: mezclar fotos de webcam, WhatsApp y de la cámara no "diluye" la
plantilla. Las huellas se guardan en caché (data/galeria_cache.npz) y solo
se recalculan las fotos nuevas o modificadas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.faces.base import FaceDetector, FaceRecognizer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Match:
    name: str | None          # None: no supera el umbral o es ambiguo
    score: float              # similitud con la persona más parecida
    best: str | None          # persona más parecida (aunque no alcance el umbral)
    second: str | None
    second_score: float
    reason: str               # "coincide", "bajo_umbral", "ambiguo", "galeria_vacia"


class Gallery:
    def __init__(self) -> None:
        self.embeddings: dict[str, np.ndarray] = {}     # persona -> matriz (n, 128)
        self.skipped: list[str] = []

    @property
    def people(self) -> list[str]:
        return sorted(self.embeddings)

    @classmethod
    def load(cls, folder: Path, detector: FaceDetector, recognizer: FaceRecognizer,
             cache: Path | None = None, max_side: int = 640) -> "Gallery":
        gallery = cls()
        if not folder.is_dir():
            return gallery
        cached: dict[str, np.ndarray] = {}
        if cache is not None and cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                keys = json.loads(str(data["keys"]))
                if json.loads(str(data["model"])) == recognizer.name:
                    cached = {key: data["vectors"][i] for i, key in enumerate(keys)}
        used: dict[str, np.ndarray] = {}
        for person in sorted(p for p in folder.iterdir() if p.is_dir()):
            vectors = []
            for photo in sorted(person.iterdir()):
                if photo.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                stat = photo.stat()
                key = f"{person.name}/{photo.name}|{stat.st_size}|{int(stat.st_mtime)}"
                vector = cached.get(key)
                if vector is None:
                    vector = cls._embed_photo(photo, detector, recognizer, max_side)
                if vector is None:
                    gallery.skipped.append(f"{person.name}/{photo.name}")
                    continue
                used[key] = vector
                vectors.append(vector)
            if vectors:
                # Además de cada foto, el promedio normalizado: con fotos muy
                # distintas entre sí a veces se parece más que cualquiera sola.
                mean = np.mean(vectors, axis=0)
                norm = float(np.linalg.norm(mean))
                if len(vectors) > 1 and norm > 1e-8:
                    vectors = [*vectors, mean / norm]
                gallery.embeddings[person.name] = np.vstack(vectors)
        if cache is not None and used:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, keys=json.dumps(list(used)), model=json.dumps(recognizer.name),
                     vectors=np.vstack(list(used.values())))
        return gallery

    @staticmethod
    def _embed_photo(photo: Path, detector: FaceDetector, recognizer: FaceRecognizer,
                     max_side: int) -> np.ndarray | None:
        image = cv2.imdecode(np.fromfile(str(photo), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        scale = min(1.0, max_side / max(image.shape[:2]))
        small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        faces = detector.detect(small)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.width * f.height)
        if scale < 1:   # alinear sobre la foto original, con más detalle
            raw = face.raw.copy()
            raw[:14] /= scale
            face = type(face)(face.x / scale, face.y / scale, face.width / scale,
                              face.height / scale,
                              tuple((x / scale, y / scale) for x, y in face.landmarks),
                              face.score, raw)
        return recognizer.embed(image, face)

    def match(self, vector: np.ndarray, threshold: float, margin: float) -> Match:
        if not self.embeddings:
            return Match(None, 0.0, None, None, 0.0, "galeria_vacia")
        scores = sorted(((float(np.max(matrix @ vector)), name)
                         for name, matrix in self.embeddings.items()), reverse=True)
        best_score, best = scores[0]
        second_score, second = scores[1] if len(scores) > 1 else (0.0, None)
        if best_score < threshold:
            return Match(None, best_score, best, second, second_score, "bajo_umbral")
        if second is not None and best_score - second_score < margin:
            return Match(None, best_score, best, second, second_score, "ambiguo")
        return Match(best, best_score, best, second, second_score, "coincide")
