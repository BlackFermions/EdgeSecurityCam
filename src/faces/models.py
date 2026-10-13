"""Modelos de rostro de OpenCV Zoo (Apache-2.0): descarga y verificación."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path


YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
MODEL_SPECS = {
    YUNET: ("https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    SFACE: ("https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}


class FaceModelError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_models(model_dir: Path) -> dict[str, Path]:
    """Devuelve las rutas de YuNet y SFace; los descarga y verifica si faltan."""
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, (url, expected) in MODEL_SPECS.items():
        path = model_dir / filename
        if not (path.is_file() and sha256(path) == expected):
            partial = path.with_suffix(path.suffix + ".part")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "CamDetector/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, \
                        partial.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
            except (OSError, urllib.error.URLError) as error:
                partial.unlink(missing_ok=True)
                raise FaceModelError(f"no se pudo descargar {filename}") from error
            if sha256(partial) != expected:
                partial.unlink(missing_ok=True)
                raise FaceModelError(f"{filename}: la verificación SHA-256 falló")
            os.replace(partial, path)
        paths[filename] = path
    return paths
