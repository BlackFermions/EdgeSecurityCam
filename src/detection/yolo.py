"""Detector de personas con YOLO (Ultralytics), clase persona solamente.

Formatos del modelo (``--yolo-formato``):

* ``pt``: PyTorch (``models/yolo26n.pt``). Un solo modelo sirve para todos los
  tamaños de entrada. Es lo que usaba siempre el servidor Windows.
* ``onnx``: ``models/yolo26n_416.onnx`` (ONNX Runtime). Mismas detecciones que
  PyTorch (conserva la cabeza sin NMS de YOLO26; verificado el 2026-11-26:
  diferencia 0,000 en 21 fotos de la cámara) y ~3× más rápido en x86.
* ``ncnn``: ``models/yolo26n_416_ncnn_model/``. Suele ser el más rápido en ARM
  (Raspberry Pi), pero Ultralytics exporta YOLO26 a NCNN con la cabeza
  clásica con NMS: las confianzas cambian ~0,02–0,06 y aparecen algunas
  detecciones débiles más. Usarlo implica revisar la calibración de [yolo].
* ``auto`` (por defecto): para cada tamaño, ONNX si está exportado, si no NCNN,
  si no PyTorch. Sin exportaciones se comporta igual que antes.

Un modelo exportado tiene la forma de entrada FIJA (Ultralytics usa la de la
exportación), por eso hay un archivo por tamaño. ``tools/exportar_yolo.py``
los exporta con la forma 16:9 del substream (416 → 256×416): igual que hace
PyTorch con una foto de 640×360 y con ~40 % menos cálculo que un cuadrado.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from pathlib import Path

import numpy as np

from src.detection.base import FrameResult


PERSON_CLASS = 0
FORMATS = ("auto", "pt", "onnx", "ncnn")
EXPORT_FORMATS = ("onnx", "ncnn")
# Librería necesaria para ejecutar cada formato exportado.
RUNTIMES = {"onnx": "onnxruntime", "ncnn": "ncnn"}

log = logging.getLogger("alarmas")


def exported_path(model: str | Path, size: int, model_format: str) -> Path:
    """Archivo (ONNX) o carpeta (NCNN) del modelo exportado para un tamaño.

    La carpeta NCNN debe terminar en ``_ncnn_model``: así la reconoce Ultralytics.
    """
    model = Path(model)
    if model_format == "onnx":
        return model.with_name(f"{model.stem}_{size}.onnx")
    if model_format == "ncnn":
        return model.with_name(f"{model.stem}_{size}_ncnn_model")
    raise ValueError(f"formato no exportable: {model_format}")


def runtime_available(model_format: str) -> bool:
    return importlib.util.find_spec(RUNTIMES[model_format]) is not None


def quiet_onnxruntime() -> None:
    """Que los hilos de ONNX Runtime no giren en espera activa entre inferencias.

    Por defecto siguen ocupando CPU tras cada inferencia: medido el 2026-11-26,
    10 inferencias separadas por 100 ms gastaron 3 s de CPU en las esperas
    (PyTorch y NCNN, 0 ms con OMP_WAIT_POLICY=PASSIVE). Ultralytics crea la
    sesión sin opciones, así que se añaden aquí a toda sesión que se cree.
    """
    import onnxruntime

    if getattr(onnxruntime.InferenceSession, "camdetector_quiet", False):
        return
    original = onnxruntime.InferenceSession

    class QuietSession(original):
        camdetector_quiet = True

        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            if sess_options is None:
                sess_options = onnxruntime.SessionOptions()
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            super().__init__(path_or_bytes, sess_options, *args, **kwargs)

    onnxruntime.InferenceSession = QuietSession


class YoloPersonDetector:
    """YOLO restringido a la clase persona; cada modelo se carga y precalienta
    una vez y lo comparten todas las cámaras."""

    def __init__(self, model: str | Path, confidence: float = 0.40,
                 image_size: int = 640, device: str = "cpu",
                 weak_confidence: float = 0.15, *, model_format: str = "pt",
                 sizes: tuple[int, ...] | list[int] = ()) -> None:
        if model_format not in FORMATS:
            raise ValueError(f"formato de YOLO desconocido: {model_format}")
        self._source = Path(model)
        self._format = model_format
        self._confidence = confidence
        self._weak_confidence = min(weak_confidence, confidence)
        self._image_size = sizes[0] if sizes else image_size
        self._device = device
        self._lock = threading.Lock()
        self._pt = None                         # modelo PyTorch (todos los tamaños)
        # tamaño -> (modelo, nombre, ¿es PyTorch?)
        self._by_size: dict[int, tuple[object, str, bool]] = {}
        for size in dict.fromkeys(sizes or (self._image_size,)):
            self._model_for(size)

    # --- modelos ---------------------------------------------------------------

    def _load_pt(self):
        if self._pt is None:
            from ultralytics import YOLO    # import diferido: es pesado

            self._pt = YOLO(str(self._source))
        return self._pt

    def _choose(self, size: int) -> tuple[object, str, bool]:
        from ultralytics import YOLO

        wanted = EXPORT_FORMATS if self._format == "auto" else (
            (self._format,) if self._format in EXPORT_FORMATS else ())
        for model_format in wanted:
            path = exported_path(self._source, size, model_format)
            if not path.exists():
                continue
            if runtime_available(model_format):
                if model_format == "onnx":
                    quiet_onnxruntime()
                return YOLO(str(path), task="detect"), path.name, False
            log.warning("YOLO %d px: existe %s pero falta la librería %s.",
                        size, path.name, RUNTIMES[model_format])
        if self._format in EXPORT_FORMATS:
            raise FileNotFoundError(
                f"no hay modelo {self._format} utilizable para {size} px "
                f"({exported_path(self._source, size, self._format).name}); "
                f"expórtalo con: python tools/exportar_yolo.py --formatos "
                f"{self._format} --tamanos {size}")
        return self._load_pt(), self._source.name, True

    def _model_for(self, size: int) -> tuple[object, str, bool]:
        """Modelo para un tamaño de entrada; se carga y precalienta la primera vez."""
        with self._lock:
            if size not in self._by_size:
                entry = self._choose(size)
                self._by_size[size] = entry
                self._predict(entry, np.zeros((360, 640, 3), dtype=np.uint8), 0.5, size)
                log.info("YOLO %d px: %s", size, entry[1])
            return self._by_size[size]

    @property
    def description(self) -> str:
        """Modelo(s) en uso, para el registro y la base (trazabilidad)."""
        names = dict.fromkeys(name for _, name, _ in self._by_size.values())
        return ", ".join(names) or self._source.name

    # --- inferencia ------------------------------------------------------------

    def _predict(self, entry, frame: np.ndarray, conf: float, size: int):
        model, _, is_pt = entry
        # A un modelo exportado NO se le pasa imgsz: Ultralytics toma su forma
        # de la exportación en la primera llamada, pero un imgsz explícito la
        # sustituiría por un cuadrado en las siguientes.
        options = {"imgsz": size} if is_pt else {}
        return model.predict(frame, conf=conf, classes=[PERSON_CLASS],
                             device=self._device, verbose=False, **options)[0]

    def detect(self, frame: np.ndarray, *, confidence: float | None = None,
               weak_confidence: float | None = None,
               image_size: int | None = None) -> FrameResult:
        """Los valores por llamada permiten parámetros distintos por cámara con
        los modelos cargados una sola vez."""
        strong_min = self._confidence if confidence is None else confidence
        weak_min = self._weak_confidence if weak_confidence is None else weak_confidence
        weak_min = min(weak_min, strong_min)
        size = image_size or self._image_size
        entry = self._model_for(size)
        with self._lock:
            prediction = self._predict(entry, frame, weak_min, size)
        strong: list[tuple[float, tuple[int, int, int, int]]] = []
        weak: list[tuple[float, tuple[int, int, int, int]]] = []
        for box in prediction.boxes:
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            (strong if score >= strong_min else weak).append((score, (x1, y1, x2, y2)))
        return FrameResult(time.monotonic(),
                           tuple(c for c, _ in strong), tuple(b for _, b in strong),
                           tuple(c for c, _ in weak), tuple(b for _, b in weak))
