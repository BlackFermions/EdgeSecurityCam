"""Elección del modelo YOLO por formato y tamaño (sin cargar modelos reales)."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.detection import yolo
from src.detection.yolo import YoloPersonDetector, exported_path


class FakeYOLO:
    loaded: list[str] = []

    def __init__(self, path, task=None):
        self.path = str(path)
        self.calls = []
        FakeYOLO.loaded.append(Path(path).name)

    def predict(self, frame, **options):
        self.calls.append(options)
        return [types.SimpleNamespace(boxes=[])]


class YoloFormatTests(unittest.TestCase):
    def setUp(self):
        FakeYOLO.loaded = []
        self.folder = Path(tempfile.mkdtemp())
        self.model = self.folder / "yolo26n.pt"
        self.model.write_bytes(b"")
        fake = types.ModuleType("ultralytics")
        fake.YOLO = FakeYOLO
        patcher = mock.patch.dict(sys.modules, {"ultralytics": fake})
        patcher.start()
        self.addCleanup(patcher.stop)

    def export(self, size, model_format):
        path = exported_path(self.model, size, model_format)
        if model_format == "ncnn":
            path.mkdir()
        else:
            path.write_bytes(b"")

    def test_exported_names(self):
        self.assertEqual(exported_path("models/yolo26n.pt", 416, "onnx").name, "yolo26n_416.onnx")
        # Ultralytics reconoce NCNN por el final "_ncnn_model".
        self.assertEqual(exported_path("models/yolo26n.pt", 320, "ncnn").name,
                         "yolo26n_320_ncnn_model")

    def test_auto_without_exports_uses_pytorch_once(self):
        detector = YoloPersonDetector(self.model, model_format="auto", sizes=[416, 320])
        self.assertEqual(FakeYOLO.loaded, ["yolo26n.pt"])      # un solo modelo PyTorch
        self.assertEqual(detector.description, "yolo26n.pt")

    def test_auto_prefers_onnx_then_ncnn_per_size(self):
        self.export(416, "onnx")
        self.export(416, "ncnn")
        self.export(320, "ncnn")
        with mock.patch.object(yolo, "runtime_available", return_value=True):
            detector = YoloPersonDetector(self.model, model_format="auto", sizes=[416, 320])
        self.assertEqual(detector.description, "yolo26n_416.onnx, yolo26n_320_ncnn_model")

    def test_auto_skips_export_without_runtime(self):
        self.export(416, "onnx")
        with mock.patch.object(yolo, "runtime_available", return_value=False), \
                self.assertLogs("alarmas", "WARNING"):
            detector = YoloPersonDetector(self.model, model_format="auto", sizes=[416])
        self.assertEqual(detector.description, "yolo26n.pt")

    def test_explicit_format_missing_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            YoloPersonDetector(self.model, model_format="ncnn", sizes=[416])

    def test_exported_model_keeps_its_input_shape(self):
        # Un imgsz explícito haría que Ultralytics usara un cuadrado en lugar
        # de la forma exportada (256x416): a un modelo exportado no se le pasa.
        self.export(416, "onnx")
        with mock.patch.object(yolo, "runtime_available", return_value=True):
            detector = YoloPersonDetector(self.model, model_format="onnx", sizes=[416])
        import numpy as np
        detector.detect(np.zeros((360, 640, 3), np.uint8), image_size=416)
        model = detector._by_size[416][0]
        self.assertTrue(all("imgsz" not in call for call in model.calls))

    def test_pytorch_receives_size_per_call(self):
        detector = YoloPersonDetector(self.model, model_format="pt", sizes=[416])
        import numpy as np
        detector.detect(np.zeros((360, 640, 3), np.uint8), image_size=320)
        self.assertEqual(detector._pt.calls[-1]["imgsz"], 320)


if __name__ == "__main__":
    unittest.main()
