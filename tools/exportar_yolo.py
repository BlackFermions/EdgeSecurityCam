"""Exporta YOLO a ONNX y NCNN (un modelo por tamaño de entrada) y lo valida.

Uso:
    python tools/exportar_yolo.py                          # ONNX y NCNN, 320 y 416 px
    python tools/exportar_yolo.py --formatos onnx --tamanos 416
    python tools/exportar_yolo.py --solo-validar

Crea models/yolo26n_<tamaño>.onnx y models/yolo26n_<tamaño>_ncnn_model/.
CamDetector los usa solos (--yolo-formato auto, el valor por defecto).

Forma de entrada: 16:9, la del substream de la W51-TY (640x360): 416 px se
exporta como 256x416, igual que lo procesa PyTorch y con ~40 % menos cálculo
que un cuadrado. Con otra relación de aspecto, --forma cuadrada.

Validación: compara cada exportación con PyTorch en las fotos de capturas/
(o en imágenes de ejemplo de Ultralytics):
* ONNX conserva la cabeza sin NMS de YOLO26: se compara con PyTorch tal cual.
* NCNN no la admite (Ultralytics usa la cabeza clásica con NMS): se compara
  con PyTorch usando esa misma cabeza, para comprobar que la exportación es
  fiel, y se informa cuánto cambia respecto a lo calibrado.
Un modelo que no pasa la validación se borra (CamDetector sigue con PyTorch).

El modelo exportado no depende del procesador: se puede exportar en el PC y
copiar a la Raspberry Pi. Necesita onnx y onnxslim (ONNX) y pnnx y ncnn
(NCNN); el instalador de la Pi los instala.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from src.detection.yolo import (EXPORT_FORMATS, PERSON_CLASS,  # noqa: E402
                                exported_path, runtime_available)

# Criterios de aceptación: la exportación debe dar lo mismo que su referencia.
MAX_COUNT_MISMATCH = 0.05   # fracción de imágenes con distinto número de detecciones
MAX_CONF_DIFF = 0.01        # mayor diferencia de confianza
MIN_IOU = 0.98              # IoU media de las cajas emparejadas
COMPARE_CONF = 0.15         # = yolo.confianza_debil por defecto
SUBSTREAM = (360, 640)      # alto, ancho


def input_shape(size: int, shape: str) -> list[int]:
    """[alto, ancho] de la entrada exportada (múltiplos de 32)."""
    if shape == "cuadrada":
        return [size, size]
    height = math.ceil(size * SUBSTREAM[0] / SUBSTREAM[1] / 32) * 32
    return [height, size]


def export(model_path: Path, size: int, model_format: str, shape: str) -> Path:
    from ultralytics import YOLO

    target = exported_path(model_path, size, model_format)
    produced = Path(YOLO(str(model_path)).export(
        format=model_format, imgsz=input_shape(size, shape), half=False,
        **({"simplify": True} if model_format == "onnx" else {})))
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    produced.rename(target)
    return target


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def load_images(limit: int, shape: str):
    """Fotos con la forma de la entrada exportada.

    Un modelo exportado tiene la forma fija; PyTorch adapta el relleno a cada
    foto. Solo con la misma relación de aspecto procesan exactamente la misma
    entrada: las demás fotos se redimensionan (640x360 o 640x640).
    """
    import cv2

    target = (SUBSTREAM[1], SUBSTREAM[0]) if shape == "16:9" else (640, 640)   # ancho, alto
    found = [path for path in sorted((ROOT / "capturas").rglob("*.jpg"))
             if "rostros" not in path.parts]          # escenas, no recortes de cara
    if not found:
        from ultralytics.utils import ASSETS
        found = [ASSETS / "bus.jpg", ASSETS / "zidane.jpg"]
    step = max(1, len(found) // limit)
    pictures = [cv2.imread(str(path)) for path in found[::step][:limit]]
    return [picture if picture.shape[1::-1] == target else cv2.resize(picture, target)
            for picture in pictures if picture is not None]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def people(model, image, size: int | None):
    options = {"imgsz": size} if size else {}
    started = time.perf_counter()
    result = model.predict(image, conf=COMPARE_CONF, classes=[PERSON_CLASS],
                           device="cpu", verbose=False, **options)[0]
    elapsed = (time.perf_counter() - started) * 1000
    return [(float(b.conf[0]), b.xyxy[0].tolist()) for b in result.boxes], elapsed


def compare(reference, ref_size, candidate, cand_size, pictures):
    """(fracción con distinto número, mayor Δ confianza, IoU media, ms ref, ms cand)."""
    for model, size in ((reference, ref_size), (candidate, cand_size)):
        people(model, pictures[0], size)        # precalentar
    mismatched, worst, ious, t_ref, t_cand = 0, 0.0, [], [], []
    for image in pictures:
        expected, ms_ref = people(reference, image, ref_size)
        got, ms_cand = people(candidate, image, cand_size)
        t_ref.append(ms_ref)
        t_cand.append(ms_cand)
        if len(expected) != len(got):
            mismatched += 1
            continue
        free = list(got)
        for conf, box in expected:
            best = max(free, key=lambda other: iou(box, other[1]))
            ious.append(iou(box, best[1]))
            worst = max(worst, abs(conf - best[0]))
            free.remove(best)
    return (mismatched / len(pictures), worst, statistics.mean(ious) if ious else 1.0,
            statistics.median(t_ref), statistics.median(t_cand))


def validate(model_path: Path, size: int, model_format: str, pictures) -> bool:
    from ultralytics import YOLO

    reference = YOLO(str(model_path))
    calibrated = None
    if model_format == "ncnn":
        calibrated = YOLO(str(model_path))
        reference.model.model[-1].end2end = False   # misma cabeza que la exportación
    exported = YOLO(str(exported_path(model_path, size, model_format)), task="detect")
    mismatch, worst, mean_iou, ms_ref, ms_cand = compare(reference, size, exported, None,
                                                         pictures)
    ok = mismatch <= MAX_COUNT_MISMATCH and worst <= MAX_CONF_DIFF and mean_iou >= MIN_IOU
    label = "PyTorch (cabeza con NMS)" if model_format == "ncnn" else "PyTorch"
    print(f"  {model_format.upper()} {size} px frente a {label} · {len(pictures)} imágenes")
    print(f"    distinto número de detecciones: {mismatch:.0%} (máx. {MAX_COUNT_MISMATCH:.0%})"
          f" · mayor Δ confianza: {worst:.3f} (máx. {MAX_CONF_DIFF})"
          f" · IoU media: {mean_iou:.3f} (mín. {MIN_IOU})")
    print(f"    tiempo mediano en este equipo: PyTorch {ms_ref:.0f} ms · "
          f"{model_format.upper()} {ms_cand:.0f} ms (la prueba de rendimiento da cifras completas)")
    if calibrated is not None:
        change, worst_change, _, _, _ = compare(calibrated, size, reference, size, pictures)
        print(f"    cambio respecto a lo calibrado (cabeza sin NMS): {change:.0%} de imágenes "
              f"con distinto número de detecciones, Δ confianza hasta {worst_change:.3f}")
    print(f"    {'VÁLIDO' if ok else 'NO VÁLIDO'}")
    return ok


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Exporta YOLO a ONNX/NCNN y lo valida")
    parser.add_argument("--modelo", type=Path, default=ROOT / "models" / "yolo26n.pt")
    parser.add_argument("--formatos", nargs="+", default=list(EXPORT_FORMATS),
                        choices=EXPORT_FORMATS)
    parser.add_argument("--tamanos", type=int, nargs="+", default=[320, 416],
                        choices=[320, 416, 480, 640])
    parser.add_argument("--forma", choices=["16:9", "cuadrada"], default="16:9",
                        help="Forma de la entrada exportada (16:9 = substream 640x360)")
    parser.add_argument("--imagenes", type=int, default=30,
                        help="Máximo de fotos de capturas/ para validar")
    parser.add_argument("--solo-validar", action="store_true",
                        help="No exporta: valida los modelos ya exportados")
    args = parser.parse_args()
    if not args.modelo.is_file():
        print(f"No existe {args.modelo}: ejecuta primero el instalador.")
        return 2

    import ultralytics
    print(f"Ultralytics {ultralytics.__version__} · modelo {args.modelo.name}")
    pictures = load_images(args.imagenes, args.forma)
    ok = True
    for model_format in args.formatos:
        if not runtime_available(model_format):
            print(f"{model_format.upper()}: falta la librería para ejecutarlo; se omite.")
            ok = False
            continue
        for size in args.tamanos:
            path = exported_path(args.modelo, size, model_format)
            if not args.solo_validar:
                shape = "x".join(map(str, input_shape(size, args.forma)))
                print(f"Exportando a {model_format.upper()}, {size} px (entrada {shape})...")
                try:
                    print(f"  → {export(args.modelo, size, model_format, args.forma)}")
                except Exception as error:          # noqa: BLE001 - se informa y se sigue
                    print(f"  ERROR al exportar: {type(error).__name__}: {error}")
                    ok = False
                    continue
            if not path.exists():
                print(f"  {path.name}: no está exportado.")
                ok = False
                continue
            if not validate(args.modelo, size, model_format, pictures):
                remove(path)
                print("    (borrado: CamDetector usará otro formato en este tamaño)")
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
