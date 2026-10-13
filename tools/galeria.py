"""Galería de rostros: enrolar personas desde la cámara y revisar la galería.

Capturar (ventana con la cámara HD; la persona mira a la cámara):
    python tools\\galeria.py capturar --camera 192.168.100.109 --nombre "Pablo Fanola"
    ESPACIO guarda la cabeza (solo si hay UNA cara completa y de buena calidad)
    q o Esc termina.

Revisar (calidad de cada foto y si alguna se confunde con otra persona):
    python tools\\galeria.py revisar --camera 192.168.100.109

La carpeta de la galería y los umbrales salen de config/camdetector.toml
(grupo [rostro] de esa cámara). Las fotos de esta cámara se parecen más a lo
que verá el reconocimiento (ángulo desde arriba, luz, lente) que las de una
webcam o un celular: suben la similitud de la persona.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.credentials import read_credentials  # noqa: E402
from src.faces.gallery import IMAGE_EXTENSIONS, Gallery  # noqa: E402
from src.faces.models import SFACE, YUNET, ensure_models  # noqa: E402
from src.faces.quality import clamp_box, expand_box, face_quality  # noqa: E402
from src.faces.sface import SFaceRecognizer  # noqa: E402
from src.faces.yunet import YuNetDetector  # noqa: E402
from src.settings import SettingsError, load_settings  # noqa: E402
from src.sources.video import open_rtsp, rtsp_url  # noqa: E402

GREEN, RED, WHITE, YELLOW = (60, 210, 60), (40, 40, 230), (240, 240, 240), (40, 220, 240)
PREVIEW_WIDTH = 1440          # YuNet trabaja sobre el HD reducido a la mitad (rápido)
HEAD_MARGIN = 0.60            # cabeza con margen alrededor de la cara


def gallery_folder(params: dict) -> Path:
    folder = Path(params["galeria"])
    return folder if folder.is_absolute() else (ROOT / folder).resolve()


def save_image(path: Path, image: np.ndarray) -> bool:
    """cv2.imwrite no acepta rutas con tildes en Windows: se codifica aparte."""
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.tobytes())
    return bool(ok)


class LatestFrame(threading.Thread):
    """Lee el vídeo HD sin parar y conserva solo el último fotograma."""

    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        self._capture = open_rtsp(url)
        self._lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self.frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def close(self) -> None:
        self.stop.set()
        self.join(timeout=3)
        self._capture.release()


def capture(args, params: dict) -> int:
    name = args.nombre.strip()
    if not name or any(ch in name for ch in '\\/:*?"<>|'):
        print("Nombre no válido (no puede tener \\ / : * ? \" < > |).")
        return 2
    models = ensure_models(ROOT / "models")
    detector = YuNetDetector(models[YUNET], score=params["yunet_puntuacion"])
    recognizer = SFaceRecognizer(models[SFACE])
    folder = gallery_folder(params)
    gallery = Gallery.load(folder, detector, recognizer)
    print(f"Galería {folder}: {', '.join(gallery.people) or 'vacía'}")
    username, password = read_credentials(args.credentials_file)
    reader = LatestFrame(rtsp_url(args.camera, username, password, path=params["hd_ruta"]))
    reader.start()
    saved = 0
    message, message_until = "", 0.0
    title = f"Galeria - capturar {name}"
    try:
        while True:
            frame = reader.latest()
            if frame is None:
                if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                    break
                continue
            height, width = frame.shape[:2]
            scale = PREVIEW_WIDTH / width
            preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            faces = detector.detect(preview)
            usable, info, color, face_full = False, "", RED, None
            if len(faces) == 1:
                small = faces[0]
                raw = small.raw.copy()
                raw[:14] /= scale
                face_full = type(small)(small.x / scale, small.y / scale, small.width / scale,
                                        small.height / scale,
                                        tuple((x / scale, y / scale) for x, y in small.landmarks),
                                        small.score, raw)
                quality = face_quality(frame, face_full)
                problems = quality.problems(params["cara_min_px"], params["frontal_min"],
                                            params["nitidez_min"], params["brillo_min"])
                info = (f"cara {quality.size_px:.0f}px  frontal {quality.frontal:.2f}  "
                        f"nitidez {quality.sharpness:.0f}  brillo {quality.brightness:.0f}")
                if problems:
                    info += "  ->  " + ", ".join(problems)
                else:
                    usable, color = True, GREEN
                    vector = recognizer.embed(frame, face_full)
                    if vector is not None and gallery.embeddings:
                        match = gallery.match(vector, params["umbral"], params["margen_segundo"])
                        info += f"  ->  se parece a {match.best} {match.score:.2f}"
                x1, y1, x2, y2 = small.box
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            elif len(faces) > 1:
                info = f"{len(faces)} caras: debe haber una sola persona frente a la camara"
                for face in faces:
                    cv2.rectangle(preview, face.box[:2], face.box[2:], RED, 2)
            else:
                info = "sin cara: la persona debe mirar a la camara"
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 58), (0, 0, 0), -1)
            cv2.putText(preview, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        GREEN if usable else WHITE, 1, cv2.LINE_AA)
            hint = (f"ESPACIO: guardar   q/Esc: salir   guardadas: {saved}   "
                    f"({folder.name}/{name})")
            cv2.putText(preview, hint, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
                        cv2.LINE_AA)
            if time.monotonic() < message_until:
                cv2.putText(preview, message, (8, preview.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2, cv2.LINE_AA)
            cv2.imshow(title, preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not usable:
                    message = "No se guarda: hace falta UNA cara completa en verde"
                else:
                    x1, y1, x2, y2 = clamp_box(expand_box(face_full.box, HEAD_MARGIN), width, height)
                    head = frame[y1:y2, x1:x2]
                    check = detector.detect(head)          # la foto guardada debe servir
                    if len(check) != 1:
                        message = "No se guarda: en el recorte no queda una sola cara"
                    else:
                        path = folder / name / f"{datetime.now():%Y%m%d_%H%M%S}_camara.jpg"
                        if save_image(path, head):
                            saved += 1
                            message = f"Guardada: {path.name}"
                            print(f"guardada {path}")
                        else:
                            message = "No se pudo guardar la imagen"
                message_until = time.monotonic() + 2.5
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        reader.close()
        cv2.destroyAllWindows()
    print(f"{saved} foto(s) nueva(s) de {name}. Reinicia CamDetector para usarlas.")
    return 0


def review(args, params: dict) -> int:
    models = ensure_models(ROOT / "models")
    detector = YuNetDetector(models[YUNET], score=params["yunet_puntuacion"])
    recognizer = SFaceRecognizer(models[SFACE])
    folder = gallery_folder(params)
    if not folder.is_dir():
        print(f"No existe la galería {folder}")
        return 2
    entries: list[tuple[str, str, np.ndarray]] = []
    print(f"Galería {folder}\n")
    print(f"{'persona / foto':<52} {'cara':>5} {'frontal':>7} {'nitidez':>8} {'brillo':>6}  estado")
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        for photo in sorted(person.iterdir()):
            if photo.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = f"{person.name}/{photo.name}"[:52]
            image = cv2.imdecode(np.fromfile(str(photo), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"{label:<52} {'':>5} {'':>7} {'':>8} {'':>6}  NO SE PUEDE LEER")
                continue
            scale = min(1.0, 640 / max(image.shape[:2]))
            small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
            faces = detector.detect(small)
            if not faces:
                print(f"{label:<52} {'':>5} {'':>7} {'':>8} {'':>6}  SIN CARA: no sirve")
                continue
            face = max(faces, key=lambda f: f.width * f.height)
            raw = face.raw.copy()
            raw[:14] /= scale
            face = type(face)(face.x / scale, face.y / scale, face.width / scale,
                              face.height / scale,
                              tuple((x / scale, y / scale) for x, y in face.landmarks),
                              face.score, raw)
            quality = face_quality(image, face)
            problems = quality.problems(params["cara_min_px"], params["frontal_min"],
                                        params["nitidez_min"], params["brillo_min"])
            vector = recognizer.embed(image, face)
            state = ", ".join(problems) if problems else "ok"
            if len(faces) > 1:
                state += f" ({len(faces)} caras: se usa la mayor)"
            print(f"{label:<52} {quality.size_px:>5.0f} {quality.frontal:>7.2f} "
                  f"{quality.sharpness:>8.0f} {quality.brightness:>6.0f}  {state}")
            if vector is not None:
                entries.append((person.name, photo.name, vector))

    people = sorted({entry[0] for entry in entries})
    if len(entries) < 2:
        return 0
    vectors = np.vstack([entry[2] for entry in entries])
    similarity = vectors @ vectors.T
    print("\nCada foto frente al resto (misma persona = más parecida; otra = menos parecida):")
    print(f"{'persona / foto':<52} {'misma':>6} {'otra':>6}  ")
    confusing = 0
    for i, (person, photo, _) in enumerate(entries):
        same = [similarity[i, j] for j, e in enumerate(entries) if j != i and e[0] == person]
        other = [(similarity[i, j], e[0]) for j, e in enumerate(entries) if e[0] != person]
        best_same = max(same) if same else float("nan")
        best_other, who = max(other) if other else (float("nan"), "-")
        warning = ""
        if same and other and best_other >= best_same:
            warning = f"  ¡se parece más a {who}! revisar"
            confusing += 1
        elif other and best_other >= params["umbral"]:
            warning = f"  supera el umbral con {who}"
            confusing += 1
        print(f"{(person + '/' + photo)[:52]:<52} {best_same:>6.2f} {best_other:>6.2f}{warning}")
    print("\nSimilitud media entre personas (la diagonal debe ser la más alta de cada fila):")
    print(" " * 20 + "".join(f"{p[:12]:>13}" for p in people))
    for a in people:
        row = []
        for b in people:
            values = [similarity[i, j] for i, ea in enumerate(entries) for j, eb in enumerate(entries)
                      if ea[0] == a and eb[0] == b and i != j]
            row.append(f"{np.mean(values):>13.2f}" if values else f"{'-':>13}")
        print(f"{a[:20]:<20}" + "".join(row))
    print(f"\n{confusing} foto(s) a revisar." if confusing else "\nNinguna foto se confunde con otra persona.")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Galería de rostros de CamDetector")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("capturar", "revisar"):
        p = sub.add_parser(command)
        p.add_argument("--camera", required=True, help="IP de la cámara (sus parámetros [rostro])")
        p.add_argument("--config", type=Path, default=ROOT / "config" / "camdetector.toml")
        p.add_argument("--credentials-file", type=Path, default=ROOT / "claves.txt")
        if command == "capturar":
            p.add_argument("--nombre", required=True, help="Nombre de la persona (carpeta)")
    args = parser.parse_args()
    try:
        params = load_settings(args.config, [args.camera])[args.camera].section("rostro")
    except SettingsError as error:
        print(f"Configuración: {error}")
        return 2
    return capture(args, params) if args.command == "capturar" else review(args, params)


if __name__ == "__main__":
    raise SystemExit(main())
