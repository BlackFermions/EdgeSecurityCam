"""Prueba de rendimiento del nodo con la cámara real (pensada para la Raspberry Pi).

Uso:
    python tools/rendimiento.py --camera 192.168.100.109
    python tools/rendimiento.py --camera 192.168.100.109 --rafagas 8 --segundos 30
    python tools/rendimiento.py --sin-camara           # solo YOLO y rostro, con fotos guardadas

Conviene que haya alguien delante de la cámara (para medir el seguimiento y
el rostro con personas reales) y que CamDetector esté DETENIDO (si no, ambos
compiten por la CPU y la cámara): sudo systemctl stop camdetector

Mide (cifras MEDIDAS en este equipo):
  1. Decodificar el substream (lo que hace cada sesión mientras dura).
  2. YOLO en cada formato disponible (PyTorch, ONNX, NCNN) a 320 y 416 px.
  3. Seguimiento por fotograma: escala de grises, puerta de movimiento y flujo óptico.
  4. Ráfagas HD para el rostro (abrir /stream1 y tomar un fotograma).
  5. Rostro: YuNet + SFace sobre la ráfaga HD (FaceEngine, igual que en vivo).
  6. RAM, temperatura y si el equipo bajó su velocidad (Raspberry Pi).

Con eso CALCULA (cifras ESTIMADAS) la carga de cada situación con los
parámetros de config/camdetector.toml y da un veredicto sobre el rostro.
Escribe el informe en Markdown en logs/rendimiento_<fecha>.md.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.credentials import read_credentials  # noqa: E402
from src.detection.yolo import (EXPORT_FORMATS, YoloPersonDetector,  # noqa: E402
                                exported_path, runtime_available)
from src.settings import SettingsError, load_settings  # noqa: E402
from src.sources.video import open_rtsp, rtsp_url  # noqa: E402
from src.tracking import FLOW_SCALE, MotionGate  # noqa: E402
from src.tracking.flow import FlowTracker  # noqa: E402

MODEL = ROOT / "models" / "yolo26n.pt"
SIZES = (320, 416)
# Datos reales (2026-10, sala de reuniones): 715 inferencias de YOLO en 10.462
# fotogramas procesados con personas → 6,8 % de los fotogramas.
YOLO_PER_FRAME_REAL = 715 / 10462
# Criterio del veredicto: dejar margen para picos, el sistema y otras tareas.
MAX_CORE_FRACTION = 0.60
HOT_C = 80.0


# --- utilidades -------------------------------------------------------------------

class CpuTimer:
    """Tiempo real y tiempo de CPU de TODO el proceso (incluye los hilos de
    FFmpeg, PyTorch, ONNX Runtime y NCNN)."""

    def __enter__(self):
        self.wall0, self.cpu0 = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *_):
        self.wall = time.perf_counter() - self.wall0
        self.cpu = time.process_time() - self.cpu0


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return None


def temperature() -> float | None:
    value = read_text("/sys/class/thermal/thermal_zone0/temp")
    return int(value) / 1000 if value and value.isdigit() else None


def throttled() -> str | None:
    """vcgencmd get_throttled: 0x0 = nunca bajó su velocidad ni le faltó energía."""
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out.split("=")[-1] or None


def memory_mb() -> tuple[float | None, float | None]:
    """(RAM actual, pico) del proceso en MB."""
    status = read_text("/proc/self/status")
    if status:
        values = {line.split(":")[0]: line.split()[1] for line in status.splitlines()
                  if line.startswith(("VmRSS", "VmHWM"))}
        return (int(values.get("VmRSS", 0)) / 1024 or None,
                int(values.get("VmHWM", 0)) / 1024 or None)
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        return info.rss / 2**20, (peak / 2**20 if peak else None)
    except ImportError:
        return None, None


def total_ram_mb() -> float | None:
    meminfo = read_text("/proc/meminfo")
    if meminfo:
        return int(meminfo.splitlines()[0].split()[1]) / 1024
    try:
        import psutil
        return psutil.virtual_memory().total / 2**20
    except ImportError:
        return None


def device_name() -> str:
    model = read_text("/proc/device-tree/model")
    if model:
        return model
    return f"{platform.system()} {platform.machine()} · {platform.processor() or 'CPU'}"


def versions() -> str:
    parts = [f"Python {platform.python_version()}", f"OpenCV {cv2.__version__}"]
    for name in ("ultralytics", "torch", "onnxruntime", "ncnn"):
        try:
            module = __import__(name)
            parts.append(f"{name} {getattr(module, '__version__', '?')}")
        except ImportError:
            pass
    return " · ".join(parts)


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def fmt(value: float | None, unit: str = "", digits: int = 0) -> str:
    return "n/d" if value is None else f"{value:.{digits}f}{unit}"


# --- mediciones -------------------------------------------------------------------

def measure_substream(url: str, seconds: float, keep: int):
    """Decodifica el substream como una sesión: fps, CPU y fotogramas para el resto."""
    frames: list[np.ndarray] = []
    count = 0
    capture = open_rtsp(url)
    if not capture.isOpened():
        capture.release()
        return None, []
    for _ in range(10):                          # descartar el arranque del RTSP
        capture.read()
    with CpuTimer() as timer:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            count += 1
            frames.append(frame)
    capture.release()
    if count == 0:
        return None, []
    step = max(1, len(frames) // keep)
    return {"fps": count / timer.wall, "cores": timer.cpu / timer.wall,
            "size": f"{frames[0].shape[1]}x{frames[0].shape[0]}"}, frames[::step][:keep]


def stored_frames(limit: int) -> list[np.ndarray]:
    paths = [path for path in sorted((ROOT / "capturas").rglob("*.jpg"))
             if "rostros" not in path.parts]
    images = [cv2.imread(str(path)) for path in paths[:limit * 3]]
    images = [image for image in images if image is not None and image.shape[:2] == (360, 640)]
    if not images:
        images = [np.full((360, 640, 3), 90, np.uint8)]
    return images[:limit]


def yolo_formats(selected: list[str]) -> list[str]:
    available = []
    for model_format in selected:
        if model_format == "pt" or (runtime_available(model_format) and any(
                exported_path(MODEL, size, model_format).exists() for size in SIZES)):
            available.append(model_format)
    return available


def measure_yolo(model_format: str, size: int, frames: list[np.ndarray], runs: int):
    if model_format != "pt" and not exported_path(MODEL, size, model_format).exists():
        return None
    detector = YoloPersonDetector(MODEL, model_format=model_format, sizes=[size])
    for frame in frames[:3]:
        detector.detect(frame, image_size=size)
    # CPU del bucle completo (el reloj de CPU de Windows avanza de 15,6 ms en
    # 15,6 ms: medido por inferencia saldría inflado).
    walls, people = [], []
    with CpuTimer() as total:
        for index in range(runs):
            frame = frames[index % len(frames)]
            started = time.perf_counter()
            result = detector.detect(frame, image_size=size)
            walls.append((time.perf_counter() - started) * 1000)
            people.append(result.people)
    rss, _ = memory_mb()
    data = {"format": model_format, "size": size, "median": statistics.median(walls),
            "p95": pct(walls, 0.95), "cpu": total.cpu * 1000 / runs,
            "people": max(people), "rss": rss, "name": detector.description}
    del detector
    gc.collect()
    return data


def measure_tracking(frames: list[np.ndarray], boxes):
    """Coste por fotograma del bucle de la sesión sin YOLO (gris + puerta + flujo)."""
    seeded = bool(boxes)
    if not seeded:                     # sin personas: una caja central de referencia
        height, width = frames[0].shape[:2]
        boxes = [(width * 0.4, height * 0.2, width * 0.6, height * 0.9)]
    gate = MotionGate()
    previous = None
    trackers: list[FlowTracker] = []
    walls = []
    with CpuTimer() as total:          # CPU del bucle completo (ver measure_yolo)
        for frame in frames:
            started = time.perf_counter()
            gray = cv2.cvtColor(cv2.resize(frame, None, fx=FLOW_SCALE, fy=FLOW_SCALE,
                                           interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if previous is None or any(tracker.lost for tracker in trackers):
                trackers = [FlowTracker(gray, box) for box in boxes]
            else:
                for tracker in trackers:
                    tracker.update(previous, gray)
            gate.update(gray, [tracker.box for tracker in trackers])
            previous = gray
            walls.append((time.perf_counter() - started) * 1000)
    return {"median": statistics.median(walls), "cpu": total.cpu * 1000 / len(frames),
            "tracks": len(boxes), "seeded": seeded}


def measure_hd(url: str, bursts: int, pause: float):
    from src.sources.hd import HdFrameGrabber

    grabber = HdFrameGrabber(url)
    walls, cpus, frame = [], [], None
    for index in range(bursts):
        with CpuTimer() as timer:
            image, _ = grabber.grab()
        if image is not None:
            walls.append(timer.wall)
            cpus.append(timer.cpu)
            frame = image
        if index + 1 < bursts:
            time.sleep(pause)
    if not walls:
        return None, None
    return {"median": statistics.median(walls), "max": max(walls),
            "cpu": statistics.mean(cpus), "ok": len(walls), "of": bursts,
            "size": f"{frame.shape[1]}x{frame.shape[0]}"}, frame


def face_models(settings):
    from src.faces.models import SFACE, YUNET, ensure_models
    from src.faces.sface import SFaceRecognizer
    from src.faces.yunet import YuNetDetector

    models = ensure_models(ROOT / "models")
    return (YuNetDetector(models[YUNET], score=settings.get("rostro", "yunet_puntuacion")),
            SFaceRecognizer(models[SFACE]))


def measure_faces_live(settings, hd_frame, sub_frame, boxes):
    """FaceEngine.analyze sobre la ráfaga HD, como en vivo (galería vacía: el
    coste de comparar con la galería es despreciable)."""
    from src.faces.engine import FaceEngine
    from src.faces.gallery import Gallery

    detector, recognizer = face_models(settings)
    engine = FaceEngine(detector, recognizer, Gallery())
    params = settings.section("rostro")
    sub_size = (sub_frame.shape[1], sub_frame.shape[0])
    results = []
    for box in boxes:
        engine.analyze(hd_frame, box, sub_size, params)        # precalentar
        with CpuTimer() as timer:
            attempt = engine.analyze(hd_frame, box, sub_size, params)
        results.append((attempt.result, timer.wall * 1000, timer.cpu * 1000))
    return results


def measure_sface_photos(settings, limit: int = 15):
    """SFace (alinear + huella) sobre caras guardadas, haya o no alguien delante."""
    detector, recognizer = face_models(settings)
    paths = sorted((ROOT / "capturas" / "rostros").rglob("*.jpg"))
    paths += sorted((ROOT / "face_gallery").rglob("*.jpg"))
    yunet, sface = [], []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        with CpuTimer() as timer:
            faces = detector.detect(image)
        if not faces:
            continue
        yunet.append(timer.wall * 1000)
        recognizer.embed(image, faces[0])               # precalentar
        with CpuTimer() as timer:
            recognizer.embed(image, faces[0])
        sface.append(timer.cpu * 1000)
        if len(sface) >= limit:
            break
    if not sface:
        return None
    return {"faces": len(sface), "yunet": statistics.median(yunet),
            "sface": statistics.median(sface)}


# --- informe ----------------------------------------------------------------------

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Prueba de rendimiento del nodo")
    parser.add_argument("--camera", help="IP de la cámara")
    parser.add_argument("--sin-camara", action="store_true",
                        help="Sin cámara: YOLO, seguimiento y SFace con fotos guardadas")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "camdetector.toml")
    parser.add_argument("--credentials-file", type=Path, default=ROOT / "claves.txt")
    parser.add_argument("--segundos", type=float, default=20.0,
                        help="Segundos de substream para medir la decodificación")
    parser.add_argument("--inferencias", type=int, default=40,
                        help="Inferencias de YOLO por formato y tamaño")
    parser.add_argument("--rafagas", type=int, default=5, help="Ráfagas HD")
    parser.add_argument("--formatos", nargs="+", default=["pt", *EXPORT_FORMATS],
                        choices=["pt", *EXPORT_FORMATS])
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()
    if not args.sin_camara and not args.camera:
        parser.error("indica --camera IP o --sin-camara")

    camera = args.camera or "0.0.0.0"
    try:
        settings = load_settings(args.config, [camera])[camera]
    except SettingsError as error:
        print(f"Configuración {args.config}: {error}")
        return 2
    cores = os.cpu_count() or 1
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    started = datetime.now()
    temp_start, throttle_start = temperature(), throttled()
    out(f"# Rendimiento de CamDetector · {started:%Y-%m-%d %H:%M}")
    out()
    out("## Equipo")
    out()
    out(f"- {device_name()} · {cores} núcleos · RAM {fmt(total_ram_mb(), ' MB')}")
    out(f"- {versions()}")
    out(f"- Temperatura al empezar: {fmt(temp_start, ' °C', 1)} · "
        f"get_throttled: {throttle_start or 'n/d'}")
    out(f"- Cámara: {args.camera or 'ninguna (--sin-camara)'} · "
        f"configuración: {args.config.name if args.config.exists() else 'valores por defecto'}")
    out()
    out("Todas las cifras de las secciones 1 a 5 son **medidas** en este equipo. "
        "La sección 6 es una **estimación** calculada a partir de ellas.")
    out()

    # 1. Substream --------------------------------------------------------------
    username = password = ""
    sub = None
    if args.camera:
        username, password = read_credentials(args.credentials_file)
        print(f"Decodificando el substream {args.segundos:.0f} s...", flush=True)
        sub, frames = measure_substream(rtsp_url(args.camera, username, password),
                                        args.segundos, keep=60)
        if sub is None:
            print("No se pudo abrir el substream; se usan fotos guardadas.")
    if sub is None:
        frames = stored_frames(40)
    out("## 1. Substream (vídeo de análisis)")
    out()
    if sub:
        out(f"{sub['size']} a {sub['fps']:.1f} fps: decodificarlo ocupa "
            f"**{sub['cores']:.2f} núcleos** durante toda la sesión.")
    else:
        out(f"Sin cámara: {len(frames)} fotos de capturas/ (no se midió la decodificación).")
    out()

    # 2. YOLO -------------------------------------------------------------------
    out("## 2. YOLO (una inferencia)")
    out()
    out("| Formato | Tamaño | Modelo | Mediana | p95 | CPU por inferencia | "
        "Personas | RAM del proceso |")
    out("|---|---|---|---|---|---|---|---|")
    yolo: list[dict] = []
    for model_format in yolo_formats(args.formatos):
        for size in SIZES:
            print(f"YOLO {model_format} {size} px...", flush=True)
            try:
                data = measure_yolo(model_format, size, frames, args.inferencias)
            except Exception as error:              # noqa: BLE001 - se informa
                out(f"| {model_format} | {size} | error: {type(error).__name__}: {error} "
                    "| | | | | |")
                continue
            if data is None:
                continue
            yolo.append(data)
            out(f"| {model_format} | {size} | {data['name']} | {data['median']:.0f} ms | "
                f"{data['p95']:.0f} ms | {data['cpu']:.0f} ms | {data['people']} | "
                f"{fmt(data['rss'], ' MB')} |")
    out()
    out("CPU por inferencia > mediana = usa varios núcleos a la vez. "
        "NCNN cambia la cabeza de YOLO26 (con NMS): ver src/detection/yolo.py.")
    out()

    # Personas de referencia para el seguimiento y el rostro.
    size_cfg = settings.get("yolo", "tamano")
    reference = YoloPersonDetector(MODEL, model_format="auto", sizes=[size_cfg])
    with_people = [(frame, reference.detect(frame, image_size=size_cfg)) for frame in frames[:15]]
    frame_ref, result_ref = max(with_people, key=lambda item: item[1].people)
    boxes = list(result_ref.boxes)

    # 3. Seguimiento --------------------------------------------------------------
    track = measure_tracking(frames, boxes)
    out("## 3. Seguimiento por fotograma (sin YOLO)")
    out()
    out(f"Gris + puerta de movimiento + flujo óptico con {track['tracks']} trayectoria(s)"
        f"{'' if track['seeded'] else ' (caja de referencia: no había nadie delante)'}: "
        f"mediana {track['median']:.1f} ms, CPU {track['cpu']:.1f} ms por fotograma.")
    out()

    # 4. Ráfagas HD -----------------------------------------------------------------
    hd, hd_frame = None, None
    out("## 4. Ráfaga HD para el rostro")
    out()
    if args.camera:
        print(f"Ráfagas HD ({args.rafagas})...", flush=True)
        hd, hd_frame = measure_hd(rtsp_url(args.camera, username, password,
                                           path=settings.get("rostro", "hd_ruta")),
                                  args.rafagas, pause=2.0)
    if hd:
        out(f"{hd['ok']}/{hd['of']} ráfagas de {hd['size']}: hasta el fotograma "
            f"mediana {hd['median']:.2f} s (máx. {hd['max']:.2f} s), "
            f"**CPU {hd['cpu']:.2f} s por ráfaga**.")
    else:
        out("No medido (sin cámara o no se pudo abrir el vídeo HD).")
    out()

    # 5. Rostro -----------------------------------------------------------------------
    out("## 5. Rostro")
    out()
    photos = measure_sface_photos(settings)
    if photos:
        out(f"Caras guardadas ({photos['faces']}): YuNet {photos['yunet']:.0f} ms (recorte), "
            f"**SFace {photos['sface']:.0f} ms de CPU por cara**.")
    else:
        out("SFace: no había caras guardadas en capturas/rostros ni face_gallery.")
    live = []
    if hd_frame is not None and boxes and track["seeded"]:
        live = measure_faces_live(settings, hd_frame, frame_ref, boxes)
        for result, wall, cpu in live:
            out(f"- En vivo (ráfaga HD, persona del substream): {result}, "
                f"{wall:.0f} ms ({cpu:.0f} ms de CPU)")
    elif args.camera:
        out("- En vivo: no medido (nadie delante de la cámara).")
    out()
    temp_end, throttle_end = temperature(), throttled()
    rss, peak = memory_mb()
    out(f"RAM del proceso al final: {fmt(rss, ' MB')} (pico {fmt(peak, ' MB')}). "
        f"Temperatura al terminar: {fmt(temp_end, ' °C', 1)} · "
        f"get_throttled: {throttle_end or 'n/d'}.")
    out()

    # 6. Estimación -------------------------------------------------------------------
    out("## 6. Carga estimada con estos parámetros (ESTIMADO)")
    out()
    if not yolo:
        out("Sin mediciones de YOLO: no se puede estimar.")
        return finish(lines, args, started)
    policy = settings.section("politica")
    session_fps = 1.0 / settings.get("sesion", "procesar_cada_s")
    if sub:
        session_fps = min(session_fps, sub["fps"])
    decode = sub["cores"] if sub else 0.0
    track_cores = track["cpu"] / 1000 * session_fps
    face = settings.section("rostro")
    hd_cores = hd["cpu"] / face["espera_entre_intentos_s"] if hd else None
    sface_cores = ((photos["sface"] / 1000) * min(face["presupuesto_sface_s"],
                                                  1 / face["espera_entre_intentos_s"] * 2)
                   if photos else 0.0)
    out(f"Supuestos: sesión a {session_fps:.1f} fotogramas/s (sesion.procesar_cada_s); "
        f"con personas YOLO corre en el {YOLO_PER_FRAME_REAL:.1%} de los fotogramas "
        "(datos reales de 2026-10); peor caso = alarma sin veredicto, YOLO cada "
        f"politica.confirmacion_s = {policy['confirmacion_s']} s; rostro peor caso = "
        f"una ráfaga HD cada rostro.espera_entre_intentos_s = "
        f"{face['espera_entre_intentos_s']} s y 2 caras por ráfaga.")
    out()
    out(f"| Formato | Tamaño | Sesión típica | Peor caso | Peor caso + rostro | "
        f"Límite ({MAX_CORE_FRACTION:.0%} de {cores}) |")
    out("|---|---|---|---|---|---|")
    budget = MAX_CORE_FRACTION * cores
    verdicts = []
    for data in yolo:
        yolo_cpu = data["cpu"] / 1000
        typical = decode + track_cores + yolo_cpu * session_fps * YOLO_PER_FRAME_REAL
        worst_rate = 1.0 / max(policy["confirmacion_s"], data["median"] / 1000)
        worst = decode + track_cores + yolo_cpu * worst_rate
        with_face = worst + hd_cores + sface_cores if hd_cores is not None else None
        fits = with_face is not None and with_face <= budget
        keeps_pace = data["p95"] / 1000 <= policy["confirmacion_s"]
        verdicts.append((data, fits, keeps_pace, with_face))
        out(f"| {data['format']} | {data['size']} | {typical:.2f} núcleos | "
            f"{worst:.2f} núcleos | {fmt(with_face, ' núcleos', 2)} | "
            f"{'cabe' if fits else ('n/d' if with_face is None else 'NO cabe')} |")
    out()

    hot = temp_end is not None and temp_end >= HOT_C
    throttle_bad = throttle_end not in (None, "0x0")
    out("## Veredicto")
    out()
    best = min(verdicts, key=lambda item: item[0]["median"])
    data, fits, keeps_pace, with_face = best
    out(f"- YOLO más rápido: **{data['format']} a {data['size']} px** "
        f"({data['median']:.0f} ms; p95 {data['p95']:.0f} ms"
        f"{'' if keeps_pace else f', MÁS LENTO que confirmacion_s = {policy['confirmacion_s']} s'}).")
    configured = [item for item in verdicts if item[0]["size"] == size_cfg]
    if configured:
        fastest_cfg = min(configured, key=lambda item: item[0]["median"])
        out(f"- Con el tamaño configurado ({size_cfg} px) el más rápido es "
            f"{fastest_cfg[0]['format']} ({fastest_cfg[0]['median']:.0f} ms).")
    if hot or throttle_bad:
        out(f"- **Atención:** temperatura {fmt(temp_end, ' °C', 1)} / get_throttled "
            f"{throttle_end}: el equipo bajó su velocidad o le faltó energía. Revisar "
            "ventilador y fuente antes de decidir; repetir la prueba.")
    if hd is None:
        out("- Rostro: **sin decidir** (falta medir la ráfaga HD con la cámara).")
    elif fits and not hot and not throttle_bad:
        out(f"- Rostro: **se puede encender** con {data['format']} a {data['size']} px "
            f"(peor caso {with_face:.2f} de {budget:.1f} núcleos permitidos).")
    else:
        out("- Rostro: **mantener apagado** con estos parámetros (peor caso por encima "
            "del límite o equipo caliente). Probar un tamaño menor o más espera entre "
            "intentos (rostro.espera_entre_intentos_s).")
    out()
    return finish(lines, args, started)


def finish(lines: list[str], args, started: datetime) -> int:
    path = args.salida or ROOT / "logs" / f"rendimiento_{started:%Y%m%d_%H%M}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Informe guardado en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
