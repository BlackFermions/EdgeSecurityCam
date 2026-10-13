"""CamDetector: alarmas de la cámara verificadas con YOLO y guardadas en SQLite.

Ejemplos:
    python escuchar_alarmas.py --camera 192.168.100.109
    python escuchar_alarmas.py --camera 192.168.100.109 --camera 192.168.100.110
    python escuchar_alarmas.py --raw --log-file logs\\camdetector.log
    python escuchar_alarmas.py --no-yolo

Sin --camera busca la cámara por MAC con ONVIF. Detener con Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

# Sin esto los hilos de PyTorch siguen ocupando CPU tras cada inferencia
# (espera activa). Debe definirse antes de importar torch.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from src.sources.camera_events import MOTION_MODES  # noqa: E402
from src.credentials import read_credentials  # noqa: E402
from src.sources.discovery import DEFAULT_CAMERA_MAC, resolve_camera_host  # noqa: E402
from src.node import CamDetectorNode  # noqa: E402
from src.settings import SettingsError, load_settings  # noqa: E402
from src.storage import NodeStore  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "yolo26n.pt"
DEFAULT_DB = BASE_DIR / "data" / "camdetector.db"
DEFAULT_CONFIG = BASE_DIR / "config" / "camdetector.toml"

# Opciones de la línea de comandos que sustituyen un parámetro del archivo de
# configuración (para todas las cámaras): opción -> (grupo, nombre).
CLI_OVERRIDES = {
    "config_interval": ("camara", "config_cada_s"),
    "motion_mode": ("camara", "modo_movimiento"),
    "yolo_confidence": ("yolo", "confianza"),
    "yolo_size": ("yolo", "tamano"),
    "idle_check_seconds": ("reposo", "cada_s"),
    "idle_min_confidence": ("reposo", "confianza_min"),
}

log = logging.getLogger("alarmas")


def configure_logging(log_file: Path | None, verbose: bool) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def load_verifier(args, hosts: list[str], username: str, password: str, viewer=None,
                  settings=None):
    # Import diferido: sin YOLO el receptor no necesita ultralytics.
    from src.analysis.session import AnalysisManager
    from src.detection.yolo import YoloPersonDetector
    from src.sources.video import rtsp_url

    model = args.model
    if not Path(model).exists():
        model = Path(model).name    # ultralytics lo descarga
    log.info("Cargando YOLO (%s)...", Path(model).name)
    try:
        detector = YoloPersonDetector(model)    # parámetros por cámara en cada llamada
    except Exception as error:      # noqa: BLE001 - se informa y se sigue sin YOLO
        log.error("YOLO no disponible (%s: %s); se registran solo las alarmas.",
                  type(error).__name__, error)
        return None
    log.info("YOLO listo: verifica cada alarma y sigue a las personas.")
    return AnalysisManager(
        detector, {host: rtsp_url(host, username, password) for host in hosts},
        args.snapshots_dir, viewer=viewer, settings=settings)


def main() -> int:
    parser = argparse.ArgumentParser(description="CamDetector: alarmas verificadas con YOLO")
    parser.add_argument("--camera", action="append", default=[],
                        help="IP de una cámara; repetir para varias")
    parser.add_argument("--camera-mac", default=DEFAULT_CAMERA_MAC,
                        help="MAC a buscar si no se indica --camera")
    parser.add_argument("--credentials-file", type=Path,
                        default=BASE_DIR / "claves.txt")
    parser.add_argument("--log-file", type=Path, help="Copia el registro a un archivo")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Base de datos SQLite del nodo")
    parser.add_argument("--no-db", action="store_true", help="No guarda en la base")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Parámetros por cámara (TOML); ver docs/parametros.md")
    parser.add_argument("--config-interval", type=float, default=None,
                        help="Segundos entre lecturas de la configuración de la cámara")
    parser.add_argument("--motion-mode", choices=MOTION_MODES, default=None,
                        help="Cómo interpretar el tema de movimiento: auto según "
                             "el modelo y Motion Detect, human o motion")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Solo registra las alarmas, sin verificar con YOLO")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Modelo YOLO (por defecto yolo26n)")
    parser.add_argument("--yolo-confidence", type=float, default=None,
                        help="Confianza mínima de YOLO para contar una persona")
    parser.add_argument("--idle-check-seconds", type=float, default=None,
                        help="Verificación en reposo: 1 fotograma cada N s aunque la "
                             "cámara no avise (0 la desactiva)")
    parser.add_argument("--idle-min-confidence", type=float, default=None,
                        help="Confianza mínima de YOLO para abrir sesión desde reposo")
    parser.add_argument("--yolo-size", type=int, default=None,
                        help="Tamaño de entrada de YOLO (320, 416 o 640)")
    parser.add_argument("--snapshots-dir", type=Path, default=BASE_DIR / "capturas",
                        help="Carpeta de las fotos de cada alarma con personas")
    parser.add_argument("--ver", action="store_true",
                        help="Abre una ventana con el vídeo analizado en tiempo real")
    parser.add_argument("--raw", action="store_true",
                        help="Muestra también cada evento recibido, no solo los cambios")
    args = parser.parse_args()
    configure_logging(args.log_file, args.raw)

    try:
        username, password = read_credentials(args.credentials_file)
    except ValueError as error:
        log.error("Credenciales: %s.", error)
        return 2

    hosts = args.camera
    if not hosts:
        host, source = resolve_camera_host("", args.camera_mac)
        if not host:
            log.error("No se encontró la cámara por ONVIF; indícala con --camera IP.")
            return 2
        log.info("Cámara encontrada: %s (%s).", host, source)
        hosts = [host]

    overrides = {CLI_OVERRIDES[option]: value for option, value in vars(args).items()
                 if option in CLI_OVERRIDES and value is not None}
    try:
        settings = load_settings(args.config, hosts, overrides)
    except SettingsError as error:
        log.error("Configuración %s: %s", args.config, error)
        return 2
    if args.config.exists():
        log.info("Parámetros: %s", args.config)
    else:
        log.info("Parámetros: valores por defecto (no existe %s; ver "
                 "config/camdetector.example.toml)", args.config)
    for host, camera_settings in settings.items():
        if camera_settings.sources:
            changed = ", ".join(f"{key}={camera_settings.get(*key.split('.'))}"
                                for key in sorted(camera_settings.sources))
            log.info("[%s] %s · ajustes: %s", host, camera_settings.name, changed)

    store = None
    if not args.no_db:
        store = NodeStore(args.db)
        log.info("Base de datos: %s", args.db)

    viewer = None
    if args.ver:
        if args.no_yolo:
            log.error("--ver necesita YOLO: quita --no-yolo.")
            return 2
        from src.sources.video import rtsp_url
        from src.outputs.viewer import LiveViewer
        viewer = LiveViewer(hosts, {host: rtsp_url(host, username, password)
                                    for host in hosts})
    verifier = (None if args.no_yolo
                else load_verifier(args, hosts, username, password, viewer, settings))
    if verifier is None and viewer is not None:
        viewer.close()
        viewer = None
    idle_sources = None
    if verifier is not None:
        from src.sources.snapshot import SnapshotSource
        from src.sources.video import rtsp_url
        idle_sources = {host: SnapshotSource(host, username, password,
                                             rtsp_url(host, username, password))
                        for host in hosts}

    node = CamDetectorNode(hosts, username, password, store, verifier=verifier,
                           settings=settings, model_name=Path(args.model).name,
                           base_dir=BASE_DIR, idle_sources=idle_sources)
    node.start()
    stop = threading.Event()
    try:
        if viewer is not None:
            log.info("Ventana abierta: q o Esc para cerrar CamDetector.")
            while viewer.refresh():
                pass
            log.info("Ventana cerrada.")
        else:
            # Esperas cortas: en Windows una espera sin límite no atiende Ctrl+C.
            while not stop.wait(1.0):
                pass
    except KeyboardInterrupt:
        log.info("Cerrando...")
    if viewer is not None:
        viewer.close()
    node.close()
    if store is not None:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
