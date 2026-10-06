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

from src.camera_events import MOTION_MODES  # noqa: E402
from src.config import read_credentials  # noqa: E402
from src.discovery import DEFAULT_CAMERA_MAC, resolve_camera_host  # noqa: E402
from src.node import CamDetectorNode  # noqa: E402
from src.storage import NodeStore  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "yolo26n.pt"
DEFAULT_DB = BASE_DIR / "data" / "camdetector.db"

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


def load_verifier(args, hosts: list[str], username: str, password: str):
    # Import diferido: sin YOLO el receptor no necesita ultralytics.
    from src.analysis import AnalysisManager
    from src.person_verifier import PersonDetector, rtsp_url

    model = args.model
    if not Path(model).exists():
        model = Path(model).name    # ultralytics lo descarga
    log.info("Cargando YOLO (%s)...", Path(model).name)
    try:
        detector = PersonDetector(model, confidence=args.yolo_confidence,
                                  image_size=args.yolo_size)
    except Exception as error:      # noqa: BLE001 - se informa y se sigue sin YOLO
        log.error("YOLO no disponible (%s: %s); se registran solo las alarmas.",
                  type(error).__name__, error)
        return None
    log.info("YOLO listo (%d px): verifica cada alarma y sigue a las personas.",
             args.yolo_size)
    return AnalysisManager(
        detector, {host: rtsp_url(host, username, password) for host in hosts},
        args.snapshots_dir)


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
    parser.add_argument("--config-interval", type=float, default=60.0,
                        help="Segundos entre lecturas de la configuración de la cámara")
    parser.add_argument("--motion-mode", choices=MOTION_MODES, default="auto",
                        help="Cómo interpretar el tema de movimiento: auto según "
                             "el modelo y Motion Detect, human o motion")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Solo registra las alarmas, sin verificar con YOLO")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Modelo YOLO (por defecto yolo26n)")
    parser.add_argument("--yolo-confidence", type=float, default=0.40,
                        help="Confianza mínima de YOLO para contar una persona")
    parser.add_argument("--yolo-size", type=int, default=416,
                        help="Tamaño de entrada de YOLO (320, 416 o 640)")
    parser.add_argument("--snapshots-dir", type=Path, default=BASE_DIR / "capturas",
                        help="Carpeta de las fotos de cada alarma con personas")
    parser.add_argument("--raw", action="store_true",
                        help="Muestra también cada evento recibido, no solo los cambios")
    args = parser.parse_args()
    configure_logging(args.log_file, args.raw)

    if args.yolo_size not in (320, 416, 480, 640):
        log.error("--yolo-size debe ser 320, 416, 480 o 640.")
        return 2
    if not 10.0 <= args.config_interval <= 3600.0:
        log.error("--config-interval debe estar entre 10 y 3600 s.")
        return 2
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

    store = None
    if not args.no_db:
        store = NodeStore(args.db)
        log.info("Base de datos: %s", args.db)

    verifier = None if args.no_yolo else load_verifier(args, hosts, username, password)
    ia = ({"modelo": Path(args.model).name, "confianza": args.yolo_confidence,
           "imgsz": args.yolo_size, "confirmar_fotogramas": 2, "decidir_s": 4.0,
           "seguimiento": "yolo+flujo-lk", "yolo_max_s_movimiento": 1.0,
           "yolo_max_s_quieto": 3.0, "disparo_min_s": 0.5}
          if verifier is not None else {"modelo": None})

    node = CamDetectorNode(hosts, username, password, store,
                           motion_mode=args.motion_mode, verifier=verifier, ia=ia,
                           config_interval=args.config_interval, base_dir=BASE_DIR)
    node.start()
    stop = threading.Event()
    try:
        # Esperas cortas: en Windows una espera sin límite no atiende Ctrl+C.
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        log.info("Cerrando...")
    node.close()
    if store is not None:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
