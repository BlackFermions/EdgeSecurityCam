"""Muestras de rendimiento cada 10 s y resumen diario cada 5 min (hora de Perú).

CSV: logs/rendimiento/AAAA-MM-DD.csv. Resumen: AAAA-MM-DD.log.
No abre la cámara ni la base de detecciones. Los picos son valores observados,
no máximos instantáneos entre muestras. Reanuda las estadísticas desde el CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import psutil


ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("monitor")
FIELDS = (
    "momento", "intervalo_s", "cpu_pct", "nucleo_max_pct", "ram_pct", "ram_usada_mb",
    "ram_disponible_mb", "ram_total_mb", "swap_pct", "swap_usada_mb", "temperatura_c",
    "frecuencia_mhz", "carga_1m", "carga_5m", "carga_15m", "disco_pct", "disco_libre_gb",
    "camdetector_pid", "camdetector_cpu_pct", "camdetector_ram_mb", "throttled_hex",
    "uptime_s", "nucleos",
)
METRICS = {
    "cpu_pct": ("CPU total", "%"),
    "nucleo_max_pct": ("Núcleo más ocupado de cada muestra", "%"),
    "ram_pct": ("RAM utilizada", "%"),
    "ram_usada_mb": ("RAM utilizada (total menos disponible)", "MiB"),
    "ram_disponible_mb": ("RAM disponible", "MiB"),
    "swap_pct": ("Swap utilizada", "%"),
    "swap_usada_mb": ("Swap utilizada", "MiB"),
    "temperatura_c": ("Temperatura SoC/CPU", "°C"),
    "frecuencia_mhz": ("Frecuencia CPU", "MHz"),
    "carga_1m": ("Carga media del sistema (1 min)", ""),
    "carga_5m": ("Carga media del sistema (5 min)", ""),
    "carga_15m": ("Carga media del sistema (15 min)", ""),
    "disco_pct": ("Disco utilizado", "%"),
    "disco_libre_gb": ("Disco libre para el proyecto", "GiB"),
    "camdetector_cpu_pct": ("CPU de CamDetector (capacidad total)", "%"),
    "camdetector_ram_mb": ("RAM residente de CamDetector", "MiB"),
}
FLAGS = {0: "bajo voltaje", 1: "frecuencia limitada", 2: "throttling", 3: "límite térmico suave"}


def command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=2,
                                check=False, creationflags=0x08000000 if os.name == "nt" else 0)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def temperature() -> float | None:
    try:
        value = float(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
        return value if math.isfinite(value) and -50 <= value <= 150 else None
    except (OSError, ValueError):
        return None


def throttled() -> int | None:
    value = command(["vcgencmd", "get_throttled"])
    try:
        return int(value.split("=", 1)[1], 16) if value else None
    except (ValueError, IndexError):
        return None


class Sampler:
    def __init__(self, zone, disk_path: Path, service: str = "camdetector"):
        self.zone, self.disk_path, self.service = zone, disk_path, service
        self.cores = psutil.cpu_count() or 1
        self.process = None
        self.previous = time.monotonic()
        # La primera lectura no mide un intervalo real: se descarta.
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        self.process_sample()

    def process_sample(self):
        raw_pid = command(["systemctl", "show", self.service, "--property=MainPID", "--value"])
        try:
            pid = int(raw_pid or 0)
            if pid <= 0:
                self.process = None
                return None, None, None
            fresh = self.process is None or self.process.pid != pid or not self.process.is_running()
            if fresh:
                self.process = psutil.Process(pid)
                self.process.cpu_percent(interval=None)
            cpu = None if fresh else self.process.cpu_percent(interval=None) / self.cores
            return pid, cpu, self.process.memory_info().rss / 2**20
        except (ValueError, psutil.Error):
            self.process = None
            return None, None, None

    def collect(self) -> dict:
        moment = time.monotonic()
        elapsed, self.previous = moment - self.previous, moment
        cpu = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        disk = psutil.disk_usage(str(self.disk_path))
        try:
            frequency = psutil.cpu_freq()
            mhz = frequency.current if frequency and frequency.current > 0 else None
        except (OSError, NotImplementedError):
            mhz = None
        try:
            loads = os.getloadavg()
        except (AttributeError, OSError):
            loads = (None, None, None)
        pid, process_cpu, rss = self.process_sample()
        flags = throttled()
        sample = dict(zip(FIELDS, (
            datetime.now(self.zone).isoformat(timespec="seconds"), elapsed, cpu,
            max(per_core) if per_core else None, memory.percent,
            (memory.total - memory.available) / 2**20, memory.available / 2**20,
            memory.total / 2**20, swap.percent, swap.used / 2**20, temperature(), mhz,
            *loads, disk.percent, disk.free / 2**30, pid, process_cpu, rss,
            hex(flags) if flags is not None else None,
            max(0, time.time() - psutil.boot_time()), self.cores,
        )))
        return {key: round(value, 3) if isinstance(value, float) else value
                for key, value in sample.items()}


@dataclass
class Metric:
    count: int = 0
    total: float = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    minimum_at: str = ""
    maximum_at: str = ""
    last: float = 0

    def add(self, value, moment: str):
        try:
            value = float(value)
        except (ValueError, TypeError):
            return
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.last = value
        if value < self.minimum:
            self.minimum, self.minimum_at = value, moment
        if value > self.maximum:
            self.maximum, self.maximum_at = value, moment


class DailyStats:
    def __init__(self, day: date):
        self.day, self.count, self.seconds = day, 0, 0.0
        self.first = self.last = ""
        self.metrics = {name: Metric() for name in METRICS}
        self.flag_samples = {bit: 0 for bit in FLAGS}
        self.flag_reads, self.history, self.cores = 0, 0, "-"

    def add(self, row: dict):
        if None in row or any(row.get(key) is None for key in FIELDS):
            # Los sensores no disponibles se serializan como cadena vacía;
            # None en una fila CSV indica una escritura incompleta.
            return
        try:
            moment = datetime.fromisoformat(row["momento"])
            elapsed = float(row["intervalo_s"])
            if moment.date() != self.day or not math.isfinite(elapsed) or elapsed <= 0:
                return
        except (ValueError, TypeError):
            return
        self.count += 1
        self.seconds += elapsed
        self.first = self.first or row["momento"]
        self.last, self.cores = row["momento"], row["nucleos"]
        for name, metric in self.metrics.items():
            metric.add(row.get(name), row["momento"])
        try:
            flags = int(row.get("throttled_hex") or "", 16)
        except (ValueError, TypeError):
            return
        self.flag_reads += 1
        self.history |= flags & 0xF0000
        for bit in FLAGS:
            self.flag_samples[bit] += bool(flags & (1 << bit))

    def render(self) -> str:
        lines = [
            f"RENDIMIENTO DIARIO - {self.day}",
            "Horas locales con desplazamiento UTC; zona por defecto America/Lima.",
            f"Primera muestra: {self.first or '-'}", f"Última muestra: {self.last or '-'}",
            f"Muestras: {self.count} | Intervalos observados: {self.seconds:.1f} s | Núcleos: {self.cores}",
            "CPU: 100% equivale a toda la capacidad del equipo; un núcleo tiene su propia escala 0-100%.",
            "CPU de CamDetector: proceso principal y sus hilos; normalizada a la capacidad total.",
            "CPU promediada en cada intervalo. Temperatura, RAM y disco: lecturas puntuales.",
            "Medias de muestras. Los picos son observados: pueden perderse eventos entre muestras.",
            "Un sensor sin lectura se muestra como n/d, nunca como cero.", "",
            "Métrica | Última | Media | Mínimo | Hora mínimo | Máximo | Hora máximo",
        ]
        for name, (label, unit) in METRICS.items():
            metric = self.metrics[name]
            if metric.count:
                lines.append(f"{label} ({unit}) | {metric.last:.2f} | {metric.total / metric.count:.2f} | "
                             f"{metric.minimum:.2f} | {metric.minimum_at} | "
                             f"{metric.maximum:.2f} | {metric.maximum_at}")
            else:
                lines.append(f"{label} ({unit}) | n/d")
        lines.extend(["", "ESTADO DE ALIMENTACIÓN Y LIMITACIÓN (vcgencmd get_throttled)",
                      f"Lecturas disponibles: {self.flag_reads}"])
        for bit, label in FLAGS.items():
            lines.append(f"Muestras con {label} activo: {self.flag_samples[bit]}"
                         if self.flag_reads else f"{label}: n/d")
        lines.append(f"Bits históricos observados: {hex(self.history) if self.flag_reads else 'n/d'}")
        lines.append("Los bits históricos son desde el arranque de la Raspberry, no eventos ocurridos necesariamente hoy.")
        return "\n".join(lines) + "\n"


class DailyLog:
    def __init__(self, folder: Path, summary_seconds: float = 300):
        self.folder, self.summary_seconds = folder, summary_seconds
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.stats = None
        self.written_at = None

    def _load(self, day: date):
        stats = DailyStats(day)
        path = self.folder / f"{day}.csv"
        if path.exists() and path.stat().st_size:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != list(FIELDS):
                    raise ValueError(f"Cabecera incompatible en {path}; no se sobrescribe.")
                for row in reader:
                    stats.add(row)
        self.stats, self.written_at = stats, None

    def append(self, sample: dict):
        day = datetime.fromisoformat(sample["momento"]).date()
        if self.stats is None or self.stats.day != day:
            self.flush()
            self._load(day)
        path = self.folder / f"{day}.csv"
        has_data = path.exists() and path.stat().st_size > 0
        if has_data:
            with path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                needs_newline = handle.read(1) != b"\n"
        else:
            needs_newline = False
        with path.open("a", encoding="utf-8", newline="") as handle:
            if needs_newline:
                handle.write("\n")
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if not has_data:
                writer.writeheader()
            writer.writerow(sample)
        # La misma representación que al reabrir el CSV, incluidos sensores ausentes.
        self.stats.add({key: "" if value is None else str(value) for key, value in sample.items()})
        if self.written_at is None or time.monotonic() - self.written_at >= self.summary_seconds:
            self.flush()

    def flush(self):
        if self.stats is None:
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.folder,
                                             prefix=".rendimiento-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(self.stats.render())
            os.replace(temporary, self.folder / f"{self.stats.day}.log")
            self.written_at = time.monotonic()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carpeta", type=Path, default=ROOT / "logs" / "rendimiento")
    parser.add_argument("--intervalo", type=float, default=10)
    parser.add_argument("--resumen-cada", type=float, default=300)
    parser.add_argument("--zona", default="America/Lima")
    parser.add_argument("--servicio", default="camdetector")
    parser.add_argument("--una-vez", action="store_true", help="Mide un intervalo de 1 s y termina")
    args = parser.parse_args()
    if not math.isfinite(args.intervalo) or args.intervalo < 1:
        parser.error("--intervalo debe ser finito y de al menos 1 segundo")
    if not math.isfinite(args.resumen_cada) or args.resumen_cada < 1:
        parser.error("--resumen-cada debe ser finito y de al menos 1 segundo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        zone = ZoneInfo(args.zona)
        output = DailyLog(args.carpeta, args.resumen_cada)
        # En Linux impide dos escritores sobre los mismos CSV diarios.
        with (args.carpeta / ".monitor.lock").open("a") as lock:
            if os.name == "posix":
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sampler = Sampler(zone, ROOT, args.servicio)
            log.info("Monitor iniciado: muestras cada %s s, resumen cada %s s; %s",
                     1 if args.una_vez else args.intervalo, args.resumen_cada, args.carpeta)
            try:
                if not stop.wait(1):
                    while not stop.is_set():
                        output.append(sampler.collect())
                        if args.una_vez or stop.wait(args.intervalo):
                            break
            finally:
                output.flush()
    except (OSError, ValueError, psutil.Error, ZoneInfoNotFoundError) as error:
        log.error("Monitor detenido: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
